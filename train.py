import os
import torch
import patchmatch_cuda

from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim, patch_norm_mse_loss, patch_norm_mse_loss_global
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from lib.models.mvs import depth_propagation, check_geometric_consistency, read_propagted_depth, depth_propagation_old
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import numpy as np
import cv2
from lib.models.trellis import parallel_rasterize
from lib.models.street_gaussian_model import quaternion_raw_multiply, matrix_to_quaternion, quaternion_to_matrix
from lib.models.mvs import densify_bkgd_by_viewpoint, vggt_region_has_missing_depth, get_actor_pose_world
from lib.datasets.vggt_prediction_io import get_vggt_pixel_source_views
from densification import reconstruct_global, reconstruct_local, reconstruct_actor
############################
import time
import matplotlib.pyplot as plt
import threading
import psutil
from pynvml import *
import queue

marker_queue = queue.Queue()
USE_TRELLIS = True
############################
import gc
import shutil
from plyfile import PlyData, PlyElement
inverse_opacity = lambda x: np.log(x/(1-x))
inverse_scale = lambda x: np.log(x)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def monitor_resources(
    interval=1.0,
    log_file=None,
    gpu_id=0,
    stop_event=None,
    marker_queue=None,
):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_id)

    current_stage = "INIT"

    while stop_event is None or not stop_event.is_set():
        # ====== 处理阶段 marker ======
        while marker_queue is not None and not marker_queue.empty():
            current_stage = marker_queue.get()
            marker_line = (
                f"\n===== STAGE: {current_stage} "
                f"@ {time.strftime('%H:%M:%S')} ====="
            )
            # print(marker_line)
            if log_file:
                with open(log_file, "a") as f:
                    f.write(marker_line + "\n")

        # ====== 资源采样 ======
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        info = nvmlDeviceGetMemoryInfo(handle)

        util = nvmlDeviceGetUtilizationRates(handle)
        gpu = util.gpu
        mem_total = round((info.total // 1048576) / 1024)
        mem_process_used = round((info.used // 1048576) / 1024)

        line = (
            f"[{time.strftime('%H:%M:%S')}] "
            f"{current_stage:<20} | "
            f"CPU: {cpu:5.1f}% | "
            f"RAM: {mem:5.1f}% | "
            f"GPU: {gpu:5.1f}% | "
            f"GPU-USED: {mem_process_used:5.1f}G |"
            f"GPU-TOTAL: {mem_total:5.1f}G |"
        )

        # print(line)
        if log_file:
            with open(log_file, "a") as f:
                f.write(line + "\n")

        time.sleep(interval)

    nvmlShutdown()


def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger()

    marker_queue.put("Loading Dataset")
    dataset = Dataset()
    marker_queue.put("Lolding Model")
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    marker_queue.put("Lolding Scene")
    scene = Scene(gaussians=gaussians, dataset=dataset)
    marker_queue.put("Start Training")

    cams_per_frame = len(data_args.get("cameras", [0, 1, 2]))

    gaussians.training_setup()
    try:
        if cfg.loaded_iter == -1:
            loaded_iter = max((int(name[10:-4]) for name in os.listdir(cfg.trained_model_dir)
                               if name.startswith('iteration_') and name.endswith('.pth')), default=-1)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_full_stack = [] # for view ID consistency. Test visibility would not be set to zero during training.
    l1 = scene.getTrainCameras().copy()
    l2 = scene.getTestCameras().copy()
    train_view_ids = [camera.id for camera in l1]
    i,j = 0,0
    while (i < len(l1) and j < len(l2)):
        if l1[i].id < l2[j].id:
            viewpoint_full_stack.append(l1[i])
            i = i + 1     
        else:   
            viewpoint_full_stack.append(l2[j])
            j = j + 1
    while i < len(l1):
        viewpoint_full_stack.append(l1[i])
        i = i + 1   
    while j < len(l2):
        viewpoint_full_stack.append(l2[j])
        j = j + 1

    FULL_STACK_LENGTH = len(viewpoint_full_stack)
    DIVERSITY_THRES = 1e-5
    view_stack_iter = start_iter // len(scene.getTrainCameras())

    include_list = list(set(gaussians.model_name_id.keys()) - set(["background"]))

    for obj_name in gaussians.model_name_id:
        obj_model = getattr(gaussians, obj_name)
        if obj_model.grape_trellis is not None:
            obj_model.grape_trellis.set_param(dataset.scene_info.metadata["c2ws"], dataset.scene_info.metadata["ixts"], data_args.selected_frames, cams_per_frame=cams_per_frame)
    N_bkgd_init = gaussians.background.get_xyz.shape[0]

    viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            view_stack_iter += 1
        # only for debug. 后续关闭
        # viewpoint_cam = viewpoint_full_stack[143]
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        randidx = viewpoint_cam.id

        gt_image = viewpoint_cam.original_image
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image
        loss_mask = viewpoint_cam.guidance['mask'] if 'mask' in viewpoint_cam.guidance else torch.ones_like(gt_image[0:1]).bool()
        loss_mask = loss_mask.cuda(non_blocking=True) if not loss_mask.is_cuda else loss_mask
        if 'lidar_depth' in viewpoint_cam.guidance:
            lidar_depth = viewpoint_cam.guidance['lidar_depth']
            lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
        if 'sky_mask' in viewpoint_cam.guidance:
            sky_mask = viewpoint_cam.guidance['sky_mask']
            sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
        # if 'obj_bound' in viewpoint_cam.guidance:
        #     obj_bound = viewpoint_cam.guidance['obj_bound']
        #     obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound
        if 'dynamic_mask' in viewpoint_cam.guidance:
            dynamic_mask = viewpoint_cam.guidance['dynamic_mask']
            dynamic_mask = dynamic_mask.cuda(non_blocking=True) if not dynamic_mask.is_cuda else dynamic_mask
        if "seg_bkgd" in viewpoint_cam.guidance:
            seg_bkgd_mask = viewpoint_cam.guidance['seg_bkgd']

        if "mono_depth" in viewpoint_cam.guidance:
            mono_depth = viewpoint_cam.guidance['mono_depth']
            mono_depth = mono_depth.cuda(non_blocking=True) if not mono_depth.is_cuda else mono_depth
        if "mono_normal" in viewpoint_cam.guidance:
            mono_normal = viewpoint_cam.guidance['mono_normal']
            mono_normal = mono_normal.cuda(non_blocking=True) if not mono_normal.is_cuda else mono_normal
            # if not viewpoint_cam.guidance['mono_normal'].is_cuda:
            #     viewpoint_cam.guidance['mono_normal'] = viewpoint_cam.guidance['mono_normal'].cuda(non_blocking=True)

        current_view, img_H, img_W = randidx, gt_image.shape[1], gt_image.shape[2]
        if "bkgd_voxel" not in viewpoint_cam.guidance: # zyk: fixed here, not growing for now.
            voxel_depth_value, voxel_depth_source, mask_visible, uvs = gaussians.background.grape_trellis.render_voxel_depth(current_view, img_H, img_W, K=viewpoint_cam.K)
            viewpoint_cam.guidance["bkgd_voxel"] = (voxel_depth_value.astype(np.float16), voxel_depth_source.astype(np.int32), mask_visible, uvs.astype(np.int16))

        # flag_global_reconstruct = False
        flag_local_reconstruct = False
        flag_actor_reconstruct = False

        ###################### hard depth #######################
        if view_stack_iter % optim_args.propagation_interval == 0 and iteration > optim_args.propagated_iteration_begin and iteration < optim_args.propagated_iteration_end:
            soft_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
            image = soft_render_pkg["rgb"]
            acc = soft_render_pkg['acc']

            _similarity = ssim(image, gt_image, mask=loss_mask)

            ######################################################## BACKGROUND CONTINUOUS ###############################################################################
            # 1. 整体过差时走现有 global 调用；本分支不进行局部采样。
            if _similarity < 0.6:
                reconstruct_global(
                    viewpoint_cam, viewpoint_full_stack, dataset, gaussians, gaussians_renderer, cams_per_frame, iteration, optim_args, gt_image, sky_mask, dynamic_mask, mono_normal, soft_render_pkg
                )

            else:
                # 2. 用共用随机 offset 生成采样网格，再读取其 segment / 动态 ID。
                # 每次检查重新抽样，仅限制总点数，不遍历整图 segment。
                grid_size, max_samples = 32, 512
                with torch.no_grad():
                    segment_img = seg_bkgd_mask.detach().cpu().numpy().transpose(1, 2, 0)
                    dynamic_img = dynamic_mask.detach().cpu().numpy()
                    background_pixels = np.all(dynamic_img == 255, axis=0) & ~sky_mask[0].cpu().numpy() 
                    # & ~np.all(segment_img == 0, axis=2)
                    # 动态区域不依赖背景 segment；背景跳过无效的分割颜色。
                    grid_origin_x, grid_origin_y = np.meshgrid(
                        np.arange(0, img_W, grid_size), np.arange(0, img_H, grid_size))
                    dx, dy = np.random.randint(0, grid_size, size=2)
                    sample_x = (grid_origin_x + dx).ravel()
                    sample_y = (grid_origin_y + dy).ravel()
                    # 边缘越界点及落在无效区域的点直接丢弃，不补采样。
                    inside = (sample_x < img_W) & (sample_y < img_H)
                    sample_x, sample_y = sample_x[inside], sample_y[inside]
                    valid = background_pixels[sample_y, sample_x]
                    sample_x, sample_y = sample_x[valid], sample_y[valid]
                    if len(sample_x) > max_samples:
                        selected_id = np.random.choice(len(sample_x), max_samples, replace=False)
                        sample_x, sample_y = sample_x[selected_id], sample_y[selected_id]

                    # 3. 与采样坐标一一对应：-1 未知，0 未缺失，1 缺失。
                    # 当前只有背景 voxel；actor 点保留未知，避免用静态重投影判断运动物体。
                    sample_vacancy_state = torch.full((len(sample_x),), -1, dtype=torch.int8, device=image.device)
                    if len(sample_x):
                        sample_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians) # , render_type="hard_depth")
                        sample_depth_map = sample_render_pkg.get('depth', torch.zeros_like(sample_render_pkg['acc']))[0]
                        sample_depth = sample_depth_map[sample_y, sample_x]

                        sample_voxel_depth = torch.as_tensor(
                            viewpoint_cam.guidance['bkgd_voxel'][0][sample_y, sample_x],
                            device=image.device, dtype=sample_depth.dtype)

                        sample_voxel_comparable = sample_voxel_depth > 0 # & sample_render_valid
                        sample_vacancy_state[sample_voxel_comparable] = (
                            sample_depth[sample_voxel_comparable] > sample_voxel_depth[sample_voxel_comparable] * 1.1
                        ).to(torch.int8)

                        # # Debug Visualization
                        # fig, ax = plt.subplots(figsize=(14, 8))
                        # ax.imshow(gt_image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
                        # for state, label, color, marker in [(-1, "Unknown", "yellow", "x"), (0, "Not missing", "lime", "+"), (1, "Missing", "red", "o")]:
                        #     selected = sample_vacancy_state.detach().cpu().numpy() == state
                        #     style = {"facecolors": "none", "edgecolors": color} if marker == "o" else {"color": color}
                        #     ax.scatter(sample_x[selected], sample_y[selected], marker=marker, s=45, linewidths=1.3, label=f"{label}: {selected.sum()}", **style)
                        # ax.set_title(f"View {viewpoint_cam.id} | Iteration {iteration}")
                        # ax.legend()
                        # ax.set_axis_off()
                        # plt.tight_layout()
                        # plt.show()

                        def check_prior_depth_reprojection(ref_uv, ref_depth, ref_K, ref_w2c,
                                                           prior_src_uv, src_K, src_w2c,
                                                           pixel_error_threshold=3.0):
                            """比较深度投影位置与 prior 对应像素，返回 (可比较 mask, 像素误差过大 mask)。"""
                            uv_h = torch.cat((ref_uv, torch.ones_like(ref_uv[:, :1])), dim=1)
                            xyz_ref = (torch.linalg.inv(ref_K) @ uv_h.T).T * ref_depth[:, None]
                            ref_to_src = src_w2c @ torch.linalg.inv(ref_w2c)
                            xyz_src = xyz_ref @ ref_to_src[:3, :3].T + ref_to_src[:3, 3]
                            projected = xyz_src @ src_K.T
                            uv_src = projected[:, :2] / projected[:, 2:3]
                            comparable = (
                                torch.isfinite(ref_uv).all(dim=1)
                                & torch.isfinite(ref_depth)
                                & (ref_depth > 0)
                                & torch.isfinite(prior_src_uv).all(dim=1)
                                & (prior_src_uv >= 0).all(dim=1)
                                & torch.isfinite(uv_src).all(dim=1)
                                & (xyz_src[:, 2] > 0)
                            )
                            pixel_error = torch.linalg.norm(uv_src - prior_src_uv, dim=1)
                            return comparable, comparable & (pixel_error > pixel_error_threshold)

                        # 只对没有 voxel 的采样点查询 prior；几何函数拒绝无效 ref 深度。
                        sample_prior_candidates = (~sample_voxel_comparable).cpu().numpy()
                        sample_voxel_info = dataset.scene_info.metadata.get('vggt_voxel_info')
                        sample_allowed_views = set(train_view_ids)
                        sample_ids_by_view = {}
                        debug_coview_counts = np.zeros(len(sample_x), dtype=np.int32)
                        for sample_id in np.flatnonzero(sample_prior_candidates):
                            source_views = get_vggt_pixel_source_views(
                                viewpoint_cam, sample_voxel_info, int(sample_x[sample_id]), int(sample_y[sample_id]))
                            # 调试数量：排除 ref，保留 train/test，区分 prior 本就无共视与后续过滤失败。
                            debug_coview_counts[sample_id] = np.count_nonzero(source_views != viewpoint_cam.id)
                            for source_id in source_views:
                                source_id = int(source_id)
                                if source_id != viewpoint_cam.id and source_id in sample_allowed_views:
                                    sample_ids_by_view.setdefault(source_id, []).append(sample_id)

                        sample_prior_valid_votes = torch.zeros(len(sample_x), dtype=torch.int32, device=sample_depth.device)
                        sample_prior_error_votes = torch.zeros_like(sample_prior_valid_votes)
                        if sample_ids_by_view:
                            sample_uv = torch.as_tensor(np.column_stack((sample_x, sample_y)), device=sample_depth.device, dtype=sample_depth.dtype)
                            sample_prior_voxel_ids = viewpoint_cam.guidance['vggt_voxel_id'][sample_y, sample_x].cpu().numpy()
                            # 每个源帧只扫描一次 ID 图，以同 ID 像素的中位位置作为 prior 对应点。
                            for source_id, sample_ids in sample_ids_by_view.items():
                                source_camera = viewpoint_full_stack[source_id]
                                source_ids = torch.as_tensor(sample_ids, device=sample_depth.device, dtype=torch.long)
                                source_voxel_ids = source_camera.guidance['vggt_voxel_id'].cpu().numpy()
                                source_static = (~source_camera.guidance['sky_mask'][0].cpu().numpy()
                                                 & torch.all(source_camera.guidance['dynamic_mask'] == 255, dim=0).cpu().numpy())
                                source_y, source_x = np.nonzero(
                                    np.isin(source_voxel_ids, sample_prior_voxel_ids[sample_ids]) & source_static)
                                if not len(source_x):
                                    continue
                                observed_ids = source_voxel_ids[source_y, source_x]
                                order = np.argsort(observed_ids)
                                voxel_ids, starts, counts = np.unique(observed_ids[order], return_index=True, return_counts=True)
                                source_uv_by_voxel = {
                                    int(voxel_id): (np.median(source_x[order[start:start + count]]),
                                                    np.median(source_y[order[start:start + count]]))
                                    for voxel_id, start, count in zip(voxel_ids, starts, counts)
                                }
                                prior_source_uv = torch.as_tensor(
                                    [source_uv_by_voxel.get(int(voxel_id), (np.nan, np.nan))
                                     for voxel_id in sample_prior_voxel_ids[sample_ids]],
                                    device=sample_depth.device, dtype=sample_depth.dtype)
                                source_pair_valid, source_inconsistent = check_prior_depth_reprojection(
                                    sample_uv[source_ids], sample_depth[source_ids], viewpoint_cam.K,
                                    viewpoint_cam.world_view_transform.T, prior_source_uv,
                                    source_camera.K, source_camera.world_view_transform.T, pixel_error_threshold=10)
                                sample_prior_valid_votes[source_ids] += source_pair_valid
                                sample_prior_error_votes[source_ids] += source_inconsistent

                        # 至少两个共视帧同意且占严格多数；证据不足或平票保持未知。
                        sample_prior_consistent_votes = sample_prior_valid_votes - sample_prior_error_votes
                        sample_prior_missing = (sample_prior_error_votes >= 2) & (sample_prior_error_votes * 2 > sample_prior_valid_votes)
                        sample_prior_not_missing = (sample_prior_consistent_votes >= 2) & (sample_prior_consistent_votes * 2 > sample_prior_valid_votes)
                        sample_vacancy_state[sample_prior_missing] = 1
                        sample_vacancy_state[sample_prior_not_missing] = 0

                        # # Debug: 仅显示 prior 候选点，颜色表示像素重投影误差过大的源视角数。
                        # debug_support = sample_prior_error_votes.cpu().numpy()
                        # debug_valid = sample_prior_valid_votes.cpu().numpy() > 0
                        # debug_known = sample_prior_candidates & debug_valid
                        # debug_unknown = sample_prior_candidates & ~debug_valid
                        # fig, ax = plt.subplots(figsize=(14, 8))
                        # ax.imshow(gt_image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
                        # ax.scatter(sample_x[debug_unknown], sample_y[debug_unknown],
                        #            color='gray', marker='x', s=40, label='No valid comparison')
                        # debug_max = max(1, int(debug_support[sample_prior_candidates].max(initial=0)))
                        # points = ax.scatter(sample_x[debug_known], sample_y[debug_known],
                        #                     c=debug_support[debug_known], s=45, edgecolors='black', linewidths=0.4,
                        #                     cmap=plt.get_cmap('turbo', debug_max + 1), vmin=-0.5, vmax=debug_max + 0.5)
                        # fig.colorbar(points, ax=ax, label='Views with large pixel reprojection error')
                        # debug_has_coview = sample_prior_candidates & (debug_coview_counts > 0)
                        # debug_no_coview = sample_prior_candidates & (debug_coview_counts == 0)
                        # debug_coview_max = max(1, int(debug_coview_counts[sample_prior_candidates].max(initial=0)))
                        # debug_coview_norm = plt.Normalize(vmin=0.5, vmax=debug_coview_max + 0.5)
                        # debug_coview_cmap = plt.get_cmap('viridis', debug_coview_max)
                        # ax.scatter(sample_x[debug_has_coview], sample_y[debug_has_coview],
                        #            facecolors='none', edgecolors=debug_coview_cmap(debug_coview_norm(debug_coview_counts[debug_has_coview])),
                        #            marker='o', s=150, linewidths=1.5, label='Prior co-visible count (ring)')
                        # ax.scatter(sample_x[debug_no_coview], sample_y[debug_no_coview],
                        #            facecolors='none', edgecolors='magenta', marker='o', s=150, linewidths=1.8,
                        #            label=f'0 prior co-visible views: {debug_no_coview.sum()}')
                        # fig.colorbar(plt.cm.ScalarMappable(norm=debug_coview_norm, cmap=debug_coview_cmap),
                        #              ax=ax, label='Prior co-visible views (excluding ref)')
                        # ax.set_title(f'Prior check | View {viewpoint_cam.id} | Iteration {iteration}')
                        # ax.legend()
                        # ax.set_axis_off()
                        # plt.tight_layout()
                        # plt.show()

            if _similarity < 0.8:

                hard_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, render_type="hard_depth")
                hard_depth = hard_render_pkg["depth"][0]

                voxel_depth_value, voxel_depth_source, mask_visible, uvs = viewpoint_cam.guidance["bkgd_voxel"]
                voxel_depth_tensor = torch.from_numpy(voxel_depth_value).cuda()

                m1 = (voxel_depth_tensor > 0) & (hard_depth > voxel_depth_tensor * 1.1) # 有初值。但是空了 missing points
                if _similarity < 0.6 or m1.sum() > (voxel_depth_tensor > 0).sum() * 0.5:
                    flag_global_reconstruct = True
                if m1.sum() > (voxel_depth_tensor > 0).sum() * 0.2:
                    flag_local_reconstruct = True

                bkgd_mask = ~sky_mask & ~torch.any(dynamic_mask != 255, axis=0, keepdim=True)
                k = int(len(acc[bkgd_mask]) * 0.25)
                v = acc[bkgd_mask].kthvalue(k).values

                if v.item() < 0.7:
                    flag_global_reconstruct = True

                if v.item() < 0.9:
                    flag_local_reconstruct = True

                obj_render_pkg = gaussians_renderer.render_object(viewpoint_cam, gaussians) #, parse_camera_again=True)
                obj_acc = obj_render_pkg["acc"]

                actor_mask = torch.any(dynamic_mask != 255, axis=0, keepdim=True)

                if len(obj_acc[actor_mask]) > 400:
                    k = int(len(obj_acc[actor_mask]) * 0.25)
                    v = obj_acc[actor_mask].kthvalue(k).values
                    if v.item() < 0.7:
                        flag_actor_reconstruct = True

        with torch.no_grad():

            ######################################################## BACKGROUND CONTINUOUS ###############################################################################
            # if flag_global_reconstruct:
            #     reconstruct_global(viewpoint_cam, viewpoint_full_stack, dataset, gaussians, gaussians_renderer,
            #         cams_per_frame, iteration, optim_args, gt_image, sky_mask, dynamic_mask, mono_normal, soft_render_pkg)

            ##################################### BACKGROUND DISCIRETE ##########################################################
            if flag_local_reconstruct and not optim_args.skip_view_selection: #  or iteration > FULL_STACK_LENGTH * 5 and iteration < FULL_STACK_LENGTH * 20 and (iteration % optim_args.propagation_interval == 0):

                voxel_depth_value, voxel_depth_source, mask_visible, uvs = viewpoint_cam.guidance["bkgd_voxel"]

                segment_img = seg_bkgd_mask.cpu().detach().numpy().transpose(1,2,0)
                no_bkgd_mask = np.logical_or(sky_mask.cpu().detach().numpy()[0], torch.any(dynamic_mask != 255, axis=0).cpu().detach().numpy())

                if USE_TRELLIS:
                    vacancy_exist, vacancy_colors = gaussians.background.grape_trellis.if_vacancy_in_ref_view_masks(voxel_depth_value, segment_img, no_bkgd_mask, vacancy_threshold=0.5)
                else:
                    vacancy_colors = []
                    voxel_info = dataset.scene_info.metadata.get('vggt_voxel_info')
                    if voxel_info is not None and 'vggt_voxel_id' in viewpoint_cam.guidance:
                        soft_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
                        # ref_depth = soft_render_pkg['depth'][0] / soft_render_pkg['acc'][0].clamp_min(1e-6)
                        ref_depth = soft_render_pkg['depth'][0]
                        source_depth_cache = {}
                        for c in np.unique(segment_img.reshape(-1, 3), axis=0):
                            if np.all(c == 0) or np.all(c == 1):
                                continue
                            obj_mask = np.all(segment_img == c, axis=2) & ~no_bkgd_mask
                            if obj_mask.sum() < 100 * 100:
                                continue
                            if vggt_region_has_missing_depth(
                                    obj_mask, viewpoint_cam, ref_depth, voxel_info,
                                    viewpoint_full_stack, gaussians, gaussians_renderer,
                                    source_depth_cache, train_view_ids):
                                vacancy_colors.append(c)
                    vacancy_exist = bool(vacancy_colors)
                if vacancy_exist:
                    # only for debug. 后续关闭
                    soft_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians) #, render_type="hard_depth")

                    rendered_depth = soft_render_pkg['depth'][0]
                    rendered_normal = soft_render_pkg['normals']

                    for c in vacancy_colors:
                        obj_mask = np.all(segment_img == c, axis=2) & ~no_bkgd_mask
                        use_trellis = USE_TRELLIS
                        if use_trellis:
                            selected_vals = obj_mask[uvs[mask_visible, 1], uvs[mask_visible, 0]]
                            mask_visible_obj = mask_visible.copy()
                            mask_visible_obj[mask_visible] &= selected_vals
                            if mask_visible_obj.sum() < 10:
                                continue
                            ref_src_views, viewset_diversity_score, _ = gaussians.background.grape_trellis.sample_viewset_from_obj_mask(
                                current_view, mask_visible_obj, point_obs_ratio=0.5, N=4)
                        else:
                            voxel_info = dataset.scene_info.metadata.get('vggt_voxel_info')
                            ref_src_views, viewset_diversity_score = gaussians.background.grape_trellis.sample_viewset_from_vggt_mask(
                                current_view, obj_mask, viewpoint_cam, voxel_info,
                                point_obs_ratio=0.5, N=4, allowed_view_ids=train_view_ids,
                            )

                        # # Debug: reference first, followed by selected prior source views.
                        # fig, axes = plt.subplots(1, len(ref_src_views), figsize=(5 * len(ref_src_views), 4), squeeze=False)
                        # for i, view_id in enumerate(ref_src_views):
                        #     camera = viewpoint_full_stack[view_id]
                        #     axes[0, i].imshow(camera.original_image.detach().permute(1, 2, 0).cpu().numpy())
                        #     axes[0, i].set_title(f"{'ref' if i == 0 else 'src'} {view_id}\n{camera.image_name}")
                        #     axes[0, i].axis('off')
                        # fig.tight_layout()
                        # plt.show()

                        reconstruct_local(ref_src_views, viewpoint_cam, viewpoint_full_stack, dataset, gaussians,
                            gaussians_renderer, mono_normal, soft_render_pkg, reconstruction_mask=obj_mask,
                            iteration=iteration, optim_args=optim_args, gt_image=gt_image,
                            sky_mask=sky_mask, dynamic_mask=dynamic_mask)

            ##################################################### OBJECT #######################################################################################
            if flag_actor_reconstruct and not optim_args.skip_view_selection: # or iteration > FULL_STACK_LENGTH * 0 and iteration < FULL_STACK_LENGTH * 20 and (iteration % optim_args.propagation_interval == 0):
                for dynamic_key in torch.unique(dynamic_mask):
                    if dynamic_key == 255:
                        continue

                    dynamic_id = dynamic_key.item()
                    obj_name = "obj_%.3d"%dynamic_id
                    if obj_name not in include_list:
                        continue

                    obj_render_pkg = gaussians_renderer.render_object(viewpoint_cam, gaussians)

                    rendered_normal = obj_render_pkg['normals']
                    render_acc = obj_render_pkg['acc']

                    if obj_name not in gaussians.graph_obj_list:
                        continue

                    obj_model = getattr(gaussians, obj_name)
                    if obj_model.random_initialization or obj_model.deformable: # should be enough
                        continue

                    obj_mask = torch.any(dynamic_mask==dynamic_id, axis=0)
                    obj_acc = render_acc[:, obj_mask]
                    if (obj_acc < 0.5).sum() < (obj_acc > 0.9).sum() * 0.1:
                        continue

                    track_id = obj_model.track_id
                    obj_views = dataset.scene_info.metadata["obj_view_dict"][track_id]
                    if randidx not in obj_views:
                        continue
                    # Keep the reference ActorPose for projection, consistency and insertion.
                    ref_o2w = get_actor_pose_world(gaussians.actor_pose, track_id, viewpoint_cam)
                    obj_rots = ref_o2w[:3, :3].unsqueeze(0)
                    obj_trans = ref_o2w[:3, 3]

                    voxel_depth_value, voxel_depth_source, mask_visible, uvs = obj_model.grape_trellis.render_voxel_depth(current_view, img_H, img_W, K=viewpoint_cam.K, obj_rots=obj_rots, obj_trans=obj_trans)
                    if mask_visible.sum() < 10:
                        continue
                    # 过低的 point_obs_ratio 可能没有输出。按逻辑而言需要在此处加入更新voxel部分
                    ref_src_views, viewset_diversity_score, rootvine_xyz = obj_model.grape_trellis.sample_viewset_from_obj_mask(current_view, mask_visible, point_obs_ratio=0.4, N=4, obj_rots=obj_rots, obj_trans=obj_trans)

                    if len(ref_src_views) <= 4 or viewset_diversity_score < DIVERSITY_THRES:
                        continue

                    # 视角已在上方选定，重建函数只负责传播、法线更新和加点。
                    reconstruct_actor(ref_src_views, dynamic_id, viewpoint_cam, viewpoint_full_stack, dataset, gaussians,
                        gaussians_renderer, iteration, optim_args, gt_image,
                        sky_mask, dynamic_mask, mono_normal, obj_render_pkg)

        voxel_depth_value, voxel_depth_source, mask_visible, uvs = viewpoint_cam.guidance["bkgd_voxel"]
        voxel_depth_tensor = torch.from_numpy(voxel_depth_value).cuda()

        if iteration > optim_args.hard_depth_start and iteration < optim_args.hard_depth_end:
            loss_hard = 0
            hard_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, render_type="hard_depth")
            hard_depth = hard_render_pkg["depth"]

            patch_range = (min(hard_depth.shape[1], hard_depth.shape[2]) // 20, max(hard_depth.shape[1], hard_depth.shape[2]) // 10) # zyk: to be tuned
            mono_depth[sky_mask] = mono_depth[~sky_mask].mean() # zyk: check if works?
            hard_depth[sky_mask] = hard_depth[~sky_mask].mean().detach()

            loss_l2_dpt = patch_norm_mse_loss(hard_depth[None,...], mono_depth[None,...], randint(patch_range[0], patch_range[1]), 0.01)
            loss_hard += 1 * loss_l2_dpt

            loss_global = patch_norm_mse_loss_global(hard_depth[None,...], mono_depth[None,...], randint(patch_range[0], patch_range[1]), 0.01)
            loss_hard += 1 * loss_global

            loss_hard.backward()
            # Optimizer step
            if iteration < training_args.iterations:                
                gaussians.update_optimizer()

        soft_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image, acc, viewspace_point_tensor, visibility_filter, radii = soft_render_pkg["rgb"], soft_render_pkg['acc'], soft_render_pkg["viewspace_points"], soft_render_pkg["visibility_filter"], soft_render_pkg["radii"]

        scalar_dict = dict()

        # rgb loss
        Ll1 = l1_loss(image, gt_image, mask=loss_mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=loss_mask))

        # sky loss
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss

        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            obj_acc_loss = torch.where(torch.any(dynamic_mask != 255, axis=0), # obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss

        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0: 
            if optim_args.use_lidar_depth and lidar_depth is not None:           
                depth_mask = torch.logical_and((lidar_depth > 0.), loss_mask)
                expected_depth = soft_render_pkg['depth'] / (soft_render_pkg['acc'] + 1e-10)  
                depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                lidar_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = lidar_depth_loss
                loss += optim_args.lambda_depth_lidar * lidar_depth_loss

            if optim_args.use_voxel_depth:
                depth_mask = torch.logical_and((voxel_depth_tensor > 0.), loss_mask)
                expected_depth = soft_render_pkg['depth'] / (soft_render_pkg['acc'] + 1e-10)  
                depth_error = torch.abs((expected_depth[depth_mask] - voxel_depth_tensor[depth_mask[0]]))
                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                voxel_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = voxel_depth_loss
                loss += optim_args.lambda_depth_lidar * voxel_depth_loss

        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss

        if optim_args.normal_loss:
            if mono_normal is not None:
                rendered_normal = soft_render_pkg['normals'] 
                normal_gt = mono_normal # 
                if sky_mask is not None: # if viewpoint_cam.sky_mask is not None:
                    filter_mask = sky_mask.to(normal_gt.device).to(torch.bool)
                    normal_gt[(filter_mask.repeat(3, 1, 1))] = -10

                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()

                lambda_l1_normal = 0.02
                lambda_cos_normal = 0.02
                loss += lambda_l1_normal * l1_normal + lambda_cos_normal * cos_normal

                viewpoint_cam.guidance['mono_normal'] = mono_normal #.cpu()

        scalar_dict['loss'] = loss.item()

        loss.backward()

        iter_end.record()

        is_save_images = True
        if is_save_images and (iteration % 100 == 0):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(soft_render_pkg['depth'].detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            # row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            row1 = torch.cat([voxel_depth_tensor[None,:,:].repeat(3,1,1) / voxel_depth_tensor.max(), image_obj, mono_depth.repeat(3,1,1)], dim=2)
            # row1 = torch.cat([normal_gt/2+0.5, image_obj, soft_render_pkg['normals']/2+0.5], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")

        with torch.no_grad():

            # Log
            tensor_dict = dict()

            if iteration % 10 == 0:                    
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if np.isnan(ema_loss_for_log):
                    ema_loss_for_log = loss.item()

                ema_psnr_for_log = 0.4 * psnr(image, gt_image, loss_mask).mean().float().item() + 0.6 * ema_psnr_for_log
                if np.isnan(ema_psnr_for_log):
                    ema_psnr_for_log = psnr(image, gt_image, loss_mask).mean().float().item()

                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}",
                                          "SSIM": f"{ssim(image, gt_image):.{4}f}",
                                          "GS":  str(gaussians.background.get_xyz.shape[0])
                                          })
            progress_bar.update(1)
            # if iteration == training_args.iterations:
            #     progress_bar.close()

            # Save ply
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )

                        scalar_dict.update(scalars)
                        tensor_dict.update(tensors)

            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)

                ######################
                # viewpoint_camera = viewpoint_full_stack[60]
                # frame_id=randidx

                gaussians.set_visibility(list(set(gaussians.model_name_id.keys())))
                gaussians.parse_camera(camera=viewpoint_cam)

                xyz = gaussians.get_xyz.detach().cpu().numpy()    
                normals = np.zeros_like(xyz)

                f = gaussians.get_features.detach().transpose(1, 2).contiguous() # [n, 3, sh_degree]
                f_dc = f[..., :1].flatten(start_dim=1).cpu().numpy()
                f_rest = f[..., 1:].flatten(start_dim=1).cpu().numpy()
                opacities = gaussians.get_opacity.detach().cpu().numpy()
                opacities = np.clip(opacities, a_min=1e-6, a_max=1.-1e-6)
                opacities = inverse_opacity(opacities)

                scale = gaussians.get_scaling.detach().cpu().numpy()
                scale = inverse_scale(scale)

                rotation = gaussians.get_rotation.detach().cpu().numpy()

                l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
                # All channels except the 3 DC
                for i in range(f_dc.shape[1]):
                    l.append('f_dc_{}'.format(i))
                for i in range(f_rest.shape[1]):
                    l.append('f_rest_{}'.format(i))
                l.append('opacity')
                for i in range(scale.shape[1]):
                    l.append('scale_{}'.format(i))
                for i in range(rotation.shape[1]):
                    l.append('rot_{}'.format(i))
                dtype_full = [(attribute, 'f4') for attribute in l]

                elements = np.empty(xyz.shape[0], dtype=dtype_full)
                attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
                elements[:] = list(map(tuple, attributes))

                save_dir = os.path.join(cfg.model_path, 'viewer', f'iteration_{iteration}_{current_view:06d}')
                pointcloud_dir = os.path.join(save_dir, 'point_cloud', f'iteration_{iteration}')
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(pointcloud_dir, exist_ok=True)
                shutil.copyfile(os.path.join(cfg.model_path, 'cameras.json'), os.path.join(save_dir, 'cameras.json'))
                shutil.copyfile(os.path.join(cfg.model_path, 'cfg_args'), os.path.join(save_dir, 'cfg_args'))
                shutil.copyfile(os.path.join(cfg.model_path, 'input.ply'), os.path.join(save_dir, 'input.ply'))

                elements = PlyElement.describe(elements, 'vertex')
                PlyData([elements]).write(os.path.join(pointcloud_dir, 'point_cloud.ply'))


def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)


    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_resources,
        kwargs={
            "interval": 1.0,
            "log_file": os.path.join(cfg.record_dir, "resource.log"),
            "gpu_id": 0,
            "stop_event": stop_event,
            "marker_queue": marker_queue,
        },
        daemon=True,
    )
    os.makedirs(cfg.record_dir, exist_ok=True)
    monitor_thread.start()
    time_start=time.time()

    try:
        training()
    finally:
        stop_event.set()
        monitor_thread.join()

    time_end=time.time()
    print('time cost', time_end-time_start,'s')
    print('scene id:', cfg.workspace)

    # All done
    print("\nTraining complete.")
