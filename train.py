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
    render_cache = {}
    next_render_cache = {}
    for iteration in range(start_iter, training_args.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            # 当前轮只读取上一完整训练轮的结果，本轮渲染另存，避免混用新旧视角。
            render_cache = next_render_cache
            next_render_cache = {}
            viewpoint_stack = scene.getTrainCameras().copy()
            view_stack_iter += 1
        # only for debug. 后续关闭
        # viewpoint_cam = viewpoint_full_stack[21]
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

        background_mask = torch.all(dynamic_mask == 255, dim=0, keepdim=True) & ~sky_mask[0]

        ###################### hard depth #######################
        if view_stack_iter % optim_args.propagation_interval == 0 and iteration > optim_args.propagated_iteration_begin and iteration < optim_args.propagated_iteration_end and randidx in render_cache:
            soft_render_pkg = {key: value.to(gt_image.device) for key, value in render_cache[randidx].items()}
            image = soft_render_pkg["rgb"]
            acc = soft_render_pkg['acc']

            # 1. 共用网格采样：背景和 actor 使用同一组坐标、状态与补全半径。
            grid_size, max_samples = 32, 512
            completion_progress = np.clip(
                (iteration - optim_args.propagated_iteration_begin) /
                (optim_args.propagated_iteration_end - optim_args.propagated_iteration_begin), 0.0, 1.0)
            sample_radius = int(round(128 + (grid_size - 128) * completion_progress))
            background_mask_np = background_mask.detach().cpu().numpy()[0]
            grid_origin_x, grid_origin_y = np.meshgrid(
                np.arange(0, img_W, grid_size), np.arange(0, img_H, grid_size))
            dx, dy = np.random.randint(0, grid_size, size=2)
            sample_x = (grid_origin_x + dx).ravel()
            sample_y = (grid_origin_y + dy).ravel()
            inside = (sample_x < img_W) & (sample_y < img_H)
            sample_x, sample_y = sample_x[inside], sample_y[inside]
            valid = ~sky_mask[0, sample_y, sample_x].detach().cpu().numpy()
            sample_x, sample_y = sample_x[valid], sample_y[valid]
            if len(sample_x) > max_samples:
                selected_id = np.random.choice(len(sample_x), max_samples, replace=False)
                sample_x, sample_y = sample_x[selected_id], sample_y[selected_id]
            sample_background_np = background_mask_np[sample_y, sample_x]
            sample_background = torch.as_tensor(sample_background_np, device=image.device)
            sample_dynamic_ids = dynamic_mask[:, sample_y, sample_x].detach().cpu().numpy()
            # -1 未知，0 未缺失，1 缺失；重叠 actor 的各自状态另外保存。
            sample_vacancy_state = torch.full((len(sample_x),), -1, dtype=torch.int8, device=image.device)
            actor_sample_states = {}

            _similarity = ssim(image, gt_image, mask=loss_mask)

            # 2. 背景分支：global 与静态区域检测 → local 二选一。
            if sample_background_np.any():
                # 2.1 仅检测静态采样点：低 acc、voxel 深度、prior 重投影。
                with torch.no_grad():
                    segment_img = seg_bkgd_mask.detach().cpu().numpy().transpose(1, 2, 0)
                    sample_render_pkg = soft_render_pkg
                    # NOTE(depth): 补全缓存已存 depth/acc；低 acc 点单独按覆盖不足判缺失。
                    sample_depth_map = sample_render_pkg.get('depth', torch.zeros_like(sample_render_pkg['acc']))[0]
                    sample_depth = sample_depth_map[sample_y, sample_x]
                    sample_acc = sample_render_pkg['acc'][0, sample_y, sample_x]
                    # 低覆盖直接视为缺失；整图 acc 和背景 voxel 均不用于 actor 检测。
                    sample_low_acc = sample_background & (sample_acc < 0.5)
                    sample_vacancy_state[sample_low_acc] = 1

                    sample_voxel_depth = torch.as_tensor(
                        viewpoint_cam.guidance['bkgd_voxel'][0][sample_y, sample_x],
                        device=image.device, dtype=sample_depth.dtype)

                    # TODO(depth): 非低 acc 点仍未验证渲染深度，0/NaN 会被判为未缺失。
                    sample_voxel_comparable = sample_background & (sample_voxel_depth > 0) # & sample_render_valid
                    # 保留 voxel 是否存在的信息供视角选择，只对非低 acc 点进行深度比较。
                    sample_voxel_check = sample_voxel_comparable & ~sample_low_acc
                    sample_vacancy_state[sample_voxel_check] = (
                        sample_depth[sample_voxel_check] > sample_voxel_depth[sample_voxel_check] * 1.1
                    ).to(torch.int8)

                    def check_prior_depth_reprojection(ref_uv, ref_depth, ref_K, ref_w2c,
                                                       prior_src_uv, src_K, src_w2c,
                                                       pixel_error_threshold=3.0):
                        """比较深度投影位置与 prior 对应像素，返回 (可比较 mask, 像素误差过大 mask)。"""
                        uv_h = torch.cat((ref_uv, torch.ones_like(ref_uv[:, :1])), dim=1)
                        # NOTE(depth): ref_depth 为补全缓存中的归一化 camera-Z，低 acc 点已在候选筛选时排除。
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

                    # 只对没有 voxel 且非低 acc 的点进行 prior 检测，避免覆盖低 acc 的缺失判定。
                    sample_prior_candidates = (sample_background & ~sample_voxel_comparable & ~sample_low_acc).cpu().numpy()
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

                    # # Debug: 仅显示 prior 候选点（无 voxel 且 acc >= 0.5）；叉号颜色表示缓存 acc。
                    # debug_support = sample_prior_error_votes.cpu().numpy()
                    # debug_acc = sample_acc.detach().cpu().numpy()
                    # debug_valid = sample_prior_valid_votes.cpu().numpy() > 0
                    # debug_known = sample_prior_candidates & debug_valid
                    # debug_unknown = sample_prior_candidates & ~debug_valid
                    # fig, ax = plt.subplots(figsize=(14, 8))
                    # ax.imshow(gt_image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
                    # # ax.imshow(segment_img)
                    # acc_points = ax.scatter(sample_x[debug_unknown], sample_y[debug_unknown],
                    #                         c=debug_acc[debug_unknown], cmap='plasma', vmin=0, vmax=1,
                    #                         marker='x', s=55, linewidths=1.5, label='No valid comparison (acc)')
                    # fig.colorbar(acc_points, ax=ax, label='Cached render acc (cross)')
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

                    bg_count = sample_background.sum().float()
                    bg_missing_ratio = (sample_background & (sample_vacancy_state == 1)).sum().float() / bg_count
                    # 没有 voxel、没有有效 prior 比较，且覆盖不足。
                    bg_unsupported_low_acc_ratio = (sample_background & ~sample_voxel_comparable & (sample_prior_valid_votes == 0) & (sample_acc < 0.8)).sum().float() / bg_count

                    # 2.2.1 整体过差时直接进行背景 global 补全。
                    if _similarity < 0.6 or bg_missing_ratio > 0.3 or bg_unsupported_low_acc_ratio > 0.1:
                        reconstruct_global(
                            viewpoint_cam, viewpoint_full_stack, dataset, gaussians, render_cache, cams_per_frame, iteration, optim_args, gt_image, sky_mask, dynamic_mask, mono_normal, soft_render_pkg
                        )
                    # 2.2.2 按静态 segment 汇集缺失点、选择视角并调用 reconstruct_local。
                    elif not optim_args.skip_view_selection:
                        sampled_states = sample_vacancy_state.cpu().numpy()
                        sampled_has_voxel = sample_voxel_comparable.cpu().numpy()
                        sampled_colors, sampled_segment_ids = np.unique(
                            segment_img[sample_y, sample_x], axis=0, return_inverse=True)
                        _, local_voxel_sources, local_visible_voxels, _ = viewpoint_cam.guidance['bkgd_voxel']
                        local_visible_ids = np.flatnonzero(local_visible_voxels)
                        local_trellis = gaussians.background.grape_trellis
                        for segment_id, color in enumerate(sampled_colors):
                            segment_samples = (sampled_segment_ids == segment_id) & sample_background_np
                            segment_missing = segment_samples & (sampled_states == 1)
                            segment_known = segment_samples & (sampled_states >= 0)
                            if segment_missing.sum() < 2: # or segment_missing.sum() <= segment_known.sum() * 0.5:
                                continue
                            segment_mask = np.all(segment_img == color, axis=2) & background_mask_np
                            segment_viewsets = []

                            voxel_missing = segment_missing & sampled_has_voxel
                            if voxel_missing.any():
                                voxel_sources = np.unique(local_voxel_sources[sample_y[voxel_missing], sample_x[voxel_missing]])
                                voxel_sources = voxel_sources[voxel_sources >= 0]
                                voxel_selection = np.zeros_like(local_visible_voxels, dtype=np.bool_)
                                # depth_source 是可见体素子集内的索引，先还原为 trellis 全局索引。
                                voxel_selection[local_visible_ids[voxel_sources]] = True
                                if voxel_selection.any() and local_trellis.get_visibility()[voxel_selection].any():
                                    trellis_ref_src_views, _, _ = local_trellis.sample_viewset_from_obj_mask(
                                        current_view, voxel_selection, point_obs_ratio=0.5, N=4)
                                    trellis_ref_src_views = [current_view] + [int(v) for v in trellis_ref_src_views[1:] if v in sample_allowed_views]
                                    if len(trellis_ref_src_views) >= 3:
                                        segment_viewsets.append((trellis_ref_src_views, voxel_missing))

                            prior_missing = segment_missing & ~sampled_has_voxel
                            if prior_missing.any():
                                prior_selection = np.zeros((img_H, img_W), dtype=np.bool_)
                                prior_selection[sample_y[prior_missing], sample_x[prior_missing]] = True
                                prior_ref_src_views, _ = local_trellis.sample_viewset_from_vggt_mask(
                                    current_view, prior_selection, viewpoint_cam, sample_voxel_info,
                                    point_obs_ratio=0.5, N=4, allowed_view_ids=train_view_ids)
                                if len(prior_ref_src_views) >= 3:
                                    segment_viewsets.append((prior_ref_src_views, prior_missing))

                            # 两套视角暂分别调用；后续在此合并视角或划分补全区域。
                            for ref_src_views, missing_samples in segment_viewsets:
                                local_render_pkg = soft_render_pkg
                                # 1125 暂时考虑只使用square mask和bkgd mask(只剔除sky与dynamic)，object mask只是用于统合视角集合的来源。
                                reconstruct_local(
                                    ref_src_views, viewpoint_cam, viewpoint_full_stack, dataset, gaussians,
                                    render_cache, mono_normal, local_render_pkg, reconstruction_mask=background_mask_np, #segment_mask,
                                    missing_uv=np.column_stack((sample_x[missing_samples], sample_y[missing_samples])),
                                    iteration=iteration, optim_args=optim_args, gt_image=gt_image,
                                    sky_mask=sky_mask, dynamic_mask=dynamic_mask, sample_radius=sample_radius)

            # 3. Actor 分支：独立 acc 检测 → 共视帧选择 → reconstruct_actor。
            # 与背景 global/local 分支平级；只渲染当前 actor 自身的 Gaussian。
            if np.any(sample_dynamic_ids != 255):
                actor_previous_include = list(gaussians.include_list)
                actor_previous_camera = gaussians.viewpoint_camera
                with torch.no_grad():
                    try:
                        for dynamic_id in np.unique(sample_dynamic_ids):
                            dynamic_id = int(dynamic_id)
                            obj_name = f"obj_{dynamic_id:03d}"
                            if dynamic_id == 255 or obj_name not in gaussians.obj_list:
                                continue
                            actor_samples_np = np.any(sample_dynamic_ids == dynamic_id, axis=0)
                            actor_samples = torch.as_tensor(actor_samples_np, device=image.device)
                            gaussians.set_visibility([obj_name])
                            gaussians.parse_camera(viewpoint_cam)
                            if obj_name not in gaussians.graph_obj_list:
                                continue
                            obj_render_pkg = gaussians_renderer.render_kernel(viewpoint_cam, gaussians)
                            actor_acc = obj_render_pkg["acc"][0, sample_y[actor_samples_np], sample_x[actor_samples_np]]
                            actor_states = (actor_acc < 0.5).to(torch.int8)
                            actor_sample_states[dynamic_id] = (actor_samples_np, actor_states.cpu().numpy())
                            # 一个像素可能属于多个 actor；合并图中任一 actor 缺失就标为缺失。
                            sample_vacancy_state[actor_samples] = torch.maximum(sample_vacancy_state[actor_samples], actor_states)
                            actor_missing = actor_states.cpu().numpy() == 1
                            if actor_missing.sum() < 2 or actor_missing.mean() < 0.1 or optim_args.skip_view_selection:
                                continue

                            obj_model = getattr(gaussians, obj_name)
                            if obj_model.random_initialization or obj_model.deformable or obj_model.grape_trellis is None:
                                continue
                            obj_views = dataset.scene_info.metadata["obj_view_dict"][obj_model.track_id]
                            if current_view not in obj_views:
                                continue
                            # voxel 只提供共视关系，不投影 voxel，也不使用其坐标或深度。
                            visibility = obj_model.grape_trellis.get_visibility()
                            actor_visibility = visibility[visibility[:, current_view]]
                            if not len(actor_visibility):
                                continue
                            candidates = [int(v) for v in train_view_ids if v != current_view and v in obj_views and actor_visibility[:, v].any()]
                            ref_src_views = [current_view]
                            uncovered = np.ones(len(actor_visibility), dtype=np.bool_)
                            for _ in range(min(4, len(candidates))):
                                coverage = actor_visibility[uncovered][:, candidates].sum(axis=0)
                                if not coverage.any():
                                    coverage = actor_visibility[:, candidates].sum(axis=0)
                                selected = int(np.random.choice(np.flatnonzero(coverage == coverage.max())))
                                source_id = candidates.pop(selected)
                                ref_src_views.append(source_id)
                                uncovered &= ~actor_visibility[:, source_id]
                            if len(ref_src_views) < 3:
                                continue

                            reconstruct_actor(ref_src_views, dynamic_id, viewpoint_cam, viewpoint_full_stack, dataset, gaussians, gaussians_renderer, 
                                iteration, optim_args, gt_image, sky_mask, dynamic_mask, mono_normal, obj_render_pkg, sample_radius=sample_radius,
                                missing_uv=np.column_stack((sample_x[actor_samples_np][actor_missing], sample_y[actor_samples_np][actor_missing]))
                            )
                    finally:
                        gaussians.set_visibility(actor_previous_include)
                        gaussians.parse_camera(actor_previous_camera)

            # 4. 统一可视化：两个分支结束后绘制；纯背景视角也可使用。
            # 圆圈为背景、方框为 actor；重叠 actor 分别绘制。
            # fig, ax = plt.subplots(figsize=(14, 8))
            # ax.imshow(gt_image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            # debug_states = sample_vacancy_state.cpu().numpy()
            # debug_groups = [('background', sample_background_np, debug_states[sample_background_np], 'o')]
            # debug_groups += [(f'obj_{actor_id:03d}', selected, states, 's')
            #                     for actor_id, (selected, states) in actor_sample_states.items()]
            # # 红色半透明区域为 actor 缺失采样点的方形邻域，并裁剪到该 actor mask。
            # debug_actor_region = np.zeros((img_H, img_W), dtype=np.bool_)
            # for actor_id, (selected, states) in actor_sample_states.items():
            #     square_region = np.zeros_like(debug_actor_region)
            #     for u, v in zip(sample_x[selected][states == 1], sample_y[selected][states == 1]):
            #         square_region[max(0, v - sample_radius):min(img_H, v + sample_radius),
            #                         max(0, u - sample_radius):min(img_W, u + sample_radius)] = True
            #     debug_actor_region |= square_region & torch.any(dynamic_mask == actor_id, dim=0).cpu().numpy()
            # debug_overlay = np.zeros((img_H, img_W, 4))
            # debug_overlay[debug_actor_region] = [1., 0., 0., 0.2]
            # ax.imshow(debug_overlay)
            # # 缺失状态最后绘制，防止重叠 actor 的健康标记遮住缺失标记。
            # for state, label, color in [(-1, 'Unknown', 'yellow'), (0, 'Covered', 'lime'), (1, 'Missing', 'red')]:
            #     for name, selected, states, marker in debug_groups:
            #         xs, ys = sample_x[selected], sample_y[selected]
            #         pick = states == state
            #         if pick.any():
            #             ax.scatter(xs[pick], ys[pick], marker=marker, s=55,
            #                         facecolors='none', edgecolors=color, linewidths=1.3,
            #                         label=f'{name} {label}: {pick.sum()}')
            # ax.set_title(f'Vacancy | View {current_view} | Iteration {iteration} | radius {sample_radius}')
            # ax.legend(fontsize=8)
            # ax.set_axis_off()
            # fig.tight_layout()
            # plt.show()

        voxel_depth_value, voxel_depth_source, mask_visible, uvs = viewpoint_cam.guidance["bkgd_voxel"]
        voxel_depth_tensor = torch.from_numpy(voxel_depth_value).cuda()

        if iteration > optim_args.hard_depth_start and iteration < optim_args.hard_depth_end:
            loss_hard = 0
            hard_render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, render_type="hard_depth")
            # TODO(depth): hard 模式只将 opacity 设为 1，仍输出 sum(w*z)；稀疏覆盖处并非真实表面 Z，当前 patch 深度损失未筛选覆盖。
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
        # 缓存下一轮需要的图像，CPU 存储且不保留训练计算图或逐高斯数据。
        next_render_cache[randidx] = {
            key: soft_render_pkg[key].detach().to('cpu', copy=True)
            for key in ('rgb', 'depth', 'normals', 'acc')}
        # 仅在独立缓存中归一化深度，供下一轮的检测、传播和深度误差比较使用。
        next_render_cache[randidx]['depth'] /= next_render_cache[randidx]['acc'].clamp_min(1e-6)
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
            depth_render = soft_render_pkg['depth']
            depth_acc = soft_render_pkg['acc']
            # 先筛选可信像素，再归一化；低 acc 不参与监督，分母保留梯度。
            depth_valid_mask = (loss_mask & torch.isfinite(depth_render) & (depth_render > 0.)
                                & torch.isfinite(depth_acc) & (depth_acc > 0.8))
            if optim_args.use_lidar_depth and lidar_depth is not None:           
                depth_mask = depth_valid_mask & torch.isfinite(lidar_depth) & (lidar_depth > 0.)
                expected_depth = depth_render[depth_mask] / depth_acc[depth_mask]
                depth_error = torch.abs(expected_depth - lidar_depth[depth_mask])
                if depth_error.numel() >= 2:
                    depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                    lidar_depth_loss = depth_error.mean()
                    scalar_dict['lidar_depth_loss'] = lidar_depth_loss
                    loss += optim_args.lambda_depth_lidar * lidar_depth_loss

            if optim_args.use_voxel_depth: # 仅background
                depth_mask = (
                    depth_valid_mask
                    & background_mask
                    & torch.isfinite(voxel_depth_tensor)
                    & (voxel_depth_tensor > 0.)
                )
                expected_depth = depth_render[depth_mask] / depth_acc[depth_mask]
                depth_error = torch.abs(expected_depth - voxel_depth_tensor[depth_mask[0]])
                if depth_error.numel() >= 2:
                    depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                    voxel_depth_loss = depth_error.mean()
                    scalar_dict['voxel_depth_loss'] = voxel_depth_loss
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
            # NOTE(depth): 此调试图显示原始 sum(w*z)，颜色变化同时受 acc 影响，不能直接用于判断几何深度是否正确。
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
                gaussians.parse_camera(viewpoint_cam)
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
