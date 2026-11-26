import numpy as np
import torch
import matplotlib.pyplot as plt
from lib.models.mvs import depth_propagation, check_geometric_consistency, get_actor_pose_world


@torch.no_grad()
def reconstruct_global(viewpoint_cam, viewpoint_full_stack, dataset, gaussians, render_cache,
                       cams_per_frame, iteration, optim_args, gt_image, sky_mask, dynamic_mask, mono_normal, soft_render_pkg):
    """Complete the reference background using the previous training stack's render cache."""
    randidx = viewpoint_cam.id
    FULL_STACK_LENGTH = len(viewpoint_full_stack)
    visibility = gaussians.background.grape_trellis.get_visibility()
    # src_idxs = [randidx+itv*cams_per_frame for itv in [-2, -1, 1, 2] if ((itv*cams_per_frame + randidx > 0) and (itv*cams_per_frame + randidx < FULL_STACK_LENGTH))] # 随机选一个视角，与前后2帧作patch matching
    src_idxs = []

    pre = randidx - cams_per_frame
    while pre >= 0 and (pre not in render_cache or visibility[:, pre].sum() == 0):
        pre -= cams_per_frame
    if pre >= 0:
        src_idxs.append(pre)

    pre -= cams_per_frame
    while pre >= 0 and (pre not in render_cache or visibility[:, pre].sum() == 0):
        pre -= cams_per_frame
    if pre >= 0:
        src_idxs.append(pre)

    post = randidx + cams_per_frame
    while post < FULL_STACK_LENGTH and (post not in render_cache or visibility[:, post].sum() == 0):
        post += cams_per_frame
    if post < FULL_STACK_LENGTH:
        src_idxs.append(post)

    post += cams_per_frame
    while post < FULL_STACK_LENGTH and (post not in render_cache or visibility[:, post].sum() == 0):
        post += cams_per_frame
    if post < FULL_STACK_LENGTH:
        src_idxs.append(post)
    if not src_idxs:
        return

    # TODO(depth): 缓存已归一化为 depth/acc；低 acc 区域的传播初值仍需考虑覆盖可信度。
    rendered_depth = soft_render_pkg['depth'][0] 
    rendered_normal = soft_render_pkg['normals']

    # get the propagated depth
    propagated_depth, cost, propagated_normal = depth_propagation(randidx, src_idxs, rendered_depth, rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
    propagated_normal = (propagated_normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                

    # propagated_depth = torch.tensor(propagated_depth).to(rendered_depth.device)
    valid_mask = propagated_depth != 300

    # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
    # NOTE(depth): 传播结果和缓存深度均按 camera-Z 定义，可比较相对深度误差。
    # abs_rel_error = torch.abs(propagated_depth - rendered_depth) / propagated_depth.clamp_min(1e-6)
    # depth_error_max_threshold = 1.0
    # depth_error_min_threshold = 0.8
    # abs_rel_error_threshold = depth_error_max_threshold - (depth_error_max_threshold - depth_error_min_threshold) * (iteration - optim_args.propagated_iteration_begin) / (optim_args.propagated_iteration_end - optim_args.propagated_iteration_begin)
    # # color error
    # render_color = soft_render_pkg['rgb']

    #for waymo, quantile 0.6; for free dataset, quantile 0.4
    # error_mask = (abs_rel_error > abs_rel_error_threshold)

    # calculate the geometric consistency
    ref_K = viewpoint_cam.K
    ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
    geometric_counts = None
    # for idx, src_idx in enumerate(src_idxs):
    for idx in range(len(src_idxs)):
        src_idx = src_idxs[idx]
        src_viewpoint = viewpoint_full_stack[src_idx]
        #c2w
        src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
        src_K = src_viewpoint.K

        src_render_pkg = {key: render_cache[src_idx][key].to(rendered_depth.device) for key in ('depth', 'normals')}
        # NOTE(depth): 源帧与参考帧均读取上一轮归一化后的缓存深度。
        src_rendered_depth = src_render_pkg['depth'][0]
        src_rendered_normal = src_render_pkg['normals']

        #get the src_depth first
        src_idxs_for_src = src_idxs[:idx] + src_idxs[idx + 1:] + [randidx]
        src_depth, src_cost, src_normal = depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth, src_rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
        mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                        ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                        src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
        if geometric_counts is None:
            geometric_counts = mask.to(torch.uint8)
        else:
            geometric_counts += mask.to(torch.uint8)

    cost = geometric_counts.squeeze() # 这里cost大约代表各视角下共享视野的部分，越高代表被共同观测且匹配成功的视角越多
    # cost_mask = cost >= 2
    cost_mask = cost >= len(src_idxs)*0.5 # 0.75

    update_mask = (cost_mask & ~torch.any(dynamic_mask != 255, axis=0)).unsqueeze(0).repeat(3, 1, 1)
    mono_normal[update_mask] = propagated_normal[update_mask] #.cpu()

    # propagated_mask = valid_mask & ~error_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
    propagated_mask = valid_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
    if sky_mask is not None:
        propagated_mask = propagated_mask & ~sky_mask[0]

    if dynamic_mask is not None:
        propagated_mask = propagated_mask & ~torch.any(dynamic_mask != 255, axis=0)

    # 这一部分补点要考虑，是否一直在反复补充同一区域特定部分，而另一部分始终未补充且导致检测缺失。
    if propagated_mask.sum() > 100: 
        K = viewpoint_cam.K
        cam2target = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
        bkgd_count = gaussians.background.get_xyz.shape[0]
        if bkgd_count < 1e6:
            target_count = propagated_mask.sum()  # min(propagated_mask.sum() // 4, 800)
        elif bkgd_count < 2e6:
            target_count = propagated_mask.sum() // 4 # min(propagated_mask.sum() // 16, 400)
        else:
            target_count = propagated_mask.sum() // 16 # min(propagated_mask.sum() // 64, 200)
        # only for debug. 后续关闭
        acc = soft_render_pkg['acc']
        gaussians.background.densify_from_depth_propagation(K, cam2target, propagated_depth, propagated_normal, propagated_mask.to(torch.bool), acc, gt_image, init_opacity=0.1, target_count=target_count) 


@torch.no_grad()
def reconstruct_local(ref_src_views, viewpoint_cam, viewpoint_full_stack, dataset, gaussians,
                      render_cache, mono_normal, soft_render_pkg, *, reconstruction_mask, missing_uv,
                      iteration, optim_args, gt_image, sky_mask, dynamic_mask, sample_radius=16):
    """Complete background squares using cached renders; sample_radius is the half-side length."""
    # # Debug: reference first, followed by selected prior source views.
    # fig, axes = plt.subplots(1, len(ref_src_views), figsize=(5 * len(ref_src_views), 4), squeeze=False)
    # for i, view_id in enumerate(ref_src_views):
    #     camera = viewpoint_full_stack[view_id]
    #     axes[0, i].imshow(camera.original_image.detach().permute(1, 2, 0).cpu().numpy())
    #     axes[0, i].set_title(f"{'ref' if i == 0 else 'src'} {view_id}\n{camera.image_name}")
    #     axes[0, i].axis('off')
    # fig.tight_layout()
    # plt.show()
    
    randidx = viewpoint_cam.id
    src_idxs = [view_id for view_id in ref_src_views[1:] if view_id in render_cache]
    if len(src_idxs) < 2:
        return
    # TODO(depth): 缓存已归一化为 depth/acc；低 acc 区域的传播初值仍需考虑覆盖可信度。
    rendered_depth = soft_render_pkg['depth'][0]
    rendered_normal = soft_render_pkg['normals']
    if torch.is_tensor(reconstruction_mask):
        reconstruction_mask = reconstruction_mask.detach().cpu().numpy()
    obj_mask = np.asarray(reconstruction_mask, dtype=np.bool_)
    # obj_mask = np.ones_like(reconstruction_mask, dtype=np.bool_)
    
    square_mask = np.zeros_like(obj_mask)
    height, width = obj_mask.shape
    for u, v in missing_uv:
        square_mask[max(0, v - sample_radius):min(height, v + sample_radius),
                    max(0, u - sample_radius):min(width, u + sample_radius)] = True
    obj_mask = obj_mask & square_mask

    # depth_propagation(randidx, src_idxs, rendered_depth.detach().cpu().numpy(), rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
    propagated_depth, cost, propagated_normal = depth_propagation(randidx, src_idxs, rendered_depth, rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
    propagated_normal = (propagated_normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
    
    propagated_depth = propagated_depth.to(rendered_depth.device)
    valid_mask = torch.isfinite(propagated_depth) & (propagated_depth > 0) & (propagated_depth != 300)

    # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
    # abs_rel_error = torch.abs(propagated_depth - rendered_depth) / propagated_depth.clamp_min(1e-6)
    # depth_error_max_threshold = 1.0
    # depth_error_min_threshold = 0.8
    # abs_rel_error_threshold = depth_error_max_threshold - (depth_error_max_threshold - depth_error_min_threshold) * (iteration - optim_args.propagated_iteration_begin) / (optim_args.propagated_iteration_end - optim_args.propagated_iteration_begin)
    # # color error
    # render_color = soft_render_pkg['rgb']

    #for waymo, quantile 0.6; for free dataset, quantile 0.4 需要确认是否起作用。
    # error_mask = (abs_rel_error > abs_rel_error_threshold)
    
    # calculate the geometric consistency
    ref_K = viewpoint_cam.K
    ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
    geometric_counts = None

    for idx in range(len(src_idxs)):
        src_idx = src_idxs[idx]
        src_viewpoint = viewpoint_full_stack[src_idx]
        #c2w
        src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
        src_K = src_viewpoint.K

        src_render_pkg = {key: render_cache[src_idx][key].to(rendered_depth.device) for key in ('depth', 'normals')}
        # NOTE(depth): 源帧与参考帧均读取上一轮归一化后的缓存深度。
        src_rendered_depth = src_render_pkg['depth'][0]
        src_rendered_normal = src_render_pkg['normals']

        #get the src_depth first
        src_idxs_for_src = src_idxs[:idx] + src_idxs[idx + 1:] + [randidx]
        src_depth, cost, src_normal = depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth, src_rendered_normal, viewpoint_full_stack, dataset, vehicle_name=None, patch_size=20)
        mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                        ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                        src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
        if geometric_counts is None:
            geometric_counts = mask.to(torch.uint8)
        else:
            geometric_counts += mask.to(torch.uint8)
            
    cost = geometric_counts.squeeze() # 这里cost大约代表各视角下共享视野的部分，越高代表被共同观测且匹配成功的视角越多
    # cost_mask = cost >= 2
    cost_mask = cost >= len(src_idxs)*0.5 #0.75
    
    #set -10 as nan     
    update_mask = (cost_mask & valid_mask & torch.from_numpy(obj_mask).to(cost_mask.device)).unsqueeze(0).repeat(3, 1, 1)
    mono_normal[update_mask] = propagated_normal[update_mask] #.cpu()
    
    # propagated_mask = valid_mask & ~error_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
    propagated_mask = valid_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
    if sky_mask is not None:
        propagated_mask = propagated_mask & ~sky_mask[0]

    if dynamic_mask is not None:
        propagated_mask = propagated_mask & torch.all(dynamic_mask == 255, axis=0)
    
    obj_mask = torch.from_numpy(obj_mask).to(propagated_mask.device)
    propagated_mask = propagated_mask & obj_mask
    
    valid_count = int(propagated_mask.sum())
    if valid_count > 3:
        K = viewpoint_cam.K
        cam2target = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
        target_count = valid_count

        # bkgd_count = gaussians.background.get_xyz.shape[0]
        # if bkgd_count < 1e6:
        #     target_count = valid_count
        # elif bkgd_count < 2e6:
        #     target_count = max(1, valid_count // 4)
        # else:
        #     target_count = max(1, valid_count // 16)
        # only for debug. 后续关闭
        acc = soft_render_pkg['acc']
        gaussians.background.densify_from_depth_propagation(K, cam2target, propagated_depth, propagated_normal, propagated_mask.to(torch.bool), acc, gt_image, init_opacity=0.3, target_count=target_count) 



@torch.no_grad()
def reconstruct_actor(ref_src_views, dynamic_id, viewpoint_cam, viewpoint_full_stack, dataset, gaussians,
                      gaussians_renderer, iteration, optim_args, gt_image, sky_mask, dynamic_mask,
                      mono_normal, obj_render_pkg, *, missing_uv, sample_radius=16):
    """Complete an actor using caller-selected views and its reference render.

    ref_src_views contains the reference ID first, followed by source IDs.
    Actor eligibility and view selection are checked by the caller.
    Insert points only within the actor mask and squares around missing samples.
    Update mono_normal in place.
    """
    
    randidx = viewpoint_cam.id
    obj_name = "obj_%.3d"%dynamic_id
    render_acc = obj_render_pkg['acc']
    # 高覆盖区域使用 camera-Z；低覆盖区域不给深度初值，让 PatchMatch 搜索。
    rendered_depth = torch.where(render_acc[0] >= 0.5,
                                 obj_render_pkg['depth'][0] / render_acc[0].clamp_min(1e-6), 0.)
    rendered_normal = obj_render_pkg['normals']
    obj_model = getattr(gaussians, obj_name)
    obj_mask = torch.any(dynamic_mask==dynamic_id, axis=0)
    square_mask = torch.zeros_like(obj_mask)
    height, width = obj_mask.shape
    for u, v in missing_uv:
        square_mask[max(0, v - sample_radius):min(height, v + sample_radius),
                    max(0, u - sample_radius):min(width, u + sample_radius)] = True
    obj_mask = obj_mask & square_mask
    if sky_mask is not None:
        obj_mask = obj_mask & ~sky_mask[0]


    track_id = obj_model.track_id
    obj_views = dataset.scene_info.metadata["obj_view_dict"][track_id]
    # Keep the reference object-to-world pose for consistency and insertion.
    ref_o2w = get_actor_pose_world(gaussians.actor_pose, track_id, viewpoint_cam)
    obj_rots = ref_o2w[:3, :3].unsqueeze(0)
    obj_trans = ref_o2w[:3, 3]

    vehicle_name = int(dynamic_id)
    # Use the supplied source IDs directly; do not sample or rank views here.
    src_idxs = [src_idx for src_idx in ref_src_views[1:] if src_idx in obj_views and src_idx != randidx]
    if len(src_idxs) < 2:
        return
    propagated_depth, cost, propagated_normal = depth_propagation(randidx, src_idxs, rendered_depth, rendered_normal, viewpoint_full_stack, dataset, vehicle_name=vehicle_name, patch_size=20, actor_pose=gaussians.actor_pose)
    if propagated_depth is None:
        print("no props")
        return 

    # depth_propagation already returns reference-camera normals.
    propagated_normal = (propagated_normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                

    # propagated_depth = propagated_depth.to(rendered_depth.device)
    valid_mask = torch.isfinite(propagated_depth) & (propagated_depth > 0) & (propagated_depth != 300)

    # Camera-to-object poses compensate motion in both reprojection directions.
    ref_K = viewpoint_cam.K
    ref_pose = (viewpoint_cam.world_view_transform.transpose(0, 1) @ ref_o2w).inverse()
    geometric_counts = None
    valid_source_count = 0
    # for idx, src_idx in enumerate(src_idxs):
    for idx in range(len(src_idxs)):
        src_idx = src_idxs[idx]
        if src_idx not in obj_views:
            continue
        src_viewpoint = viewpoint_full_stack[src_idx]
        src_K = src_viewpoint.K
        src_o2w = get_actor_pose_world(gaussians.actor_pose, track_id, src_viewpoint)
        src_pose = (src_viewpoint.world_view_transform.transpose(0, 1) @ src_o2w).inverse()

        gaussians.set_visibility([obj_name])
        gaussians.parse_camera(src_viewpoint)
        src_render_pkg = gaussians_renderer.render_kernel(src_viewpoint, gaussians)
        src_acc = src_render_pkg['acc'][0]
        src_rendered_depth = torch.where(src_acc >= 0.5,
                                         src_render_pkg['depth'][0] / src_acc.clamp_min(1e-6), 0.)
        src_rendered_normal = src_render_pkg['normals']

        #get the src_depth first
        # depth_propagation(src_viewpoint, torch.zeros_like(src_projected_depth).cuda(), viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
        src_idxs_for_src = src_idxs[:idx] + src_idxs[idx + 1:] + [randidx]
        # depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth.detach().cpu().numpy(), src_rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_full_stack, dataset, vehicle_name=vehicle_name, patch_size=20)
        src_depth, cost, src_normal = depth_propagation(src_idx, src_idxs_for_src, src_rendered_depth, src_rendered_normal, viewpoint_full_stack, dataset, vehicle_name=vehicle_name, patch_size=20, actor_pose=gaussians.actor_pose)
        if src_depth is None:
            print("no props")
            continue 
        valid_source_count += 1

        # src_depth, cost, src_normal = read_propagted_depth('./cache/propagated_depth')
        # src_depth = torch.tensor(src_depth).cuda()
        mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                        ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                        src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
        if geometric_counts is None:
            geometric_counts = mask.to(torch.uint8)
        else:
            geometric_counts += mask.to(torch.uint8)

    if geometric_counts is None or valid_source_count < 2:
        return
    cost = geometric_counts.squeeze() # 这里cost大约代表各视角下共享视野的部分，越高代表被共同观测且匹配成功的视角越多
    # cost_mask = cost >= 2
    cost_mask = cost >= valid_source_count*0.5

    # Update the supplied normal map; the caller writes it back to camera guidance.
    update_mask = (cost_mask & valid_mask & obj_mask).unsqueeze(0).repeat(3, 1, 1)
    mono_normal[update_mask] = propagated_normal[update_mask]

    propagated_mask = valid_mask & cost_mask
    if sky_mask is not None:
        propagated_mask = propagated_mask & ~sky_mask[0]

    # if obj_bound is not None:
    #     propagated_mask = propagated_mask & ~obj_bound[0]

    if obj_mask is not None:
        propagated_mask = propagated_mask & obj_mask

    valid_count = int(propagated_mask.sum())
    if valid_count > 3:
        K = viewpoint_cam.K
        cam2target = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
        
        target_count = valid_count
        
        # actor_count = obj_model.get_xyz.shape[0]
        # if actor_count < 1000:
        #     target_count = min(valid_count // 2, 800)
        # elif actor_count < 2000:
        #     target_count = min(valid_count // 4, 500)
        # elif actor_count < 4000:
        #     target_count = min(valid_count // 8, 200)
        # else:
        #     target_count = min(valid_count // 16, 100)

        # target_count = min(valid_count, max(10, target_count))
        obj_model.densify_from_depth_propagation(K, cam2target, propagated_depth, propagated_normal, propagated_mask.to(torch.bool), render_acc, gt_image, obj_rots, obj_trans, init_opacity=0.3, target_count=target_count) 
