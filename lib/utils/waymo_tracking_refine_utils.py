import os
import shutil

import numpy as np


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _tracking_refine_config(cfg):
    raw = cfg.data.get("tracking_refine", False)
    if isinstance(raw, bool):
        enabled = raw
        config = cfg.data
    else:
        enabled = raw.get("enabled", True)
        config = raw
    return enabled, config


def _make_tracking_context(
    image_filenames,
    ixts,
    c2ws,
    ego_frame_poses,
    obj_view_dict,
    object_tracklets_vehicle,
    object_info,
    cameras,
    bbox_to_corner3d,
    quaternion_to_matrix_numpy,
):
    return {
        "ego_frame_poses": ego_frame_poses,
        "image_filenames": image_filenames,
        "ixts": np.asarray(ixts),
        "c2ws": np.asarray(c2ws),
        "obj_view_dict": obj_view_dict,
        "obj_tracklets": object_tracklets_vehicle,
        "object_info": object_info,
        "selected_cameras": list(cameras),
        "bbox_to_corner3d": bbox_to_corner3d,
        "quat_to_mat": quaternion_to_matrix_numpy,
        "num_cameras": len(cameras),
    }


def _copy_refined_to_standard(output_file, object_id, pcd, refined_visibility):
    o3d = __import__("open3d")

    input_ply_dir = os.path.join(output_file, "input_ply")
    standard_ply = os.path.join(input_ply_dir, f"points3D_obj_{object_id:03d}.ply")
    standard_visibility = os.path.join(input_ply_dir, f"points3D_obj_{object_id:03d}.npy")
    o3d.io.write_point_cloud(standard_ply, pcd)
    np.save(standard_visibility, refined_visibility)


def _existing_refined_outputs(output_file, object_ids):
    o3d = __import__("open3d")
    refined = {}
    input_ply_dir = os.path.join(output_file, "input_ply")
    for object_id in object_ids:
        ply_path = os.path.join(input_ply_dir, f"points3D_obj_{object_id:03d}_refined.ply")
        vis_path = os.path.join(input_ply_dir, f"points3D_obj_{object_id:03d}_refined_visibility.npy")
        meta_path = os.path.join(input_ply_dir, f"points3D_obj_{object_id:03d}_tracking_refine.npz")
        if not (os.path.exists(ply_path) and os.path.exists(vis_path) and os.path.exists(meta_path)):
            continue
        pcd = o3d.io.read_point_cloud(ply_path)
        refined_visibility = np.load(vis_path)
        meta = np.load(meta_path, allow_pickle=True)
        version = int(meta["tracking_visibility_version"]) if "tracking_visibility_version" in meta.files else 0
        if version != 2:
            continue
        refined_pose_dict = {
            int(key): value
            for key, value in zip(meta["refined_pose_keys"], meta["refined_pose_values"])
        }
        refined[object_id] = {
            "pcd": pcd,
            "refined_visibility": refined_visibility,
            "refined_pose_dict": refined_pose_dict,
        }
    return refined


def load_refined_pose_metadata(output_file, object_ids):
    """Load versioned tracking-refined poses without loading point clouds."""
    refined_pose_by_object = {}
    input_ply_dir = os.path.join(output_file, "input_ply")
    for object_id in object_ids:
        meta_path = os.path.join(input_ply_dir, f"points3D_obj_{int(object_id):03d}_tracking_refine.npz")
        if not os.path.exists(meta_path):
            continue
        with np.load(meta_path, allow_pickle=True) as meta:
            version = int(meta["tracking_visibility_version"]) if "tracking_visibility_version" in meta.files else 0
            if version != 2:
                continue
            refined_pose_by_object[int(object_id)] = {
                int(key): value
                for key, value in zip(meta["refined_pose_keys"], meta["refined_pose_values"])
            }
    return refined_pose_by_object


def visualize_object_voxel_visibility(
    pcd,
    visibility,
    refined_pose_dict,
    ego_frame_poses,
    c2ws,
    start_frame,
    num_cameras,
    max_voxels=1000,
    max_views_per_voxel=6,
    ray_length=0.5,
):
    """Draw refined voxels and their camera rays in canonical object space."""
    o3d = __import__("open3d")
    points_all = np.asarray(pcd.points)
    colors_all = np.asarray(pcd.colors)
    visibility = np.asarray(visibility, dtype=bool)
    if len(points_all) == 0:
        return
    keep = np.arange(len(points_all))
    if max_voxels > 0 and len(keep) > max_voxels:
        keep = np.linspace(0, len(keep) - 1, max_voxels).astype(np.int64)
    points = points_all[keep]
    colors = colors_all[keep] if len(colors_all) == len(points_all) else np.full_like(points, 0.7)
    visible = visibility[keep]

    local_camera_centers = {}
    for view_id in np.flatnonzero(visible.any(axis=0)):
        frame_id = int(start_frame + view_id // num_cameras)
        obj_pose_vehicle = refined_pose_dict.get(frame_id)
        if obj_pose_vehicle is None or view_id >= len(c2ws) or frame_id >= len(ego_frame_poses):
            continue
        camera_world = np.append(np.asarray(c2ws[view_id])[:3, 3], 1.0)
        camera_vehicle = camera_world @ np.linalg.inv(ego_frame_poses[frame_id]).T
        camera_object = camera_vehicle @ np.linalg.inv(obj_pose_vehicle).T
        local_camera_centers[int(view_id)] = camera_object[:3]

    line_points, lines, line_colors = [], [], []
    for point, view_mask in zip(points, visible):
        view_ids = [view for view in np.flatnonzero(view_mask) if int(view) in local_camera_centers]
        if max_views_per_voxel > 0 and len(view_ids) > max_views_per_voxel:
            sample = np.linspace(0, len(view_ids) - 1, max_views_per_voxel).astype(np.int64)
            view_ids = [view_ids[idx] for idx in sample]
        for view_id in view_ids:
            direction = local_camera_centers[int(view_id)] - point
            norm = np.linalg.norm(direction)
            if norm <= 1e-8:
                continue
            line_start = len(line_points)
            line_points.extend([point, point + direction / norm * ray_length])
            lines.append([line_start, line_start + 1])
            line_colors.append([0.1, 0.85, 1.0])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    geometries = [cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)]
    if lines:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(line_points))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))
        geometries.append(line_set)
    print(f"object visibility visualization: {len(points)} voxels, {len(lines)} view rays")
    o3d.visualization.draw_geometries(geometries)


def run_tracking_refine_from_dataparser(
    cfg,
    datadir,
    image_filenames,
    ixts,
    c2ws,
    ego_frame_poses,
    obj_view_dict,
    object_tracklets_vehicle,
    object_info,
    cameras,
    start_frame,
    end_frame_inclusive,
    bbox_to_corner3d,
    quaternion_to_matrix_numpy,
):
    enabled, refine_cfg = _tracking_refine_config(cfg)
    if not enabled:
        return {}

    # Import lazily so normal dataparser usage does not require CoTracker/SciPy until the feature is enabled.
    from script import test as tracking_tools

    output_file = cfg.model_path
    save_dir = os.path.join(output_file, "input_ply")
    start = int(start_frame)
    end = int(end_frame_inclusive) + 1
    device = _cfg_get(refine_cfg, "device", "cuda")
    checkpoint = _cfg_get(refine_cfg, "checkpoint", _cfg_get(refine_cfg, "cotracker_checkpoint", None))
    force_cotracker = bool(_cfg_get(refine_cfg, "force_cotracker", False))
    force_refine = bool(_cfg_get(refine_cfg, "force_refine", False))
    image_step = int(_cfg_get(refine_cfg, "image_step", 2))
    max_queries_per_call = int(_cfg_get(refine_cfg, "max_queries_per_call", 1024))
    voxel_size = float(_cfg_get(refine_cfg, "voxel_size", 0.15))
    cam_offset = _cfg_get(refine_cfg, "cam_offset", None)
    keyframe_stride = int(_cfg_get(refine_cfg, "keyframe_stride", 5))
    refine_min_visible_points = int(_cfg_get(refine_cfg, "min_visible_points", _cfg_get(refine_cfg, "refine_min_visible_points", 20)))
    refine_min_visible_ratio = float(_cfg_get(refine_cfg, "min_visible_ratio", _cfg_get(refine_cfg, "refine_min_visible_ratio", 0.05)))
    refine_trajectory_weight = float(_cfg_get(refine_cfg, "trajectory_weight", _cfg_get(refine_cfg, "refine_trajectory_weight", 5.0)))
    refine_anchor_max_nfev = int(_cfg_get(refine_cfg, "anchor_max_nfev", _cfg_get(refine_cfg, "refine_anchor_max_nfev", 80)))
    refine_frame_max_nfev = int(_cfg_get(refine_cfg, "frame_max_nfev", _cfg_get(refine_cfg, "refine_frame_max_nfev", 30)))
    enable_point_filter = bool(_cfg_get(refine_cfg, "point_filter", True))
    point_filter_min_observations = int(_cfg_get(refine_cfg, "point_filter_min_observations", 3))
    point_filter_median_px = float(_cfg_get(refine_cfg, "point_filter_median_px", 30.0))
    point_filter_bad_px = float(_cfg_get(refine_cfg, "point_filter_bad_px", 60.0))
    point_filter_max_bad_ratio = float(_cfg_get(refine_cfg, "point_filter_max_bad_ratio", 0.5))
    write_debug = bool(_cfg_get(refine_cfg, "write_debug", False))
    visualize_open3d = bool(_cfg_get(refine_cfg, "visualize_open3d", _cfg_get(refine_cfg, "show_open3d", False)))
    visualize_visibility = bool(_cfg_get(refine_cfg, "visualize_visibility", False))
    visibility_max_voxels = int(_cfg_get(refine_cfg, "visibility_max_voxels", 1000))
    visibility_max_views_per_voxel = int(_cfg_get(refine_cfg, "visibility_max_views_per_voxel", 6))
    visibility_ray_length = float(_cfg_get(refine_cfg, "visibility_ray_length", 0.5))
    max_visualize_objects = _cfg_get(refine_cfg, "max_visualize_objects", None)
    if max_visualize_objects is not None:
        max_visualize_objects = int(max_visualize_objects)

    context = _make_tracking_context(
        image_filenames,
        ixts,
        c2ws,
        ego_frame_poses,
        obj_view_dict,
        object_tracklets_vehicle,
        object_info,
        cameras,
        bbox_to_corner3d,
        quaternion_to_matrix_numpy,
    )

    object_ids = sorted(int(object_id) for object_id in object_info.keys() if object_id in obj_view_dict)
    object_ids = [object_id for object_id in object_ids if object_id in tracking_tools.list_object_ids(output_file)]
    if not object_ids:
        print("tracking refine skipped: no dynamic object point cloud was generated.")
        return {}

    if not force_refine:
        existing = _existing_refined_outputs(output_file, object_ids)
        if len(existing) == len(object_ids):
            object_bank = None
            if visualize_open3d:
                object_bank = tracking_tools.load_object_bank(output_file, object_ids)
            for object_id, data in existing.items():
                _copy_refined_to_standard(
                    output_file,
                    object_id,
                    data["pcd"],
                    data["refined_visibility"],
                )
            if visualize_open3d:
                for vis_idx, object_id in enumerate(sorted(existing.keys())):
                    if max_visualize_objects is not None and vis_idx >= max_visualize_objects:
                        break
                    tracking_tools.visualize_open3d_debug(
                        object_bank[object_id]["positions"],
                        object_bank[object_id]["colors"],
                        existing[object_id]["pcd"],
                    )
            if visualize_visibility:
                for vis_idx, object_id in enumerate(sorted(existing.keys())):
                    if max_visualize_objects is not None and vis_idx >= max_visualize_objects:
                        break
                    data = existing[object_id]
                    visualize_object_voxel_visibility(
                        data["pcd"], data["refined_visibility"], data["refined_pose_dict"],
                        context["ego_frame_poses"], context["c2ws"], start,
                        context["num_cameras"], visibility_max_voxels,
                        visibility_max_views_per_voxel, visibility_ray_length,
                    )
            print("tracking refine reused existing refined object outputs.")
            return {object_id: data["refined_pose_dict"] for object_id, data in existing.items()}

    object_bank = tracking_tools.load_object_bank(output_file, object_ids)

    cotracker_cache = tracking_tools.tracking_cache_path(
        save_dir,
        object_ids,
        start,
        end,
        list(cameras),
        image_step,
        max_queries_per_call,
        voxel_size,
        0,
        0.0,
    )
    cached_tracking = None if force_cotracker else tracking_tools.load_tracking_cache(cotracker_cache)
    if cached_tracking is None:
        if checkpoint is None:
            raise ValueError("data.tracking_refine.checkpoint is required when CoTracker cache is missing.")
        torch = tracking_tools.require_torch()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("tracking refine requested CUDA but torch.cuda.is_available() is False.")
        from cotracker.predictor import CoTrackerPredictor

        model = CoTrackerPredictor(checkpoint=checkpoint, v2=False).to(device)
        obj_src_by_vehicle, cross_cam_observations = tracking_tools.track_visible_objects_all_cameras(
            model,
            context,
            object_bank,
            start,
            end,
            image_step,
            device,
            max_queries_per_call,
        )
        tracking_tools.save_tracking_cache(cotracker_cache, obj_src_by_vehicle, cross_cam_observations)
    else:
        obj_src_by_vehicle, cross_cam_observations = cached_tracking

    tracking_tools.save_cross_camera_observations(save_dir, cross_cam_observations)
    if write_debug:
        tracking_tools.save_tracking_debug_videos(
            save_dir,
            context["image_filenames"],
            context["ego_frame_poses"],
            context["ixts"],
            context["c2ws"],
            context["obj_view_dict"],
            context["object_info"],
            obj_src_by_vehicle,
            start,
            end,
            context["num_cameras"],
            float(_cfg_get(refine_cfg, "video_fps", 10.0)),
            int(_cfg_get(refine_cfg, "video_point_stride", 4)),
            int(_cfg_get(refine_cfg, "video_max_points", 2000)),
        )

    refine_cam_offset = tracking_tools.choose_camera_offset(object_bank, context["num_cameras"], cam_offset)
    print(f"tracking refine uses camera offset {refine_cam_offset}")

    refined_pose_by_object = {}
    refined_results = {}
    for object_id in object_ids:
        if object_id not in obj_src_by_vehicle:
            print(f"object {object_id} has no CoTracker source view, skip tracking refine")
            continue

        obj_src_single = {}
        for source_view, entry in obj_src_by_vehicle[object_id].items():
            if int(entry.get("source_cam_offset", source_view % context["num_cameras"])) != refine_cam_offset:
                continue
            obj_src_single[int(entry["source_frame"])] = entry
        if len(obj_src_single) == 0:
            print(f"object {object_id} has no source frame on camera offset {refine_cam_offset}, skip")
            continue

        refine_paths = tracking_tools.refine_cache_paths(
            save_dir,
            object_id,
            start,
            end,
            refine_cam_offset,
            keyframe_stride,
            voxel_size,
            refine_min_visible_points,
            refine_min_visible_ratio,
            refine_trajectory_weight,
            refine_anchor_max_nfev,
            refine_frame_max_nfev,
            enable_point_filter,
            point_filter_min_observations,
            point_filter_median_px,
            point_filter_bad_px,
            point_filter_max_bad_ratio,
        )
        cached_refine = None if force_refine else tracking_tools.load_refine_cache(refine_paths)
        if cached_refine is None:
            try:
                pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors = tracking_tools.refine_object_pose_and_points(
                    obj_src_single,
                    object_bank[object_id]["positions"],
                    object_bank[object_id]["colors"],
                    object_bank[object_id]["normals"],
                    object_bank[object_id]["visibility"],
                    context["obj_view_dict"],
                    object_id,
                    context["ego_frame_poses"],
                    context["ixts"],
                    context["c2ws"],
                    start,
                    end,
                    refine_cam_offset,
                    context["num_cameras"],
                    keyframe_stride,
                    voxel_size,
                    refine_min_visible_points,
                    refine_min_visible_ratio,
                    refine_trajectory_weight,
                    refine_anchor_max_nfev,
                    refine_frame_max_nfev,
                    enable_point_filter,
                    point_filter_min_observations,
                    point_filter_median_px,
                    point_filter_bad_px,
                    point_filter_max_bad_ratio,
                    tracking_visibility_sources=obj_src_by_vehicle[object_id],
                )
            except ValueError as exc:
                print(f"object {object_id} skipped during tracking refine: {exc}")
                continue
            tracking_tools.save_refine_cache(refine_paths, pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors)
        else:
            pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors = cached_refine

        tracking_tools.save_outputs(save_dir, object_id, pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors)
        _copy_refined_to_standard(
            output_file,
            object_id,
            pcd,
            refined_visibility,
        )
        refined_pose_by_object[object_id] = refined_pose_dict
        refined_results[object_id] = {
            "pcd": pcd,
            "refined_visibility": refined_visibility,
            "refined_pose_dict": refined_pose_dict,
            "delta_r_dict": delta_r_dict,
            "delta_anchors": delta_anchors,
        }
        if visualize_open3d and (max_visualize_objects is None or len(refined_results) <= max_visualize_objects):
            tracking_tools.visualize_open3d_debug(
                object_bank[object_id]["positions"],
                object_bank[object_id]["colors"],
                pcd,
            )
        if visualize_visibility and (max_visualize_objects is None or len(refined_results) <= max_visualize_objects):
            visualize_object_voxel_visibility(
                pcd, refined_visibility, refined_pose_dict,
                context["ego_frame_poses"], context["c2ws"], start,
                context["num_cameras"], visibility_max_voxels,
                visibility_max_views_per_voxel, visibility_ray_length,
            )

    if write_debug and refined_results:
        tracking_tools.save_refine_debug_videos(
            save_dir,
            context["image_filenames"],
            context["ego_frame_poses"],
            context["ixts"],
            context["c2ws"],
            context["obj_view_dict"],
            context["object_info"],
            object_bank,
            refined_results,
            start,
            end,
            context["num_cameras"],
            float(_cfg_get(refine_cfg, "video_fps", 10.0)),
            int(_cfg_get(refine_cfg, "video_point_stride", 4)),
            int(_cfg_get(refine_cfg, "video_max_points", 2000)),
        )

    return refined_pose_by_object
