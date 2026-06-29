import argparse
import gc
import hashlib
import json
import math
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


WAYMO_TRACK2LABEL = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "sign": 3,
    "misc": -1,
}


def require_open3d():
    import open3d as o3d

    return o3d


def require_ply_data():
    from plyfile import PlyData

    return PlyData


def require_torch():
    import torch

    return torch


def require_box_utils():
    from lib.utils.box_utils import bbox_to_corner3d, get_bound_2d_mask, inbbox_points

    return bbox_to_corner3d, get_bound_2d_mask, inbbox_points


def require_general_utils():
    from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy

    return matrix_to_quaternion, quaternion_to_matrix_numpy


def require_least_squares():
    from scipy.optimize import least_squares

    return least_squares


def require_spherical_in():
    from scipy.special import spherical_in

    return spherical_in


def make_cache_tag(parts):
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def parse_image_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" in stem:
        frame_str, cam_str = stem.rsplit("_", 1)
        return int(frame_str), int(cam_str)
    return int(stem[:-1]), int(stem[-1])


def load_camera_info(datadir, num_dataset_cameras):
    ego_pose_dir = os.path.join(datadir, "ego_pose")
    extrinsics_dir = os.path.join(datadir, "extrinsics")
    intrinsics_dir = os.path.join(datadir, "intrinsics")

    intrinsics = []
    extrinsics = []
    for cam in range(num_dataset_cameras):
        raw_intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{cam}.txt"))
        fx, fy, cx, cy = raw_intrinsic[:4]
        intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{cam}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for _ in range(num_dataset_cameras)]
    for ego_pose_name in sorted(os.listdir(ego_pose_dir)):
        pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_name))
        if "_" not in os.path.splitext(ego_pose_name)[0]:
            ego_frame_poses.append(pose)
        else:
            _, cam = parse_image_name(ego_pose_name)
            if cam < num_dataset_cameras:
                ego_cam_poses[cam].append(pose)

    ego_frame_poses = np.asarray(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point

    ego_cam_poses = np.asarray([np.asarray(poses) for poses in ego_cam_poses])
    ego_cam_poses[:, :, :3, 3] -= center_point
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


def make_obj_pose(ego_pose, box_info):
    torch = require_torch()
    matrix_to_quaternion, _ = require_general_utils()
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle_mat = np.eye(4)
    obj_pose_vehicle_mat[:3, :3] = rotz_matrix
    obj_pose_vehicle_mat[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world_mat = ego_pose @ obj_pose_vehicle_mat

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle_mat[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_pose_vehicle = np.concatenate([obj_pose_vehicle_mat[:3, 3], obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world_mat[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_pose_world = np.concatenate([obj_pose_world_mat[:3, 3], obj_quaternion_world])
    return obj_pose_vehicle, obj_pose_world


def get_obj_pose_tracking(datadir, selected_frames, ego_poses, cameras, num_dataset_cameras):
    tracklet_path = os.path.join(datadir, "track/track_info.txt")
    tracklet_camera_vis_path = os.path.join(datadir, "track/track_camera_vis.json")
    print(f"Loading tracklets from: {tracklet_path}")

    with open(tracklet_path, "r") as f:
        tracklets_str = f.read().splitlines()[1:]
    with open(tracklet_camera_vis_path, "r") as f:
        tracklet_camera_vis = json.load(f)

    start_frame, end_frame = selected_frames
    image_dir = os.path.join(datadir, "images")
    n_frames = len(glob(os.path.join(image_dir, "*.png"))) // num_dataset_cameras
    n_obj_in_frame = np.zeros(n_frames)
    tracklets_ls = []
    objects_info = {}

    for tracklet_str in tracklets_str:
        tracklet = tracklet_str.split()
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        object_class = tracklet[2]
        if object_class in ["sign", "misc"]:
            continue

        cameras_vis = tracklet_camera_vis[str(track_id)][str(frame_id)]
        if len(set(cameras) & set(cameras_vis)) == 0:
            continue

        if track_id not in objects_info:
            objects_info[track_id] = {
                "track_id": track_id,
                "class": object_class,
                "class_label": WAYMO_TRACK2LABEL[object_class],
                "height": float(tracklet[4]),
                "width": float(tracklet[5]),
                "length": float(tracklet[6]),
            }
        else:
            objects_info[track_id]["height"] = max(objects_info[track_id]["height"], float(tracklet[4]))
            objects_info[track_id]["width"] = max(objects_info[track_id]["width"], float(tracklet[5]))
            objects_info[track_id]["length"] = max(objects_info[track_id]["length"], float(tracklet[6]))

        tr_array = np.concatenate(
            [np.array(tracklet[:2]).astype(np.float64), np.array([WAYMO_TRACK2LABEL[object_class]]), np.array(tracklet[4:]).astype(np.float64)]
        )
        tracklets_ls.append(tr_array)
        n_obj_in_frame[frame_id] += 1

    tracklets_array = np.asarray(tracklets_ls)
    num_frames = end_frame - start_frame + 1
    max_obj_per_frame = max(1, int(n_obj_in_frame[start_frame : end_frame + 1].max()))
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    for tracklet in tracklets_array:
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        if start_frame <= frame_id <= end_frame:
            obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_poses[frame_id], tracklet[6:10])
            frame_idx = frame_id - start_frame
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()
            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world

    for key in list(objects_info.keys()):
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) == 0:
            objects_info.pop(key)
            continue
        obj_world_positions = visible_objects_pose_world[all_obj_idx][:, :3]
        distance = np.linalg.norm(obj_world_positions[0] - obj_world_positions[-1])
        dynamic = np.any(np.std(obj_world_positions, axis=0) > 0.5) or distance > 2
        if not dynamic:
            visible_objects_ids[all_obj_idx] = -1.0
            visible_objects_pose_vehicle[all_obj_idx] = -1.0
            visible_objects_pose_world[all_obj_idx] = -1.0
            objects_info.pop(key)

    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = int(np.sum(mask, axis=1).max()) if np.any(mask) else 1
    if max_obj_per_frame_new < max_obj_per_frame:
        ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for col in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, col]
                if obj_id >= 0:
                    obj_column = np.argwhere(ids_new[frame_idx, :] < 0).min()
                    ids_new[frame_idx, obj_column] = obj_id
                    pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, col]
                    pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, col]
        visible_objects_ids = ids_new
        visible_objects_pose_vehicle = pose_vehicle_new
        visible_objects_pose_world = pose_world_new

    box_scale = 1.0
    frames = np.arange(start_frame, end_frame + 1, dtype=np.int32)
    for key, obj in objects_info.items():
        obj["deformable"] = obj["class"] == "pedestrian"
        obj["width"] *= box_scale
        obj["length"] *= box_scale
        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0].astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj["start_frame"] = int(np.min(obj_frames))
        obj["end_frame"] = int(np.max(obj_frames))
        obj["obj_frames"] = obj_frames

    objects_tracklets_vehicle = np.concatenate([visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1)
    return objects_tracklets_vehicle, objects_info


def build_scene_context(datadir, selected_frames, cameras, num_dataset_cameras):
    bbox_to_corner3d, get_bound_2d_mask, _ = require_box_utils()
    _, quaternion_to_matrix_numpy = require_general_utils()
    image_dir = os.path.join(datadir, "images")
    image_filenames_all = sorted(glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg")))
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir, num_dataset_cameras)
    object_tracklets_vehicle, object_info = get_obj_pose_tracking(
        datadir, selected_frames, ego_frame_poses, cameras, num_dataset_cameras
    )

    start_frame, end_frame = selected_frames
    frames_idx = []
    cams = []
    image_filenames = []
    ixts = []
    exts = []
    c2ws = []

    for image_filename in image_filenames_all:
        frame, cam = parse_image_name(image_filename)
        if start_frame <= frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            pose = ego_cam_poses[cam, frame]
            c2w = pose @ ext
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)
            ixts.append(ixt)
            exts.append(ext)
            c2ws.append(c2w)

    obj_view_dict = {}
    for view_idx, image_filename in tqdm(list(enumerate(image_filenames)), desc="building object views"):
        cam = cams[view_idx]
        img = cv2.imread(image_filename)
        if img is None:
            raise FileNotFoundError(image_filename)
        h, w = img.shape[:2]
        obj_tracklets = object_tracklets_vehicle[frames_idx[view_idx]]
        ixt, ext = ixts[view_idx], exts[view_idx]
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id < 0:
                continue
            obj_pose_vehicle = np.eye(4)
            obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
            obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
            obj_length = object_info[track_id]["length"]
            obj_width = object_info[track_id]["width"]
            obj_height = object_info[track_id]["height"]
            bbox = np.array([[-obj_length, -obj_width, -obj_height], [obj_length, obj_width, obj_height]]) * 0.5
            corners_local = bbox_to_corner3d(bbox)
            corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
            corners_vehicle = corners_local @ obj_pose_vehicle.T
            obj_mask = get_bound_2d_mask(corners_vehicle[..., :3], ixt, np.linalg.inv(ext), h, w)
            if obj_mask.sum() < 20 * 20 * 10:
                continue
            obj_view_dict.setdefault(track_id, {})[view_idx] = [obj_mask.sum(), obj_pose_vehicle]

    return {
        "ego_frame_poses": ego_frame_poses,
        "image_filenames": image_filenames,
        "ixts": np.asarray(ixts),
        "c2ws": np.asarray(c2ws),
        "obj_view_dict": obj_view_dict,
        "obj_tracklets": object_tracklets_vehicle,
        "object_info": object_info,
        "selected_cameras": cameras,
        "bbox_to_corner3d": bbox_to_corner3d,
        "quat_to_mat": quaternion_to_matrix_numpy,
        "num_cameras": len(cameras),
    }


def list_object_ids(output_file):
    input_ply_dir = os.path.join(output_file, "input_ply")
    object_ids = []
    for path in sorted(glob(os.path.join(input_ply_dir, "points3D_obj_*.ply"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem.replace("points3D_obj_", "")
        if suffix.isdigit():
            object_ids.append(int(suffix))
    return object_ids


def load_object_points(output_file, vehicle_name):
    PlyData = require_ply_data()
    obj_visibility = np.load(os.path.join(output_file, f"input_ply/points3D_obj_{vehicle_name:03d}.npy"))
    plydata = PlyData.read(os.path.join(output_file, f"input_ply/points3D_obj_{vehicle_name:03d}.ply"))
    vertices = plydata["vertex"]
    obj_positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    obj_colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    obj_normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return obj_positions, obj_colors, obj_normals, obj_visibility


def load_object_bank(output_file, object_ids):
    object_bank = {}
    for object_id in tqdm(object_ids, desc="loading object point clouds"):
        object_bank[object_id] = {}
        positions, colors, normals, visibility = load_object_points(output_file, object_id)
        object_bank[object_id]["positions"] = positions
        object_bank[object_id]["colors"] = colors
        object_bank[object_id]["normals"] = normals
        object_bank[object_id]["visibility"] = visibility
    return object_bank


def object_bank_cache_path(save_dir, object_ids, start, end, cameras, voxel_size, sh_degree, sh_kappa):
    tag = make_cache_tag(
        {
            "object_ids": sorted(object_ids),
            "start": start,
            "end": end,
            "cameras": cameras,
            "voxel_size": voxel_size,
            "visibility_sh_degree": sh_degree,
            "visibility_sh_kappa": sh_kappa,
        }
    )
    cache_dir = os.path.join(save_dir, "pointcloud_cache")
    return os.path.join(cache_dir, f"object_bank_{tag}.npz")


def save_object_bank_cache(path, object_bank):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, object_bank=np.array(object_bank, dtype=object))
    print(f"Saved generated object-bank cache: {path}")


def load_object_bank_cache(path):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    print(f"Loaded generated object-bank cache: {path}")
    return data["object_bank"].item()


def real_sh_basis_np(directions, sh_degree):
    directions = np.asarray(directions, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-8)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    result = [np.full_like(x, 0.28209479177387814)]
    if sh_degree >= 1:
        result.extend([
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
        ])
    if sh_degree >= 2:
        result.extend([
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z * z - 1.0),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ])
    if sh_degree > 2:
        raise NotImplementedError("visibility SH currently supports degree <= 2.")
    return np.stack(result, axis=-1)


def vmf_radial_coeffs_np(sh_degree, k):
    spherical_in = require_spherical_in()
    k = float(k)
    coeffs = []
    for ell in range(sh_degree + 1):
        n_l0 = np.sqrt((2 * ell + 1) / (4 * np.pi))
        coeff = 4 * np.pi * n_l0 * np.exp(-k) * spherical_in(ell, k)
        coeff /= n_l0
        coeffs.append(coeff)
    return np.asarray(coeffs, dtype=np.float32)


def visibility_views_to_vmf_sh(points_xyz, points_visibility, c2ws, sh_degree=2, kappa=8.0):
    num_coeffs = (sh_degree + 1) ** 2
    coeffs = np.zeros((points_xyz.shape[0], num_coeffs), dtype=np.float32)
    radial = vmf_radial_coeffs_np(sh_degree, kappa)
    degree_per_coeff = []
    for ell in range(sh_degree + 1):
        degree_per_coeff.extend([ell] * (2 * ell + 1))
    degree_per_coeff = np.asarray(degree_per_coeff)

    camera_centers = np.asarray(c2ws)[:, :3, 3]
    for point_id in range(points_xyz.shape[0]):
        view_ids = np.where(points_visibility[point_id])[0]
        if view_ids.size == 0:
            continue
        directions = camera_centers[view_ids] - points_xyz[point_id][None]
        basis = real_sh_basis_np(directions, sh_degree)
        coeffs[point_id] = (basis * radial[degree_per_coeff][None]).mean(axis=0)
    return coeffs


def voxel_downsample_with_visibility(points_xyz, points_rgb, points_normal, points_visibility, c2ws, voxel_size, sh_degree, sh_kappa):
    o3d = require_open3d()
    if points_xyz.shape[0] == 0:
        empty_visibility = np.zeros((0, points_visibility.shape[1]), dtype=bool)
        empty_sh = np.zeros((0, (sh_degree + 1) ** 2), dtype=np.float32)
        return points_xyz, points_rgb, points_normal, empty_visibility, empty_sh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points_rgb)
    downsampled_pcd, _, point_indices_for_each_voxel = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=(pcd.get_min_bound() // voxel_size) * voxel_size,
        max_bound=(pcd.get_max_bound() // voxel_size + 1) * voxel_size,
    )

    voxel_visibility = []
    voxel_normals = []
    for indices in point_indices_for_each_voxel:
        view_mask = np.zeros(points_visibility.shape[1], dtype=bool)
        normals = []
        for point_idx in indices:
            view_mask = np.logical_or(view_mask, points_visibility[point_idx])
            normals.append(points_normal[point_idx])
        voxel_visibility.append(view_mask)
        normal = np.mean(np.asarray(normals), axis=0)
        normal = normal / max(np.linalg.norm(normal), 1e-8)
        voxel_normals.append(normal)

    voxel_xyz = np.asarray(downsampled_pcd.points)
    voxel_rgb = np.asarray(downsampled_pcd.colors)
    voxel_normal = np.asarray(voxel_normals)
    voxel_visibility = np.asarray(voxel_visibility)
    voxel_visibility_sh = visibility_views_to_vmf_sh(voxel_xyz, voxel_visibility, c2ws, sh_degree, sh_kappa)
    return voxel_xyz, voxel_rgb, voxel_normal, voxel_visibility, voxel_visibility_sh


def build_object_bank_from_dataparser(
    datadir,
    context,
    object_ids,
    start,
    end,
    voxel_size,
    visibility_sh_degree,
    visibility_sh_kappa,
):
    _, _, inbbox_points = require_box_utils()
    pointcloud_path = os.path.join(datadir, "pointcloud.npz")
    if not os.path.exists(pointcloud_path):
        raise FileNotFoundError(pointcloud_path)

    pointcloud_npz = np.load(pointcloud_path, allow_pickle=True)
    pts3d_dict = pointcloud_npz["pointcloud"].item()
    pts2d_dict = pointcloud_npz["camera_projection"].item()
    num_cameras = context["num_cameras"]
    n_views = (end - start) * num_cameras

    raw_bank = {
        object_id: {"positions": [], "colors": [], "normals": [], "visibility": []}
        for object_id in object_ids
    }

    for frame in tqdm(range(start, end), desc="building object point clouds"):
        rel_frame = frame - start
        raw_3d = pts3d_dict[frame]
        raw_2d = pts2d_dict[frame]
        observed_by_selected = np.zeros(raw_2d.shape[0], dtype=bool)
        for proj_offset in range(0, raw_2d.shape[1], 3):
            projected_cam = raw_2d[:, proj_offset].astype(np.int64)
            projected_x = raw_2d[:, proj_offset + 1]
            projected_y = raw_2d[:, proj_offset + 2]
            valid_projection = (projected_x > 0) & (projected_y > 0)
            observed_by_selected |= np.isin(projected_cam, context["selected_cameras"]) & valid_projection

        if observed_by_selected.sum() == 0:
            continue

        points_xyz_vehicle = np.concatenate(
            [raw_3d[observed_by_selected], np.ones((observed_by_selected.sum(), 1), dtype=raw_3d.dtype)],
            axis=-1,
        )
        points_xyz_world = points_xyz_vehicle @ context["ego_frame_poses"][frame].T
        points_rgb = np.ones((points_xyz_vehicle.shape[0], 3), dtype=np.float32)
        points_normal = np.ones((points_xyz_vehicle.shape[0], 3), dtype=np.float32)
        points_visibility = np.zeros((points_xyz_vehicle.shape[0], n_views), dtype=bool)

        for cam_offset, cam in enumerate(context["selected_cameras"]):
            view_idx = rel_frame * num_cameras + cam_offset
            image = cv2.imread(context["image_filenames"][view_idx])
            if image is None:
                raise FileNotFoundError(context["image_filenames"][view_idx])
            image_rgb = image[..., [2, 1, 0]] / 255.0
            normal_filename = context["image_filenames"][view_idx].replace("images", "normal_img").replace(".jpg", ".png")
            normal_img = cv2.imread(normal_filename)
            if normal_img is None:
                normals_world = np.zeros_like(image_rgb)
                normals_world[..., 2] = 1.0
            else:
                normal_dsine = normal_img / 255.0 * 2.0 - 1.0
                normal_cam = np.zeros_like(normal_dsine)
                normal_cam[..., 0] = -normal_dsine[..., 2]
                normal_cam[..., 1] = -normal_dsine[..., 1]
                normal_cam[..., 2] = -normal_dsine[..., 0]
                normals_world = normal_cam @ context["c2ws"][view_idx][:3, :3].T

            uv, depth = project_world_points(points_xyz_world[:, :3], context["c2ws"][view_idx], context["ixts"][view_idx])
            valid = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < image_rgb.shape[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < image_rgb.shape[0])
                & (depth > 2.0)
            )
            us = uv[valid, 0].astype(np.int16)
            vs = uv[valid, 1].astype(np.int16)
            points_rgb[valid] = image_rgb[vs, us]
            points_normal[valid] = normals_world[vs, us]
            points_visibility[valid, view_idx] = True

        object_tracklets = context["obj_tracklets"][rel_frame]
        ego_pose = context["ego_frame_poses"][frame]
        for tracklet in object_tracklets:
            track_id = int(tracklet[0])
            if track_id not in raw_bank:
                continue
            obj_pose_vehicle = np.eye(4)
            obj_pose_vehicle[:3, :3] = context["quat_to_mat"](tracklet[4:8])
            obj_pose_vehicle[:3, 3] = tracklet[1:4]
            vehicle2local = np.linalg.inv(obj_pose_vehicle)
            points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
            points_xyz_obj = points_xyz_obj[:, :3]
            points_normal_obj = points_normal @ np.linalg.inv(ego_pose[:3, :3]).T @ vehicle2local[:3, :3].T

            obj_meta = context["object_info"][track_id]
            bbox = [
                [-obj_meta["length"] / 2, -obj_meta["width"] / 2, -obj_meta["height"] * 0.6],
                [obj_meta["length"] / 2, obj_meta["width"] / 2, obj_meta["height"] * 0.6],
            ]
            corners = context["bbox_to_corner3d"](bbox)
            in_bbox = inbbox_points(points_xyz_obj, corners)
            if in_bbox.sum() == 0:
                continue
            raw_bank[track_id]["positions"].append(points_xyz_obj[in_bbox])
            raw_bank[track_id]["colors"].append(points_rgb[in_bbox])
            raw_bank[track_id]["normals"].append(points_normal_obj[in_bbox])
            raw_bank[track_id]["visibility"].append(points_visibility[in_bbox])

    object_bank = {}
    for object_id, obj_data in raw_bank.items():
        if len(obj_data["positions"]) == 0:
            continue
        points_xyz = np.concatenate(obj_data["positions"], axis=0)
        points_rgb = np.concatenate(obj_data["colors"], axis=0)
        points_normal = np.concatenate(obj_data["normals"], axis=0)
        points_visibility = np.concatenate(obj_data["visibility"], axis=0)
        voxel_xyz, voxel_rgb, voxel_normal, voxel_visibility, voxel_visibility_sh = voxel_downsample_with_visibility(
            points_xyz,
            points_rgb,
            points_normal,
            points_visibility,
            context["c2ws"],
            voxel_size,
            visibility_sh_degree,
            visibility_sh_kappa,
        )
        if voxel_xyz.shape[0] == 0:
            continue
        object_bank[object_id] = {
            "positions": voxel_xyz,
            "colors": voxel_rgb,
            "normals": voxel_normal,
            "visibility": voxel_visibility,
            "visibility_sh": voxel_visibility_sh,
        }
    return object_bank


def choose_camera_offset(object_bank, num_cameras, explicit_offset=None):
    if explicit_offset is not None:
        return explicit_offset
    visibility_sum = None
    for obj_data in object_bank.values():
        obj_visibility = obj_data["visibility"]
        if visibility_sum is None:
            visibility_sum = obj_visibility.sum(0)
        else:
            visibility_sum = visibility_sum + obj_visibility.sum(0)
    if visibility_sum is None:
        raise ValueError("No object point cloud was loaded.")
    candidate_indices = np.where(visibility_sum > 10)[0]
    if candidate_indices.size == 0:
        raise ValueError("Object visibility contains no view with more than 10 visible points.")
    values, counts = np.unique(candidate_indices % num_cameras, return_counts=True)
    return int(values[np.argmax(counts)])


def build_video(image_filenames, start, end, cam_offset, num_cameras, step, device):
    torch = require_torch()
    frames = []
    for rel_frame in range(end - start):
        view_idx = rel_frame * num_cameras + cam_offset
        img = cv2.imread(image_filenames[view_idx])
        if img is None:
            raise FileNotFoundError(image_filenames[view_idx])
        frames.append(img[::step, ::step, :])
    video = np.stack(frames)
    return torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device)


def project_points(obj_positions_local, obj_pose_vehicle, ego_pose, c2w, ixt):
    points_xyz_obj_world = obj_positions_local @ obj_pose_vehicle.T @ ego_pose.T
    view_pos_cam = points_xyz_obj_world @ np.linalg.inv(c2w).T
    uv = view_pos_cam[:, :3] @ ixt.T
    return uv[:, :2] / uv[:, 2:3]


def project_points_with_depth(obj_positions_local, obj_pose_vehicle, ego_pose, c2w, ixt):
    points_xyz_obj_world = obj_positions_local @ obj_pose_vehicle.T @ ego_pose.T
    view_pos_cam = points_xyz_obj_world @ np.linalg.inv(c2w).T
    uvw = view_pos_cam[:, :3] @ ixt.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, view_pos_cam[:, 2]


def project_world_points(points_xyz_world, c2w, ixt):
    points_world = np.concatenate([points_xyz_world[:, :3], np.ones_like(points_xyz_world[:, :1])], axis=-1)
    view_pos_cam = points_world @ np.linalg.inv(c2w).T
    uvw = view_pos_cam[:, :3] @ ixt.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, view_pos_cam[:, 2]


def compact_view_index(frame, start, num_cameras, cam_offset):
    return (frame - start) * num_cameras + cam_offset


def run_cotracker_queries(model, video, queries, step, max_queries_per_call, device):
    torch = require_torch()
    tracks = []
    visibility = []
    num_queries = queries.shape[1]
    for start_idx in range(0, num_queries, max_queries_per_call):
        end_idx = min(start_idx + max_queries_per_call, num_queries)
        pred_tracks, pred_visibility, _ = model(
            video,
            grid_size=10,
            grid_query_frame=0,
            backward_tracking=True,
            queries=queries[:, start_idx:end_idx],
        )
        tracks.append(pred_tracks[0].detach().cpu().numpy() * step)
        visibility.append(pred_visibility[0].detach().cpu().numpy())
        del pred_tracks, pred_visibility
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.concatenate(tracks, axis=1), np.concatenate(visibility, axis=1)


def collect_object_queries_for_view(
    image_shape,
    src_view,
    frame_id,
    cam_offset,
    object_bank,
    obj_view_dict,
    ego_frame_poses,
    ixts,
    c2ws,
):
    img_h, img_w = image_shape[:2]
    query_chunks = []
    query_meta = []
    visible_views_by_object = {}

    for object_id, obj_data in object_bank.items():
        if object_id not in obj_view_dict or src_view not in obj_view_dict[object_id]:
            continue

        obj_visibility = obj_data["visibility"]
        if src_view >= obj_visibility.shape[1]:
            continue

        obj_pose_vehicle = obj_view_dict[object_id][src_view][1]
        obj_positions_local = np.concatenate(
            [obj_data["positions"], np.ones_like(obj_data["positions"][..., :1])], axis=-1
        )
        points_xyz_obj_world = obj_positions_local @ obj_pose_vehicle.T @ ego_frame_poses[frame_id].T
        obj_view_mask = obj_visibility[:, src_view]
        if obj_view_mask.sum() == 0:
            continue

        view_pos_world = np.concatenate(
            [points_xyz_obj_world[obj_view_mask, :3], np.ones_like(points_xyz_obj_world[obj_view_mask, :1])], axis=-1
        )
        view_pos_cam = view_pos_world @ np.linalg.inv(c2ws[src_view]).T
        tmp = view_pos_cam[:, :3] @ ixts[src_view].T
        us = (tmp[:, 0] / tmp[:, 2]).astype(np.int16)
        vs = (tmp[:, 1] / tmp[:, 2]).astype(np.int16)
        in_image = (us >= 0) & (us < img_w) & (vs >= 0) & (vs < img_h) & (tmp[:, 2] > 0)
        if in_image.sum() == 0:
            continue

        query_mask = np.zeros_like(obj_view_mask, dtype=bool)
        query_mask[obj_view_mask] = in_image
        query_pixels = np.stack([us[in_image], vs[in_image]], axis=1)
        slice_start = sum(chunk.shape[0] for chunk in query_chunks)
        slice_end = slice_start + query_pixels.shape[0]
        query_chunks.append(query_pixels)
        visible_views_by_object[object_id] = np.where(obj_visibility[query_mask].sum(axis=0) > 0)[0]
        query_meta.append(
            {
                "object_id": object_id,
                "slice": slice(slice_start, slice_end),
                "point_mask": obj_view_mask,
                "query_mask": query_mask,
                "project_mask": in_image,
                "source_view": src_view,
                "source_cam_offset": cam_offset,
            }
        )

    if not query_chunks:
        return None, [], visible_views_by_object
    return np.concatenate(query_chunks, axis=0), query_meta, visible_views_by_object


def track_visible_objects(
    model,
    video,
    image_filenames,
    ixts,
    c2ws,
    ego_frame_poses,
    obj_view_dict,
    object_bank,
    start,
    end,
    cam_offset,
    num_cameras,
    step,
    device,
    max_queries_per_call,
):
    torch = require_torch()
    obj_src_by_vehicle = {object_id: {} for object_id in object_bank.keys()}
    cross_cam_observations = {object_id: {} for object_id in object_bank.keys()}

    for t1 in tqdm(range(start, end), desc="co-tracker"):
        src_view = compact_view_index(t1, start, num_cameras, cam_offset)
        img = cv2.imread(image_filenames[src_view])
        if img is None:
            raise FileNotFoundError(image_filenames[src_view])
        query_pixels, query_meta, visible_views = collect_object_queries_for_view(
            img.shape,
            src_view,
            t1,
            cam_offset,
            object_bank,
            obj_view_dict,
            ego_frame_poses,
            ixts,
            c2ws,
        )
        if query_pixels is None:
            print(f"source view {src_view} has no projected query point, skip")
            continue

        grid_pts = query_pixels.reshape(1, -1, 2) / step
        tmp_grid_pts = torch.from_numpy(grid_pts).float().to(device)
        queries = torch.cat([torch.ones_like(tmp_grid_pts[:, :, :1]) * (t1 - start), tmp_grid_pts], dim=2)
        pred_tracks, pred_visibility = run_cotracker_queries(
            model,
            video,
            queries,
            step,
            max_queries_per_call,
            device,
        )

        for meta in query_meta:
            object_id = meta["object_id"]
            point_slice = meta["slice"]
            obj_src_by_vehicle[object_id][src_view] = {
                "pred_tracks": pred_tracks[:, point_slice, :],
                "pred_visibility": pred_visibility[:, point_slice],
                "point_mask": meta["point_mask"],
                "project_mask": meta["project_mask"],
                "query_mask": meta["query_mask"],
                "source_view": meta["source_view"],
                "source_frame": t1,
                "source_cam_offset": meta["source_cam_offset"],
            }
            cross_cam_observations[object_id][src_view] = visible_views.get(object_id, np.array([], dtype=np.int64))

        del pred_tracks, pred_visibility, queries
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    obj_src_by_vehicle = {k: v for k, v in obj_src_by_vehicle.items() if len(v) > 0}
    cross_cam_observations = {k: v for k, v in cross_cam_observations.items() if len(v) > 0}
    return obj_src_by_vehicle, cross_cam_observations


def track_visible_objects_all_cameras(
    model,
    context,
    object_bank,
    start,
    end,
    step,
    device,
    max_queries_per_call,
):
    merged_obj_src = {object_id: {} for object_id in object_bank.keys()}
    merged_cross_cam = {object_id: {} for object_id in object_bank.keys()}

    for cam_offset in range(context["num_cameras"]):
        video = build_video(
            context["image_filenames"],
            start,
            end,
            cam_offset,
            context["num_cameras"],
            step,
            device,
        )
        obj_src, cross_cam = track_visible_objects(
            model,
            video,
            context["image_filenames"],
            context["ixts"],
            context["c2ws"],
            context["ego_frame_poses"],
            context["obj_view_dict"],
            object_bank,
            start,
            end,
            cam_offset,
            context["num_cameras"],
            step,
            device,
            max_queries_per_call,
        )
        for object_id, entries in obj_src.items():
            merged_obj_src.setdefault(object_id, {}).update(entries)
        for object_id, entries in cross_cam.items():
            merged_cross_cam.setdefault(object_id, {}).update(entries)

    merged_obj_src = {k: v for k, v in merged_obj_src.items() if len(v) > 0}
    merged_cross_cam = {k: v for k, v in merged_cross_cam.items() if len(v) > 0}
    return merged_obj_src, merged_cross_cam


def tracking_cache_path(save_dir, object_ids, start, end, cameras, image_step, max_queries_per_call, voxel_size, sh_degree, sh_kappa):
    tag = make_cache_tag(
        {
            "object_ids": sorted(object_ids),
            "start": start,
            "end": end,
            "cameras": cameras,
            "image_step": image_step,
            "max_queries_per_call": max_queries_per_call,
            "voxel_size": voxel_size,
            "visibility_sh_degree": sh_degree,
            "visibility_sh_kappa": sh_kappa,
            "multi_camera": True,
        }
    )
    cache_dir = os.path.join(save_dir, "tracking_cache")
    return os.path.join(cache_dir, f"cotracker_all_objects_{tag}.npz")


def save_tracking_cache(path, obj_src_by_vehicle, cross_cam_observations):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        obj_src_by_vehicle=np.array(obj_src_by_vehicle, dtype=object),
        cross_cam_observations=np.array(cross_cam_observations, dtype=object),
    )
    print(f"Saved CoTracker cache: {path}")


def load_tracking_cache(path):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    obj_src_by_vehicle = data["obj_src_by_vehicle"].item()
    cross_cam_observations = data["cross_cam_observations"].item()
    print(f"Loaded CoTracker cache: {path}")
    return obj_src_by_vehicle, cross_cam_observations


def yaw_from_rot(rot):
    return np.arctan2(rot[1, 0], rot[0, 0])


def yaw_pitch_to_rot(yaw_pitch):
    yaw, pitch = float(yaw_pitch[0]), float(yaw_pitch[1])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rot_yaw = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot_pitch = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    return rot_yaw @ rot_pitch


def residuals_anchor(
    delta_yaw_pitch,
    n_src,
    n_dst,
    r_vec_arr,
    t_vec_arr,
    obj_positions_local,
    visibility_arr,
    msk_obs_arr,
    ego_pose_arr,
    c2w_arr,
    ixt,
    uv_obs_arr,
    frame_positions=None,
    trajectory_weight=0.0,
    smooth_weight=0.0,
    target_yaw_arr=None,
):
    res = []
    refined_yaws = []
    for i in range(n_src):
        points = obj_positions_local[visibility_arr[i]]
        delta_pose_i = np.eye(4)
        delta_pose_i[:3, :3] = yaw_pitch_to_rot(delta_yaw_pitch[i * 2 : i * 2 + 2])
        refined_points = points @ np.linalg.inv(delta_pose_i.T)

        for j in range(n_dst):
            refined_pose_j = np.eye(4)
            origin_r = cv2.Rodrigues(r_vec_arr[j])[0]
            delta_r = yaw_pitch_to_rot(delta_yaw_pitch[j * 2 : j * 2 + 2])
            refined_pose_j[:3, :3] = (delta_r.T @ origin_r.T).T
            refined_pose_j[:3, 3] = t_vec_arr[j]
            if i == 0:
                refined_yaws.append(yaw_from_rot(refined_pose_j[:3, :3]))
            obs_mask = msk_obs_arr[i][j]
            if obs_mask.sum() == 0:
                continue
            uv_pred = project_points(refined_points[obs_mask], refined_pose_j, ego_pose_arr[j], c2w_arr[j], ixt)
            res.append(np.sqrt(np.sum((uv_pred - uv_obs_arr[i][j][obs_mask]) ** 2, axis=1)))
    if trajectory_weight > 0 and target_yaw_arr is not None and len(refined_yaws) == len(target_yaw_arr):
        target_yaw_arr = np.asarray(target_yaw_arr)
        refined_yaws = np.asarray(refined_yaws)
        valid = np.isfinite(target_yaw_arr)
        if valid.any():
            res.append(wrap_angle(refined_yaws[valid] - target_yaw_arr[valid]) * trajectory_weight)
    if smooth_weight > 0 and len(refined_yaws) >= 3:
        smooth_yaws = np.unwrap(np.asarray(refined_yaws))
        res.append((smooth_yaws[2:] - 2.0 * smooth_yaws[1:-1] + smooth_yaws[:-2]) * smooth_weight)
    return np.hstack(res).ravel() if res else np.zeros(0)


def residuals_single_delta(
    delta_yaw_pitch,
    n_src,
    n_dst,
    r_vec_arr,
    t_vec_arr,
    obj_positions_local,
    visibility_arr,
    msk_obs_arr,
    ego_pose_arr,
    c2w_arr,
    ixt,
    uv_obs_arr,
):
    res = []
    for i in range(n_src):
        points = obj_positions_local[visibility_arr[i]]
        delta_pose_i = np.eye(4)
        delta_pose_i[:3, :3] = yaw_pitch_to_rot(delta_yaw_pitch[i * 2 : i * 2 + 2])
        refined_points = points @ np.linalg.inv(delta_pose_i.T)

        for j in range(n_dst):
            pose_j = np.eye(4)
            pose_j[:3, :3] = cv2.Rodrigues(r_vec_arr[j])[0]
            pose_j[:3, 3] = t_vec_arr[j]
            obs_mask = msk_obs_arr[i][j]
            if obs_mask.sum() == 0:
                continue
            uv_pred = project_points(refined_points[obs_mask], pose_j, ego_pose_arr[j], c2w_arr[j], ixt)
            res.append(np.sqrt(np.sum((uv_pred - uv_obs_arr[i][j][obs_mask]) ** 2, axis=1)))
    return np.hstack(res).ravel() if res else np.zeros(0)


def build_tracking_consistency_point_mask(
    obj_src,
    obj_positions_local,
    refined_pose_dict,
    delta_r_dict,
    delta_canonical,
    ego_frame_poses,
    ixts,
    c2ws,
    start,
    end,
    cam_offset,
    num_cameras,
    min_observations,
    median_px_threshold,
    bad_px_threshold,
    max_bad_ratio,
):
    point_errors = [[] for _ in range(obj_positions_local.shape[0])]
    for source_frame, entry in obj_src.items():
        if source_frame not in delta_r_dict:
            continue
        query_indices = np.where(entry["query_mask"])[0]
        if query_indices.shape[0] == 0:
            continue

        source_delta = delta_r_dict[source_frame]
        source_points = obj_positions_local[query_indices] @ np.linalg.inv(source_delta.T) @ delta_canonical.T
        n_track_frames = min(end - start, entry["pred_tracks"].shape[0])
        for rel_frame in range(n_track_frames):
            target_frame = start + rel_frame
            if target_frame == source_frame:
                continue
            if target_frame not in refined_pose_dict:
                continue
            obs_mask = entry["pred_visibility"][rel_frame].astype(bool)
            if obs_mask.sum() == 0:
                continue

            current_view = compact_view_index(target_frame, start, num_cameras, cam_offset)
            uv_pred, depth = project_points_with_depth(
                source_points[obs_mask],
                refined_pose_dict[target_frame],
                ego_frame_poses[target_frame],
                c2ws[current_view],
                ixts[current_view],
            )
            uv_obs = entry["pred_tracks"][rel_frame][obs_mask]
            valid = (
                np.isfinite(uv_pred).all(axis=1)
                & np.isfinite(uv_obs).all(axis=1)
                & np.isfinite(depth)
                & (depth > 0)
            )
            if valid.sum() == 0:
                continue

            residual = np.linalg.norm(uv_pred[valid] - uv_obs[valid], axis=1)
            valid_query_indices = query_indices[obs_mask][valid]
            for point_id, err in zip(valid_query_indices, residual):
                point_errors[int(point_id)].append(float(err))

    keep_mask = np.ones(obj_positions_local.shape[0], dtype=bool)
    tested_count = 0
    for point_id, errors in enumerate(point_errors):
        if len(errors) < min_observations:
            continue
        tested_count += 1
        errors = np.asarray(errors)
        median_error = float(np.median(errors))
        bad_ratio = float(np.mean(errors > bad_px_threshold))
        if median_error > median_px_threshold or bad_ratio > max_bad_ratio:
            keep_mask[point_id] = False

    removed_count = int((~keep_mask).sum())
    print(
        "tracking consistency point filter: "
        f"tested={tested_count}, removed={removed_count}, kept={int(keep_mask.sum())}"
    )
    return keep_mask


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def frame_refine_visibility_stats(obj_src, obj_visibility, frame, current_view):
    visible_count = int(obj_visibility[:, current_view].sum())
    visible_ratio = float(visible_count) / max(1, obj_visibility.shape[0])
    query_count = int(obj_src[frame]["query_mask"].sum()) if frame in obj_src else 0
    return visible_count, query_count, visible_ratio


def frame_is_refine_usable(obj_src, obj_visibility, frame, current_view, min_visible_points, min_visible_ratio):
    visible_count, query_count, visible_ratio = frame_refine_visibility_stats(obj_src, obj_visibility, frame, current_view)
    if visible_count < min_visible_points:
        return False
    if query_count < min_visible_points:
        return False
    if visible_ratio < min_visible_ratio:
        return False
    return True


def build_trajectory_target_yaws(rot_arr, pos_arr):
    pos_arr = np.asarray(pos_arr)
    target_yaws = np.full(len(pos_arr), np.nan, dtype=np.float64)
    for i in range(len(pos_arr)):
        if len(pos_arr) < 2:
            break
        if i == 0:
            tangent = pos_arr[1, :2] - pos_arr[0, :2]
        elif i == len(pos_arr) - 1:
            tangent = pos_arr[-1, :2] - pos_arr[-2, :2]
        else:
            tangent = pos_arr[i + 1, :2] - pos_arr[i - 1, :2]
        if np.linalg.norm(tangent) < 1e-4:
            continue
        tangent_yaw = np.arctan2(tangent[1], tangent[0])
        origin_yaw = yaw_from_rot(rot_arr[i])
        candidates = np.array([tangent_yaw, tangent_yaw + np.pi])
        errors = np.abs(wrap_angle(origin_yaw - candidates))
        target_yaws[i] = candidates[int(np.argmin(errors))]
    return target_yaws


def pose_from_tracklet(tracklet, quat_to_mat):
    pose = np.eye(4)
    pose[:3, :3] = quat_to_mat(tracklet[4:8])
    pose[:3, 3] = tracklet[1:4]
    return pose


def collect_object_pose_by_frame(context, object_id, start, end):
    poses = {}
    for rel_frame, frame in enumerate(range(start, end)):
        for tracklet in context["obj_tracklets"][rel_frame]:
            if int(tracklet[0]) == int(object_id):
                poses[frame] = pose_from_tracklet(tracklet, context["quat_to_mat"])
                break
    return poses


def build_multicamera_observations(obj_src, point_keep_mask, start, num_cameras, max_points_per_observation):
    observations = []
    for source_view, entry in sorted(obj_src.items()):
        source_frame = int(entry["source_frame"])
        source_cam_offset = int(entry["source_cam_offset"])
        query_indices = np.where(entry["query_mask"])[0]
        query_keep = point_keep_mask[query_indices]
        if query_keep.sum() == 0:
            continue

        for rel_frame in range(entry["pred_tracks"].shape[0]):
            target_frame = start + rel_frame
            target_view = rel_frame * num_cameras + source_cam_offset
            visible = entry["pred_visibility"][rel_frame].astype(bool)
            usable = np.logical_and(visible, query_keep)
            if usable.sum() == 0:
                continue
            local_query_indices = np.where(usable)[0]
            if max_points_per_observation > 0 and local_query_indices.size > max_points_per_observation:
                sample_ids = np.linspace(0, local_query_indices.size - 1, max_points_per_observation).astype(np.int64)
                local_query_indices = local_query_indices[sample_ids]
            observations.append(
                {
                    "source_view": int(source_view),
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "target_view": target_view,
                    "point_indices": query_indices[local_query_indices],
                    "uv": entry["pred_tracks"][rel_frame, local_query_indices],
                }
            )
    return observations


def project_local_points_with_pose(points_local_h, obj_pose_vehicle, ego_pose, c2w, ixt):
    uv, depth = project_points_with_depth(points_local_h, obj_pose_vehicle, ego_pose, c2w, ixt)
    return uv, depth


def multicamera_refine_residuals(
    delta_rvec,
    observations,
    frame_to_var,
    frame_list,
    pose_by_frame,
    obj_positions_h,
    context,
    reproj_weight,
):
    res = []
    delta_rot = {}
    refined_rot = {}
    for frame in frame_list:
        idx = frame_to_var[frame]
        delta_rot[frame] = cv2.Rodrigues(delta_rvec[idx * 3 : idx * 3 + 3])[0]
        refined_rot[frame] = pose_by_frame[frame][:3, :3] @ delta_rot[frame]

    for obs in observations:
        if obs["source_frame"] not in frame_to_var or obs["target_frame"] not in frame_to_var:
            continue
        src_delta = np.eye(4)
        src_delta[:3, :3] = delta_rot[obs["source_frame"]]
        refined_points = obj_positions_h[obs["point_indices"]] @ np.linalg.inv(src_delta.T)

        target_pose = np.array(pose_by_frame[obs["target_frame"]], copy=True)
        target_pose[:3, :3] = refined_rot[obs["target_frame"]]
        uv_pred, depth = project_local_points_with_pose(
            refined_points,
            target_pose,
            context["ego_frame_poses"][obs["target_frame"]],
            context["c2ws"][obs["target_view"]],
            context["ixts"][obs["target_view"]],
        )
        valid = depth > 0
        if valid.sum() > 0:
            residual = (uv_pred[valid] - obs["uv"][valid]).reshape(-1) * reproj_weight
            res.append(residual)

    return np.concatenate(res) if res else np.zeros(0)


def filter_observations(observations, source_frames=None, target_frames=None):
    if source_frames is not None:
        source_frames = set(source_frames)
    if target_frames is not None:
        target_frames = set(target_frames)
    filtered = []
    for obs in observations:
        if source_frames is not None and obs["source_frame"] not in source_frames:
            continue
        if target_frames is not None and obs["target_frame"] not in target_frames:
            continue
        filtered.append(obs)
    return filtered


def choose_sparse_keyframes(observations, pose_by_frame, start, keyframe_stride):
    source_frames = sorted(set(obs["source_frame"] for obs in observations) & set(pose_by_frame.keys()))
    if not source_frames:
        return []
    keyframes = [frame for frame in source_frames if (frame - start) % keyframe_stride == 0]
    if len(keyframes) < 2:
        keyframes = source_frames[:: max(1, keyframe_stride)]
    if source_frames[0] not in keyframes:
        keyframes.insert(0, source_frames[0])
    if source_frames[-1] not in keyframes:
        keyframes.append(source_frames[-1])
    return sorted(set(keyframes))


def optimize_rotation_deltas_for_frames(observations, frames, pose_by_frame, obj_positions_h, context, reproj_weight):
    least_squares = require_least_squares()
    frame_to_var = {frame: i for i, frame in enumerate(frames)}
    x0 = np.zeros(len(frames) * 3, dtype=np.float64)
    res = least_squares(
        multicamera_refine_residuals,
        x0,
        loss="soft_l1",
        f_scale=10.0,
        max_nfev=60,
        args=(
            observations,
            frame_to_var,
            frames,
            pose_by_frame,
            obj_positions_h,
            context,
            reproj_weight,
        ),
        verbose=1,
    )
    delta_by_frame = {}
    refined_pose_by_frame = {}
    for frame in frames:
        idx = frame_to_var[frame]
        delta_rot = cv2.Rodrigues(res.x[idx * 3 : idx * 3 + 3])[0]
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = delta_rot
        refined_pose = np.array(pose_by_frame[frame], copy=True)
        refined_pose[:3, :3] = pose_by_frame[frame][:3, :3] @ delta_rot
        delta_by_frame[frame] = delta_pose
        refined_pose_by_frame[frame] = refined_pose
    return refined_pose_by_frame, delta_by_frame, res


def expand_rotation_delta_residuals(
    delta_rvec,
    observations,
    source_frame,
    key_refined_pose_by_frame,
    obj_positions_h,
    pose_by_frame,
    context,
    reproj_weight,
):
    delta_pose = np.eye(4)
    delta_pose[:3, :3] = cv2.Rodrigues(delta_rvec)[0]
    refined_points_all = obj_positions_h @ np.linalg.inv(delta_pose.T)
    res = []
    for obs in observations:
        if obs["source_frame"] != source_frame or obs["target_frame"] not in key_refined_pose_by_frame:
            continue
        refined_points = refined_points_all[obs["point_indices"]]
        target_pose = key_refined_pose_by_frame[obs["target_frame"]]
        uv_pred, depth = project_local_points_with_pose(
            refined_points,
            target_pose,
            context["ego_frame_poses"][obs["target_frame"]],
            context["c2ws"][obs["target_view"]],
            context["ixts"][obs["target_view"]],
        )
        valid = depth > 0
        if valid.sum() > 0:
            res.append((uv_pred[valid] - obs["uv"][valid]).reshape(-1) * reproj_weight)
    return np.concatenate(res) if res else np.zeros(0)


def expand_rotation_delta_for_frame(
    frame,
    observations,
    key_refined_pose_by_frame,
    obj_positions_h,
    pose_by_frame,
    context,
    reproj_weight,
):
    least_squares = require_least_squares()
    frame_observations = filter_observations(observations, source_frames=[frame], target_frames=key_refined_pose_by_frame.keys())
    if len(frame_observations) == 0:
        delta_pose = np.eye(4)
        return np.array(pose_by_frame[frame], copy=True), delta_pose, None
    res = least_squares(
        expand_rotation_delta_residuals,
        np.zeros(3, dtype=np.float64),
        loss="soft_l1",
        f_scale=10.0,
        max_nfev=40,
        args=(
            frame_observations,
            frame,
            key_refined_pose_by_frame,
            obj_positions_h,
            pose_by_frame,
            context,
            reproj_weight,
        ),
        verbose=0,
    )
    delta_rot = cv2.Rodrigues(res.x)[0]
    delta_pose = np.eye(4)
    delta_pose[:3, :3] = delta_rot
    refined_pose = np.array(pose_by_frame[frame], copy=True)
    refined_pose[:3, :3] = pose_by_frame[frame][:3, :3] @ delta_rot
    return refined_pose, delta_pose, res


def multicamera_refine_object(
    object_id,
    obj_src,
    obj_data,
    context,
    start,
    end,
    max_points_per_observation,
    reproj_weight,
    keyframe_stride,
):
    pose_by_frame = collect_object_pose_by_frame(context, object_id, start, end)
    if len(pose_by_frame) < 2:
        raise ValueError("not enough bbox poses")

    point_keep = np.ones(obj_data["positions"].shape[0], dtype=bool)
    observations = build_multicamera_observations(
        obj_src,
        point_keep,
        start,
        context["num_cameras"],
        max_points_per_observation,
    )
    if len(observations) == 0:
        raise ValueError("no usable tracking observations after filtering")

    obj_positions_h = np.concatenate([obj_data["positions"], np.ones_like(obj_data["positions"][:, :1])], axis=-1)
    keyframes = choose_sparse_keyframes(observations, pose_by_frame, start, keyframe_stride)
    if len(keyframes) < 2:
        raise ValueError("not enough sparse keyframes")

    anchor_observations = filter_observations(observations, source_frames=keyframes, target_frames=keyframes)
    if len(anchor_observations) == 0:
        raise ValueError("no keyframe-to-keyframe observations")

    key_refined_pose_dict, key_delta_r_dict, anchor_res = optimize_rotation_deltas_for_frames(
        anchor_observations,
        keyframes,
        pose_by_frame,
        obj_positions_h,
        context,
        reproj_weight,
    )

    source_frames = sorted(set(obs["source_frame"] for obs in observations) & set(pose_by_frame.keys()))
    refined_pose_dict = dict(key_refined_pose_dict)
    delta_r_dict = dict(key_delta_r_dict)
    expansion_costs = []
    for frame in tqdm(source_frames, desc=f"expand obj_{object_id:03d} bbox rotations"):
        if frame in refined_pose_dict:
            continue
        refined_pose, delta_pose, res = expand_rotation_delta_for_frame(
            frame,
            observations,
            key_refined_pose_dict,
            obj_positions_h,
            pose_by_frame,
            context,
            reproj_weight,
        )
        refined_pose_dict[frame] = refined_pose
        delta_r_dict[frame] = delta_pose
        if res is not None:
            expansion_costs.append(float(res.cost))

    filtered_data = {
        "positions": obj_data["positions"][point_keep],
        "colors": obj_data["colors"][point_keep],
        "normals": obj_data["normals"][point_keep],
        "visibility": obj_data["visibility"][point_keep],
        "visibility_sh": obj_data["visibility_sh"][point_keep],
    }
    info = {
        "frames": np.array(sorted(refined_pose_dict.keys())),
        "keyframes": np.array(keyframes),
        "point_keep_mask": point_keep,
        "num_points_before": int(point_keep.shape[0]),
        "num_points_after": int(point_keep.sum()),
        "outlier_stats": np.array([], dtype=object),
        "cost": float(anchor_res.cost),
        "expansion_cost_mean": float(np.mean(expansion_costs)) if expansion_costs else 0.0,
        "success": bool(anchor_res.success),
        "message": anchor_res.message,
    }
    return filtered_data, refined_pose_dict, delta_r_dict, info


def refine_object_pose_and_points(
    obj_src,
    obj_positions,
    obj_colors,
    obj_normals,
    obj_visibility,
    obj_view_dict,
    vehicle_name,
    ego_frame_poses,
    ixts,
    c2ws,
    start,
    end,
    cam_offset,
    num_cameras,
    keyframe_stride,
    voxel_size,
    min_visible_points,
    min_visible_ratio,
    trajectory_weight,
    anchor_max_nfev,
    frame_max_nfev,
    enable_point_filter,
    point_filter_min_observations,
    point_filter_median_px,
    point_filter_bad_px,
    point_filter_max_bad_ratio,
):
    least_squares = require_least_squares()
    t_candidates = [
        t
        for t in range(start, end, keyframe_stride)
        if t in obj_src and compact_view_index(t, start, num_cameras, cam_offset) in obj_view_dict[vehicle_name]
    ]
    t_candidates = [
        t
        for t in t_candidates
        if frame_is_refine_usable(
            obj_src,
            obj_visibility,
            t,
            compact_view_index(t, start, num_cameras, cam_offset),
            min_visible_points,
            min_visible_ratio,
        )
    ]
    if len(t_candidates) < 2:
        raise ValueError("Need at least two valid keyframes for pose refinement.")
    t1_list = t_candidates
    t2_list = t_candidates
    ixt = ixts[cam_offset]

    obj_positions_local = np.concatenate([obj_positions, np.ones_like(obj_positions[..., :1])], axis=-1)
    visibility_arr = [obj_src[t]["query_mask"] for t in t1_list]

    r_vec_arr = []
    t_vec_arr = []
    ego_pose_arr = []
    c2w_arr = []
    rot_arr = []
    for t2 in t2_list:
        current_view = compact_view_index(t2, start, num_cameras, cam_offset)
        obj_pose_vehicle = obj_view_dict[vehicle_name][current_view][1]
        rot_arr.append(obj_pose_vehicle[:3, :3])
        r_vec_arr.append(cv2.Rodrigues(obj_pose_vehicle[:3, :3])[0])
        t_vec_arr.append(obj_pose_vehicle[:3, 3])
        ego_pose_arr.append(ego_frame_poses[t2])
        c2w_arr.append(c2ws[current_view])

    uv_obs_arr = [[obj_src[t1]["pred_tracks"][t2 - start] for t2 in t2_list] for t1 in t1_list]
    msk_obs_arr = [[obj_src[t1]["pred_visibility"][t2 - start] for t2 in t2_list] for t1 in t1_list]
    delta_yaw_pitch = np.hstack([[0.001, 0.0] for _ in t2_list])
    target_yaw_arr = build_trajectory_target_yaws(rot_arr, t_vec_arr)

    res = least_squares(
        residuals_anchor,
        delta_yaw_pitch,
        method="lm",
        args=(
            len(t1_list),
            len(t2_list),
            r_vec_arr,
            t_vec_arr,
            obj_positions_local,
            visibility_arr,
            msk_obs_arr,
            ego_pose_arr,
            c2w_arr,
            ixt,
            uv_obs_arr,
            None,
            trajectory_weight,
            trajectory_weight,
            target_yaw_arr,
        ),
        max_nfev=anchor_max_nfev,
        verbose=2,
    )
    delta_anchors = np.asarray(res.x)

    obj_position_refined = []
    obj_colors_refined = []
    obj_normals_refined = []
    observed_view_refined = []
    obj_indices_refined = []
    delta_r_dict = {}
    refined_pose_dict = {}

    delta_canonical = np.eye(4)
    delta_canonical[:3, :3] = yaw_pitch_to_rot(delta_anchors[:2])

    for i, t2 in enumerate(t2_list):
        current_view = compact_view_index(t2, start, num_cameras, cam_offset)
        origin_pose_i = obj_view_dict[vehicle_name][current_view][1]
        delta_r_i = np.eye(4)
        delta_r_i[:3, :3] = yaw_pitch_to_rot(delta_anchors[i * 2 : i * 2 + 2])
        delta_r_dict[t2] = delta_r_i
        refined_pose_dict[t2] = np.eye(4)
        refined_pose_dict[t2][:3, :3] = (np.linalg.inv(delta_canonical.T) @ delta_r_i.T @ origin_pose_i.T).T[:3, :3]
        refined_pose_dict[t2][:3, 3] = origin_pose_i[:3, 3]

        mask = obj_visibility[:, current_view]
        points = obj_positions_local[mask] @ np.linalg.inv(delta_r_i.T) @ delta_canonical.T
        obj_position_refined.append(points)
        obj_colors_refined.append(obj_colors[mask])
        obj_normals_refined.append(obj_normals[mask])
        observed_view_refined.append([current_view] * points.shape[0])
        obj_indices_refined.append(np.where(mask)[0])

    r_vec_arr_ref = []
    t_vec_arr_ref = []
    ego_pose_arr_ref = []
    c2w_arr_ref = []
    for j, t2 in enumerate(t2_list):
        current_view = compact_view_index(t2, start, num_cameras, cam_offset)
        obj_pose_vehicle = obj_view_dict[vehicle_name][current_view][1]
        ref_rot = (yaw_pitch_to_rot(delta_anchors[j * 2 : j * 2 + 2]).T @ obj_pose_vehicle[:3, :3].T).T
        r_vec_arr_ref.append(cv2.Rodrigues(ref_rot)[0])
        t_vec_arr_ref.append(obj_pose_vehicle[:3, 3])
        ego_pose_arr_ref.append(ego_frame_poses[t2])
        c2w_arr_ref.append(c2ws[current_view])

    for t0 in tqdm(range(start, end), desc="expanding pose refinement"):
        if t0 in t2_list:
            continue
        if t0 not in obj_src:
            continue
        current_view = compact_view_index(t0, start, num_cameras, cam_offset)
        if current_view not in obj_view_dict[vehicle_name]:
            continue
        obj_view_mask = obj_visibility[:, current_view]
        query_mask = obj_src[t0]["query_mask"]
        origin_pose_i = obj_view_dict[vehicle_name][current_view][1]
        if not frame_is_refine_usable(
            obj_src,
            obj_visibility,
            t0,
            current_view,
            min_visible_points,
            min_visible_ratio,
        ):
            delta_r_i = np.array(delta_canonical, copy=True)
            delta_r_dict[t0] = delta_r_i
            refined_pose_dict[t0] = np.array(origin_pose_i, copy=True)

            points = obj_positions_local[obj_view_mask]
            obj_position_refined.append(points)
            obj_colors_refined.append(obj_colors[obj_view_mask])
            obj_normals_refined.append(obj_normals[obj_view_mask])
            observed_view_refined.append([current_view] * points.shape[0])
            obj_indices_refined.append(np.where(obj_view_mask)[0])
            continue

        uv_obs = [[obj_src[t0]["pred_tracks"][t2 - start] for t2 in t2_list]]
        msk_obs = [[obj_src[t0]["pred_visibility"][t2 - start] for t2 in t2_list]]

        res_i = least_squares(
            residuals_single_delta,
            np.array([0.001, 0.0]),
            method="lm",
            args=(
                1,
                len(t2_list),
                r_vec_arr_ref,
                t_vec_arr_ref,
                obj_positions_local,
                [query_mask],
                msk_obs,
                ego_pose_arr_ref,
                c2w_arr_ref,
                ixt,
                uv_obs,
            ),
            max_nfev=frame_max_nfev,
            verbose=0,
        )

        delta_r_i = np.eye(4)
        delta_r_i[:3, :3] = yaw_pitch_to_rot(res_i.x)
        delta_r_dict[t0] = delta_r_i
        refined_pose_dict[t0] = np.eye(4)
        refined_pose_dict[t0][:3, :3] = (np.linalg.inv(delta_canonical.T) @ delta_r_i.T @ origin_pose_i.T).T[:3, :3]
        refined_pose_dict[t0][:3, 3] = origin_pose_i[:3, 3]

        points = obj_positions_local[obj_view_mask] @ np.linalg.inv(delta_r_i.T) @ delta_canonical.T
        obj_position_refined.append(points)
        obj_colors_refined.append(obj_colors[obj_view_mask])
        obj_normals_refined.append(obj_normals[obj_view_mask])
        observed_view_refined.append([current_view] * points.shape[0])
        obj_indices_refined.append(np.where(obj_view_mask)[0])

    obj_position_refined = np.vstack(obj_position_refined)
    obj_colors_refined = np.vstack(obj_colors_refined)
    obj_normals_refined = np.vstack(obj_normals_refined)
    views_refined = np.hstack(observed_view_refined)
    obj_indices_refined = np.hstack(obj_indices_refined)

    if enable_point_filter:
        point_keep = build_tracking_consistency_point_mask(
            obj_src,
            obj_positions_local,
            refined_pose_dict,
            delta_r_dict,
            delta_canonical,
            ego_frame_poses,
            ixts,
            c2ws,
            start,
            end,
            cam_offset,
            num_cameras,
            point_filter_min_observations,
            point_filter_median_px,
            point_filter_bad_px,
            point_filter_max_bad_ratio,
        )
        refined_keep = point_keep[obj_indices_refined]
        if refined_keep.any():
            obj_position_refined = obj_position_refined[refined_keep]
            obj_colors_refined = obj_colors_refined[refined_keep]
            obj_normals_refined = obj_normals_refined[refined_keep]
            views_refined = views_refined[refined_keep]
        else:
            print("tracking consistency point filter removed all accumulated points; keep unfiltered points instead")

    o3d = require_open3d()
    pcd_refine = o3d.geometry.PointCloud()
    pcd_refine.points = o3d.utility.Vector3dVector(obj_position_refined[:, :3])
    pcd_refine.colors = o3d.utility.Vector3dVector(obj_colors_refined)
    pcd_refine.normals = o3d.utility.Vector3dVector(obj_normals_refined)
    downsampled_pcd, _, point_indices_for_each_voxel = pcd_refine.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=(pcd_refine.get_min_bound() // voxel_size) * voxel_size,
        max_bound=(pcd_refine.get_max_bound() // voxel_size + 1) * voxel_size,
    )

    refined_visibility = np.zeros([np.asarray(downsampled_pcd.points).shape[0], (end - start) * num_cameras], dtype=bool)
    for point_id, indices in enumerate(point_indices_for_each_voxel):
        for idx in indices:
            refined_visibility[point_id, views_refined[idx]] = True

    return downsampled_pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors


def save_outputs(output_dir, vehicle_name, pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors):
    o3d = require_open3d()
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"points3D_obj_{vehicle_name:03d}_refined.ply")
    vis_path = os.path.join(output_dir, f"points3D_obj_{vehicle_name:03d}_refined_visibility.npy")
    meta_path = os.path.join(output_dir, f"points3D_obj_{vehicle_name:03d}_tracking_refine.npz")
    o3d.io.write_point_cloud(ply_path, pcd)
    np.save(vis_path, refined_visibility)
    np.savez(
        meta_path,
        refined_pose_keys=np.array(sorted(refined_pose_dict.keys())),
        refined_pose_values=np.array([refined_pose_dict[k] for k in sorted(refined_pose_dict.keys())]),
        delta_pose_keys=np.array(sorted(delta_r_dict.keys())),
        delta_pose_values=np.array([delta_r_dict[k] for k in sorted(delta_r_dict.keys())]),
        delta_anchors=delta_anchors,
    )
    print(f"Saved refined point cloud: {ply_path}")
    print(f"Saved refined visibility: {vis_path}")
    print(f"Saved pose metadata: {meta_path}")


def multicamera_refine_cache_paths(
    save_dir,
    vehicle_name,
    start,
    end,
    cameras,
    voxel_size,
    max_points_per_observation,
    keyframe_stride,
):
    tag = make_cache_tag(
        {
            "vehicle_name": vehicle_name,
            "start": start,
            "end": end,
            "cameras": cameras,
            "voxel_size": voxel_size,
            "max_points_per_observation": max_points_per_observation,
            "keyframe_stride": keyframe_stride,
            "version": "multicamera_rotation_delta_keyframe_v1",
        }
    )
    cache_dir = os.path.join(save_dir, "multicamera_refine")
    prefix = os.path.join(cache_dir, f"obj_{vehicle_name:03d}_{tag}")
    return {
        "ply": f"{prefix}_filtered.ply",
        "visibility": f"{prefix}_visibility.npy",
        "visibility_sh": f"{prefix}_visibility_sh.npy",
        "meta": f"{prefix}_meta.npz",
    }


def save_multicamera_refine_result(paths, filtered_data, refined_pose_dict, delta_r_dict, info):
    o3d = require_open3d()
    os.makedirs(os.path.dirname(paths["ply"]), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_data["positions"])
    pcd.colors = o3d.utility.Vector3dVector(filtered_data["colors"])
    pcd.normals = o3d.utility.Vector3dVector(filtered_data["normals"])
    o3d.io.write_point_cloud(paths["ply"], pcd)
    np.save(paths["visibility"], filtered_data["visibility"])
    np.save(paths["visibility_sh"], filtered_data["visibility_sh"])
    np.savez(
        paths["meta"],
        refined_pose_keys=np.array(sorted(refined_pose_dict.keys())),
        refined_pose_values=np.array([refined_pose_dict[k] for k in sorted(refined_pose_dict.keys())]),
        delta_pose_keys=np.array(sorted(delta_r_dict.keys())),
        delta_pose_values=np.array([delta_r_dict[k] for k in sorted(delta_r_dict.keys())]),
        point_keep_mask=info["point_keep_mask"],
        frames=info["frames"],
        keyframes=info["keyframes"],
        num_points_before=info["num_points_before"],
        num_points_after=info["num_points_after"],
        outlier_stats=info["outlier_stats"],
        cost=info["cost"],
        expansion_cost_mean=info["expansion_cost_mean"],
        success=info["success"],
        message=info["message"],
    )
    print(f"Saved multi-camera refine point cloud: {paths['ply']}")
    print(f"Saved multi-camera refine metadata: {paths['meta']}")


def load_multicamera_refine_result(paths):
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    o3d = require_open3d()
    pcd = o3d.io.read_point_cloud(paths["ply"])
    meta = np.load(paths["meta"], allow_pickle=True)
    filtered_data = {
        "positions": np.asarray(pcd.points),
        "colors": np.asarray(pcd.colors),
        "normals": np.asarray(pcd.normals),
        "visibility": np.load(paths["visibility"]),
        "visibility_sh": np.load(paths["visibility_sh"]),
    }
    refined_pose_dict = {
        int(key): value
        for key, value in zip(meta["refined_pose_keys"], meta["refined_pose_values"])
    }
    delta_r_dict = {
        int(key): value
        for key, value in zip(meta["delta_pose_keys"], meta["delta_pose_values"])
    }
    info = {
        "point_keep_mask": meta["point_keep_mask"],
        "frames": meta["frames"],
        "keyframes": meta["keyframes"] if "keyframes" in meta.files else np.array([], dtype=np.int64),
        "num_points_before": int(meta["num_points_before"]),
        "num_points_after": int(meta["num_points_after"]),
        "outlier_stats": meta["outlier_stats"],
        "cost": float(meta["cost"]),
        "expansion_cost_mean": float(meta["expansion_cost_mean"]) if "expansion_cost_mean" in meta.files else 0.0,
        "success": bool(meta["success"]),
        "message": str(meta["message"]),
    }
    print(f"Loaded multi-camera refine metadata: {paths['meta']}")
    return filtered_data, refined_pose_dict, delta_r_dict, info


def refine_cache_paths(
    save_dir,
    vehicle_name,
    start,
    end,
    cam_offset,
    keyframe_stride,
    voxel_size,
    min_visible_points,
    min_visible_ratio,
    trajectory_weight,
    anchor_max_nfev,
    frame_max_nfev,
    enable_point_filter,
    point_filter_min_observations,
    point_filter_median_px,
    point_filter_bad_px,
    point_filter_max_bad_ratio,
):
    tag = make_cache_tag(
        {
            "vehicle_name": vehicle_name,
            "start": start,
            "end": end,
            "cam_offset": cam_offset,
            "keyframe_stride": keyframe_stride,
            "voxel_size": voxel_size,
            "min_visible_points": min_visible_points,
            "min_visible_ratio": min_visible_ratio,
            "trajectory_weight": trajectory_weight,
            "anchor_max_nfev": anchor_max_nfev,
            "frame_max_nfev": frame_max_nfev,
            "enable_point_filter": enable_point_filter,
            "point_filter_min_observations": point_filter_min_observations,
            "point_filter_median_px": point_filter_median_px,
            "point_filter_bad_px": point_filter_bad_px,
            "point_filter_max_bad_ratio": point_filter_max_bad_ratio,
            "version": "single_camera_refine_yaw_pitch_traj_point_filter_v1",
        }
    )
    cache_dir = os.path.join(save_dir, "refine_cache")
    prefix = os.path.join(cache_dir, f"obj_{vehicle_name:03d}_{tag}")
    return {
        "ply": f"{prefix}.ply",
        "visibility": f"{prefix}_visibility.npy",
        "meta": f"{prefix}_meta.npz",
    }


def save_refine_cache(paths, pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors):
    o3d = require_open3d()
    os.makedirs(os.path.dirname(paths["ply"]), exist_ok=True)
    o3d.io.write_point_cloud(paths["ply"], pcd)
    np.save(paths["visibility"], refined_visibility)
    np.savez(
        paths["meta"],
        refined_pose_keys=np.array(sorted(refined_pose_dict.keys())),
        refined_pose_values=np.array([refined_pose_dict[k] for k in sorted(refined_pose_dict.keys())]),
        delta_pose_keys=np.array(sorted(delta_r_dict.keys())),
        delta_pose_values=np.array([delta_r_dict[k] for k in sorted(delta_r_dict.keys())]),
        delta_anchors=delta_anchors,
    )
    print(f"Saved pose refine cache: {paths['meta']}")


def load_refine_cache(paths):
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    o3d = require_open3d()
    pcd = o3d.io.read_point_cloud(paths["ply"])
    refined_visibility = np.load(paths["visibility"])
    meta = np.load(paths["meta"], allow_pickle=True)
    refined_pose_dict = {
        int(key): value
        for key, value in zip(meta["refined_pose_keys"], meta["refined_pose_values"])
    }
    delta_r_dict = {
        int(key): value
        for key, value in zip(meta["delta_pose_keys"], meta["delta_pose_values"])
    }
    delta_anchors = meta["delta_anchors"]
    print(f"Loaded pose refine cache: {paths['meta']}")
    return pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors


def save_cross_camera_observations(output_dir, cross_cam_observations):
    if not cross_cam_observations:
        return
    cross_cam_dir = os.path.join(output_dir, "tracking_cache")
    os.makedirs(cross_cam_dir, exist_ok=True)
    for object_id, frames in cross_cam_observations.items():
        path = os.path.join(cross_cam_dir, f"obj_{object_id:03d}_cross_camera_observations.npz")
        np.savez(
            path,
            frames=np.array(sorted(frames.keys())),
            visible_views=np.array([frames[k] for k in sorted(frames.keys())], dtype=object),
        )
        print(f"Saved cross-camera visibility records: {path}")


def object_debug_color(object_id):
    rng = np.random.default_rng(int(object_id) * 1009 + 17)
    color = rng.integers(64, 256, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_points(canvas, points, color, radius=2, max_points=None):
    if max_points is not None and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[indices]
    h, w = canvas.shape[:2]
    for x, y in points.astype(np.int32):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1)


def draw_projected_local_points(
    canvas,
    points_local_h,
    obj_pose_vehicle,
    ego_pose,
    c2w,
    ixt,
    color,
    radius=2,
    point_stride=1,
    max_points=None,
):
    if points_local_h.shape[0] == 0:
        return 0
    uv, depth = project_points_with_depth(points_local_h, obj_pose_vehicle, ego_pose, c2w, ixt)
    h, w = canvas.shape[:2]
    valid = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
        & (depth > 0)
    )
    indices = np.where(valid)[0]
    if point_stride > 1:
        indices = indices[::point_stride]
    if max_points is not None and indices.size > max_points:
        sample_ids = np.linspace(0, indices.size - 1, max_points).astype(np.int64)
        indices = indices[sample_ids]
    for x, y in uv[indices].astype(np.int32):
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1)
    return int(indices.size)


def make_bbox_corners_h(length, width, height):
    x0, x1 = -length / 2, length / 2
    y0, y1 = -width / 2, width / 2
    z0, z1 = -height / 2, height / 2
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    return np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=-1)


def draw_projected_bbox(canvas, corners_uv, corners_depth, color, thickness=2):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    h, w = canvas.shape[:2]
    valid_depth = corners_depth > 0
    pts = corners_uv.astype(np.int32)
    for a, b in edges:
        if not (valid_depth[a] and valid_depth[b]):
            continue
        ok, p0, p1 = cv2.clipLine((0, 0, w, h), tuple(pts[a]), tuple(pts[b]))
        if ok:
            cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)


def save_tracking_debug_videos(
    save_dir,
    image_filenames,
    ego_frame_poses,
    ixts,
    c2ws,
    obj_view_dict,
    object_info,
    obj_src_by_vehicle,
    start,
    end,
    num_cameras,
    fps,
    point_stride,
    max_points_per_track,
):
    video_dir = os.path.join(save_dir, "vis", "tracking_videos")
    os.makedirs(video_dir, exist_ok=True)

    entries_by_cam = {cam_offset: [] for cam_offset in range(num_cameras)}
    for object_id, src_entries in obj_src_by_vehicle.items():
        for source_view, entry in src_entries.items():
            cam_offset = int(entry.get("source_cam_offset", source_view % num_cameras))
            entries_by_cam.setdefault(cam_offset, []).append((object_id, source_view, entry))

    for cam_offset in range(num_cameras):
        first_view = compact_view_index(start, start, num_cameras, cam_offset)
        first_image = cv2.imread(image_filenames[first_view])
        if first_image is None:
            continue
        height, width = first_image.shape[:2]
        out_path = os.path.join(video_dir, f"cam_{cam_offset}_tracking.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        for frame_id in tqdm(range(start, end), desc=f"tracking video cam {cam_offset}"):
            rel_frame = frame_id - start
            view_idx = compact_view_index(frame_id, start, num_cameras, cam_offset)
            image = cv2.imread(image_filenames[view_idx])
            if image is None:
                image = np.zeros((height, width, 3), dtype=np.uint8)

            canvas = image.copy()
            tracked_counts = {}

            for object_id, source_view, entry in entries_by_cam.get(cam_offset, []):
                pred_tracks = entry["pred_tracks"]
                pred_visibility = entry["pred_visibility"]
                if rel_frame >= pred_tracks.shape[0]:
                    continue

                visible = pred_visibility[rel_frame].astype(bool)
                if point_stride > 1:
                    stride_mask = np.zeros_like(visible)
                    stride_mask[::point_stride] = True
                    visible = np.logical_and(visible, stride_mask)
                if visible.sum() == 0:
                    continue

                color = object_debug_color(object_id)
                points = pred_tracks[rel_frame, visible]
                draw_points(canvas, points, color, radius=2, max_points=max_points_per_track)
                tracked_counts[object_id] = tracked_counts.get(object_id, 0) + int(visible.sum())

                if int(source_view) == int(view_idx):
                    source_points = pred_tracks[rel_frame, visible]
                    draw_points(canvas, source_points, (0, 255, 255), radius=4, max_points=max_points_per_track)
                    draw_points(canvas, source_points, (0, 0, 0), radius=1, max_points=max_points_per_track)

            cv2.putText(
                canvas,
                f"cam={cam_offset} frame={frame_id} tracked_objs={len(tracked_counts)}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y0 = 60
            for row, object_id in enumerate(sorted(tracked_counts.keys())[:12]):
                color = object_debug_color(object_id)
                cv2.putText(
                    canvas,
                    f"obj_{object_id:03d}: {tracked_counts[object_id]}",
                    (16, y0 + row * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            for object_id, view_entries in obj_view_dict.items():
                if view_idx not in view_entries or object_id not in object_info:
                    continue
                obj_meta = object_info[object_id]
                bbox_corners_local = make_bbox_corners_h(obj_meta["length"], obj_meta["width"], obj_meta["height"])
                obj_pose_vehicle = view_entries[view_idx][1]
                bbox_uv, bbox_depth = project_points_with_depth(
                    bbox_corners_local,
                    obj_pose_vehicle,
                    ego_frame_poses[frame_id],
                    c2ws[view_idx],
                    ixts[view_idx],
                )
                draw_projected_bbox(canvas, bbox_uv, bbox_depth, object_debug_color(object_id), thickness=2)

            writer.write(canvas)

        writer.release()
        print(f"Saved tracking debug video: {out_path}")


def save_refine_debug_videos(
    save_dir,
    image_filenames,
    ego_frame_poses,
    ixts,
    c2ws,
    obj_view_dict,
    object_info,
    object_bank,
    refined_results,
    start,
    end,
    num_cameras,
    fps,
    point_stride,
    max_points_per_object,
):
    if not refined_results:
        return
    video_dir = os.path.join(save_dir, "vis", "refine_videos")
    os.makedirs(video_dir, exist_ok=True)

    raw_point_cache = {}
    refined_point_cache = {}
    bbox_cache = {}
    for object_id, result in refined_results.items():
        if object_id not in object_bank or object_id not in object_info:
            continue
        raw_points = object_bank[object_id]["positions"]
        raw_point_cache[object_id] = np.concatenate([raw_points, np.ones_like(raw_points[:, :1])], axis=-1)
        refined_points = np.asarray(result["pcd"].points)
        refined_point_cache[object_id] = np.concatenate([refined_points, np.ones_like(refined_points[:, :1])], axis=-1)
        obj_meta = object_info[object_id]
        bbox_cache[object_id] = make_bbox_corners_h(obj_meta["length"], obj_meta["width"], obj_meta["height"])

    for cam_offset in range(num_cameras):
        first_view = compact_view_index(start, start, num_cameras, cam_offset)
        first_image = cv2.imread(image_filenames[first_view])
        if first_image is None:
            continue
        height, width = first_image.shape[:2]
        out_path = os.path.join(video_dir, f"cam_{cam_offset}_refine.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        for frame_id in tqdm(range(start, end), desc=f"refine video cam {cam_offset}"):
            view_idx = compact_view_index(frame_id, start, num_cameras, cam_offset)
            image = cv2.imread(image_filenames[view_idx])
            if image is None:
                image = np.zeros((height, width, 3), dtype=np.uint8)
            canvas = image.copy()
            ego_pose = ego_frame_poses[frame_id]
            c2w = c2ws[view_idx]
            ixt = ixts[view_idx]
            drawn_objects = 0

            for object_id in sorted(refined_results.keys()):
                if object_id not in raw_point_cache or object_id not in bbox_cache:
                    continue
                if object_id not in obj_view_dict or view_idx not in obj_view_dict[object_id]:
                    continue
                result = refined_results[object_id]
                if frame_id not in result["refined_pose_dict"]:
                    continue

                old_pose = obj_view_dict[object_id][view_idx][1]
                refined_pose = result["refined_pose_dict"][frame_id]
                raw_count = draw_projected_local_points(
                    canvas,
                    raw_point_cache[object_id],
                    old_pose,
                    ego_pose,
                    c2w,
                    ixt,
                    (0, 255, 255),
                    radius=2,
                    point_stride=point_stride,
                    max_points=max_points_per_object,
                )
                refined_count = draw_projected_local_points(
                    canvas,
                    refined_point_cache[object_id],
                    refined_pose,
                    ego_pose,
                    c2w,
                    ixt,
                    (0, 0, 255),
                    radius=2,
                    point_stride=point_stride,
                    max_points=max_points_per_object,
                )
                if raw_count == 0 and refined_count == 0:
                    continue

                bbox_corners_local = bbox_cache[object_id]
                old_bbox_uv, old_bbox_depth = project_points_with_depth(
                    bbox_corners_local,
                    old_pose,
                    ego_pose,
                    c2w,
                    ixt,
                )
                refined_bbox_uv, refined_bbox_depth = project_points_with_depth(
                    bbox_corners_local,
                    refined_pose,
                    ego_pose,
                    c2w,
                    ixt,
                )
                draw_projected_bbox(canvas, old_bbox_uv, old_bbox_depth, (0, 255, 255), thickness=2)
                draw_projected_bbox(canvas, refined_bbox_uv, refined_bbox_depth, (0, 0, 255), thickness=2)
                drawn_objects += 1

            cv2.putText(
                canvas,
                f"cam={cam_offset} frame={frame_id} refined_objs={drawn_objects}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "raw: yellow  refined: red",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(canvas)

        writer.release()
        print(f"Saved refine debug video: {out_path}")


def draw_pose_projection_debug_images(
    save_dir,
    vehicle_name,
    object_info,
    image_filenames,
    ego_frame_poses,
    ixts,
    c2ws,
    obj_view_dict,
    obj_positions,
    obj_visibility,
    refined_pcd,
    refined_visibility,
    refined_pose_dict,
    start,
    end,
    cam_offset,
    num_cameras,
    frame_stride,
    max_frames,
):
    vis_dir = os.path.join(save_dir, "vis", f"obj_{vehicle_name:03d}")
    os.makedirs(vis_dir, exist_ok=True)

    obj_positions_local = np.concatenate([obj_positions, np.ones_like(obj_positions[..., :1])], axis=-1)
    refined_points = np.asarray(refined_pcd.points)
    refined_points_local = np.concatenate([refined_points, np.ones_like(refined_points[..., :1])], axis=-1)
    obj_meta = object_info[vehicle_name]
    bbox_corners_local = make_bbox_corners_h(obj_meta["length"], obj_meta["width"], obj_meta["height"])

    saved = 0
    for frame_id in range(start, end, frame_stride):
        if saved >= max_frames:
            break
        current_view = compact_view_index(frame_id, start, num_cameras, cam_offset)
        if frame_id not in refined_pose_dict:
            continue
        if vehicle_name not in obj_view_dict or current_view not in obj_view_dict[vehicle_name]:
            continue

        img = cv2.imread(image_filenames[current_view])
        if img is None:
            continue
        h, w = img.shape[:2]
        ixt = ixts[current_view]
        ego_pose = ego_frame_poses[frame_id]
        c2w = c2ws[current_view]

        old_pose = obj_view_dict[vehicle_name][current_view][1]
        raw_uv, raw_depth = project_points_with_depth(obj_positions_local, old_pose, ego_pose, c2w, ixt)
        old_valid = (
            (raw_uv[:, 0] >= 0)
            & (raw_uv[:, 0] < w)
            & (raw_uv[:, 1] >= 0)
            & (raw_uv[:, 1] < h)
            & (raw_depth > 0)
        )

        refined_uv, refined_depth = project_points_with_depth(
            refined_points_local,
            refined_pose_dict[frame_id],
            ego_pose,
            c2w,
            ixt,
        )
        refined_valid = (
            (refined_uv[:, 0] >= 0)
            & (refined_uv[:, 0] < w)
            & (refined_uv[:, 1] >= 0)
            & (refined_uv[:, 1] < h)
            & (refined_depth > 0)
        )

        canvas = img.copy()
        for x, y in raw_uv[old_valid].astype(np.int32):
            cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1)
        for x, y in refined_uv[refined_valid].astype(np.int32):
            cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

        old_bbox_uv, old_bbox_depth = project_points_with_depth(bbox_corners_local, old_pose, ego_pose, c2w, ixt)
        refined_bbox_uv, refined_bbox_depth = project_points_with_depth(
            bbox_corners_local,
            refined_pose_dict[frame_id],
            ego_pose,
            c2w,
            ixt,
        )
        draw_projected_bbox(canvas, old_bbox_uv, old_bbox_depth, (0, 255, 255), thickness=2)
        draw_projected_bbox(canvas, refined_bbox_uv, refined_bbox_depth, (0, 0, 255), thickness=2)

        out_path = os.path.join(vis_dir, f"frame_{frame_id:06d}_cam_{cam_offset}.png")
        cv2.imwrite(out_path, canvas)
        saved += 1


def visualize_open3d_debug(original_points, original_colors, refined_pcd):
    o3d = require_open3d()

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_pcd.colors = o3d.utility.Vector3dVector(original_colors)

    original_pcd.paint_uniform_color([1.0, 0.85, 0.0])
    refined_debug_pcd = o3d.geometry.PointCloud(refined_pcd)
    refined_debug_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    debug_geometries = [
        original_pcd,
        refined_debug_pcd,
        coord_frame,
    ]

    # Keep the geometry variables above split across lines for breakpoint inspection.
    o3d.visualization.draw_geometries(debug_geometries, point_show_normal=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-camera CoTracker tracking for generated object point clouds, with pose refine intentionally skipped by default.")
    parser.add_argument("--datadir", required=True, help="Waymo-style data directory containing images/track/ego_pose/intrinsics/extrinsics.")
    parser.add_argument("--output-file", required=True, help="Training output directory containing input_ply/points3D_obj_XXX.{ply,npy}.")
    parser.add_argument("--vehicle-name", type=int, nargs="*", default=None, help="Optional object track ids. Omit to process every object ply found in input_ply.")
    parser.add_argument("--checkpoint", default=None, help="CoTracker checkpoint path, e.g. scaled_offline.pth. Required only when CoTracker cache is missing or --force-cotracker is set.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive start frame.")
    parser.add_argument("--end", type=int, default=60, help="Exclusive end frame for tracking.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2], help="Camera ids included in the scene context.")
    parser.add_argument("--num-dataset-cameras", type=int, default=5, help="Total camera count in the raw dataset image folder.")
    parser.add_argument("--cam-offset", type=int, default=None, help="Single camera offset inside --cameras order; auto-selected from visibility by default.")
    parser.add_argument("--image-step", type=int, default=2, help="Downsample image/video by this factor before CoTracker.")
    parser.add_argument("--keyframe-stride", type=int, default=5, help="Stride for anchor keyframes in bbox pose refinement.")
    parser.add_argument("--voxel-size", type=float, default=0.15, help="Voxel size for refined object point cloud.")
    parser.add_argument("--refine-min-visible-points", type=int, default=20, help="Minimum visible/query point count required to refine a frame.")
    parser.add_argument("--refine-min-visible-ratio", type=float, default=0.05, help="Minimum object point visibility ratio required to refine a frame.")
    parser.add_argument("--refine-trajectory-weight", type=float, default=5.0, help="Weight for sparse-keyframe trajectory heading regularization.")
    parser.add_argument("--refine-anchor-max-nfev", type=int, default=80, help="Maximum function evaluations for sparse keyframe bbox refinement.")
    parser.add_argument("--refine-frame-max-nfev", type=int, default=30, help="Maximum function evaluations for per-frame bbox refinement expansion.")
    parser.add_argument("--disable-point-filter", action="store_true", help="Disable tracking reprojection consistency filtering for refined object points.")
    parser.add_argument("--point-filter-min-observations", type=int, default=3, help="Minimum valid tracking reprojection observations before a point can be filtered.")
    parser.add_argument("--point-filter-median-px", type=float, default=30.0, help="Filter a point when its median tracking reprojection residual exceeds this many pixels.")
    parser.add_argument("--point-filter-bad-px", type=float, default=60.0, help="Pixel residual threshold used to count bad tracking reprojection observations.")
    parser.add_argument("--point-filter-max-bad-ratio", type=float, default=0.5, help="Filter a point when this ratio of valid observations exceeds --point-filter-bad-px.")
    parser.add_argument("--visibility-sh-degree", type=int, default=2, help="SH degree for voxel visibility source-view encoding.")
    parser.add_argument("--visibility-sh-kappa", type=float, default=8.0, help="vMF concentration used by visibility SH encoding.")
    parser.add_argument("--max-queries-per-call", type=int, default=1024, help="Split concatenated CoTracker queries into chunks to reduce peak memory.")
    parser.add_argument("--debug-frame-stride", type=int, default=1, help="Frame stride for saved projection debug images.")
    parser.add_argument("--max-debug-frames", type=int, default=100, help="Maximum saved projection debug images per object.")
    parser.add_argument("--skip-tracking-videos", action="store_true", help="Do not save per-camera tracking debug videos after loading/saving tracking cache.")
    parser.add_argument("--skip-refine-videos", action="store_true", help="Do not save per-camera old/refined object projection videos after pose refinement.")
    parser.add_argument("--tracking-video-fps", type=float, default=10.0, help="FPS for per-camera tracking debug videos.")
    parser.add_argument("--tracking-video-point-stride", type=int, default=4, help="Draw one tracked point every N query points in tracking debug videos.")
    parser.add_argument("--tracking-video-max-points", type=int, default=2000, help="Maximum drawn points per object/source track on each video frame.")
    parser.add_argument("--show-open3d", action="store_true", help="Open an Open3D window for each refined object after saving outputs.")
    parser.add_argument("--force-cotracker", action="store_true", help="Ignore existing CoTracker cache and run the model again.")
    parser.add_argument("--force-pointcloud", action="store_true", help="Ignore generated object-bank cache and rebuild point clouds from dataparser inputs.")
    parser.add_argument("--force-refine", action="store_true", help="Ignore existing pose refine cache and optimize again.")
    parser.add_argument("--run-refine", action="store_true", help="Run the single-camera bbox pose refinement path after tracking.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults to <output-file>/input_ply.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start.")
    if args.max_queries_per_call <= 0:
        raise ValueError("--max-queries-per-call must be positive.")
    if args.debug_frame_stride <= 0:
        raise ValueError("--debug-frame-stride must be positive.")
    if args.tracking_video_point_stride <= 0:
        raise ValueError("--tracking-video-point-stride must be positive.")
    if args.tracking_video_fps <= 0:
        raise ValueError("--tracking-video-fps must be positive.")
    if args.point_filter_min_observations < 1:
        raise ValueError("--point-filter-min-observations must be positive.")
    if args.point_filter_median_px <= 0 or args.point_filter_bad_px <= 0:
        raise ValueError("--point-filter pixel thresholds must be positive.")
    if args.point_filter_max_bad_ratio < 0 or args.point_filter_max_bad_ratio > 1:
        raise ValueError("--point-filter-max-bad-ratio must be in [0, 1].")
    if args.refine_min_visible_points < 0:
        raise ValueError("--refine-min-visible-points must be non-negative.")
    if not (0.0 <= args.refine_min_visible_ratio <= 1.0):
        raise ValueError("--refine-min-visible-ratio must be in [0, 1].")
    if args.refine_trajectory_weight < 0:
        raise ValueError("--refine-trajectory-weight must be non-negative.")
    if args.refine_anchor_max_nfev <= 0:
        raise ValueError("--refine-anchor-max-nfev must be positive.")
    if args.refine_frame_max_nfev <= 0:
        raise ValueError("--refine-frame-max-nfev must be positive.")
    torch = require_torch()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    selected_frames = [args.start, args.end - 1]
    context = build_scene_context(args.datadir, selected_frames, args.cameras, args.num_dataset_cameras)
    object_ids = args.vehicle_name if args.vehicle_name else sorted(context["object_info"].keys())
    object_ids = [object_id for object_id in object_ids if object_id in context["obj_view_dict"]]
    if not object_ids:
        raise ValueError("No requested object has valid object views in the selected frame range.")
    save_dir = args.save_dir or os.path.join(args.output_file, "input_ply")

    object_bank_cache = object_bank_cache_path(
        save_dir,
        object_ids,
        args.start,
        args.end,
        args.cameras,
        args.voxel_size,
        args.visibility_sh_degree,
        args.visibility_sh_kappa,
    )
    object_bank = None if args.force_pointcloud else load_object_bank_cache(object_bank_cache)
    if object_bank is None:
        object_bank = build_object_bank_from_dataparser(
            args.datadir,
            context,
            object_ids,
            args.start,
            args.end,
            args.voxel_size,
            args.visibility_sh_degree,
            args.visibility_sh_kappa,
        )
        save_object_bank_cache(object_bank_cache, object_bank)
    object_ids = [object_id for object_id in object_ids if object_id in object_bank]
    if not object_ids:
        raise ValueError("No requested object has generated point cloud in the selected frame range.")

    cotracker_cache = tracking_cache_path(
        save_dir,
        object_ids,
        args.start,
        args.end,
        args.cameras,
        args.image_step,
        args.max_queries_per_call,
        args.voxel_size,
        args.visibility_sh_degree,
        args.visibility_sh_kappa,
    )
    cached_tracking = None if args.force_cotracker else load_tracking_cache(cotracker_cache)
    if cached_tracking is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when CoTracker cache is missing or --force-cotracker is set.")
        from cotracker.predictor import CoTrackerPredictor

        model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=False).to(args.device)
        obj_src_by_vehicle, cross_cam_observations = track_visible_objects_all_cameras(
            model,
            context,
            object_bank,
            args.start,
            args.end,
            args.image_step,
            args.device,
            args.max_queries_per_call,
        )
        save_tracking_cache(cotracker_cache, obj_src_by_vehicle, cross_cam_observations)
    else:
        obj_src_by_vehicle, cross_cam_observations = cached_tracking

    save_cross_camera_observations(save_dir, cross_cam_observations)
    if not args.skip_tracking_videos:
        save_tracking_debug_videos(
            save_dir,
            context["image_filenames"],
            context["ego_frame_poses"],
            context["ixts"],
            context["c2ws"],
            context["obj_view_dict"],
            context["object_info"],
            obj_src_by_vehicle,
            args.start,
            args.end,
            context["num_cameras"],
            args.tracking_video_fps,
            args.tracking_video_point_stride,
            args.tracking_video_max_points,
        )

    if not args.run_refine:
        print("Tracking cache is ready. Skip bbox refinement; pass --run-refine to run single-camera pose refinement.")
        return

    refine_cam_offset = choose_camera_offset(object_bank, context["num_cameras"], args.cam_offset)
    print(f"Using camera offset {refine_cam_offset} for single-camera pose refinement")

    refined_results = {}
    for vehicle_name in object_ids:
        if vehicle_name not in obj_src_by_vehicle:
            print(f"object {vehicle_name} has no CoTracker source view, skip single-camera refinement")
            continue

        obj_src_single = {}
        for source_view, entry in obj_src_by_vehicle[vehicle_name].items():
            if int(entry.get("source_cam_offset", source_view % context["num_cameras"])) != refine_cam_offset:
                continue
            obj_src_single[int(entry["source_frame"])] = entry
        if len(obj_src_single) == 0:
            print(f"object {vehicle_name} has no source frame on camera offset {refine_cam_offset}, skip")
            continue

        obj_data = object_bank[vehicle_name]
        refine_paths = refine_cache_paths(
            save_dir,
            vehicle_name,
            args.start,
            args.end,
            refine_cam_offset,
            args.keyframe_stride,
            args.voxel_size,
            args.refine_min_visible_points,
            args.refine_min_visible_ratio,
            args.refine_trajectory_weight,
            args.refine_anchor_max_nfev,
            args.refine_frame_max_nfev,
            not args.disable_point_filter,
            args.point_filter_min_observations,
            args.point_filter_median_px,
            args.point_filter_bad_px,
            args.point_filter_max_bad_ratio,
        )
        cached_refine = None if args.force_refine else load_refine_cache(refine_paths)
        if cached_refine is None:
            try:
                pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors = refine_object_pose_and_points(
                    obj_src_single,
                    obj_data["positions"],
                    obj_data["colors"],
                    obj_data["normals"],
                    obj_data["visibility"],
                    context["obj_view_dict"],
                    vehicle_name,
                    context["ego_frame_poses"],
                    context["ixts"],
                    context["c2ws"],
                    args.start,
                    args.end,
                    refine_cam_offset,
                    context["num_cameras"],
                    args.keyframe_stride,
                    args.voxel_size,
                    args.refine_min_visible_points,
                    args.refine_min_visible_ratio,
                    args.refine_trajectory_weight,
                    args.refine_anchor_max_nfev,
                    args.refine_frame_max_nfev,
                    not args.disable_point_filter,
                    args.point_filter_min_observations,
                    args.point_filter_median_px,
                    args.point_filter_bad_px,
                    args.point_filter_max_bad_ratio,
                )
            except ValueError as exc:
                print(f"object {vehicle_name} skipped during single-camera refinement: {exc}")
                continue
            save_refine_cache(refine_paths, pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors)
        else:
            pcd, refined_visibility, refined_pose_dict, delta_r_dict, delta_anchors = cached_refine

        refined_results[vehicle_name] = {
            "pcd": pcd,
            "refined_visibility": refined_visibility,
            "refined_pose_dict": refined_pose_dict,
            "delta_r_dict": delta_r_dict,
            "delta_anchors": delta_anchors,
        }

        print(
            f"object {vehicle_name}: single-camera refinement produced "
            f"{len(refined_pose_dict)} refined poses and {np.asarray(pcd.points).shape[0]} points"
        )

        save_outputs(
            save_dir,
            vehicle_name,
            pcd,
            refined_visibility,
            refined_pose_dict,
            delta_r_dict,
            delta_anchors,
        )
        draw_pose_projection_debug_images(
            save_dir,
            vehicle_name,
            context["object_info"],
            context["image_filenames"],
            context["ego_frame_poses"],
            context["ixts"],
            context["c2ws"],
            context["obj_view_dict"],
            obj_data["positions"],
            obj_data["visibility"],
            pcd,
            refined_visibility,
            refined_pose_dict,
            args.start,
            args.end,
            refine_cam_offset,
            context["num_cameras"],
            args.debug_frame_stride,
            args.max_debug_frames,
        )

        if args.show_open3d:
            visualize_open3d_debug(obj_data["positions"], obj_data["colors"], pcd)

    if not args.skip_refine_videos:
        save_refine_debug_videos(
            save_dir,
            context["image_filenames"],
            context["ego_frame_poses"],
            context["ixts"],
            context["c2ws"],
            context["obj_view_dict"],
            context["object_info"],
            object_bank,
            refined_results,
            args.start,
            args.end,
            context["num_cameras"],
            args.tracking_video_fps,
            args.tracking_video_point_stride,
            args.tracking_video_max_points,
        )


if __name__ == "__main__":
    main()
