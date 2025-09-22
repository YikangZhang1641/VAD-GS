import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm 
from lib.config import cfg
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
from lib.utils.data_utils import get_val_frames
from lib.utils.graphics_utils import get_rays, sphere_intersection
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.datasets.base_readers import storePly, get_Sphere_Norm
from scipy.spatial.transform import Rotation as R

# waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}
waymo_track2label = {"vehicle": 0, "Vehicle": 0,
                     "pedestrian": 1, "Pedestrian": 1, 
                     "cyclist": 2, 
                     "sign": 3, 
                     "misc": -1}

# _camera2label = {
#     'FRONT': 0,
#     'FRONT_LEFT': 1,
#     'FRONT_RIGHT': 2,
#     'SIDE_LEFT': 3,
#     'SIDE_RIGHT': 4,
# }

# _label2camera = {
#     0: 'FRONT',
#     1: 'FRONT_LEFT',
#     2: 'FRONT_RIGHT',
#     3: 'SIDE_LEFT',
#     4: 'SIDE_RIGHT',
# }
# image_heights = [1280, 1280, 1280, 886, 886]
# image_widths = [1920, 1920, 1920, 1920, 1920]

image_heights = [900, 900, 900, 900, 900, 900]
image_widths = [1600, 1600, 1600, 1600, 1600, 1600]


image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
# image_filename_to_frame = lambda x: int(x.split('.')[0][:6])
image_filename_to_frame = lambda x: int(x.split('.')[0][:3])

# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info_ds(datadir, colmap_basedir):
    lidar_pose_dir = os.path.join(datadir, 'lidar_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    
    intrinsics = [] # 5cam
    extrinsics = [] # 5cam
    for i in range(6):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        
        lidar_to_world_start = np.loadtxt(os.path.join(lidar_pose_dir, "000.txt"))
        camera_front_start = np.loadtxt(
            os.path.join(extrinsics_dir, f"000_{i:01d}.txt")
        )
        ext = np.linalg.inv(lidar_to_world_start) @  camera_front_start
        extrinsics.append(ext)

    # try:
    #     ego_cam_poses = [[] for i in range(6)]
    #     cam_extrinsics = read_extrinsics_binary(os.path.join(colmap_basedir, "triangulated/sparse/model/images.bin"))
    #     for k in sorted(cam_extrinsics.keys()):
    #         v = cam_extrinsics[k]
    #         cam_id = v.camera_id
    #         rquat = np.array(v.qvec)
    #         rquat[3], rquat[0], rquat[1], rquat[2] = rquat[0], rquat[1], rquat[2], rquat[3]

    #         transform_inv = np.eye(4)
    #         transform_inv[:3,:3] = R.from_quat(rquat).as_matrix()
    #         transform_inv[:3, 3] = v.tvec
    #         # transform_inv
    #         # cam2world = np.linalg.inv(lidar_to_world_start) @ np.linalg.inv(transform_inv)
    #         cam2world = np.linalg.inv(transform_inv)
    #         ego_cam_poses[cam_id].append(cam2world)

    #     ego_frame_poses = []
    #     for i in range(len(ego_cam_poses[0])):
    #         ego_frame_poses.append( ego_cam_poses[0][i] @ np.linalg.inv(extrinsics[0]) )

    #     # ego_cam_poses = np.array(ego_cam_poses)
    #     ego_cam_poses = np.array([a for a in ego_cam_poses if len(a) > 0])
    #     ego_frame_poses = np.array(ego_frame_poses)

    # except:
    ## Nuscenes
    ego_cam_poses = [[] for i in range(6)]
    extrinsics_paths = sorted(os.listdir(extrinsics_dir))
    for ext_path in extrinsics_paths:
        cam2world = np.loadtxt(os.path.join(extrinsics_dir, ext_path))
        cam_id = int(ext_path.split(".")[0].split("_")[-1])
        cam2world = np.linalg.inv(lidar_to_world_start) @ cam2world
        ego_cam_poses[cam_id].append(cam2world)

    ego_frame_poses = []
    ego_pose_paths = sorted(os.listdir(lidar_pose_dir))
    for ego_pose_path in ego_pose_paths:
        lidar_to_world = np.loadtxt(os.path.join(lidar_pose_dir, ego_pose_path))
        lidar_to_world = np.linalg.inv(lidar_to_world_start) @ lidar_to_world
        ego_frame_poses.append(lidar_to_world)
    
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_cam_poses = np.array(ego_cam_poses)

    
    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    # center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    # ego_frame_poses[:, :3, 3] -= center_point # [num_frames, 4, 4]
    
#     ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(6)]
    ego_cam_poses = np.array(ego_cam_poses)
    # ego_cam_poses[:, :, :3, 3] -= center_point # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses
        # 内参，首帧各cam对lidar相对位姿，世界坐标系lidar位姿，世界坐标系各cam位姿  

# # calculate obj pose in world frame
# # box_info: box_center_x box_center_y box_center_z box_heading
# def make_obj_pose(ego_pose, box_info):
#     tx, ty, tz, heading = box_info
#     c = math.cos(heading)
#     s = math.sin(heading)
#     rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

#     obj_pose_vehicle = np.eye(4)
#     obj_pose_vehicle[:3, :3] = rotz_matrix
#     obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
#     obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

#     obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
#     obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
#     obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
#     obj_position_vehicle = obj_pose_vehicle[:3, 3]
#     obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

#     obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
#     obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
#     obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
#     obj_position_world = obj_pose_world[:3, 3]
#     obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])
    
#     return obj_pose_vehicle, obj_pose_world


def make_obj_pose_ds(ego_pose_world, obj_pose_world):
    # obj_pose_world = np.matmul(ego_pose_world, obj_pose_vehicle)
    obj_pose_vehicle = np.matmul(np.linalg.inv(ego_pose_world), obj_pose_world)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle, obj_pose_world



def get_obj_pose_tracking(datadir, model_path, selected_frames, ego_frame_poses, cameras=[0, 1, 2, 3, 4]):
    tracklets_ls = []    
    objects_info = {}

    frame_instances_path = os.path.join(datadir, 'instances/frame_instances.json')
    instances_info_path = os.path.join(datadir, 'instances/instances_info.json')

    with open(frame_instances_path, "r") as f:
        frame_instances = json.load(f)

    with open(instances_info_path, "r") as f:
        instances_info = json.load(f)


    start_frame, end_frame = selected_frames[0], selected_frames[1]

    image_dir = os.path.join(datadir, 'images')
    n_cameras = 5
    n_images = len(os.listdir(image_dir))
    n_frames = n_images // n_cameras
    n_obj_in_frame = np.zeros(n_frames)
    
    for track_id_str in instances_info.keys():
        track_id = int(track_id_str)
        track_instance = instances_info[track_id_str]

        object_class = track_instance["class_name"]
        if "vehicle" in object_class:
            object_class = "vehicle"
        elif "pedestrian" in object_class:
            object_class = "pedestrian"

        if track_id not in objects_info.keys():
            objects_info[track_id] = dict()
            objects_info[track_id]['track_id'] = track_id
            objects_info[track_id]['class'] = object_class 
            objects_info[track_id]['class_label'] = waymo_track2label[object_class]
            objects_info[track_id]['height'] = track_instance["frame_annotations"]["box_size"][0][2]
            objects_info[track_id]['width'] = track_instance["frame_annotations"]["box_size"][0][1]
            objects_info[track_id]['length'] = track_instance["frame_annotations"]["box_size"][0][0]

        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], track_instance["frame_annotations"]["box_size"][0][2])
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], track_instance["frame_annotations"]["box_size"][0][1])
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], track_instance["frame_annotations"]["box_size"][0][0])
        
    tracklets_array = np.array(tracklets_ls)
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    
    for frame_id in frame_instances.keys():
        n_obj_in_frame[int(frame_id)] = len(frame_instances[frame_id])

    num_frames = end_frame - start_frame + 1
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())



    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    lidar_to_world_start = np.loadtxt(os.path.join(datadir, 'lidar_pose', "000.txt"))
    cnt = 0
    for k, v in instances_info.items():
        track_id = int(k)

        for frame_id, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"], v["frame_annotations"]["obj_to_world"], v["frame_annotations"]["box_size"]):

    # ################# 此处 DS所用ID track_id，与 STG所用ID tracklet_camera_vis是不一致的。
    #         cameras_vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]
    #         join_cameras_list = list(set(cameras) & set(cameras_vis_list))
    #         if len(join_cameras_list) == 0:
    #             continue
    # ####################
            if start_frame <= frame_id <= end_frame:            

                cnt += 1

                frame_idx = frame_id - start_frame

                obj_pose_world_mat = np.array(obj_to_world).reshape(4, 4)
                obj_pose_world_mat = np.linalg.inv(lidar_to_world_start) @ obj_pose_world_mat

                ego_pose_world = ego_frame_poses[frame_id]
                obj_pose_vehicle, obj_pose_world = make_obj_pose_ds(ego_pose_world, obj_pose_world_mat)

                obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

                visible_objects_ids[frame_idx, obj_column] = track_id
                visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
                visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world

    # Remove static objects
    pointcloud_dir = os.path.join(model_path, 'input_ply')
    print("Removing static objects")
    for key in objects_info.copy().keys():
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
            dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
            # invalid = len(glob(os.path.join(pointcloud_dir, "*obj*"))) > 0 and not os.path.exists(os.path.join(pointcloud_dir, f'points3D_obj_{key:03d}.ply'))
            # if not dynamic or invalid:
            if not dynamic: 
                visible_objects_ids[all_obj_idx] = -1.
                visible_objects_pose_vehicle[all_obj_idx] = -1.
                visible_objects_pose_world[all_obj_idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)
            
    # # zyk todo: remove those with few points by reading input_ply
    # if os.path.exists(pointcloud_dir):
    #     for key in objects_info.copy().keys():
    #         ply_path = os.path.join(pointcloud_dir, f'points3D_obj_{key:03d}.ply')
    #         if not os.path.exists(ply_path):
    #             objects_info.pop(key)


#     if pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')

    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0    
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = cfg.data.get('box_scale', 1.0)
    print('box scale: ', box_scale)
    
    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info   
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False
        
        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale
        
        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj['start_frame'] = np.min(obj_frames)
        obj['end_frame'] = np.max(obj_frames)
        
        objects_info[key] = obj

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )
    
    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )
    
    
    return objects_tracklets_world, objects_tracklets_vehicle, objects_info

def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]
    
    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)
    
    if max_timestamp > frame_timestamps[1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)
        
    return tracklets, frame_timestamps
    
###############################################################     
def compute_sh_coefficients(normals):
    assert normals.shape[1] == 3, "输入法向量应为 (N, 3) 形状"

    # 归一化法向量，确保单位长度
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    # SH 基函数的归一化因子
    k0 = 0.5 * np.sqrt(1 / np.pi)
    k1 = np.sqrt(3 / (4 * np.pi))

    # 计算 SH 系数
    Y_0_0 = k0 * np.ones(normals.shape[0])  # L=0, Y_0^0
    Y_1_m1 = k1 * normals[:, 1]  # L=1, Y_1^-1 (y)
    Y_1_0 = k1 * normals[:, 2]   # L=1, Y_1^0  (z)
    Y_1_1 = k1 * normals[:, 0]   # L=1, Y_1^1  (x)

    # SH 系数按 (Y_0^0, Y_1^-1, Y_1^0, Y_1^1) 顺序存储
    sh_coeffs = np.array([Y_0_0.mean(), Y_1_m1.mean(), Y_1_0.mean(), Y_1_1.mean()])

    return sh_coeffs


def compute_sh_principal_direction(sh_coeffs):
    assert len(sh_coeffs) >= 4, "SH 系数必须至少包含 4 个 (L=1) 分量"

    # 这里假设 SH 系数顺序是 (Y_0^0, Y_1^-1, Y_1^0, Y_1^1)
    c1_m1, c1_0, c1_1 = sh_coeffs[1], sh_coeffs[2], sh_coeffs[3]

    # 方向向量
    principal_direction = np.array([c1_1, c1_m1, c1_0])  # [x, y, z]
    norm = np.linalg.norm(principal_direction)

    if norm > 1e-6:
        principal_direction /= norm  # 归一化
    else:
        principal_direction = np.array([0, 0, 1])  # 默认方向

    return principal_direction
    
# Normal prior 可视化
def my_vis(points_xyz_world, points_rgb, points_normal, coord_size=15):
    normal_lines = []
    line_colors = []
    # tmp = points_xyz_vehicle[mask_cam] @ ego_pose.T
    tmp = points_xyz_world[:,:3]

    for j in range(tmp.shape[0]):
        p1 = tmp[j]
        p2 = tmp[j] + points_normal[j] * 0.2  # 放大法向量
        normal_lines.append([p1, p2])
        line_colors.append([1, 0, 0])  # 颜色（红色表示法向量）

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack(normal_lines))
    line_set.lines = o3d.utility.Vector2iVector(np.arange(len(normal_lines) * 2).reshape(-1, 2))
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz_world[:,:3])                
    pcd.colors = o3d.utility.Vector3dVector(points_rgb)
    
    if coord_size is None:
        o3d.visualization.draw_geometries([pcd, line_set], point_show_normal=True)
        return

    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_size, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, line_set, FOR1], point_show_normal=True)

def create_custom_camera_visualization(
    view_width_px,
    view_height_px,
    intrinsic,
    extrinsic,
    scale=1.0,
    line_radius=0.005,
    color=[0.2, 0.6, 1.0]
):
    def get_camera_frustum_points(K, width, height, scale):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        z = scale

        corners = np.array([
            [(0 - cx)     / fx * z, (0 - cy)      / fy * z, z],
            [(width - cx) / fx * z, (0 - cy)      / fy * z, z],
            [(width - cx) / fx * z, (height - cy) / fy * z, z],
            [(0 - cx)     / fx * z, (height - cy) / fy * z, z],
        ])  # shape: (4, 3)

        origin = np.zeros((1, 3))
        points = np.vstack([origin, corners])  # shape: (5, 3)
        return points

    def transform_points(points, extrinsic):
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (points_h @ extrinsic.T)[:, :3]
        return transformed

    def create_line_cylinder(start, end, radius, color):
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None
        direction /= length

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        cylinder.paint_uniform_color(color)
        cylinder.compute_vertex_normals()

        # align with Z axis
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction))
        if np.linalg.norm(axis) < 1e-6:
            R = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis) * angle
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis)

        cylinder.rotate(R, center=np.zeros(3))

        mid_point = (start + end) / 2.0
        cylinder.translate(mid_point)
        
        return cylinder

    # Get frustum points (origin + 4 image plane corners)
    frustum_pts = get_camera_frustum_points(intrinsic, view_width_px, view_height_px, scale)
    frustum_pts = transform_points(frustum_pts, extrinsic)
    
    # Line connections: origin to corners and corners to each other
    lines = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # from camera center to image corners
        (1, 2), (2, 3), (3, 4), (4, 1)   # image plane rectangle
    ]

    cylinders = []
    for start_idx, end_idx in lines:
        cyl = create_line_cylinder(frustum_pts[start_idx], frustum_pts[end_idx], line_radius, color)
        if cyl:
            cylinders.append(cyl)
    return cylinders


# camera_meshes = create_custom_camera_visualization(
#     view_width_px=img.shape[1],
#     view_height_px=img.shape[0],
#     intrinsic=ixt[:3, :3],
#     extrinsic=extrinsic,
#     scale=1,
#     line_radius=0.01,
#     color=[0, 0, 1]  
# )
# FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

# o3d.visualization.draw_geometries([FOR1] + camera_meshes)


def generate_dataparser_outputs(
        datadir, 
        selected_frames=None, 
        build_pointcloud=True, 
        cameras=[0, 1, 2, 3, 4]
    ):
    
    image_dir = os.path.join(datadir, 'images')
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.jpg')))
    num_cameras = len(cameras)
    num_frames_all = len(image_filenames_all) // num_cameras
    
    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    # load calibration and ego pose
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info_ds(datadir, os.path.join(f'{cfg.model_path}/colmap'))    
    # intrinsics 内参，
    # extrinsics 首帧各 cam 对 lidar 相对位姿，
    # ego_frame_poses 世界坐标系 lidar 位姿，
    # ego_cam_poses 世界坐标系各cam位姿

    # intrinsics, extrinsics, ego_frame_poses = load_camera_info_ds(datadir)    

    # load camera, frame, path
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []
    
    ixts = []
    exts = []
    poses = []
    c2ws = []
    
    frames_timestamps = []
    cams_timestamps = []
        
    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )
    
    # timestamp_path = os.path.join(datadir, 'timestamps.json')
    # with open(timestamp_path, 'r') as f:
    #     timestamps = json.load(f)
        
    for frame in range(start_frame, end_frame+1):
    #     frames_timestamps.append(timestamps['FRAME'][f'{frame:06d}'])
        frames_timestamps.append(frame)
        
    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if frame >= start_frame and frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            # pose = ego_cam_poses[cam, frame]
            pose = ego_frame_poses[frame]
            # c2w = pose @ ext
            c2w = ego_cam_poses[cam, frame]

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)
            
            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)
            
            # camera_name = _label2camera[cam]
            # timestamp = timestamps[camera_name][f'{frame:06d}']
            timestamp = frame
            cams_timestamps.append(timestamp)
        
    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    # timestamp_offset = min(cams_timestamps + frames_timestamps)
    timestamp_offset = 0
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp, max_timestamp = min(cams_timestamps.min(), frames_timestamps.min()), max(cams_timestamps.max(), frames_timestamps.max())
 
    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking(
        datadir, 
        cfg.model_path,
        selected_frames, 
        ego_frame_poses,
        cameras,
    )
        
    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]['start_frame']
        object_end_frame = object_info[track_id]['end_frame']
        # object_start_timestamp = timestamps['FRAME'][f'{object_start_frame:06d}'] - timestamp_offset - 0.1
        # object_end_timestamp = timestamps['FRAME'][f'{object_end_frame:06d}'] - timestamp_offset + 0.1
        object_start_timestamp = object_start_frame
        object_end_timestamp = object_end_frame
      
        object_info[track_id]['start_timestamp'] = max(object_start_timestamp, min_timestamp)
        object_info[track_id]['end_timestamp'] = min(object_end_timestamp, max_timestamp)
        object_info[track_id]['start_timestamp'] = object_info[track_id]["start_frame"]
        object_info[track_id]['end_timestamp'] = object_info[track_id]["end_frame"]

        
    result = dict()
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info 
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps
    result["ego_frame_poses"] = ego_frame_poses

    # get object bounding mask
    obj_bounds = []
    obj_view_dict = {}
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = image_heights[cam], image_widths[cam]
        obj_bound = np.zeros((h, w)).astype(np.uint8)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, ext = ixts[i], exts[i]
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0:
                obj_pose_vehicle = np.eye(4)    
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]['length']
                obj_width = object_info[track_id]['width']
                obj_height = object_info[track_id]['height']
                bbox = np.array([[-obj_length, -obj_width, -obj_height], 
                                 [obj_length, obj_width, obj_height]]) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                corners_vehicle = corners_local @ obj_pose_vehicle.T # 3D bounding box in vehicle frame
                mask = get_bound_2d_mask(
                    corners_3d=corners_vehicle[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(ext), 
                    H=h, W=w
                )
                obj_bound = np.logical_or(obj_bound, mask)

                if mask.sum() < 20*20*10:
                    continue
                if track_id not in obj_view_dict:
                    obj_view_dict[track_id] = {}
                obj_view_dict[track_id][i] = [mask.sum(), obj_pose_vehicle] # 画面中点数/obj pose

        obj_bounds.append(obj_bound)
    result['obj_bounds'] = obj_bounds         
    result['obj_view_dict'] = obj_view_dict         

    # os.makedirs('obj_bounds', exist_ok=True)
    # for i, x in enumerate(obj_bounds):
    #     x = x.astype(np.uint8) * 255
    #     cv2.imwrite(f'obj_bounds/{i}.png', x)
    
    # run colmap
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
        from script.waymo.colmap_drivestudio_full import run_colmap_waymo
        run_colmap_waymo(result)
    
    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)
        
        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_normal_dict = dict()
        points_view_dict = dict()
        
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        points_normal_dict['bkgd'] = []
        points_view_dict['bkgd'] = []

        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{track_id:03d}'] = []
            points_rgb_dict[f'obj_{track_id:03d}'] = []
            points_normal_dict[f'obj_{track_id:03d}'] = []
            points_view_dict[f'obj_{track_id:03d}'] = []


        print('initialize from sfm pointcloud')
        points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        points_colmap_xyz, points_colmap_rgb, points_colmap_error, points_colmap_tracks = read_points3D_binary(points_colmap_path)
        points_colmap_rgb = points_colmap_rgb / 255.
        colmap_images = read_extrinsics_binary(points_colmap_path.replace("points3D", "images"))

        _mask = points_colmap_error[:, 0] < 0.6
        points_colmap_xyz = points_colmap_xyz[_mask]
        points_colmap_rgb = points_colmap_rgb[_mask]
        points_colmap_tracks = [arr for arr, mask in zip(points_colmap_tracks, _mask) if mask]

        print('initialize from lidar pointcloud')
        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
        pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()
        # from waymo:
            # Can be projected to multi-cameras, order: [FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT].
            # Only save the first projection camera,

            # camera_projection
            # Inner dimensions are:
            # channel 0: CameraName.Name of 1st projection. Set to UNKNOWN if no projection.
            # channel 1: x (axis along image width)
            # channel 2: y (axis along image height)
            # channel 3: CameraName.Name of 2nd projection. Set to UNKNOWN if no projection.
            # channel 4: x (axis along image width)
            # channel 5: y (axis along image height)
            # Note: pixel 0 corresponds to the left edge of the first pixel in the image.

        normals_world_all = []
        N_VIEWS = (end_frame-start_frame+1)*num_cameras
        for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
            idxs = list(range(i * num_cameras, (i+1) * num_cameras))
            cams_frame = [cams[idx] for idx in idxs]
            image_filenames_frame = [image_filenames[idx] for idx in idxs]
            
            raw_3d = pts3d_dict[frame]
            raw_2d = pts2d_dict[frame]
            
            # use the first projection camera
            # points_camera_all = raw_2d[..., 0] 
            
            # each point should be observed by at least one camera in camera lists
            mask = np.array([raw_2d[i, 0] in cameras or raw_2d[i, 3] in cameras for i in range(raw_2d.shape[0])]).astype(np.bool_)
            # mask = np.array([c in cameras for c in points_camera_all]).astype(np.bool_)
            
            # get filtered LiDAR pointcloud position and color        
            points_xyz_vehicle = raw_3d[mask] # local_lidar_points

            # transfrom LiDAR pointcloud from vehicle frame to world frame
            ego_pose = ego_frame_poses[frame]
            points_xyz_vehicle = np.concatenate(
                [points_xyz_vehicle, 
                np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
            )
            points_xyz_world = points_xyz_vehicle @ ego_pose.T
            
            points_rgb = np.ones_like(points_xyz_vehicle[:, :3])
            points_normal = np.ones_like(points_xyz_vehicle[:, :3])
            # points_trackview = np.ones_like(points_xyz_vehicle[:, :1]).astype(np.int16) * -1
            points_visibility = np.zeros([points_xyz_vehicle.shape[0], N_VIEWS]).astype(np.bool_) 

            # points_camera = points_camera_all[mask]
            # points_projw = points_projw_all[mask]
            # points_projh = points_projh_all[mask]

            for cam, image_filename, idx in zip(cams_frame, image_filenames_frame, idxs):
                # mask_cam = (points_camera == cam)
                image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.
                normal_filename = image_filename.replace("images", "normal_img").replace("jpg", "png")
                normal_dsine = cv2.imread(normal_filename) / 255 * 2 - 1
                # DSINE 估计结果为左手系
                normals_transformed = np.zeros_like(normal_dsine)
                normals_transformed[..., 0] = -normal_dsine[..., 2]  # X_new = -X_old
                normals_transformed[..., 1] = -normal_dsine[..., 1]   # Y_new = Z_old
                normals_transformed[..., 2] = -normal_dsine[..., 0]

                ext = extrinsics[cam]
                # pose = ego_cam_poses[cam, frame]
                pose = ego_frame_poses[frame]
                # c2w = pose @ ext
                ixt = ixts[idx]
                c2w = c2ws[idx]
                # normals_world = normals_transformed @ ego_pose[:3, :3].T # (H, W, 3) * (3, 3)
                normals_world = normals_transformed @ c2w[:3,:3].T # @ np.linalg.inv(ext)[:3, :3] 
                # points_xyz_cam = points_xyz @ w2c.T
                normals_world_all.append(normals_world)

                view_pos_world = np.concatenate([points_xyz_world[:,:3], np.ones_like(points_xyz_world[:,:1])], axis=-1)
                view_pos_cam = view_pos_world @ np.linalg.inv(c2w).T

                tmp = view_pos_cam[:,:3] @ ixt.T
                us = (tmp[:,0] / tmp[:,2]) #.astype(np.int16)
                vs = (tmp[:,1] / tmp[:,2]) #.astype(np.int16)

                mask = np.logical_and(us >= 0, us < image.shape[1])
                mask = np.logical_and(mask, vs >= 0)
                mask = np.logical_and(mask, vs < image.shape[0])
                mask = np.logical_and(mask, tmp[:, 2] > 2)

                # image2 = np.array(image)
                # image2[vs[mask], us[mask], 0] = 1
                # image2[vs[mask], us[mask], 1] = 0
                # image2[vs[mask], us[mask], 2] = 0
                # plt.imshow(image2)
                mask_projw = us.astype(np.int16)[mask]
                mask_projh = vs.astype(np.int16)[mask]


                # mask_projw = points_projw[mask_cam]
                # mask_projh = points_projh[mask_cam]
                
                # image2 = np.array(image)
                # image2[mask_projh, mask_projw] = 0
                # image2[mask_projh, mask_projw, 0] = 1
                # plt.imshow(image2)

                # 每个lidar point对绑定图像取色
                mask_rgb = image[mask_projh, mask_projw]
                mask_normal = normals_world[mask_projh, mask_projw]

                # points_rgb[mask_cam] = mask_rgb
                # points_normal[mask_cam] = mask_normal
                points_rgb[mask] = mask_rgb
                points_normal[mask] = mask_normal
                # points_trackview[mask_cam] = idx
                points_visibility[mask, idx] = True

            ## Normal Prior Visualization
            # my_vis(points_xyz_world, points_rgb, points_normal)

            # filer points in tracking bbox
            points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

            for tracklet in object_tracklets_vehicle[i]:
                track_id = int(tracklet[0])
                if track_id >= 0:
                    obj_pose_vehicle = np.eye(4)                    
                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                    obj_pose_vehicle[:3, 3] = tracklet[1:4]
                    vehicle2local = np.linalg.inv(obj_pose_vehicle)
                    
                    # 注：world与vehicle关系: points_xyz_world = points_xyz_vehicle @ ego_pose.T
                    points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
                    points_xyz_obj = points_xyz_obj[..., :3]
                    points_normal_obj = points_normal @ np.linalg.inv(ego_pose[:3,:3]).T @ vehicle2local[:3,:3].T
                    # my_vis(points_xyz_obj, points_rgb, points_normal_obj)

                    length = object_info[track_id]['length']
                    width = object_info[track_id]['width']
                    height = object_info[track_id]['height'] * 1.2
                    bbox = [[-length/2, -width/2, -height/2], [length/2, width/2, height/2]]
                    obj_corners_3d_local = bbox_to_corner3d(bbox)
                    
                    points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)

                    points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                    points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                    points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])
                    points_normal_dict[f'obj_{track_id:03d}'].append(points_normal_obj[points_xyz_inbbox])
                    points_view_dict[f'obj_{track_id:03d}'].append(points_visibility[points_xyz_inbbox])
                    # my_vis(points_xyz_obj[points_xyz_inbbox], points_rgb[points_xyz_inbbox], points_normal_obj[points_xyz_inbbox])

            points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
            points_lidar_rgb = points_rgb[~points_xyz_obj_mask]
            points_lidar_normal = points_normal[~points_xyz_obj_mask]
            points_lidar_visibility = points_visibility[~points_xyz_obj_mask]
            
            points_xyz_dict['bkgd'].append(points_lidar_xyz)
            points_rgb_dict['bkgd'].append(points_lidar_rgb)
            points_normal_dict['bkgd'].append(points_lidar_normal)
            points_view_dict['bkgd'].append(points_lidar_visibility)

        initial_num_obj = 20000
        voxels_xyz_dict = {}
        voxels_rgb_dict = {}
        voxels_normal_sh_dict = {}
        voxels_view_dict = {}
        

        ################## # 如果要修改Initialization，那么需要在此处voxelization进行处理 #####################
        ################## BACKGROUNDS ###################
        # voxels: lidar(1-1) + colmap(1-m) -> voxel(sh)       
        points_bkgd_lidar_xyz = np.concatenate(points_xyz_dict["bkgd"], axis=0)
        points_bkgd_lidar_rgb = np.concatenate(points_rgb_dict["bkgd"], axis=0)
        points_bkgd_lidar_normal = np.concatenate(points_normal_dict["bkgd"], axis=0)
        points_bkgd_lidar_view = np.concatenate(points_view_dict["bkgd"], axis=0)
                
        # 应当在bkgd 加入colmap， xyz拼接做voxel
        # 拼接 -> 一起体素化with track -> 指向源id -> colmap 3d pix id 访问 track view id -> 
        BKGD_LIDAR_LENGTH = points_bkgd_lidar_xyz.shape[0]
        # BKGD_COLMAP_LENGTH = points_colmap_xyz.shape[0]

        lidar_sphere_normalization = get_Sphere_Norm(points_bkgd_lidar_xyz)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

        # todo: 必须做sphere的剔除，但是不确定list(compress)操作是否正确。

        # 拼接lidar colmap为open3d pcd
        # points_lidar_cam_xyz = np.vstack([points_bkgd_lidar_xyz, points_colmap_xyz])
        # points_lidar_cam_rgb = np.vstack([points_bkgd_lidar_rgb, points_colmap_rgb])

        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_lidar_cam_xyz[:,:3])                
        # pcd.colors = o3d.utility.Vector3dVector(points_lidar_cam_rgb)
        pcd.points = o3d.utility.Vector3dVector(points_bkgd_lidar_xyz[:,:3])                
        pcd.colors = o3d.utility.Vector3dVector(points_bkgd_lidar_rgb)

        voxel_size = 0.15
        downsampled_pcd, _, point_indices_for_each_voxel = pcd.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound=(pcd.get_min_bound() // voxel_size) * voxel_size, max_bound=(pcd.get_max_bound() // voxel_size+1)*voxel_size)
        downsample_outlier_pcd, downsample_outlier_indice = downsampled_pcd.remove_radius_outlier(nb_points=10, radius=0.5)

        views_for_voxels = []
        normals_for_voxels = []
        for _tmp in downsample_outlier_indice:
            indices = point_indices_for_each_voxel[_tmp]
            # xyzs_voxel = []
            # rgbs_voxel = []
            # views_voxel = []
            views_voxel = np.zeros([N_VIEWS]).astype(np.bool_)
            normals_voxel = []
            
            for p3d_idx_for_voxel in indices:
                # if p3d_idx_for_voxel < BKGD_LIDAR_LENGTH: 
                    # lidar点，一一对应，有points_bkgd_lidar_normal:nparray，有points_bkgd_lidar_view:list
                    # xyzs_voxel.append(   points_bkgd_lidar_xyz[p3d_idx_for_voxel])
                    # rgbs_voxel.append(   points_bkgd_lidar_rgb[p3d_idx_for_voxel])
                    # views_voxel.append(points_bkgd_lidar_view[p3d_idx_for_voxel].item())
                views_voxel = np.logical_or(views_voxel, points_bkgd_lidar_view[p3d_idx_for_voxel])
                normals_voxel.append(points_bkgd_lidar_normal[p3d_idx_for_voxel])
                # else:
                #     # colmap点，一对多，points_colmap_tracks包涵【img_id, point2d_id】，需要从 normals_world_all[img_id][u,v]获取
                #     p3d_colmap_id = p3d_idx_for_voxel - BKGD_LIDAR_LENGTH
                #     p_tracks = points_colmap_tracks[p3d_colmap_id]
                    
                #     for track_id in range(len(p_tracks)):
                #         colmap_view_id = p_tracks[track_id][0] # colmap计数，cam0 (1-199), cam1 (200-398), cam2 (399-597)
                #         point2d_id = p_tracks[track_id][1] 
                #         v, u = colmap_images[colmap_view_id].xys[point2d_id].astype(np.int16)
                        
                #         _cam_id = (colmap_view_id - 1) // num_frames
                #         _stamp_id = (colmap_view_id - 1) % num_frames
                #         _my_view_id = _stamp_id * num_cameras + _cam_id

                #         p_n = normals_world_all[_my_view_id][u, v]
                #         # xyzs_voxel.append(   points_colmap_xyz[p3d_colmap_id])
                #         # rgbs_voxel.append(   points_colmap_rgb[p3d_colmap_id])
                #         views_voxel.append(  _my_view_id ) # zyk: colmap id starts from 1, while lidar starts from 0
                #         normals_voxel.append(p_n)                
                
            # xyzs_for_voxels.append(xyzs_voxel)
            # rgbs_for_voxels.append(rgbs_voxel)
            views_for_voxels.append(views_voxel)
            normals_for_voxels.append(np.mean(np.array(normals_voxel).reshape(-1, 3), axis=0))



        # combine SfM pointcloud with LiDAR pointcloud
        try:
            if cfg.data.filter_colmap:
                points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=np.bool_)
                for i, ext in enumerate(exts):
                    # if frames_idx[i] not in train_frames:
                    #     continue
                    camera_position = c2ws[i][:3, 3]
                    radius = np.linalg.norm(points_colmap_xyz - camera_position, axis=-1)
                    mask = np.logical_or(radius < cfg.data.get('extent', 10), points_colmap_xyz[:, 2] < camera_position[2])
                    points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)     
                
                points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
                mask = points_colmap_dist < 2 * sphere_radius
                points_colmap_mask = np.logical_and(points_colmap_mask, mask)     

                points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
                points_colmap_rgb = points_colmap_rgb[points_colmap_mask]
                
            for p3d_colmap_id in range(len(points_colmap_tracks)):
                if not points_colmap_mask[p3d_colmap_id]:
                    continue

                # views_voxel = []
                views_voxel = np.zeros([N_VIEWS]).astype(np.bool_)
                normals_voxel = []
                p_tracks = points_colmap_tracks[p3d_colmap_id]
                
                for track_id in range(len(p_tracks)):
                    colmap_view_id = p_tracks[track_id][0] # colmap计数，cam0 (1-199), cam1 (200-398), cam2 (399-597)
                    point2d_id = p_tracks[track_id][1] 
                    v, u = colmap_images[colmap_view_id].xys[point2d_id].astype(np.int16)
                    
                    _cam_id = (colmap_view_id - 1) // num_frames
                    _stamp_id = (colmap_view_id - 1) % num_frames
                    _my_view_id = _stamp_id * num_cameras + _cam_id

                    if start_frame <= _stamp_id <= end_frame:  
                        p_n = normals_world_all[_my_view_id][u, v]
                        # xyzs_voxel.append(   points_colmap_xyz[p3d_colmap_id])
                        # rgbs_voxel.append(   points_colmap_rgb[p3d_colmap_id])
                        # views_voxel.append(  _my_view_id ) # zyk: colmap id starts from 1, while lidar starts from 0
                        views_voxel[_my_view_id] = True
                        normals_voxel.append(p_n)        

                views_for_voxels.append(views_voxel)
                normals_for_voxels.append(np.mean(np.array(normals_voxel).reshape(-1, 3), axis=0))

        
            points_bkgd_xyz = np.concatenate([np.array(downsample_outlier_pcd.points), points_colmap_xyz], axis=0) 
            points_bkgd_rgb = np.concatenate([np.array(downsample_outlier_pcd.colors), points_colmap_rgb], axis=0)
        except:
            print('No colmap pointcloud')
            points_bkgd_xyz = np.array(downsample_outlier_pcd.points)
            points_bkgd_rgb = np.array(downsample_outlier_pcd.colors)






        # # 可视化看一下voxel * points_per_voxel 的所有源法线
        # vis_xyz, vis_rgb, vis_normal = [], [], []
        # for i in range(len(xyzs_for_voxels)):
        #     for j in range(len(xyzs_for_voxels[i])):
        #         vis_xyz.append(xyzs_for_voxels[i][j])
        #         vis_rgb.append(rgbs_for_voxels[i][j])
        #         vis_normal.append(normals_for_voxels[i][j])
        # my_vis(np.array(vis_xyz), np.array(vis_rgb), np.array(vis_normal))

        normals = np.array(normals_for_voxels).reshape(-1, 3)
        points_bkgd_normal = normals / np.linalg.norm(normals, axis=1)[:,None]


        # # 可视化看一下每个voxel的sh拟合法线朝向
        # normal_each_voxel = []
        # for i in range(len(normal_sh_for_voxels)):
        #     normal_each_voxel.append(compute_sh_principal_direction(normal_sh_for_voxels[i]))
        # my_vis(np.array(downsample_outlier_pcd.points), np.array(downsample_outlier_pcd.colors), np.array(normal_each_voxel))


        voxels_xyz_dict['bkgd'] = points_bkgd_xyz
        voxels_rgb_dict['bkgd'] = points_bkgd_rgb
        voxels_view_dict['bkgd'] = np.array(views_for_voxels)
        voxels_normal_sh_dict['bkgd']= points_bkgd_normal







        ################## OBJECTS ###################


        for k, v in points_xyz_dict.items():
            if len(v) == 0 or k == "bkgd":
                continue
            if k.startswith('obj'): # zyk: todo: 是否需要剔除地面
                points_obj_xyz = np.concatenate(points_xyz_dict[k], axis=0)
                points_obj_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                points_obj_view = np.concatenate(points_view_dict[k], axis=0)
                points_obj_normal = np.concatenate(points_normal_dict[k], axis=0)


                # 新增：基于normal prior剔除错误分类的bkgd
                # _mask_a = points_obj_normal[:,1] * points_obj_xyz[:,1] > 0 # 指向y中轴面
                # _mask_b = points_obj_normal[:,2] * points_obj_xyz[:,2] > -0.9 # 指向z中轴面(直接剔除地面)
                # mask_right_direction = np.logical_and(_mask_a, _mask_b)
                _mask_a = points_obj_normal[:,1] * points_obj_xyz[:,1] > 0 # 指向y中轴面
                _mask_gnd = np.logical_and(points_obj_normal[:,2] > 0.9, points_obj_xyz[:,2] < 0) # 指向z中轴面(直接剔除地面)
                if _mask_gnd.sum() > 100:
                    gnd_height = points_obj_xyz[:, 2].min() + (points_obj_xyz[_mask_gnd, 2].mean() - points_obj_xyz[:, 2].min()) * 2
                    _mask_b = np.logical_or(points_obj_xyz[:,2] > gnd_height, points_obj_normal[:,2] < 0.7)
                    mask_right_direction = np.logical_and(_mask_a, _mask_b)

                    points_obj_xyz = points_obj_xyz[mask_right_direction]
                    points_obj_rgb = points_obj_rgb[mask_right_direction]
                    points_obj_view = points_obj_view[mask_right_direction]
                    points_obj_normal = points_obj_normal[mask_right_direction]


                pcd_obj = o3d.geometry.PointCloud()
                pcd_obj.points = o3d.utility.Vector3dVector(points_obj_xyz)
                pcd_obj.colors = o3d.utility.Vector3dVector(points_obj_rgb)    

                if points_obj_xyz.shape[0] > initial_num_obj:
                    # voxel_size = 1.0
                    downsampled_pcd_obj, _, point_indices_for_each_voxel = pcd_obj.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound=(pcd_obj.get_min_bound() // voxel_size) * voxel_size, max_bound=(pcd_obj.get_max_bound() // voxel_size+1)*voxel_size)
                    points_xyz = np.array(downsampled_pcd_obj.points)
                    points_rgb = np.array(downsampled_pcd_obj.colors)

                    views_for_voxels = []
                    normals_for_voxels = []
                    for indices in point_indices_for_each_voxel:
                        views_voxel = np.zeros([N_VIEWS]).astype(np.bool_)
                        normals_voxel = []
                        
                        for p3d_idx_for_voxel in indices:
                            # lidar点，一一对应，有points_bkgd_lidar_normal:nparray，有points_bkgd_lidar_view:list
                            views_voxel = np.logical_or(views_voxel, points_obj_view[p3d_idx_for_voxel])  
                            normals_voxel.append(points_obj_normal[p3d_idx_for_voxel])
                            
                        views_for_voxels.append(views_voxel)
                        # normals_for_voxels.append(normals_voxel)
                        normals_for_voxels.append(np.mean(normals_voxel, axis=0))

                    normal_sh_for_voxels = []
                    for i in range(len(normals_for_voxels)):
                        normals = np.array(normals_for_voxels[i]).reshape(-1 ,3)
                        normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # 归一化
                        
                        # sh_coeffs = compute_sh_coefficients(normals)
                        # normal_sh_for_voxels.append(sh_coeffs)
                        normal_sh_for_voxels.append(normals)


                    point_view = np.array(views_for_voxels)
                    point_normal = np.concatenate(normal_sh_for_voxels)
                    
                else:
                    points_xyz = points_obj_xyz
                    points_rgb = points_obj_rgb
                    # point_view = points_obj_view # Waymo
                    point_view = np.zeros_like(points_obj_view) 
                    point_view[:, np.where(points_obj_view.sum(0)> 0)] = True  # For those extremely sparse, eg. Nuscenes
                    # point_normal = compute_sh_coefficients(points_obj_normal)
                    point_normal = points_obj_normal


                # if len(points_xyz) > initial_num_obj:
                #     random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                #     points_xyz = points_xyz[random_indices]
                #     points_rgb = points_rgb[random_indices]
                    
                voxels_xyz_dict[k] = points_xyz
                voxels_rgb_dict[k] = points_rgb
                voxels_view_dict[k] = point_view
                voxels_normal_sh_dict[k] = point_normal
            
            else:
                raise NotImplementedError()

        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb
            
        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict


        for k in voxels_xyz_dict.keys():
            points_xyz = voxels_xyz_dict[k]
            points_rgb = voxels_rgb_dict[k]
            points_normal_sh = voxels_normal_sh_dict[k]
            points_visibility = voxels_view_dict[k]
            
            if (np.isnan(points_normal_sh).sum()):
                print("nan in", k)

            if points_xyz.shape[0] < 100:
                continue

            ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')

            try:
                storePly(ply_path, points_xyz, points_rgb, normals=points_normal_sh[:, 0:])

                np.save(ply_path[:-3]+"npy", points_visibility)
                
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue
    return result