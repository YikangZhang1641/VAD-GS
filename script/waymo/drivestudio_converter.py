import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
import cv2
import math
import imageio
import argparse
import json
from tqdm import tqdm, trange
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
from lib.utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
from lib.utils.img_utils import draw_3d_box_on_img
from lib.utils.graphics_utils import project_numpy
import glob
import matplotlib.pyplot as plt

# castrack_path = '/nas/home/yanyunzhi/waymo/castrack/seq_infos/val/result.json'
# with open(castrack_path, 'r') as f:
#     castrack_infos = json.load(f)



camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT', 
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT', 
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}

image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}

opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

def get_extrinsic(camera_calibration):
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4) # camera to vehicle
    extrinsic = np.matmul(camera_extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
    return extrinsic
    
def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    return intrinsic

def project_label_to_image(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3], 
        K=intrinsic, 
        RT=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    return points_uv, valid

def project_label_to_mask(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    
    return mask
    
    
def parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, start_idx=None, end_idx=None):
    print(f'Processing sequence {seq_name}...')
    print(f'Saving to {seq_save_dir}')

    nuscenes_cams = [0,1,2,3,4,5]
    WIDTH  = 1600 // 2
    HEIGHT = 900 // 2

    lidar_ds_path = os.path.join(root_dir, "lidar/*.bin")
    lidar_filepaths = sorted(glob.glob(lidar_ds_path))

    # try:
    #     with open(track_file, 'r') as f:
    #         castrack_infos = json.load(f)
    # except:
    #     castrack_infos = dict()

    # os.makedirs(seq_save_dir, exist_ok=True)
    
    # seq_path = os.path.join(root_dir, seq_name+'.tfrecord')
    
    # # set start and end timestep
    # datafile = WaymoDataFileReader(seq_path)
    # num_frames = len(datafile.get_record_table())

    num_frames = len(lidar_filepaths)

    start_idx = start_idx or 0
    end_idx = end_idx or num_frames - 1
    
    camera_front_start = np.loadtxt(
        os.path.join(root_dir, "extrinsics", f"{start_idx:03d}_0.txt")
    )


    if 'lidar' in process_list:
        pts_3d_all = dict()
        pts_2d_all = dict()
        print("Processing LiDAR data...")

        data_path = root_dir
        points = []
        colors = []
        # datafile = WaymoDataFileReader(seq_path)
        
        # for frame_id, frame in tqdm(enumerate(datafile)):
        for t in tqdm(
            range(num_frames), 
            # desc="Projecting lidar pts on images for camera {}".format(cam),
            dynamic_ncols=True
        ):
        
            frame_idx = t
            pts_3d = [] # LiDAR point cloud in world frame
            # pts_2d = [] # LiDAR point cloud projection in camera [camera_name, w, h] 
            
            ################## LIDAR coordinate ########################
            lidar_to_world = np.loadtxt(
                os.path.join(data_path, "lidar_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            lidar_to_world = np.linalg.inv(camera_front_start) @ lidar_to_world
            
            lidar_info = np.fromfile(lidar_filepaths[t], dtype=np.float32).reshape(-1, 4)
            local_lidar_points = lidar_info[:, :3]
            
            camera_projection = np.zeros([local_lidar_points.shape[0], 6]).astype(np.int16)
            camera_projection[:,0] = -1
            camera_projection[:,3] = -1
            
            global_lidar_points = (
                lidar_to_world[:3, :3] @ local_lidar_points.T
                + lidar_to_world[:3, 3:4] 
            ).T
        #     ############################################
        #     local_lidar_points = local_lidar_points @ LIDAR_TRANS
            
            pts_3d.append(local_lidar_points)


            for cam_id in nuscenes_cams:
                image = plt.imread(os.path.join(data_path, "images", f"{t:03d}_{cam_id}.jpg"))
                image = image[::2, ::2, :]
                
                intrinsic = np.loadtxt(
                    os.path.join(data_path, "intrinsics", f"{cam_id}.txt")
                )
                
                fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
                k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]

                fx /= 2
                fy /= 2
                cx /= 2
                cy /= 2
                
                intrinsic_4x4 = np.zeros([4,4])
                intrinsic_4x4[:3, :3] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                intrinsic_4x4[3, 3] = 1.0

                
                cam2world = np.loadtxt(
                    os.path.join(data_path, "extrinsics", f"{t:03d}_{cam_id}.txt")
                )

                # Align camera poses with the first camera pose
                cam2world = np.linalg.inv(camera_front_start) @ cam2world
        #         cam2world = cam2world @ OPENCV2DATASET

                lidar2img = intrinsic_4x4 @ np.linalg.inv(cam2world)
                lidar_points = (
                    lidar2img[:3, :3] @ global_lidar_points.T + lidar2img[:3, 3:4]
                ).T # (num_pts, 3)

                depth = lidar_points[:, 2]
                cam_points = lidar_points[:, :2] / (depth.reshape(-1,1) + 1e-6) # (num_pts, 2)
                cam_points = cam_points.astype(np.int16)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 1] < HEIGHT)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 0] < WIDTH)
                    & (depth > 0)
                )
                _cam_points = cam_points[valid_mask]
                points_color = image[
                    _cam_points[:, 1].astype(np.int16), _cam_points[:, 0].astype(np.int16)
                ]
            
                valid_mask1 = valid_mask & (camera_projection[:,0] == -1)
                valid_mask2 = valid_mask & (camera_projection[:,0] != -1)

                camera_projection[valid_mask1, 0] = cam_id
                camera_projection[valid_mask1, 1] = cam_points[valid_mask1, 0]
                camera_projection[valid_mask1, 2] = cam_points[valid_mask1, 1]
            
                camera_projection[valid_mask2, 3] = cam_id
                camera_projection[valid_mask2, 4] = cam_points[valid_mask2, 0]
                camera_projection[valid_mask2, 5] = cam_points[valid_mask2, 1]
            
            
                v = (camera_projection[:,0] != -1)
                w = camera_projection[v, 1]
                h = camera_projection[v, 2]

                points.append(global_lidar_points[v])
                colors.append(image[h, w])

            pts_3d_all[t] = local_lidar_points
            pts_2d_all[t] = camera_projection

            
        np.savez_compressed(f'{seq_save_dir}/pointcloud.npz', 
                            pointcloud=pts_3d_all, 
                            camera_projection=pts_2d_all)
        print("Processing LiDAR data done...")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_list', type=str, nargs='+', default=['pose', 'calib', 'image', 'lidar', 'track', 'dynamic_mask'])
    parser.add_argument('--root_dir', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--save_dir', type=str, default='/nas/home/yanyunzhi/waymo/street_gaussian/training/surrounding')
    parser.add_argument('--track_file', type=str, default='/nas/home/yanyunzhi/waymo/castrack/seq_infos/val/result.json')
    # parser.add_argument('--split_file', type=str)
    parser.add_argument('--segment_file', type=str)
    args = parser.parse_args()
    
    process_list = args.process_list
    root_dir = args.root_dir
    save_dir = args.save_dir
    track_file = args.track_file
    # split_file = open(args.split_file, "r").readlines()[1:]
    # scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    # seq_names = [line.strip().split(",")[1] for line in split_file]
    # segment_file = args.segment_file

    # seq_lists = open(segment_file).read().splitlines()
    # seq_lists = open(os.path.join(root_dir, 'segment_list.txt')).read().splitlines()
    # os.makedirs(save_dir, exist_ok=True)
    # for i, scene_id in enumerate(scene_ids_list):
    #     assert seq_names[i][3:] == seq_lists[scene_id][8:14]
    #     seq_save_dir = os.path.join(save_dir, str(scene_id).zfill(3))
    #     parse_seq_rawdata(
    #         process_list=process_list,
    #         root_dir=root_dir,
    #         seq_name=seq_lists[scene_id],
    #         seq_save_dir=seq_save_dir,
    #         track_file=track_file,
    #     )

    parse_seq_rawdata(
        process_list=process_list,
        root_dir=root_dir,
        seq_name=None,
        seq_save_dir=root_dir,
        track_file=track_file,
    )

if __name__ == '__main__':
    main()