from lib.utils.waymo_utils import generate_dataparser_outputs
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, get_PCA_Norm, get_Sphere_Norm
from lib.config import cfg
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import cv2
import sys
import copy
import shutil
sys.path.append(os.getcwd())


def load_vggt_predictions(path, image_filenames):
    """Load per-view VGGT guidance and the compact future-vine voxel table."""
    predictions_path = cfg.data.get('vggt_predictions', None)
    if not predictions_path:
        return {}, None
    if not os.path.isabs(predictions_path):
        data_candidate = os.path.join(path, predictions_path)
        output_candidate = os.path.join(cfg.model_path, 'visibility', predictions_path)
        predictions_path = data_candidate if os.path.exists(data_candidate) else output_candidate
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f'VGGT predictions not found: {predictions_path}')

    with np.load(predictions_path, allow_pickle=True) as predictions:
        required = {'image_names', 'depth'}
        missing = required.difference(predictions.files)
        if missing:
            raise KeyError(f'VGGT predictions missing fields for guidance: {sorted(missing)}')
        names = [os.path.splitext(os.path.basename(str(name)))[0]
                 for name in predictions['image_names']]
        depths = predictions['depth'].astype(np.float32, copy=False)
        if depths.ndim == 4 and depths.shape[-1] == 1:
            depths = depths[..., 0]
        transforms = predictions['pixel_transform'] if 'pixel_transform' in predictions.files else None
        pixel_voxel_ids = (predictions['vggt_pixel_voxel_ids']
                           if 'vggt_pixel_voxel_ids' in predictions.files else None)
        voxel_keys = {
            'vggt_voxel_points', 'vggt_voxel_ids',
            'vggt_voxel_source_offsets', 'vggt_voxel_source_view_ids',
            'vggt_invalid_voxel_id', 'vggt_voxel_size',
        }
        voxel_info = ({key: predictions[key] for key in voxel_keys}
                      if voxel_keys.issubset(predictions.files) else None)

    if len(names) != len(depths):
        raise ValueError(f'VGGT image_names/depth length mismatch: {len(names)} vs {len(depths)}')
    prediction_by_name = {name: idx for idx, name in enumerate(names)}
    result = {}
    for image_filename in image_filenames:
        stem = os.path.splitext(os.path.basename(image_filename))[0]
        pred_idx = prediction_by_name.get(stem)
        if pred_idx is None:
            continue
        result[stem] = {
            'depth': depths[pred_idx],
            'voxel_ids': None if pixel_voxel_ids is None else pixel_voxel_ids[pred_idx],
            'pixel_transform': None if transforms is None else transforms[pred_idx],
        }
    if voxel_info is not None:
        scene_view_by_name = {
            os.path.splitext(os.path.basename(filename))[0]: idx
            for idx, filename in enumerate(image_filenames)
        }
        voxel_info['prediction_to_scene_view'] = np.asarray(
            [scene_view_by_name.get(name, -1) for name in names], dtype=np.int32,
        )
    print(f'Loaded VGGT unscaled depth for {len(result)}/{len(image_filenames)} Waymo views')
    return result, voxel_info

def readWaymoFullInfo(path, images='images', split_train=-1, split_test=-1, **kwargs):

    selected_frames = cfg.data.get('selected_frames', None)
    if cfg.debug:
        selected_frames = [0, 0]

    if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(load_dir, save_dir)
        
        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
        save_dir = os.path.join(cfg.model_path, 'colmap')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(colmap_dir, save_dir)
        
    bkgd_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_bkgd.ply')
    build_pointcloud = (cfg.mode == 'train') and (not os.path.exists(bkgd_ply_path) or cfg.data.get('regenerate_pcd', False))
    
    # dynamic mask
    dynamic_mask_dir = os.path.join(path, 'sam_masks')
    bkgd_mask_dir = os.path.join(path, 'sam_bkgd_masks')
    load_dynamic_mask = True

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    load_sky_mask = (cfg.mode == 'train') and os.path.exists(sky_mask_dir)
    
    # lidar depth
    lidar_depth_dir = os.path.join(path, 'lidar_depth')
    load_lidar_depth = (cfg.mode == 'train') and os.path.exists(lidar_depth_dir)

    # zyk
    mono_depth_dir = os.path.join(path, 'depth_v2')
    load_mono_depth = True # (cfg.mode == 'train') and os.path.exists(mono_depth_dir)

    normal_dir = os.path.join(path, "normal_img")
    load_normal = True


    output = generate_dataparser_outputs(
        datadir=path, 
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cfg.data.get('cameras', [0, 1, 2]),
    )

    exts = output['exts']
    ixts = output['ixts']
    poses = output['poses']
    c2ws = output['c2ws']
    image_filenames = output['image_filenames']
    obj_tracklets = output['obj_tracklets']
    obj_info = output['obj_info']
    frames, cams = output['frames'], output['cams']
    frames_idx = output['frames_idx']
    num_frames = output['num_frames']
    cams_timestamps = output['cams_timestamps']
    tracklet_timestamps = output['tracklet_timestamps']
    obj_bounds = output['obj_bounds']

    vggt_unscaled_depth, vggt_voxel_info = load_vggt_predictions(path, image_filenames)

    ego_frame_poses = output['ego_frame_poses']
    obj_view_dict = output['obj_view_dict']

    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['obj_tracklets'] = obj_tracklets
    scene_metadata['tracklet_timestamps'] = tracklet_timestamps
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = len(cfg.data.cameras)
    scene_metadata['num_frames'] = num_frames

    scene_metadata['ego_frame_poses'] = ego_frame_poses
    scene_metadata['obj_view_dict'] = obj_view_dict
    scene_metadata['c2ws'] = c2ws
    scene_metadata['ixts'] = ixts
    scene_metadata['vggt_voxel_info'] = vggt_voxel_info
    

    camera_timestamps = dict()
    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []      

    ########################################################################################################################
    cam_infos = []
    for i in tqdm(range(len(exts))):
        # generate pose and image
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)    
        
        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()
        
        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]

        if frames_idx[i] in train_frames:
            metadata['is_val'] = False
            camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        else:
            metadata['is_val'] = True
            camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])
        
        guidance = dict()

        if image_name in vggt_unscaled_depth:
            guidance['unscaled_depth'] = {
                'depth': vggt_unscaled_depth[image_name]['depth'],
                'pixel_transform': vggt_unscaled_depth[image_name]['pixel_transform'],
                'raw_width': width,
                'raw_height': height,
            }
            if vggt_unscaled_depth[image_name]['voxel_ids'] is not None:
                guidance['vggt_voxel_id'] = {
                    'voxel_ids': vggt_unscaled_depth[image_name]['voxel_ids'],
                    'invalid_id': (np.iinfo(np.uint32).max if vggt_voxel_info is None else
                                   int(np.asarray(vggt_voxel_info['vggt_invalid_voxel_id']).item())),
                    'pixel_transform': vggt_unscaled_depth[image_name]['pixel_transform'],
                    'raw_width': width,
                    'raw_height': height,
                }

        # load dynamic mask
        if load_dynamic_mask:
            dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.png')
            dynamic_mask = cv2.imread(dynamic_mask_path)
            guidance['dynamic_mask'] = dynamic_mask

            seg_bkgd_mask_path = os.path.join(bkgd_mask_dir, f'{image_name}.png')
            seg_bkgd_mask = cv2.imread(seg_bkgd_mask_path)
            guidance["seg_bkgd"] = seg_bkgd_mask

            guidance['obj_bound'] = Image.fromarray(obj_bounds[i])

        # load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f'{image_name}.npy')
            depth = np.load(depth_path, allow_pickle=True)
            depth = dict(depth.item())
            mask = depth['mask']
            value = depth['value']
            depth = np.zeros_like(mask).astype(np.float32)
            depth[mask] = value
            guidance['lidar_depth'] = depth
            
        # load sky mask
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name[3:]}.png')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            guidance['sky_mask'] = Image.fromarray(sky_mask)

        # # zyk
        if load_mono_depth:
            depth_v2_path = os.path.join(mono_depth_dir, f'{image_name}.png')
            mono_depth = 255 - cv2.imread(depth_v2_path)[:,:,0]
            guidance['mono_depth'] = Image.fromarray(mono_depth)

        # # # zyk: load normal map. Taking too much time if using npy. Use png instead.
        if load_normal:
            normal_img_path = os.path.join(normal_dir, f'{image_name}.png')
            tmp = cv2.imread(normal_img_path) / 255 * 2 - 1
            ref_norm = np.zeros(tmp.shape)
            # normal = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB) / 255 * 2 - 1
            ref_norm = -tmp[:,:,[2,1,0]]

            guidance['mono_normal'] = ref_norm


        mask = None        
        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)
        
        # sys.stdout.write('\n')
    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]
    
    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps
        
    novel_view_cam_infos = []
    
    #######################################################################################################################3
    # Get scene extent
    # 1. Default nerf++ setting
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. The radius we obtain should not be too small (larger than 10 here)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    
    # 3. If we have extent set in config, we ignore previous setting
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent
    
    # 4. We write scene radius back to config
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. We write scene center and radius to scene metadata    
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    # Get sphere center
    lidar_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_lidar.ply')
    if os.path.exists(lidar_ply_path):
        sphere_pcd: BasicPointCloud = fetchPly(lidar_ply_path)
    else:
        sphere_pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    
    sphere_normalization = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata['sphere_center'] = sphere_normalization['center']
    scene_metadata['sphere_radius'] = sphere_normalization['radius']
    print(f'Sphere extent: {sphere_normalization["radius"]}')

    pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    if cfg.mode == 'train':
        point_cloud = pcd
    else:
        point_cloud = None
        bkgd_ply_path = None

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )
    
    return scene_info
    
    
    
