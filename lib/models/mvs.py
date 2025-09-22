import torch
import math
import numpy as np

import cv2
import os, shutil

import struct 

import matplotlib.pyplot as plt
import subprocess
from random import randint
import patchmatch_cuda

def write_cam_txt(cam_path, K, w2c, depth_range):
    with open(cam_path, "w") as file:
        file.write("extrinsic\n")
        for row in w2c:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")

        file.write("\nintrinsic\n")
        for row in K:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")
        
        file.write("\n")
        
        file.write(" ".join(str(element) for element in depth_range))
        file.write("\n")

def cam_to_tensor(K, w2c, depth_range):
    row = [w2c.reshape(-1,1), K.reshape(-1,1), depth_range]


def writeDepthDmb(file_path, depth):
    h, w = depth.shape  # 获取高度和宽度
    type = 1  # 标识类型（与 readDepthDmb 中的 type 需匹配）
    nb = 1  # 每个像素的通道数，这里是单通道

    with open(file_path, "wb") as outimage:
        # 写入头部信息（整数4字节）
        outimage.write(struct.pack("i", type))  # 文件类型
        outimage.write(struct.pack("i", h))  # 高度
        outimage.write(struct.pack("i", w))  # 宽度
        outimage.write(struct.pack("i", nb))  # 通道数

        # 写入深度数据（float32，每个像素4字节）
        outimage.write(depth.astype(np.float32).tobytes())



def readDepthDmb(file_path):
    inimage = open(file_path, "rb")
    if not inimage:
        print("Error opening file", file_path)
        return -1

    type = -1

    type = struct.unpack("i", inimage.read(4))[0]
    h = struct.unpack("i", inimage.read(4))[0]
    w = struct.unpack("i", inimage.read(4))[0]
    nb = struct.unpack("i", inimage.read(4))[0]

    if type != 1:
        inimage.close()
        return -1

    dataSize = h * w * nb

    depth = np.zeros((h, w), dtype=np.float32)
    depth_data = np.frombuffer(inimage.read(dataSize * 4), dtype=np.float32)
    depth_data = depth_data.reshape((h, w))
    np.copyto(depth, depth_data)

    inimage.close()
    return depth

def readNormalDmb(file_path):
    try:
        with open(file_path, 'rb') as inimage:
            type = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            h = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            w = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            nb = np.fromfile(inimage, dtype=np.int32, count=1)[0]

            if type != 1:
                print("Error: Invalid file type")
                return -1

            dataSize = h * w * nb

            normal = np.zeros((h, w, 3), dtype=np.float32)
            normal_data = np.fromfile(inimage, dtype=np.float32, count=dataSize)
            normal_data = normal_data.reshape((h, w, nb))
            normal[:, :, :] = normal_data[:, :, :3]

            return normal

    except IOError:
        print("Error opening file", file_path)
        return -1

def read_propagted_depth(path):    
    cost = readDepthDmb(os.path.join(path, 'costs.dmb'))
    cost[np.isnan(cost)] = 2
    cost[cost < 0] = 2
    # mask = cost > 0.5

    depth = readDepthDmb(os.path.join(path, 'depths.dmb'))
    # depth[mask] = 300
    depth[np.isnan(depth)] = 300
    depth[depth < 0] = 300
    depth[depth > 300] = 300
    
    normal = readNormalDmb(os.path.join(path, 'normals.dmb'))

    return depth, cost, normal

def load_pairs_relation(path):
    pairs_relation = []
    num = 0
    with open(path, 'r') as file:
        num_images = int(file.readline())
        for i in range(num_images):

            ref_image_id = int(file.readline())
            if i != ref_image_id:
                print(ref_image_id)
                print(i)

            src_images_infos = file.readline().split()
            num_src_images = int(src_images_infos[0])
            src_images_infos = src_images_infos[1:]
            
            pairs = []
            #only fetch the first 4 src images
            for j in range(num_src_images):
                id, score = int(src_images_infos[2*j]), int(src_images_infos[2*j+1])
                #the idx needs to align to the training images
                if score <= 0.0 or id % 8 == 0:
                    continue
                id = (id // 8) * 7 + (id % 8) - 1
                pairs.append(id)
                
                if len(pairs) > 3:
                    break
                
            if ref_image_id % 8 != 0:
                #only load the training images
                pairs_relation.append(pairs)
            else:
                num = num + 1
            
    return pairs_relation

def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


# project the reference point cloud into the source view, then project back
#extrinsics here refers c2w
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    batch, height, width = depth_ref.shape
    
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(torch.inverse(extrinsics_ref), extrinsics_src),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1=1, thre2=0.01):
    batch, height, width = depth_ref.shape
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = torch.logical_and(dist < thre1, relative_depth_diff < thre2)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff


# def prepare_view(current_view, idx, rendered_depth, rendered_normals, viewpoint_stack, dataset, vehicle_name=None):
#     viewpoint_cam = viewpoint_stack[current_view]
#     frame_id = current_view // 3

#     cdata_image_path = './cache/images'
#     cdata_mask_path = './cache/masks'
#     cdata_camera_path = './cache/cams'
#     cdata_depth_path = './cache/depths'
#     cdata_normal_path = './cache/normals'

#     dataset_path = dataset.source_path

#     # ### 生成./cache/masks     
#     sky_mask = viewpoint_cam.guidance['sky_mask']    
#     dynamic_mask = cv2.imread(os.path.join(dataset_path, "sam_masks", viewpoint_cam.image_name + ".png"))[:,:,0]
#     scale = min(1.0, 1600 / dynamic_mask.shape[1])
#     resolution = (int(dynamic_mask.shape[0] * scale), int(dynamic_mask.shape[1] * scale))
#     dynamic_mask = np.resize(dynamic_mask, resolution)

#     if vehicle_name is None: # bkgd
#         all_mask = sky_mask[0].cpu().detach().numpy().astype(np.bool_)
#         all_mask = np.logical_or(all_mask, dynamic_mask.astype(np.bool_))
#     else:
#         all_mask = sky_mask[0].cpu().detach().numpy().astype(np.bool_)
#     plt.imsave(os.path.join(cdata_mask_path, str(idx)+".png"), ~all_mask)


#     ### 生成./cache/images (ref)
#     ref_img = viewpoint_cam.original_image
#     ref_img = ref_img * 255
#     ref_img = ref_img.permute((1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
#     ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
#     ref_img[all_mask] = 0
#     cv2.imwrite(os.path.join(cdata_image_path, str(idx)+".jpg"), ref_img)


#     if idx == 0:
#         ### 生成./cache/normals 
#         # tmp = cv2.imread(os.path.join(dataset_path, "normal_img", viewpoint_cam.image_name + ".png"))
#         # scale = min(1.0, 1600 / tmp.shape[1])
#         # resolution = (int(tmp.shape[0] * scale), int(tmp.shape[1] * scale), 3)
#         # tmp = np.resize(tmp, resolution)

#         # ref_norm = np.zeros(tmp.shape)
#         # ref_norm[:,:,0] = tmp[:,:,2]
#         # ref_norm[:,:,1] = tmp[:,:,0]
#         # ref_norm[:,:,2] = tmp[:,:,1] # fix

#         cv2.imwrite(os.path.join(cdata_normal_path, "0.png"), (rendered_normals/2+0.5)*255)
#         # cv2.imwrite(os.path.join(cdata_normal_path, "0.png"), ref_norm)


#     ### 生成./cache/cams
#     ref_K = viewpoint_cam.K.detach().cpu().numpy()
#     ref_w2c = viewpoint_cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
#     depth_min = 0.1 
#     depth_max = rendered_depth.max()  
#     if vehicle_name is None:
#         ref_ext = ref_w2c
#     else:
#         obj_view_dict = dataset.scene_info.metadata["obj_view_dict"]
#         if current_view not in obj_view_dict[vehicle_name]:
#             print("obj not exist in ref", current_view)
#             return False
        
#         obj_pose_vehicle = obj_view_dict[vehicle_name][current_view][1]

#         ego_pose = dataset.scene_info.metadata["ego_frame_poses"][frame_id]
#         ref_o2c = ref_w2c @ ego_pose @ obj_pose_vehicle
#         ref_ext = ref_o2c

#     write_cam_txt(os.path.join(cdata_camera_path, str(idx)+".txt"), ref_K, ref_ext, [depth_min, (depth_max-depth_min)/192.0, 192.0, depth_max])

#     if idx == 0:
#         ### 生成./cache/depths 
#         cv2.imwrite(os.path.join(cdata_depth_path, "0.png"), rendered_depth.astype(np.uint8))
#         writeDepthDmb(os.path.join(cdata_depth_path, "0.dmb"), rendered_depth)

#         # anchor_depth = np.zeros([ref_img.shape[0], ref_img.shape[1]]).astype(np.float32) # 当前帧注册的可见点云，认为绝对准确
#         # anchor_depth = viewpoint_cam.guidance['lidar_depth'].cpu().detach().numpy()[0]
#         # anchor_depth.tofile(os.path.join(cdata_depth_path, "0.raw"))
#     return ref_ext


# def depth_propagation(current_view, src_idxs, rendered_depth, rendered_normals, viewpoint_stack, dataset, vehicle_name=None, patch_size=20):
#     # pass data to c++ api for mvs
#     # viewpoint_cam = viewpoint_stack[current_view]
#     # frame_id = current_view // 3
        
#     # dataset_path = dataset.source_path
    
#     cdata_image_path = './cache/images'
#     cdata_mask_path = './cache/masks'
#     cdata_camera_path = './cache/cams'
#     cdata_depth_path = './cache/depths'
#     cdata_normal_path = './cache/normals'

#     for path in [cdata_image_path, cdata_mask_path, cdata_camera_path, cdata_depth_path, cdata_normal_path]:
#         if os.path.exists(path):
#             shutil.rmtree(path)  
#         os.mkdir(path)


#     ref_ext = prepare_view(current_view, 0, rendered_depth, rendered_normals, viewpoint_stack, dataset, vehicle_name=vehicle_name)
        
#     for idx, src_idx in enumerate(src_idxs):
#         prepare_view(src_idx, idx+1, None, None, viewpoint_stack, dataset, vehicle_name=vehicle_name)

#     # c++ api for depth propagation
#     propagation_command = '/home/zyk/Projects/_DL/_Reconstruction/street_gaussians/submodules/Propagation/Propagation ./cache 0 "1 2 3 4" ' + str(patch_size)
#     os.system(propagation_command)
#     return ref_ext

def depth_propagation(current_view, src_idxs, rendered_depth, rendered_normals, viewpoint_stack, dataset, vehicle_name=None, patch_size=20):
    num_cams = dataset.scene_info.metadata["num_cams"]
    frame_id = current_view // num_cams
    viewpoint_cam = viewpoint_stack[current_view]

    img_H, img_W = rendered_depth.shape
    # channel = randint(0,2)

    ref_K = viewpoint_cam.K.cpu().detach()
    ref_w2c = viewpoint_cam.world_view_transform.transpose(0, 1).cpu().detach()

    ### 生成./cache/masks     
    sky_mask = viewpoint_cam.guidance['sky_mask']
    dynamic_mask = viewpoint_cam.guidance['dynamic_mask']

    depth_min = 0.1 
    depth_max = max(rendered_depth.max(), 80)
    if vehicle_name is None: # bkgd
        all_mask = ~sky_mask[0] & ~torch.any(dynamic_mask != 255, axis=0)
        ref_ext = ref_w2c
    else:
        all_mask = ~sky_mask[0] & torch.any(dynamic_mask==vehicle_name, axis=0)
        obj_view_dict = dataset.scene_info.metadata["obj_view_dict"]
        if current_view not in obj_view_dict[vehicle_name]:
            print("obj not exist in ref", current_view)
            return None, None, None
        obj_pose_vehicle = obj_view_dict[vehicle_name][current_view][1]
        ego_pose = dataset.scene_info.metadata["ego_frame_poses"][frame_id]
        
        ref_o2c = ref_w2c @ ego_pose @ obj_pose_vehicle
        ref_ext = ref_o2c
    mask_tensors = [255*all_mask.byte().contiguous().cuda()]

    ### 生成./cache/images (ref)
    # image_tensors = [viewpoint_cam.original_image[channel].contiguous().cuda()]
    image_tensors = [viewpoint_cam.original_image.mean(axis=0).contiguous().cuda()]
    ### 生成./cache/normals 
    normal_guess_tensor = rendered_normals.permute(1,2,0).contiguous().cuda()

    ### 生成./cache/depths 

    depth_tensor = rendered_depth.contiguous().cuda()
    anchor_depth_tensor = depth_tensor.clone()  #viewpoint_cam.guidance['lidar_depth'][0].contiguous().cuda()

    ### 生成./cache/cams (ref)
    row = torch.concat([ref_ext[:3,:3].reshape(-1), ref_ext[:3,3].reshape(-1), ref_K.reshape(-1), torch.tensor([img_H, img_W, depth_min, depth_max])])
    camera_tensors = [row.float().unsqueeze(0).contiguous().cuda()]

    for idx, src_idx in enumerate(src_idxs):
        frame_id = src_idx // num_cams
        src_viewpoint = viewpoint_stack[src_idx]

        src_w2c = src_viewpoint.world_view_transform.transpose(0, 1).detach().cpu()
        src_K = src_viewpoint.K.detach().cpu()

        ### 生成./cache/masks 
        sky_mask = src_viewpoint.guidance['sky_mask']
        dynamic_mask = src_viewpoint.guidance['dynamic_mask']
        
        if vehicle_name is None: # bkgd
            all_mask = ~sky_mask[0] & ~torch.any(dynamic_mask != 255, axis=0)
            src_ext = src_w2c
        else:
            all_mask = ~sky_mask[0] & torch.any(dynamic_mask==vehicle_name, axis=0)
            if src_idx not in obj_view_dict[vehicle_name]:
                print("obj not exist in src", src_idx)
                continue
            obj_pose_vehicle = obj_view_dict[vehicle_name][src_idx][1]

            ego_pose = dataset.scene_info.metadata["ego_frame_poses"][frame_id]
            src_o2c = src_w2c @ ego_pose @ obj_pose_vehicle
            src_ext = src_o2c

        mask_tensors.append(255*all_mask.byte().contiguous().cuda())
        # image_tensors.append(src_viewpoint.original_image[channel].contiguous().cuda())
        image_tensors.append(src_viewpoint.original_image.mean(axis=0).contiguous().cuda())
        row = torch.concat([src_ext[:3,:3].reshape(-1), src_ext[:3,3].reshape(-1), src_K.reshape(-1), torch.tensor([img_H, img_W, depth_min, depth_max])])
        camera_tensors.append(row.float().unsqueeze(0).contiguous().cuda())

    if len(image_tensors) < 2:
        return None, None, None
    
    try:
        plane_out, cost_out = patchmatch_cuda.run_propagation(
            image_tensors,
            mask_tensors,
            camera_tensors,
            depth_tensor,
            anchor_depth_tensor * 0,
            normal_guess_tensor
        )
        depth_out = plane_out[..., 3]
        normal_out = plane_out[..., :3]
        
    except:
        print("failed somewhere in propagation")
        torch.save({"image_tensors":image_tensors, "mask_tensors":mask_tensors, "camera_tensors":camera_tensors, "depth_tensor":depth_tensor, "anchor_depth_tensor":anchor_depth_tensor, "normal_guess_tensor":normal_guess_tensor}, "./text.pth")
        depth_out, cost_out, normal_out = None, None, None

    # 0627 tmp zyk  Nuscenes
    nuscenes_tmp = normal_out.clone()
    nuscenes_tmp[:,:,0]=normal_out[:,:,0]
    nuscenes_tmp[:,:,1]=-normal_out[:,:,2]
    nuscenes_tmp[:,:,2]=normal_out[:,:,1]

    return depth_out, cost_out, nuscenes_tmp

#############################################################################

def depth_propagation_old(current_view, src_idxs, rendered_depth, rendered_normals, viewpoint_stack, dataset, vehicle_name=None, patch_size=20):
    # pass data to c++ api for mvs
    viewpoint_cam = viewpoint_stack[current_view]
    frame_id = current_view // 3
        
    dataset_path = dataset.source_path
    
    cdata_image_path = './cache/images'
    cdata_mask_path = './cache/masks'
    cdata_camera_path = './cache/cams'
    cdata_depth_path = './cache/depths'
    cdata_normal_path = './cache/normals'

    for path in [cdata_image_path, cdata_mask_path, cdata_camera_path, cdata_depth_path, cdata_normal_path]:
        if os.path.exists(path):
            shutil.rmtree(path)  
        os.mkdir(path)


    ### 生成./cache/masks     
    sky_mask = viewpoint_cam.guidance['sky_mask']
    dynamic_mask = cv2.imread(os.path.join(dataset_path, "sam_masks", viewpoint_cam.image_name + ".png"))#[:,:,0]
    dynamic_mask = cv2.resize(dynamic_mask, (sky_mask.shape[2], sky_mask.shape[1]), interpolation=cv2.INTER_NEAREST)
    if vehicle_name is None: # bkgd
        all_mask = sky_mask[0].cpu().detach().numpy().astype(np.bool_)
        all_mask = np.logical_or(all_mask, np.any(dynamic_mask.astype(np.bool_), axis=2))
    else:
        all_mask = sky_mask[0].cpu().detach().numpy().astype(np.bool_)
    plt.imsave(os.path.join(cdata_mask_path, "0.png"), ~all_mask)


    ### 生成./cache/images (ref)
    ref_img = viewpoint_cam.original_image
    ref_img = ref_img * 255
    ref_img = ref_img.permute((1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    # ref_img[all_mask] = 0
    cv2.imwrite(os.path.join(cdata_image_path, "0.jpg"), ref_img)


    ### 生成./cache/normals 
    # 后面改成渲染结果
    # tmp = cv2.imread(os.path.join(dataset_path, "normal_img", viewpoint_cam.image_name + ".png"))
    # scale = min(1.0, 1600 / tmp.shape[1])
    # resolution = (int(tmp.shape[0] * scale), int(tmp.shape[1] * scale), 3)
    # tmp = np.resize(tmp, resolution)
    # ref_norm = np.zeros(tmp.shape)
    # ref_norm[:,:,0] = tmp[:,:,2]
    # ref_norm[:,:,1] = tmp[:,:,0]
    # ref_norm[:,:,2] = tmp[:,:,1] # fix

    # rendered_normals
    cv2.imwrite(os.path.join(cdata_normal_path, "0.png"), (rendered_normals/2+0.5)*255)
    # cv2.imwrite(os.path.join(cdata_normal_path, "0.png"), ref_norm)

    ### 生成./cache/depths 
    ref_K = viewpoint_cam.K.detach().cpu().numpy()
    ref_w2c = viewpoint_cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()


    if vehicle_name is not None:
        obj_view_dict = dataset.scene_info.metadata["obj_view_dict"]
        if current_view not in obj_view_dict[vehicle_name]:
            print("obj not exist in ref", current_view)
            return False
        obj_pose_vehicle = obj_view_dict[vehicle_name][current_view][1]
        ego_pose = dataset.scene_info.metadata["ego_frame_poses"][frame_id]
        
        ref_o2w = ego_pose @ obj_pose_vehicle
        ref_o2c = ref_w2c @ ref_o2w

    cv2.imwrite(os.path.join(cdata_depth_path, "0.png"), rendered_depth.astype(np.uint8))
    writeDepthDmb(os.path.join(cdata_depth_path, "0.dmb"), rendered_depth)

    # voxel depth?
    anchor_depth = np.zeros([ref_img.shape[0], ref_img.shape[1]]).astype(np.float32) # 当前帧注册的可见点云，认为绝对准确
    anchor_depth = viewpoint_cam.guidance['lidar_depth'].cpu().detach().numpy()[0]
    anchor_depth.tofile(os.path.join(cdata_depth_path, "0.raw"))

    ### 生成./cache/cams (ref)
    depth_min = 1 
    if vehicle_name is None:
        depth_max = max(rendered_depth.max(), 80)
        ref_ext = ref_w2c
    else:
        depth_max = max(rendered_depth.max(), 80)
        ref_ext = ref_o2c
    write_cam_txt(os.path.join(cdata_camera_path, "0.txt"), ref_K, ref_ext, [depth_min, (depth_max-depth_min)/192.0, 192.0, depth_max])

    for idx, src_idx in enumerate(src_idxs):
        frame_id = src_idx // 3
        src_viewpoint = viewpoint_stack[src_idx]

        ### 生成./cache/masks 
        sky_mask = src_viewpoint.guidance['sky_mask']
        dynamic_mask = cv2.imread(os.path.join(dataset_path, "sam_masks", src_viewpoint.image_name + ".png"))#[:,:,0]
        dynamic_mask = cv2.resize(dynamic_mask, (sky_mask.shape[2], sky_mask.shape[1]), interpolation=cv2.INTER_NEAREST)


        if vehicle_name is None: # bkgd
            all_mask = sky_mask[0].cpu().detach().numpy().astype(np.bool_)
            all_mask = np.logical_or(all_mask, np.any(dynamic_mask.astype(np.bool_), axis=2))
        else:
            all_mask = sky_mask[0].cpu().detach().numpy().astype(np.bool_)
        plt.imsave(os.path.join(cdata_mask_path, str(idx+1)+".png"), ~all_mask)

        ### 生成./cache/images (src)
        src_img = src_viewpoint.original_image
        src_img = src_img * 255
        src_img = src_img.permute((1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        # src_img[all_mask] = 0
        cv2.imwrite(os.path.join(cdata_image_path, str(idx+1)+".jpg"), src_img)

        ### 生成./cache/cams (src)
        src_w2c = src_viewpoint.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        src_K = src_viewpoint.K.detach().cpu().numpy()

        if vehicle_name is None:
            src_ext = src_w2c
        else:
            if src_idx not in obj_view_dict[vehicle_name]:
                print("obj not exist in src", src_idx)
                continue
            obj_pose_vehicle = obj_view_dict[vehicle_name][src_idx][1]

            ego_pose = dataset.scene_info.metadata["ego_frame_poses"][frame_id]
            src_o2c = src_w2c @ ego_pose @ obj_pose_vehicle
            src_ext = src_o2c

        write_cam_txt(os.path.join(cdata_camera_path, str(idx+1)+".txt"), src_K,   src_ext, [depth_min, (depth_max-depth_min)/192.0, 192.0, depth_max])
        
    # c++ api for depth propagation
    # propagation_command = '/home/zyk/Projects/_DL/_Reconstruction/street_gaussians/submodules/Propagation/Propagation ./cache 0 "1 2 3 4" ' + str(patch_size)
    # os.system(propagation_command)

    propagation_command = [
        '/home/zyk/Projects/_DL/_Reconstruction/street_gaussians/submodules/Propagation/Propagation',
        './cache',
        '0',
        '1 2 3 4',
        str(patch_size)
    ]
    result = subprocess.run(propagation_command, capture_output=True, text=True)

    return ref_ext




def generate_edge_mask(propagated_depth, patch_size):
    # img gradient
    x_conv = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).float().cuda()
    y_conv = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).float().cuda()
    gradient_x = torch.abs(torch.nn.functional.conv2d(propagated_depth.unsqueeze(0).unsqueeze(0), x_conv, padding=1))
    gradient_y = torch.abs(torch.nn.functional.conv2d(propagated_depth.unsqueeze(0).unsqueeze(0), y_conv, padding=1))
    gradient = gradient_x + gradient_y

    # edge mask
    edge_mask = (gradient > 5).float()

    # dilation
    kernel = torch.ones(1, 1, patch_size, patch_size).float().cuda()
    dilated_mask = torch.nn.functional.conv2d(edge_mask, kernel, padding=(patch_size-1)//2)
    dilated_mask = torch.round(dilated_mask).squeeze().to(torch.bool)
    dilated_mask = ~dilated_mask

    return dilated_mask




def densify_bkgd_by_viewpoint(randidx, gt_image, optim_args, dataset, gaussians, gaussians_renderer, viewpoint_stack, iteration, voxel_depth_value, voxel_depth_source, bkgd_positions_visibile, bkgd_colors_visibile, sky_mask=None, obj_bound=None, vacancy_threshold=0.5, view_diversity_threshold=0.001):
    viewpoint_cam = viewpoint_stack[randidx]
    render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, render_type="rgb", include_list=["background"])
    # image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    rendered_depth = render_pkg['depth'][0] # [1, H, W]
    rendered_normals = render_pkg['normals']

    grape_trellis = gaussians.background.grape_trellis

    segment_img = viewpoint_cam.meta["segment_img"]
    no_bkgd_mask = np.logical_or(sky_mask.cpu().detach().numpy()[0], obj_bound.cpu().detach().numpy()[0])
    # voxel_depth_value, voxel_depth_source, bkgd_positions_visibile, bkgd_colors_visibile = grape_trellis.render_voxel_depth(randidx, segment_img.shape[0], segment_img.shape[1])
    vacancy_exist, vacancy_colors = grape_trellis.if_vacancy_in_ref_view_masks(voxel_depth_value, segment_img, no_bkgd_mask, vacancy_threshold=vacancy_threshold)

    if vacancy_exist: 
        for i in range(len(vacancy_colors)):
            c = vacancy_colors[i]
            obj_mask = np.all(segment_img==c, axis=2)
            
            # ref_src_views, viewset_diversity_score, rootvine_xyz = grape_trellis.sample_viewset_from_obj_mask(randidx, obj_mask, voxel_depth_value, voxel_depth_source, bkgd_positions_visibile)
            ref_src_views, viewset_diversity_score, rootvine_xyz = grape_trellis.sample_viewset_from_obj_mask(randidx, bkgd_positions_visibile)

            # visualize 
            # grape_trellis.visualize_viewset(ref_src_views, rootvine_xyz, bkgd_positions_visibile, bkgd_colors_visibile)
            src_idxs = ref_src_views[1:]
            if len(src_idxs) < 2:
                continue

            # intervals = [-6, -3, 3, 6, 9] # waymo, 3 cam per timestamp
            # src_idxs = [randidx+itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < len(viewpoint_stack)))]


            depth_propagation(randidx, src_idxs, rendered_depth.detach().cpu().numpy(), rendered_normals.detach().cpu().numpy().transpose(1,2,0), viewpoint_stack, dataset, vehicle_name=None, patch_size=20)
            propagated_depth, cost, normal = read_propagted_depth('./cache/propagated_depth')

            cost = torch.tensor(cost).to(rendered_depth.device)
            normal = torch.tensor(normal).to(rendered_depth.device)

            R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
            normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
            
            propagated_depth = torch.tensor(propagated_depth).to(rendered_depth.device)
            valid_mask = propagated_depth != 300

            abs_rel_error = torch.abs(propagated_depth - rendered_depth) / propagated_depth
            depth_error_max_threshold = 1.0
            depth_error_min_threshold = 0.8
            abs_rel_error_threshold = depth_error_max_threshold - (depth_error_max_threshold - depth_error_min_threshold) * (iteration - optim_args.propagated_iteration_begin) / (optim_args.propagated_iteration_after - optim_args.propagated_iteration_begin)
            # color error
            # render_color = render_pkg['rgb']
            # color_error = torch.abs(render_color - viewpoint_cam.original_image)
            # color_error = color_error.mean(dim=0).squeeze()
            #for waymo, quantile 0.6; for free dataset, quantile 0.4
            error_mask = (abs_rel_error > abs_rel_error_threshold)


            ref_K = viewpoint_cam.K
            ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
            geometric_counts = None
            for idx, src_idx in enumerate(src_idxs):
                src_viewpoint = viewpoint_stack[src_idx]
                #c2w
                src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                src_K = src_viewpoint.K

                src_render_pkg = gaussians_renderer.render(src_viewpoint, gaussians, render_type="rgb", include_list=["background"])
                src_rendered_depth = src_render_pkg['depth'][0]
                src_rendered_normal = src_render_pkg['normals']
                
                #get the src_depth first
                depth_propagation(src_idx, src_idxs, src_rendered_depth.detach().cpu().numpy(), src_rendered_normal.detach().cpu().numpy().transpose(1,2,0), viewpoint_stack, dataset, vehicle_name=None, patch_size=20)
                src_depth, cost, src_normal = read_propagted_depth('./cache/propagated_depth')
                src_depth = torch.tensor(src_depth).cuda()
                mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                                ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                                src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=1, thre2=0.01)
                if geometric_counts is None:
                    geometric_counts = mask.to(torch.uint8)
                else:
                    geometric_counts += mask.to(torch.uint8)
                    
            cost = geometric_counts.squeeze() # 这里cost大约代表各视角下共享视野的部分，越高代表被共同观测且匹配成功的视角越多
            cost_mask = cost >= len(src_idxs)*0.75

            normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
            
            propagated_mask = valid_mask & ~error_mask & cost_mask # propagation有值 & 由渲染获得的D误差足够大 & 被多个视角共同观测 
            if sky_mask is not None:
                propagated_mask = propagated_mask & ~sky_mask[0]

            if obj_bound is not None:
                propagated_mask = propagated_mask & ~obj_bound[0]


            propogated_vine_mask = np.logical_and(propagated_mask.cpu().detach().numpy(), obj_mask)
            propogated_vine_mask = np.logical_and(propogated_vine_mask, voxel_depth_source==-1)

            if propogated_vine_mask.sum() > 100  and  viewset_diversity_score > view_diversity_threshold:
                ref_visibility = np.zeros([len(viewpoint_stack)]).astype(np.bool_)
                ref_visibility[ref_src_views[0]] = True
                grape_trellis.generate_vine_voxel_from_depth_propogation(viewpoint_cam, propagated_depth, propogated_vine_mask, ref_visibility, viewset_diversity_score, gt_image, normal) # normal from prop or pred?

            print("depth_propagation", i, "/", len(vacancy_colors), "finished, view", randidx, "points", propogated_vine_mask.sum())

        # grape_trellis.visualize_root_and_vine()
            # if propagated_mask.sum() > 100:
            #     gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool), gt_image) 

            # # Visualize: 
            # import numpy as np
            # import open3d as o3d
            # fx, fy = ref_K[0,0].item(), ref_K[1,1].item()
            # cx, cy = ref_K[0,2].item(), ref_K[1,2].item()
            # ref_img = viewpoint_cam.original_image.numpy().transpose(1,2,0)

            # y, x = np.meshgrid(np.arange(viewpoint_cam.image_height), np.arange(viewpoint_cam.image_width), indexing='ij')
            # x_dir = (x - cx) / fx # 右
            # y_dir = (y - cy) / fy # 下
            # z_dir = np.ones_like(x_dir) # 前
            # directions = np.stack([x_dir, y_dir, z_dir], axis=-1)
            # xyz = directions * propagated_depth.cpu().detach().numpy()[:,:,None]
            # # _msk = propagated_mask.cpu().detach().numpy()
            # _msk = propogated_vine_mask
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz[_msk])
            # pcd.colors = o3d.utility.Vector3dVector(ref_img[_msk])
            # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd, FOR1])

            # projected_depth = rendered_depth.detach().cpu().numpy()
            # xyz = directions * projected_depth[:,:,None]
            # _c = ref_img[_msk] / 255
            # _c[:,0] = 1
            # _c[:,1] = 0
            # _c[:,2] = 0
            # pcd_proj = o3d.geometry.PointCloud()
            # pcd_proj.points = o3d.utility.Vector3dVector(xyz[_msk])
            # pcd_proj.colors = o3d.utility.Vector3dVector(_c)
            # o3d.visualization.draw_geometries([pcd_proj, FOR1])

            # o3d.visualization.draw_geometries([pcd, pcd_proj, FOR1])


            # # Visualize whole environment pointcloud:
            # import numpy as np
            # import open3d as o3d
            # from lib.utils.sh_utils import SH2RGB
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(gaussians.background.get_xyz.cpu().detach().numpy())
            # pcd.colors = o3d.utility.Vector3dVector(SH2RGB(gaussians.background.get_features_dc)[:,0,:].cpu().detach().numpy())
            # o3d.visualization.draw_geometries([pcd])
