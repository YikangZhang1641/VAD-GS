import torch
import torch.nn as nn

import numpy as np
import open3d as o3d
from collections import defaultdict  #希望能提速

import matplotlib.pyplot as plt
from numba import njit, prange

from lib.utils.sh_utils import RGB2SH, IDFT
from lib.utils.general_utils import inverse_sigmoid, get_expon_lr_func, quaternion_to_matrix, quaternion_raw_multiply
from scipy.spatial.transform import Rotation as R
import copy


class GrapeTrellis:
    def __init__(self, points_xyz, points_rgb, points_normal, points_visibility, voxel_size = 0.15):
        self.voxel_size = voxel_size
        self.min_bound=(points_xyz.min(axis=0) // self.voxel_size) * self.voxel_size
        self.max_bound=(points_xyz.max(axis=0) // self.voxel_size + 1) * self.voxel_size
        
        self.N_views = points_visibility.shape[1]
        self.root_table = RootTable(self.min_bound, self.max_bound, voxel_size)
        self.vine_table = VineTable(self.min_bound, self.max_bound, self.N_views, voxel_size)

        self.c2ws = None
        self.ixts = None

        ctr2corners = []
        for i in [-1,1]:
            for j in [-1,1]:
                for k in [-1,1]:
                    ctr2corners.append([i,j,k])
        self.ctr2corners = np.array(ctr2corners) * voxel_size / 2

        self.root_table.build_hash_table(points_xyz, points_rgb, points_normal, points_visibility)

        # xyz:        geo
        # xyz_offset:       photo 
        # rgb:              photo 
        # scale:      geo & photo
        # rot:        geo 
        # rot_offset:       photo
        # opaticy:    geo & photo


        # Trellis 只维护anchor xyz和rot的tensor或array。不存nn.parameter。nn.parameter一直在GaussianModelBkgd内进行迭代和训练。新增部分应当只空白地cat在最后，而非直接由Trellis获取并重复初始化。

    def save(self, path):
        data = {
            "voxel_size": self.voxel_size,
            "min_bound": self.min_bound,
            "max_bound": self.max_bound,
            "N_views": self.N_views,

            "root_points_xyz": self.root_table.points_xyz,
            "root_points_color": self.root_table.points_color,
            "root_points_normal": self.root_table.points_normal,
            "root_points_visibility": self.root_table.points_visibility,
            "root_hash_voxel_id": self.root_table.hash_voxel_id,

            "vine_valid_cnt": self.vine_table.valid_cnt,
            "vine_points_xyz": self.vine_table.points_xyz,
            "vine_points_color": self.vine_table.points_color,
            "vine_points_normal": self.vine_table.points_normal,
            "vine_points_visibility": self.vine_table.points_visibility,
            "vine_points_ray_vector": self.vine_table.points_ray_vector,
            "vine_view_diversity": self.vine_table.view_diversity,
            "vine_hash_voxel_table_id": self.vine_table.hash_voxel_table_id,
        }
        np.savez(path, **data)
        

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.root_table.points_xyz = data["root_points_xyz"]
        self.root_table.points_color = data["root_points_color"]
        self.root_table.points_normal = data["root_points_normal"]
        self.root_table.points_visibility = data["root_points_visibility"]
        self.root_table.hash_voxel_id = data["root_hash_voxel_id"].item()
        self.root_table.voxel_size = data["voxel_size"].item()
        self.root_table.min_bound = data["min_bound"]
        self.root_table.max_bound = data["max_bound"]


        self.vine_table.valid_cnt = data["vine_valid_cnt"].item()
        self.vine_table.points_xyz = data["vine_points_xyz"]
        self.vine_table.points_color = data["vine_points_color"]
        self.vine_table.points_normal = data["vine_points_normal"]
        self.vine_table.points_visibility = data["vine_points_visibility"]
        self.vine_table.points_ray_vector = data["vine_points_ray_vector"]
        self.vine_table.view_diversity = data["vine_view_diversity"]
        self.vine_table.hash_voxel_table_id = data["vine_hash_voxel_table_id"].item()
        self.vine_table.N_views = data["N_views"].item()
        self.vine_table.voxel_size = data["voxel_size"].item()
        self.vine_table.min_bound = data["min_bound"]
        self.vine_table.max_bound = data["max_bound"]


    def set_param(self, c2ws, ixts, selected_frames, cams_per_frame):
        self.c2ws = c2ws
        self.ixts = ixts
        self.start_frame = selected_frames[0] * cams_per_frame
        self.end_frame = (selected_frames[1] + 1) * cams_per_frame

    def get_voxel_center_xyz(self):
        root_xyz = self.root_table.points_xyz
        vine_xyz = self.vine_table.get_xyz()
        return np.concatenate([root_xyz, vine_xyz], axis=0)

    # def get_xyz_offset(self):
    #     # 只在densiffication后修改torch tensor，其余时刻只读取nn.parameter
    #     pass

    def get_voxel_color(self):
        root_color = self.root_table.points_color
        vine_color = self.vine_table.get_color()
        return np.concatenate([root_color, vine_color], axis=0)


    def get_voxel_size(self):
        return self.root_table.points_xyz.shape[0] + self.vine_table.valid_cnt


    def get_visibility(self):
        root_vis = self.root_table.points_visibility #[:, self.start_frame:self.end_frame]
        vine_vis = self.vine_table.get_visibility()
        return np.concatenate([root_vis, vine_vis], axis=0)

    def get_normal(self):
        root_normal = self.root_table.points_normal
        vine_normal = self.vine_table.get_normal()

        anchor_normal = np.concatenate([root_normal, vine_normal], axis=0)
        return anchor_normal

    def get_voxel_visibility_from_xyz(self, x, y, z):
        # key = self.hashcode(x,y,z)

        vid = self.root_table.get_voxel_id_from_point(x, y, z)
        if vid is not None:
            return self.root_table.points_visibility[vid]

        voxel_name = self.vine_table.hashcode_voxel(x, y, z)
        if voxel_name in self.vine_table.hash_voxel_table_id:
            vid = self.vine_table.hash_voxel_table_id[voxel_name]
            return self.vine_table.points_visibility[vid]
        
        return np.zeros([self.N_views]).astype(np.bool_)
    
    
    # def render_voxel_depth(self, current_view, img_H, img_W):
    #     bkgd_positions = self.get_voxel_center_xyz()
    #     bkgd_colors = self.get_voxel_color()
    #     bkgd_view_mask = self.get_visibility()[:, current_view]

    #     # view_pos_world_voxel_corners = bkgd_positions[mask_root, None, :].repeat(8,1) + self.ctr2corners
    #     view_pos_world_voxel_corners = bkgd_positions[:, None, :].repeat(8,1) + self.ctr2corners
    #     view_pos_world_voxel_corners = np.concatenate([view_pos_world_voxel_corners, np.ones_like(view_pos_world_voxel_corners[..., :1])], axis=-1)
    #     view_pos_cam = view_pos_world_voxel_corners @ np.linalg.inv(self.c2ws[current_view]).T

    #     tmp = view_pos_cam[...,:3] @ self.ixts[current_view].T
    #     corners_2d = tmp[..., :2] / tmp[..., 2:]
    #     corners_2d = np.round(corners_2d).astype(int)

    #     # 判断条件是否足够？如边界等
    #     mask_visible = np.logical_and(corners_2d[:,0,0] >= 0, corners_2d[:,0,0] < img_W)
    #     mask_visible = np.logical_and(mask_visible, corners_2d[:,0,1] >= 0)
    #     mask_visible = np.logical_and(mask_visible, corners_2d[:,0,1] < img_H)
    #     mask_visible = np.logical_and(mask_visible, tmp[:,0,2] > 0.2333)
    #     mask_visible = np.logical_and(mask_visible, bkgd_view_mask) # 遮挡关系

    #     corners_2d = np.round(corners_2d[mask_visible]).astype(int)
    #     voxel_depth_value, voxel_depth_source = parallel_rasterize(corners_2d, tmp[mask_visible,0,2], img_H, img_W)
    #     voxel_depth_value[voxel_depth_value==np.inf] = 0
    #     return voxel_depth_value, voxel_depth_source, bkgd_positions[mask_visible], bkgd_colors[mask_visible]


    def render_voxel_depth(self, current_view, img_H, img_W, obj_rots=None, obj_trans=None):
        actor_positions = self.get_voxel_center_xyz()
        actor_colors = self.get_voxel_color()
        actor_view_mask = self.get_visibility()[:, current_view]

        # track_id = obj_model.track_id
        # obj_rot = gaussians.actor_pose.get_tracking_rotation(track_id, viewpoint_cam)
        # obj_trans = gaussians.actor_pose.get_tracking_translation(track_id, viewpoint_cam)                
        # ego_pose = viewpoint_cam.ego_pose
        # ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
        # obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
        # obj_rots = quaternion_to_matrix(obj_rot)
        # obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]
        if obj_rots is not None and obj_trans is not None:
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, torch.from_numpy(actor_positions).cuda()) + obj_trans
            xyzs_obj = xyzs_obj.cpu().detach().numpy()
        else:
            xyzs_obj = actor_positions

        # view_pos_world_voxel_corners = bkgd_positions[mask_root, None, :].repeat(8,1) + self.ctr2corners
        view_pos_world_voxel_corners = xyzs_obj[:, None, :].repeat(8,1) + self.ctr2corners
        view_pos_world_voxel_corners = np.concatenate([view_pos_world_voxel_corners, np.ones_like(view_pos_world_voxel_corners[..., :1])], axis=-1)
        view_pos_cam = view_pos_world_voxel_corners @ np.linalg.inv(self.c2ws[current_view]).T

        # self.ixts[current_view].T #viewpoint_cam.K.cpu().detach().numpy().T
        # scale = 960/1920 # waymo
        scale = 960/1600 # Nuscenes
        K = copy.deepcopy(self.ixts[current_view])
        K[:2] *= scale        
        
        tmp = view_pos_cam[...,:3] @ K.T
        # tmp = (xyzs_obj[:, None, :].repeat(8,1) + self.ctr2corners) @ self.ixts[current_view].T #viewpoint_cam.K.cpu().detach().numpy().T
        corners_2d = tmp[..., :2] / tmp[..., 2:]
        corners_2d = np.round(corners_2d).astype(int)

        # 判断条件是否足够？如边界等
        mask_visible = np.logical_and(corners_2d[:,0,0] >= 0, corners_2d[:,0,0] < img_W)
        mask_visible = np.logical_and(mask_visible, corners_2d[:,0,1] >= 0)
        mask_visible = np.logical_and(mask_visible, corners_2d[:,0,1] < img_H)
        mask_visible = np.logical_and(mask_visible, tmp[:,0,2] > 0.2333)
        mask_visible = np.logical_and(mask_visible, actor_view_mask) # 遮挡关系. zyk: nuscenes过于稀疏，先这样尝试，后续按visibility应当删除

        # corners_2d = np.round(corners_2d[mask_visible]).astype(int)
        voxel_depth_value, voxel_depth_source = parallel_rasterize(corners_2d[mask_visible], tmp[mask_visible,0,2], img_H, img_W)
        voxel_depth_value[voxel_depth_value==np.inf] = 0
        return voxel_depth_value, voxel_depth_source, mask_visible, corners_2d[:, 0, :2] # actor_positions[mask_visible], actor_colors[mask_visible]



    def if_vacancy_in_ref_view_masks(self, voxel_depth_value, semantic_mask, no_bkgd_mask, vacancy_threshold=0.3):
        # 过滤所有semantic_mask id，取出存在voxel depth无法覆盖的mask颜色
        img_H, img_W = voxel_depth_value.shape[0], voxel_depth_value.shape[1]
        samentic_colors = np.unique(semantic_mask.reshape([-1,3]), axis=0)
        vacancy_colors = []
        for i in range(samentic_colors.shape[0]):
            c = samentic_colors[i]
            if np.all(c==0) or np.all(c==1):
                continue

            obj_mask = np.all(semantic_mask==c, axis=2)
            if obj_mask.sum() < 100*100:
                continue
            # zyk: manual designed filter 
            min_idx = np.argwhere(obj_mask).min(0)
            max_idx = np.argwhere(obj_mask).max(0)
            diff_idx = max_idx - min_idx
            if diff_idx[0] < 100 and diff_idx[1] < 100:
                continue

            img_border_ratio = 0.2
            if max_idx[0] < img_H * img_border_ratio or min_idx[0] > img_H * (1 - img_border_ratio):
                continue
            
            if max_idx[1] < img_W * img_border_ratio or min_idx[1] > img_W * (1 - img_border_ratio):
                continue

            # dynamic mask missing !!! sky mask missing !!!
            vacancy_out_of_scope = np.logical_and(no_bkgd_mask, obj_mask)
            no_bkgd_ratio = vacancy_out_of_scope.sum() / obj_mask.sum()
            if no_bkgd_ratio > 0.5:
                semantic_mask[obj_mask] = 0
                continue

            vacancy_in_obj_mask = np.logical_and(voxel_depth_value == 0, obj_mask)
            vacancy_ratio = vacancy_in_obj_mask.sum() / obj_mask.sum()
            if vacancy_ratio < vacancy_threshold:
                # print("no need to extend")
                semantic_mask[obj_mask] = 0
            else:
                print("need extension") 
                vacancy_colors.append(c)
        return len(vacancy_colors) > 0, vacancy_colors
    


    def calculate_belief_view_pair(self, target_points, ref_mat, src_mat):
        src_R = src_mat[:3,:3]
        src_T = src_mat[:3, 3]

        ref_R = ref_mat[:3,:3]
        ref_T = ref_mat[:3, 3]

        # baseline_t = (src_T - ref_T) @ np.linalg.inv(ref_R).T
        baseline_t = (src_T - ref_T) @ np.linalg.inv(ref_R).T

        B = np.sqrt(np.sum(baseline_t[:2] ** 2)) / np.abs(baseline_t[2])
        # B = np.sum((src_T - ref_T)**2)
        r = R.from_matrix(np.linalg.inv(ref_R) @ src_R)

        # theta = 1-(r.as_quat(scalar_first=True)[0])**2
        theta = 2 * np.arccos(r.as_quat()[-1])
        return np.sqrt(B * theta) / (np.linalg.norm(target_points - src_T, axis=1).mean() * np.linalg.norm(target_points - ref_T, axis=1).mean())

    # 根据给定的一组视角，计算其整体视角差异置信度：
    def calculate_belief_views(self, rootvine_xyz, ref_src_view_ids):
        prop_belief = []
        for i in range(len(ref_src_view_ids)-1):
            for j in range(i+1, len(ref_src_view_ids)):
                view_A = ref_src_view_ids[i]
                view_B = ref_src_view_ids[j]
                prop_belief.append(self.calculate_belief_view_pair(rootvine_xyz, self.c2ws[view_A], self.c2ws[view_B]))
        return np.median(prop_belief)
    
    
    def greedy_sample_src_views_by_ref_points(self, rootvine_xyz, ref_view, mask_visible, point_obs_ratio=0.75, N=5, obj_rots=None, obj_trans=None):
        # rootvine_xyz = self.get_voxel_center_xyz()[mask_visible]

        # vine_visibility = []
        # for i in range(rootvine_xyz.shape[0]):
        #     x,y,z = rootvine_xyz[i]
        #     vibility = self.get_voxel_visibility_from_xyz(x,y,z)
        #     vine_visibility.append(vibility)
        vine_visibility = self.get_visibility()[mask_visible] # bkgd或obj所有voxel中被观测到的部分

        obs_per_voxel = vine_visibility.sum(1) # 每个voxel 被观测到的视角数，用于找到被多个视角共同观测的锚点
        # thres = max(obs_per_voxel.mean(), 5) # Waymo
        thres = obs_per_voxel.mean()           # Nuscenes
        obs_voxel_idx = np.argwhere(obs_per_voxel >= thres)[:, 0] # 必要voxel 
        
        obs_per_view = vine_visibility[obs_voxel_idx].sum(0) # 包含 必要voxel 的view
        # thres2 = obs_per_view.mean() 
        candidates = np.where(obs_per_view >= obs_per_view.max() * point_obs_ratio)[0]
        # np.random.shuffle(candidates)
        # if is_obj:
        #     candidates = np.where(obs_per_view > rootvine_xyz.shape[0] * point_obs_ratio)[0]
        
        ref_src_view_ids = [ref_view]
        
        if candidates.shape[0] <= N:
            for c in candidates:
                if c in ref_src_view_ids:
                    continue
                ref_src_view_ids.append(c)
            return ref_src_view_ids
        
        belief_sum = np.zeros(candidates.shape[0])
        for _ in range(N):
            for i in range(candidates.shape[0]):
                v = candidates[i]
                if v in ref_src_view_ids:
                    belief_sum[i] = 0
                    continue
                tmp = self.calculate_belief_view_pair(rootvine_xyz[obs_voxel_idx], self.c2ws[ref_src_view_ids[-1]], self.c2ws[v])
                belief_sum[i] += 0 if np.isnan(tmp) else tmp
            # src1 = candidates[np.argmax(belief_sum)]
            src1 = candidates[np.random.choice(len(belief_sum), p=belief_sum/belief_sum.sum())]

            ref_src_view_ids.append(src1)

        return ref_src_view_ids


    # def sample_viewset_from_obj_mask(self, current_view, obj_mask, voxel_depth_value, voxel_depth_source, mask_visible, point_obs_ratio=0.8):
    def sample_viewset_from_obj_mask(self, current_view, mask_visible, point_obs_ratio=0.8, N=4, obj_rots=None, obj_trans=None):
        # root_in_obj_mask = np.logical_and(voxel_depth_value > 0, obj_mask)
        # root_in_obj_idx = np.unique(voxel_depth_source[root_in_obj_mask])
        # rootvine_xyz = bkgd_positions_visibile[root_in_obj_idx]
        rootvine_xyz = self.get_voxel_center_xyz()[mask_visible]
        if obj_rots is not None and obj_trans is not None:
            rootvine_xyz = torch.einsum('bij, bj -> bi', obj_rots, torch.from_numpy(rootvine_xyz).cuda()) + obj_trans
            rootvine_xyz = rootvine_xyz.cpu().detach().numpy()


        ref_src_views = self.greedy_sample_src_views_by_ref_points(rootvine_xyz, current_view, mask_visible, point_obs_ratio=point_obs_ratio, N=4)
        view_diversity = self.calculate_belief_views(rootvine_xyz, ref_src_views)
        return ref_src_views, view_diversity, rootvine_xyz

    def visualize_viewset(self, ref_src_views, rootvine_xyz, bkgd_positions_visibile, bkgd_colors_visibile):
        # 补充视角可视化，定性评估视角多元化置信度指标
        pcd_bkgd = o3d.geometry.PointCloud()
        pcd_bkgd.points = o3d.utility.Vector3dVector(bkgd_positions_visibile)
        pcd_bkgd.colors = o3d.utility.Vector3dVector(bkgd_colors_visibile)
        voxel_bkgd = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bkgd, voxel_size=self.voxel_size)

        lines = []
        for view in ref_src_views:
            cam_dir = np.array([0,0,1,1]) @ self.c2ws[view].T # @ np.linalg.inv(ego_pose).T @ np.linalg.inv(obj_pose_vehicle).T
            view_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.array([rootvine_xyz.mean(0), cam_dir[:3]])),
                lines=o3d.utility.Vector2iVector(np.array([[0,1]])),
            )
            if len(lines) == 0:
                view_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            else:
                view_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
            lines.append(view_line_set)
        o3d.visualization.draw_geometries([voxel_bkgd] + lines)
    
    def visualize_root_and_vine(self):
        pcd_bkgd = o3d.geometry.PointCloud()
        xyz = np.vstack([self.root_table.points_xyz, self.vine_table.get_xyz().reshape([-1,3])])
        # c1 = np.ones(self.root_table.points_xyz.shape) * 0.9
        c1 = self.root_table.points_color
        c2 = np.zeros(self.vine_table.get_xyz().shape).reshape([-1,3])
        c2[:, 0] = 1
        color = np.vstack([c1, c2])
        pcd_bkgd.points = o3d.utility.Vector3dVector(xyz)
        pcd_bkgd.colors = o3d.utility.Vector3dVector(color)
        voxel_bkgd = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bkgd, voxel_size=self.voxel_size)
        o3d.visualization.draw_geometries([voxel_bkgd])

    def visualize_root_and_vine_visibility(self):
        pcd_bkgd = o3d.geometry.PointCloud()
        xyz = np.vstack([self.root_table.points_xyz, self.vine_table.get_xyz().reshape([-1,3])])
        c1 = self.root_table.points_visibility.sum(1, keepdims=True).repeat(3,1).astype(np.float64)
        c2 = np.zeros(self.vine_table.get_xyz().shape).reshape([-1,3])
        c1[:, 0] = c1[:, 0] / c1[:, 0].max()
        c1[:, 1:] = 0
        c2[:, 0] = 1
        color = np.vstack([c1, c2]) 
        pcd_bkgd.points = o3d.utility.Vector3dVector(xyz)
        pcd_bkgd.colors = o3d.utility.Vector3dVector(color)
        voxel_bkgd = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bkgd, voxel_size=self.voxel_size)
        o3d.visualization.draw_geometries([voxel_bkgd])


    def rot_for_training(self):
        # zyk: normal 对应 rot scaling 如何计算? 
        # zyk: 肯定有问题。先定z轴为平面法线方向。
        anchor_normal = self.get_normal()

        axis = np.array([-anchor_normal[:,1], anchor_normal[:,0], np.zeros_like(anchor_normal[:,0])]).T
        axis_norm = np.linalg.norm(axis, axis=1)

        mask = axis_norm > 1e-6
        axis[mask] /= axis_norm[mask].reshape(-1,1)
        axis[~mask][:, 0] = 0
        axis[~mask][:, 1] = 0
        axis[~mask][:, 2] = 1

        half_theta = np.arccos(anchor_normal[:,2]) / 2
        q_w = np.cos(half_theta).reshape(-1,1)
        q_xyz = axis * np.sin(half_theta).reshape(-1,1)
        rots = np.concatenate([q_w, q_xyz], axis=1)
        return rots


    def generate_vine_voxel_from_depth_propogation(self, viewpoint_cam, propagated_depth, vine_mask, ref_visibility, ref_viewset_diversity_score, ref_gt_image, ref_prop_normal):
        # 首先拿到所有点。对于root已有pixel（voxel depth有值），过滤；否则记录voxel_xyz和视角信息
        cam_id = viewpoint_cam.id
        K = viewpoint_cam.K
        cam2world = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()

        height, width = propagated_depth.shape
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        coordinates = torch.stack([x.to(propagated_depth.device), y.to(propagated_depth.device), torch.ones_like(propagated_depth)], dim=-1)
        coordinates_3D = propagated_depth[vine_mask].unsqueeze(1) * (K.inverse() @coordinates[vine_mask].T).T
        ray_vectors = (cam2world[:3, :3] @ coordinates_3D.T).T
        world_coordinates_3D = ray_vectors + cam2world[:3, 3]

        u, indices = np.unique(world_coordinates_3D.cpu().detach().numpy() // self.voxel_size, axis=0, return_index=True)
        # voxel_pos = torch.unique((world_coordinates_3D) // self.voxel_size, dim=0).to(torch.int16).cpu().detach().numpy()
        for vid in indices:
            p_xyz = world_coordinates_3D[vid].cpu().detach().numpy() # zyk: turn to tensor cuda later
            p_color = ref_gt_image[:,vine_mask][:,vid].cpu().detach().numpy()
            p_normal = ref_prop_normal[:,vine_mask][:,vid].cpu().detach().numpy()
            p_ray = ray_vectors[vid].cpu().detach().numpy()

            # voxel_name = self.vine_table.hashcode_voxel(vxyz[0], vxyz[1], vxyz[2])
            voxel_name = self.vine_table.hashcode_point(p_xyz[0], p_xyz[1], p_xyz[2])
            if voxel_name in self.root_table.hash_voxel_id.keys():
                # 是否已经在root table中
                continue

            # 是否已经在vine table中
            searched_ray_nbr = False
            # 其实应当有在2D上投影重叠但3D中并不重叠的情况
            # 在视角方向上寻找数格，若需要其他视角voxel则认为是重叠（root-vine-vine-root联结方案）。搜寻长度以diversity score为基准。
            
            # 需要对置信度有所更改
            ray_nbr = voxel_traversal(p_xyz - p_ray*0.1, p_xyz + p_ray*0.1, self.voxel_size)
            for i,j,k in ray_nbr:
                nbr_name = self.vine_table.hashcode_voxel(i,j,k)
                if nbr_name in self.root_table.hash_voxel_id:
                    break

                if nbr_name in self.vine_table.hash_voxel_table_id:
                    # nbr = self.vine_table.get_xyz(nbr_name)
                    nbr_idx = self.vine_table.hash_voxel_table_id[nbr_name]
                    searched_ray_nbr = True
                    if self.vine_table.view_diversity[nbr_idx] > ref_viewset_diversity_score: # vine-vine联结判别方案可继续累加，如必须同向视角才可替代、如必须视角diversity足够才可增删，etc
                        # 若nbr置信度足够大，则直接认为当前vine是落在nbr上的
                        self.vine_table.add_observation(nbr_name, ref_visibility, ref_viewset_diversity_score, p_ray)
                    else:
                        # 若新生vine置信度反而大于nbr，则将nbr移动并合并到当前vine
                        self.vine_table.add_observation(nbr_name, ref_visibility, ref_viewset_diversity_score, p_ray)
                        self.vine_table.points_xyz[nbr_idx] = p_xyz
                        self.vine_table.hash_voxel_table_id.pop(nbr_name)
                        self.vine_table.hash_voxel_table_id[voxel_name] = nbr_idx
                    break
    
            if not searched_ray_nbr:
                # current_voxel = self.vine_table.spawn_voxel(voxel_name)
                # current_voxel.add_observation(ref_visibility, ref_viewset_diversity_score, ray_vec)
                self.vine_table.push_back(p_xyz, p_color, p_normal, ref_visibility, ref_viewset_diversity_score, p_ray) # right?



# class VineVoxel:
#     def __init__(self, name):
#         self.name = name 
#         self.visibility = None
#         self.viewset_diversity = -1
#         self.ray_vector = None

#     def add_observation(self, visibility, diversity_score, ray_vector):
#         if self.visibility is None:
#             self.visibility = np.zeros(visibility.shape).astype(np.bool_)
#         self.visibility = np.logical_or(self.visibility, visibility)
#         if self.viewset_diversity < diversity_score:
#             self.viewset_diversity = diversity_score
#             self.ray_vector = ray_vector


class VineTable:
    def __init__(self, min_bound, max_bound, N_views=597, voxel_size=0.15):
        self.voxel_size = voxel_size
        self.N_views = N_views
        self.min_bound = min_bound
        self.max_bound = max_bound
        # self.hash_voxel_table = defaultdict(list)
        self.hash_voxel_table_id = defaultdict(list)

        self.CAPACITY = 2048
        self.points_xyz = np.zeros([self.CAPACITY, 3]).astype(np.float16)
        self.points_color = np.zeros([self.CAPACITY, 3]).astype(np.float16)   # or np.float?
        self.points_normal = np.zeros([self.CAPACITY, 3]).astype(np.float16)
        
        self.points_visibility = np.zeros([self.CAPACITY, N_views]).astype(np.bool_)
        self.points_ray_vector = np.zeros([self.CAPACITY, 3]).astype(np.float16)
        self.view_diversity = np.zeros([self.CAPACITY, 1]).astype(np.float16)

        self.valid_cnt = 0 # xxx[:valid_cnt] for query
    #     self.last_updated_cnt = 0
    #     # self.hash_voxel_id = defaultdict(list)

    # def update(self):
    #     new_voxels = 

    #     self.last_updated_cnt = self.valid_cnt

    def add_observation(self, voxel_name, ref_visibility, ref_viewset_diversity_score, ray_vector):
        idx = -1
        if voxel_name in self.hash_voxel_table_id:
            idx = self.hash_voxel_table_id[voxel_name]
            if idx < 0 or idx >= self.valid_cnt:
                return None
            
        self.points_visibility[idx] = np.logical_or(self.points_visibility[idx], ref_visibility)
        if self.view_diversity[idx] < ref_viewset_diversity_score:
            self.view_diversity[idx] = ref_viewset_diversity_score
            self.points_ray_vector[idx] = ray_vector
    
    def get_xyz(self, voxel_name=None):
        if voxel_name is None:
            return self.points_xyz[:self.valid_cnt]

        idx = -1
        if voxel_name in self.hash_voxel_table_id:
            idx = self.hash_voxel_table_id[voxel_name]
            if idx < 0 or idx >= self.valid_cnt:
                return None
        return self.points_xyz[idx]
    
    def get_color(self):
        return self.points_color[:self.valid_cnt]

    def get_normal(self):
        return self.points_normal[:self.valid_cnt]
    
    def get_visibility(self):
        return self.points_visibility[:self.valid_cnt]
    
    def get_ray_vector(self):
        return self.points_ray_vector[:self.valid_cnt]

    def push_back(self, new_xyz, new_color, new_normal, new_visibility, new_view_diversity, new_ray_vector):
        new_xyz = new_xyz.reshape(-1, 3)
        new_color = new_color.reshape(-1, 3)
        new_normal = new_normal.reshape(-1, 3)
        new_visibility = new_visibility.reshape(-1, self.N_views)
        new_view_diversity = new_view_diversity.reshape(-1, 1)
        new_ray_vector = new_ray_vector.reshape(-1, 3)

        num_new_points = new_xyz.shape[0]
        if self.valid_cnt + num_new_points > self.CAPACITY:
            self._double_resize()

        for i in range(num_new_points):
            name = self.hashcode_point(new_xyz[i,0], new_xyz[i,1], new_xyz[i,2])
            if name not in self.hash_voxel_table_id:
                self.hash_voxel_table_id[name] = self.valid_cnt

                self.points_xyz[self.valid_cnt] = new_xyz
                self.points_color[self.valid_cnt] = new_color # float? int?
                self.points_normal[self.valid_cnt] = new_normal
                self.points_visibility[self.valid_cnt] = new_visibility
                self.points_ray_vector[self.valid_cnt] = new_ray_vector
                self.view_diversity[self.valid_cnt] = new_view_diversity

                self.valid_cnt += 1


    def _double_resize(self):
        self.CAPACITY = self.CAPACITY * 2
        self.points_xyz = self._enlarge_and_copy(self.points_xyz, self.CAPACITY)
        self.points_color = self._enlarge_and_copy(self.points_color, self.CAPACITY)
        self.points_normal = self._enlarge_and_copy(self.points_normal, self.CAPACITY)
        self.points_visibility = self._enlarge_and_copy(self.points_visibility, self.CAPACITY)
        self.points_ray_vector = self._enlarge_and_copy(self.points_ray_vector, self.CAPACITY)
        self.view_diversity = self._enlarge_and_copy(self.view_diversity, self.CAPACITY)


    def _enlarge_and_copy(self, x, size):
        new_x = np.zeros([size, x.shape[1]]) 
        new_x[:self.valid_cnt] = x[:self.valid_cnt]
        return new_x


    def hashcode_point(self, x, y, z):
        return str(int(x // self.voxel_size)) + " " + str(int(y // self.voxel_size)) + " " + str(int(z // self.voxel_size))

    def hashcode_voxel(self, vx, vy, vz):
        # if isinstance(vx, int) and isinstance(vy, int) and isinstance(vz, int):
        return str(vx) + " " + str(vy) + " " + str(vz)

    def hash_to_xyz(self, voxel_name):
        return [int(x) for x in voxel_name.split(" ")]
        



class RootTable: # 对于root，每个voxel有且仅有一个点。不需要额外功能， 只需要提供查找给定float点对应的voxel和邻接voxel即可。
    def __init__(self, min_bound, max_bound, voxel_size = 0.15):
        self.voxel_size = voxel_size
        self.min_bound = min_bound
        self.max_bound = max_bound
        
    def hashcode(self, x, y, z):
        return str(int(x // self.voxel_size)) + " " + str(int(y // self.voxel_size)) + " " + str(int(z // self.voxel_size))

    def build_hash_table(self, points_xyz, points_color, points_normal, points_visibility):
        self.hash_voxel_id = defaultdict(list)
        keep_idx = []

        valid_i = 0
        for i in range(points_xyz.shape[0]):
            key = self.hashcode(points_xyz[i,0], points_xyz[i,1], points_xyz[i,2])
            
            if key in self.hash_voxel_id:
                # 可能存在一些隐患。在处理duplicates时，可以忽略xyz/color，但是visibility不应忽略，需要累加
                continue

            self.hash_voxel_id[key] = valid_i
            valid_i += 1            
            keep_idx.append(i)
        keep_idx = np.array(keep_idx)
        
        self.points_xyz = points_xyz[keep_idx]

        self.points_color = points_color[keep_idx]
        self.points_normal = points_normal[keep_idx]
        
        self.points_visibility = points_visibility[keep_idx]
        return self.points_xyz, self.points_color, self.points_normal, self.points_visibility
        
    def get_voxel_id_from_point(self, x, y, z):
        key = self.hashcode(x,y,z)
        if key not in self.hash_voxel_id:
            return None
        v = self.hash_voxel_id[key]
        return v
    
    def visualize(self):
        voxel_meshes = []

        normal_lines = []
        line_colors = []    
        
        pcd_bkgd = o3d.geometry.PointCloud()
        pcd_bkgd.points = o3d.utility.Vector3dVector(np.array(self.points_xyz))
        pcd_bkgd.colors = o3d.utility.Vector3dVector(np.array(self.points_color))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bkgd, voxel_size=self.voxel_size)

        normal_lines = np.vstack([self.points_xyz, self.points_xyz + self.points_normal * 0.2])
        line_colors = np.zeros(self.points_xyz.shape)
        line_colors[:,1:] = 0

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(normal_lines)
        line_set.lines = o3d.utility.Vector2iVector(np.arange(normal_lines.shape[0] * 2).reshape(-1, 2))
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
            
        return voxel_meshes + [line_set, voxel_grid]




@njit(parallel=True)
def parallel_rasterize(corners_2d, depths, img_h, img_w):
    voxel_depth_source = -np.ones((img_h, img_w), dtype=np.int32)
    voxel_depth_value = np.full((img_h, img_w), np.inf)
    
    for i in prange(corners_2d.shape[0]):
        # 计算当前体素投影包围盒
        x_coords = corners_2d[i, :, 0]
        y_coords = corners_2d[i, :, 1]
        min_x = max(0, np.min(x_coords))
        max_x = min(img_w-1, np.max(x_coords))
        min_y = max(0, np.min(y_coords))
        max_y = min(img_h-1, np.max(y_coords))
        
        # 遍历包围盒内像素
        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):
                if point_in_voxel_projection(x, y, corners_2d[i]):  # 实现快速点包含判断
                    z = depths[i]
                    if z < voxel_depth_value[y, x]:
                        voxel_depth_value[y, x] = z
                        voxel_depth_source[y, x] = i
    return voxel_depth_value, voxel_depth_source

@njit
def point_in_voxel_projection(x, y, corners):
    cross_count = 0
    
    faces = np.array([[0, 1, 3, 2, 0], [4, 5, 7, 6, 5], [0, 1, 5, 4, 0], [2, 3, 7, 6, 2], [0, 2, 6, 4, 0], [1, 3, 7, 5, 1]]).astype(np.int8)
    for i in range(faces.shape[0]):
        face = faces[i]
        
        triangle_vertices = corners[face[:3]]
        if point_in_2d_triangle(x, y, triangle_vertices):
            return True

        triangle_vertices = corners[face[2:]]
        if point_in_2d_triangle(x, y, triangle_vertices):
            return True
            
    return False

@njit 
def point_in_2d_triangle(x, y, triangle_vertices):
    x0, y0 = triangle_vertices[0][0], triangle_vertices[0][1]
    x1, y1 = triangle_vertices[1][0], triangle_vertices[1][1]
    x2, y2 = triangle_vertices[2][0], triangle_vertices[2][1]
  
    P01 = (x0 - x) * (y1 - y) - (x1 - x) * (y0 - y)
    P12 = (x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)
    P20 = (x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)
    
    return (P01 > 0 and P12 > 0 and P20 > 0) or (P01 < 0 and P12 < 0 and P20 < 0)



def voxel_traversal(P0, P1, voxel_size):
    x0, y0, z0 = P0
    x1, y1, z1 = P1
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    
    # 步进方向
    stepX = 1 if dx > 0 else -1
    stepY = 1 if dy > 0 else -1
    stepZ = 1 if dz > 0 else -1
    
    # 初始体素索引
    i = int(x0 // voxel_size)
    j = int(y0 // voxel_size)
    k = int(z0 // voxel_size)
    end_i = int(x1 // voxel_size)
    end_j = int(y1 // voxel_size)
    end_k = int(z1 // voxel_size)
    
    # 边界距离初始化
    tMaxX = (voxel_size * (i + (stepX > 0)) - x0) / dx if dx != 0 else float('inf')
    tMaxY = (voxel_size * (j + (stepY > 0)) - y0) / dy if dy != 0 else float('inf')
    tMaxZ = (voxel_size * (k + (stepZ > 0)) - z0) / dz if dz != 0 else float('inf')
    
    tDeltaX = voxel_size / abs(dx) if dx != 0 else float('inf')
    tDeltaY = voxel_size / abs(dy) if dy != 0 else float('inf')
    tDeltaZ = voxel_size / abs(dz) if dz != 0 else float('inf')
    
    voxels = [(i, j, k)]
    # while (i, j, k) != (end_i, end_j, end_k):
    total_steps = (abs(end_i - i) + abs(end_j - j) + abs(end_k - k)) + 3
    for _ in range(total_steps):
        if tMaxX < tMaxY and tMaxX < tMaxZ:
            i += stepX
            tMaxX += tDeltaX
        elif tMaxY < tMaxZ:
            j += stepY
            tMaxY += tDeltaY
        else:
            k += stepZ
            tMaxZ += tDeltaZ
        voxels.append((i, j, k))
    return voxels
