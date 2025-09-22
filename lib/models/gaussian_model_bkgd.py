import torch
import numpy as np
import torch.nn as nn
import os
from lib.config import cfg
from lib.utils.graphics_utils import BasicPointCloud
from lib.datasets.base_readers import fetchPly
from lib.models.gaussian_model import GaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer

from lib.models.trellis import GrapeTrellis

from lib.utils.sh_utils import RGB2SH, IDFT, SH2RGB
from simple_knn._C import distCUDA2
from lib.utils.general_utils import inverse_sigmoid, get_expon_lr_func, quaternion_to_matrix, quaternion_raw_multiply
import open3d as o3d
import matplotlib.pyplot as plt
from lib.utils.waymo_utils import my_vis
# from lib.utils.general_utils import parallel_rasterize, point_in_voxel_projection, point_in_2d_triangle



class GaussianModelBkgd(GaussianModel):
    def __init__(
        self, 
        model_name='background', 
        scene_center=np.array([0, 0, 0]),
        scene_radius=20,
        sphere_center=np.array([0, 0, 0]),
        sphere_radius=20,
    ):
        self.scene_center = torch.from_numpy(scene_center).float().cuda()
        self.scene_radius = torch.tensor([scene_radius]).float().cuda()
        self.sphere_center = torch.from_numpy(sphere_center).float().cuda()
        self.sphere_radius = torch.tensor([sphere_radius]).float().cuda()
        num_classes = cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0
        self.background_mask = None

        super().__init__(model_name=model_name, num_classes=num_classes)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, train_views: np.array): 
        print('Create background model')
        # pointcloud_path_sky =  os.path.join(cfg.model_path, 'input_ply', 'points3D_sky.ply')
        # include_sky = cfg.model.nsg.get('include_sky', False)
        # if os.path.exists(pointcloud_path_sky) and not include_sky:
        #     pcd_sky = fetchPly(pointcloud_path_sky)
        #     pointcloud_xyz = np.concatenate((pcd.points, pcd_sky.points), axis=0)
        #     pointcloud_rgb = np.concatenate((pcd.colors, pcd_sky.colors), axis=0)
        #     pointcloud_normal = np.zeros_like(pointcloud_xyz)
        #     pcd = BasicPointCloud(pointcloud_xyz, pointcloud_rgb, pointcloud_normal)

        # return super().create_from_pcd(pcd, spatial_lr_scale, N_views)

        # self.spatial_lr_scale = spatial_lr_scale

        bkgd_path  = os.path.join(cfg.model_path, 'input_ply/points3D_bkgd.ply')   
        assert os.path.exists(bkgd_path) 

        bkgd_pcd = fetchPly(bkgd_path)

        points_xyz = np.asarray(bkgd_pcd.points)
        points_rgb = np.asarray(bkgd_pcd.colors)
        points_normal = np.asarray(bkgd_pcd.normals)[:,[2,0,1]] # 一阶球谐省略求解，直接计算方向
        points_normal = points_normal / np.linalg.norm(points_normal, axis=1, keepdims=True)
        points_visibility = np.load(os.path.join(cfg.model_path, "input_ply/points3D_bkgd.npy"))
 
        preserve_mask = np.zeros_like(points_visibility, dtype=bool)
        preserve_mask[:, train_views] = True
        filtered_visibility = np.logical_and(points_visibility, preserve_mask)

        # self.voxel_size = 0.15 # Waymo
        self.voxel_size = 0.15 # Nuscenes

        self.grape_trellis = GrapeTrellis(points_xyz, points_rgb, points_normal, filtered_visibility, voxel_size=self.voxel_size)
        # self.last_update_root = self.grape_trellis.root_table.points_xyz.shape[0]
        # self.last_update_vine = self.grape_trellis.vine_table.valid_cnt

        return super().create_from_pcd(pcd, spatial_lr_scale, train_views)


        # #   gaussians: [xyz, features, scaling, rotation, opacity, max_radii2D(?), anchor_id] 
        # fused_point_cloud = torch.tensor(self.grape_trellis.get_voxel_center_xyz()).float().cuda()
        # print(f"Number of points at initialisation for {self.model_name}: ", fused_point_cloud.shape[0])
        # # self._xyz_root = nn.Parameter(fused_point_cloud.requires_grad_(False))
        # self._xyz_anchor = fused_point_cloud.requires_grad_(False)
        # self._xyz_offset = nn.Parameter(torch.zeros(fused_point_cloud.shape, device="cuda").requires_grad_(True))
        

        # ###### zyk: rot anchor 
        # rots = torch.from_numpy(self.grape_trellis.rot_for_training()).float().cuda()
        # # 是否计算正确？
        # # self._rotation_root = nn.Parameter(rots.requires_grad_(False))
        # self._rotation_anchor = rots.requires_grad_(False)
        # rot_offset = torch.zeros(rots.shape)
        # rot_offset[:, 0] = 1
        # self._rotation_offset = nn.Parameter(rot_offset.cuda().requires_grad_(True))


        # fused_color = RGB2SH(torch.tensor(self.grape_trellis.get_voxel_color()).float().cuda())
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() 
        # # N * rgb * sh_coeffs
        # features[..., 0] = fused_color
        # self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        

        # # zyk: 需要么？是否直接voxel size即可 --> 糊成一团，不确定是否直接导致大量floater
        # # dist = torch.clamp_min(torch.from_numpy(self.voxel_size * np.ones(fused_point_cloud.shape)).float().cuda(), 0.0000001)
        # # scales = torch.log(dist)
        # # scales[:,-1] = np.log(0.001) # zyk: set z shortest
        # dist2 = torch.clamp_min(distCUDA2(self._xyz_anchor), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # self._scaling = nn.Parameter(scales.requires_grad_(True))


        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # # self._semantic = nn.Parameter(semamtics.requires_grad_(True))
        

        # self.max_radii2D = torch.zeros((fused_point_cloud.shape[0]), device="cuda")
        # self._anchor_id = torch.arange(self._xyz_anchor.shape[0], device="cuda") # already int?




    def set_background_mask(self, camera: Camera):
        pass
    
    @property
    def get_scaling(self):
        scaling = super().get_scaling
        # scaling = self.scaling_activation(self._scaling)
        return scaling if self.background_mask is None else scaling[self.background_mask]

    @property
    def get_rotation(self):
        rotation = super().get_rotation
        # rotation = quaternion_raw_multiply(self._rotation_anchor[self._anchor_id], self._rotation_offset)
        return rotation if self.background_mask is None else rotation[self.background_mask]

    @property
    def get_xyz(self):
        xyz = super().get_xyz
        # xyz = self._xyz_anchor[self._anchor_id] + self._xyz_offset
        return xyz if self.background_mask is None else xyz[self.background_mask]        
    
    @property
    def get_features(self):
        features = super().get_features
        # features = torch.cat([self._features_dc, self._features_rest], dim=1)
        return features if self.background_mask is None else features[self.background_mask]        
    
    @property
    def get_opacity(self):
        opacity = super().get_opacity
        # opacity = self.opacity_activation(self._opacity)
        return opacity if self.background_mask is None else opacity[self.background_mask]
    
    @property
    def get_semantic(self):
        semantic = super().get_semantic
        return semantic if self.background_mask is None else semantic[self.background_mask]


    def get_anchor_id(self):
        return self._anchor_id if self.background_mask is None else self._anchor_id [self.background_mask]


    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        max_grad = cfg.optim.get('densify_grad_threshold_bkgd', max_grad)
        if cfg.optim.get('densify_grad_abs_bkgd', False):
            grads = self.xyz_gradient_accum[:, 1:2] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        grads[grads.isnan()] = 0.0
        self.scalar_dict.clear()
        self.tensor_dict.clear()
        self.scalar_dict['points_total'] = self.get_xyz.shape[0]

        # Clone and Split
        extent = self.scene_radius
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = torch.logical_or(prune_mask, torch.all(self.get_scaling < 0.001, axis=1).squeeze())
        self.scalar_dict['points_below_min_opacity'] = prune_mask.sum().item()

        # Prune big points in world space 
        if prune_big_points:
            dists = torch.linalg.norm(self.get_xyz - self.sphere_center, dim=1)            
            big_points_ws = torch.max(self.get_scaling, dim=1).values > extent * self.percent_big_ws
            big_points_ws[dists > 2 * self.sphere_radius] = False
            
            over_small_points_ws = (self.max_radii2D > 0) & (self.max_radii2D <= 1)

            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            prune_mask = torch.logical_or(prune_mask, over_small_points_ws) # zyk: overfitting
            
            self.scalar_dict['points_big_ws'] = big_points_ws.sum().item()




        self.scalar_dict['points_pruned'] = prune_mask.sum().item()
        self.prune_points(prune_mask)
        
        # Reset 
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

        return self.scalar_dict, self.tensor_dict