from torch.utils import data
import open3d as o3d
import os 
import numpy as np
import torch
from utils.data_util import scene_cube
import cv2


class NoisyOtherMatterPortDataset(data.Dataset):
    def __init__(self,data_root,train_list,expand = False,voxel_size = 0.1,layer = 4,snum = 1000000):
        self.train_list = []
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.octree = scene_cube(layer_num = 7,voxel_size = voxel_size,device = "cpu",lowest=7)
        self.pcd_list = []
        self.mesh_list = []
        self.expand = expand
        self.image_size = (640,512)
        self.layer = layer
        self.snum = snum
        for scan in train_list:
            scan_dir = os.path.join(self.data_root,scan)
            meshs = os.listdir(scan_dir)
            mesh_num = len(meshs)
            c = mesh_num
            for i in range(c - 1,0,-1):
                if os.path.exists(os.path.join(scan_dir,f"mesh_{i}.ply")):
                    break
                else:
                    mesh_num = mesh_num - 1
            self.mesh_list += [os.path.join(scan_dir,f"mesh_{i}.ply") for i in range(mesh_num)]
            self.pcd_list += [os.path.join(scan_dir,f"pcd_{i}.ply") for i in range(mesh_num)]

        print(len(self.mesh_list))
    def __len__(self):
        return len(self.mesh_list)
    
    def preturb_points(self,points,normals,voxel_size,num = 1):
        sdf = torch.randn((points.shape[0],num,1)) * 0.01
        
        p = points.unsqueeze(1).repeat(1,num,1)
        n = normals.unsqueeze(1).repeat(1,num,1)
        p = p + n * sdf * voxel_size
        sdf = sdf.view(-1,1)
        p_points = p.view(-1,3)

        return p_points,sdf

    def __getitem__(self, idx):
        if idx < 0:
            assert -idx <= len(self)
            idx = len(self) + idx
        if not os.path.exists(self.pcd_list[idx]):
            zero = torch.tensor([])
            return zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
        pcd = o3d.io.read_point_cloud(self.pcd_list[idx])
        pcd = pcd.voxel_down_sample(0.01)
        # sdf_path = self.sdf_list[idx]
        gt_mesh = o3d.io.read_triangle_mesh(self.mesh_list[idx])
        gt_mesh.compute_vertex_normals()
        gt_pcd = gt_mesh.sample_points_uniformly(500000)
        gt_pcd = gt_pcd.voxel_down_sample(0.005)
        # dic = np.load(sdf_path)

        
        points= torch.from_numpy(np.asarray(pcd.points)).float()
        normals= torch.from_numpy(np.asarray(pcd.normals)).float()

        gt_points = torch.from_numpy(np.asarray(gt_pcd.points)).float()
        gt_normals = torch.from_numpy(np.asarray(gt_pcd.normals)).float()

        min_bound = torch.min(torch.cat([gt_points,points],dim = 0),dim = 0)[0] - 0.5 * self.voxel_size
        max_bound = torch.max(torch.cat([gt_points,points],dim = 0),dim = 0)[0] + 0.5 * self.voxel_size
        bound = max_bound - min_bound
        self.octree.bound_min = min_bound
        self.octree.set_bound(bound.numpy())
        structures = self.octree.get_structure(gt_points,L = self.layer - 1,expand=True)

        return points,normals,structures,gt_points,gt_normals,min_bound,bound

