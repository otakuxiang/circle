from torch.utils import data
import open3d as o3d
import os 
import numpy as np
import torch
from data_proc.trainScannnetPcd import scene_cube
class SynthesisDataset(data.Dataset):
    def __init__(self,data_root,stage):
        if stage == "train":
            self.list_file = os.path.join(data_root,"train.lst")
        else:
            self.list_file = os.path.join(data_root,"val.lst")
        self.lists = []
        self.sdf_lists = []
        f = open(self.list_file)
        for line in f:
            room_path = os.path.join(data_root,line.strip())
            pc_path = os.path.join(room_path,"pointcloud")
            pcds = ["pointcloud_%02d.ply" % j for j in range(10)]
            sdfs = ["sdf_%02d.npz" % j for j in range(10)]
            pcds = [os.path.join(pc_path,pcd) for pcd in pcds]
            sdfs =  [os.path.join(pc_path,sdf) for sdf in sdfs]
            self.lists += pcds
            self.sdf_lists += sdfs
        self.data_root = data_root
        self.matrix = np.array(
            [
                [1,0,0,0.525],
                [0,0,1,0.525],
                [0,1,0,0.55],
                [0,0,0,1]
            ]
        )
        self.octree = scene_cube(layer_num = 5,voxel_size = 0.01,device = "cpu",lowest=7)

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):
        if idx < 0:
            assert -idx <= len(self)
            idx = len(self) + idx
        pcd_name = self.lists[idx]
        pcd = o3d.io.read_point_cloud(pcd_name)
        sdf_path = self.sdf_lists[idx]
        points = torch.from_numpy(np.asarray(pcd.points)).float()
        normals = torch.from_numpy(np.asarray(pcd.normals)).float()
        points_hole,normals_hole = self.octree.random_delete(points,normals,50)
        structures = self.octree.get_structure(points,L = 3)
        dic = np.load(sdf_path)
        sdf = torch.from_numpy(dic['sdf']).float()
        xyz = torch.from_numpy(dic['xyz']).float()     
        voxel_resolution = dic['voxel_resolution']
        return points,normals,points_hole.float(),normals_hole.float(),structures,sdf,xyz,voxel_resolution
