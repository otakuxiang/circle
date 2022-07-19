
import cv2
import os
import yaml
import numpy as np
import open3d as o3d
import time
import torch
import sys
import random
import time,pdb
from torch.utils import data


class MatterPort3dImageLoader(data.Dataset):
    def __init__(self,data_root,scene,region,resolution=(512, 640)):
        
        sequence_path = os.path.join(data_root,"scans",scene)
        region_path = os.path.join(data_root,"data_pro",scene,region)
        self.rgb_path = os.path.join(sequence_path,"undistorted_color_images")
        f_file = open(os.path.join(region_path,"frame_list.txt"))
        self.frame_list = []
        for line in f_file:
            self.frame_list.append(line.strip().split(" "))
        self.img_size = (resolution[1],resolution[0])
        self.depth_path = os.path.join(region_path,"depth")
        self.pose_path = os.path.join(sequence_path,"matterport_camera_poses")
        self.intri_path = os.path.join(sequence_path,"matterport_camera_intrinsics")
        self.depthMapFactor = 4000
        self.res = resolution
    
    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        frame_id = idx
        tripod_number,camera_id,frame_idx = self.frame_list[idx]
        f = open(os.path.join(self.pose_path,tripod_number+"_pose_"+camera_id+"_"+frame_idx+".txt"))
        pose = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")
            for k in range(0,4):
                pose[idx,k] = float(ss[k])
        # pose = np.linalg.inv(pose)
        pose = torch.from_numpy(pose).float()
        
        f.close()
        K_depth = np.zeros((3,3))
        f = open(os.path.join(self.intri_path,tripod_number+"_intrinsics_"+camera_id+".txt"))
        p = np.zeros((4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")   
            for j in range(4):
                p[j] = float(ss[j+2])
        f.close()
        K_depth[0,0] = p[0]
        K_depth[1,1] = p[1]
        K_depth[2,2] = 1
        K_depth[0,2] = p[2]
        K_depth[1,2] = p[3]
       
        depth_path = os.path.join(self.depth_path,f"{frame_id}.png")
        depth =cv2.imread(depth_path,-1)
        K_depth[0,:] *= float(self.res[1]) / depth.shape[1]
        K_depth[1,:] *= float(self.res[0]) / depth.shape[0]  
        
        ins = torch.from_numpy(K_depth).float()
        if depth is None:
            print("get None image!")
            print(depth_path)
            return None
        
        depth = cv2.resize(depth,self.img_size,interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / self.depthMapFactor
        depth = torch.from_numpy(depth).float()
        depth[torch.logical_or(depth < 0.1,depth > 4.0)] = np.nan
        
        return depth,ins,pose