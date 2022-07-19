import cv2
import os
import yaml
import numpy as np
import open3d as o3d
from system.ext import unproject_depth, remove_radius_outlier, \
    estimate_normals, filter_depth, compute_sdf
import time
import torch
import sys
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
import random
import utils.net_util as net_util

class ScannetImageLoader: 
    def __init__(self,sequence_path,config_path,frame_list = [],w = 640,h = 480):
        self.data_root = sequence_path
        rgb_path = os.path.join(sequence_path,"color")
        depth_path = os.path.join(sequence_path,"depth")
        self.rgb_files = os.listdir(rgb_path)
        post_name = self.rgb_files[0][-4:]
        self.depth_files = os.listdir(depth_path)
        self.rgb_files = [os.path.join(rgb_path,str(i)+post_name) for i in range(len(self.rgb_files))]
        self.depth_files = [os.path.join(depth_path,str(i)+".png") for i in range(len(self.depth_files))]

        f = open(os.path.join(config_path,"intrinsic_depth.txt"))
        p = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")   
            for j in range(4):
                p[idx,j] = float(ss[j])
        f.close()
        self.K_depth = np.zeros((3,3))
        self.K_rgb = np.zeros((3,3))
        self.K_depth[0,0] = p[0,0]
        self.K_depth[1,1] = p[1,1]
        self.K_depth[2,2] = 1
        self.K_depth[0,2] = p[0,2]
        self.K_depth[1,2] = p[1,2]
        f = open(os.path.join(config_path,"intrinsic_color.txt"))
        p = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")   
            for j in range(4):
                p[idx,j] = float(ss[j])
        f.close()
        
        self.K_rgb[0,0] = p[0,0]
        self.K_rgb[1,1] = p[1,1]
        self.K_rgb[2,2] = 1
        self.K_rgb[0,2] = p[0,2]
        self.K_rgb[1,2] = p[1,2]
        self.depthMapFactor = 1000
        rgb = cv2.imread(os.path.join(self.data_root,self.rgb_files[0]),-1)
        depth = cv2.imread(os.path.join(self.data_root,self.depth_files[0]),-1)

        self.K_depth[0,:] *= w / depth.shape[1]
        self.K_depth[1,:] *= h / depth.shape[0]  

        self.K_rgb[0,:] *= w / rgb.shape[1]
        self.K_rgb[1,:] *= h / rgb.shape[0]  
        
        self.img_size = (w,h)
        self.frame_id = 0
        self.max_frame = len(self.depth_files)

    def loadTraj(self,traj_path):
        pose_file_list = os.listdir(traj_path)
        self.poses = []
        for j in range(len(pose_file_list)):
            pose_file = str(j)+".txt"
            pose_file = os.path.join(traj_path,pose_file)
            # print(pose_file)
            f = open(pose_file)
            p = np.zeros((4,4))
            for idx,line in enumerate(f):
                ss = line.strip().split(" ")
                for i in range(0,4):
                    p[idx,i] = float(ss[i])
            self.poses.append(p)
    
    def getDepthIns(self):
        return self.K_depth
    def getColorIns(self):
        return self.K_rgb

    def getGtPose(self,frame):
        if frame >= len(self.poses):
            print("frame id > pose id!")
            return None
        
        return self.poses[frame]
    def hasNext(self):
        return self.frame_id < self.max_frame
    
    def getImage(self):
        rgb_path = os.path.join(self.data_root,self.rgb_files[self.frame_id])
        depth_path = os.path.join(self.data_root,self.depth_files[self.frame_id])

        
        rgb = cv2.imread(rgb_path,-1)
        depth = cv2.imread(depth_path,-1)
        if rgb is None or depth is None:
            print("get None image!")
            return None,None
        rgb = cv2.resize(rgb,self.img_size)
        depth = cv2.resize(depth,self.img_size,interpolation=cv2.INTER_NEAREST)
        self.frame_id += 1
        depth = depth.astype(np.float32) / self.depthMapFactor
        
        return rgb,depth

    def getImage(self,frame_id):
        rgb_path = os.path.join(self.data_root,self.rgb_files[frame_id])
        depth_path = os.path.join(self.data_root,self.depth_files[frame_id])
        rgb = cv2.imread(rgb_path,-1)
        depth =cv2.imread(depth_path,-1)
        if rgb is None or depth is None:
            print("get None image!")
            return None,None
        rgb = cv2.resize(rgb,self.img_size)
        depth = cv2.resize(depth,self.img_size,interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / self.depthMapFactor
        
        return rgb,depth
    def size(self):
        return len(self.rgb_files)

