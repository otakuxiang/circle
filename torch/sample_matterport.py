import os
import sys
from typing_extensions import final
from tqdm import tqdm 
import open3d as o3d
import numpy as np
from system.ext import valid_mask
import torch
import pdb,traceback,cv2
from system.ext import unproject_depth, \
    estimate_normals, filter_depth
from torch.utils import data
import subprocess
class MatterPort3dImageLoader(data.Dataset):
    def __init__(self,scene_path,random_ratio):
        
        self.depth_path = os.path.join(scene_path,"matterport_depth_images")
        self.pose_path = os.path.join(scene_path,"matterport_camera_poses")
        self.intri_path = os.path.join(scene_path,"matterport_camera_intrinsics")
        tripod_numbers = [ins[:ins.find("_")] for ins in os.listdir(self.intri_path)]
        self.frames = []
        for tripod_number in tripod_numbers:
            for camera_id in range(3):
                for frame_id in range(6):
                    self.frames.append([tripod_number,camera_id,frame_id])
        # print(len(self.frames))
        # print(len(self.frames),)
        self.select_ids = np.random.choice(a = len(self.frames),size = int(len(self.frames) * random_ratio),replace=False)
        self.depthMapFactor = 4000
        
    def __len__(self):
        return len(self.select_ids)

    def __getitem__(self, idx):
        frame_id = self.select_ids[idx]
        tripod_number,camera_id,frame_idx = self.frames[frame_id]
        f = open(os.path.join(self.pose_path,f"{tripod_number}_pose_{camera_id}_{frame_idx}.txt"))
        pose = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")
            for k in range(0,4):
                pose[idx,k] = float(ss[k])
        # pose = np.linalg.inv(pose)
        pose = torch.from_numpy(pose).float()
        
        f.close()
        K_depth = np.zeros((3,3))
        f = open(os.path.join(self.intri_path,f"{tripod_number}_intrinsics_{camera_id}.txt"))
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
       
        depth_path = os.path.join(self.depth_path,tripod_number+"_d"+str(camera_id)+"_"+str(frame_idx)+".png")
        depth =cv2.imread(depth_path,-1)
        ins = torch.from_numpy(K_depth).float()
        if depth is None:
            print("get None image!")
            print(depth_path)
            return None
        
        depth = depth.astype(np.float32) / self.depthMapFactor
        depth = torch.from_numpy(depth).float()

        
        return frame_id,depth,pose,ins

class MatterPort3dRegionImageLoader(data.Dataset):
    def __init__(self,scene_path,region_name,random_ratio):
        self.depth_path = os.path.join(scene_path,"matterport_depth_images")
        self.pose_path = os.path.join(scene_path,"matterport_camera_poses")
        self.intri_path = os.path.join(scene_path,"matterport_camera_intrinsics")
        self.region_mesh_path = os.path.join(scene_path,"region_segmentations",f"{region_name}.ply")
        
        tripod_numbers = [ins[:ins.find("_")] for ins in os.listdir(self.intri_path)]
        tripod_numbers = np.unique(tripod_numbers)
        tripods = self.get_region_cameras(tripod_numbers)
        # print(tripods)
        
        self.frames = []
        for tripod_number in tripods:
            for camera_id in range(3):
                for frame_id in range(6):
                    self.frames.append([tripod_number,camera_id,frame_id])
        self.select_ids = np.random.choice(a = len(self.frames),size = int(len(self.frames) * random_ratio) ,replace=False)
        self.depthMapFactor = 4000
    def __len__(self):
        return len(self.select_ids)
    
    def get_region_cameras(self,tripod_numbers):
        region_mesh = o3d.io.read_triangle_mesh(self.region_mesh_path)
        pcd = region_mesh.sample_points_uniformly(10000000)
        pcd = pcd.voxel_down_sample(0.01)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        min_bound = pcd.get_min_bound()
        tripods = []
        for tripod in tripod_numbers:
            center = np.zeros((3))
            for i in range(0,3):
                for j in range(0,6):
                    f = open(os.path.join(self.pose_path,tripod+"_pose_"+str(i)+"_"+str(j)+".txt"))
                    p = np.zeros((4,4))
                    for idx,line in enumerate(f):
                        ss = line.strip().split(" ")
                        for k in range(0,4):
                            p[idx,k] = float(ss[k])
                    center += p[0:3,3]
            center /= 18
            center[2] = min_bound[2] + 0.05
            nn = kdtree.search_radius_vector_3d(center,0.1)
            if nn[0] > 10:
                tripods.append(tripod)
        return tripods
    
    def __getitem__(self, idx):
        frame_id = self.select_ids[idx]
        tripod_number,camera_id,frame_idx = self.frames[frame_id]
        f = open(os.path.join(self.pose_path,f"{tripod_number}_pose_{camera_id}_{frame_idx}.txt"))
        pose = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")
            for k in range(0,4):
                pose[idx,k] = float(ss[k])
        # pose = np.linalg.inv(pose)
        pose = torch.from_numpy(pose).float()
        
        f.close()
        K_depth = np.zeros((3,3))
        f = open(os.path.join(self.intri_path,f"{tripod_number}_intrinsics_{camera_id}.txt"))
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
       
        depth_path = os.path.join(self.depth_path,tripod_number+"_d"+str(camera_id)+"_"+str(frame_idx)+".png")
        depth =cv2.imread(depth_path,-1)
        ins = torch.from_numpy(K_depth).float()
        if depth is None:
            print("get None image!")
            print(depth_path)
            return None
        
        depth = depth.astype(np.float32) / self.depthMapFactor
        depth = torch.from_numpy(depth).float()
        
        return frame_id,depth,pose,ins
    
                
import shutil

random_ratio = float(sys.argv[2])
rname = int(random_ratio * 100)
# the original matterport3D dataset location, should cotains v1 folder
src_root = f"/home/chx/data_disk/MatterPort3D/"
# the target sampled matterport3d location
tar_root = f"/home/chx/nas/disk_0/matterport_result/{sys.argv[1]}/"

file_path= os.path.join(src_root,"v1/scans")
f = open(os.path.join(src_root,f"scenes_{sys.argv[1]}.txt"))
dirList = [line.strip() for line in f]
save_root = tar_root
f.close()


with torch.no_grad():
    for scene_dir in tqdm(dirList):
        save_path = os.path.join(save_root,scene_dir)
        os.makedirs(save_path,exist_ok = True)
        sequence_path = os.path.join(file_path, scene_dir)
        regions_all = os.listdir(os.path.join(sequence_path,"region_segmentations"))
        regions = []
        for region in regions_all:
            if region.find("ply") != -1:
                regions.append(region[:-4])
        for region in regions:
            result_folder = os.path.join(save_path,region)
            # if os.path.exists(os.path.join(result_folder,f"frame_list_{rname}.txt")) \
            #     and os.path.exists(os.path.join(result_folder,f"depth_{rname}")):
            #     num = len(os.listdir(os.path.join(result_folder,f"depth_{rname}")))
            #     f = open(os.path.join(result_folder,f"frame_list_{rname}.txt"))
            #     num_f = len(f.readlines())
            #     # import pdb; pdb.set_trace()
            #     if num_f == num and num != 0:
            #         print(f"Continue because being fucked.")
            #         continue
            os.makedirs(result_folder,exist_ok = True)
            dataset = MatterPort3dRegionImageLoader(sequence_path,region,random_ratio)
            
            # dataset = MatterPort3dImageLoader(sequence_path,random_ratio)
            region_mesh = o3d.io.read_triangle_mesh(os.path.join(sequence_path,"region_segmentations",f"{region}.ply"))
            region_pcd = region_mesh.sample_points_uniformly(10000000)
            region_pcd = region_pcd.voxel_down_sample(0.005)
            region_pcd = torch.from_numpy(np.asarray(region_pcd.points)).float().cuda()
            region_pcd = torch.cat([region_pcd,torch.zeros(region_pcd.size(0),1).cuda()],dim = -1)
            depth_dir = os.path.join(result_folder,f"depth_{rname}")
            if os.path.exists(depth_dir):
                shutil.rmtree(depth_dir)
            os.makedirs(depth_dir,exist_ok = True)


            f = open(os.path.join(result_folder,f"frame_list_{rname}.txt"),"w")
            dataloader = torch.utils.data.DataLoader(dataset,num_workers = 10,batch_size = 1,shuffle = False)
            num = 0
            for [idx,depth,pose,K_depth] in tqdm(dataloader):  
                frame = dataset.frames[idx[0]]
                # rgb = rgb.squeeze(0)
                pose = pose.cuda().squeeze(0)
                K_depth = K_depth.squeeze(0)

                depth = depth.squeeze(0).cuda()
                if torch.any(torch.isinf(pose)) or torch.any(torch.isnan(pose)):
                    continue
                depth_data = depth.clone()
                h,w = depth_data.shape
                depth_data[torch.logical_or(depth_data < 0.1,depth_data > 4)] = np.nan
                pc_data = unproject_depth(depth_data, K_depth[0,0], K_depth[1,1],
                                                    K_depth[0,2], K_depth[1,2])
                pc_data = pc_data.view(-1,3,1)                 
                pc_data = (torch.matmul(pose[0:3,0:3],pc_data) + pose[0:3,3:4]).squeeze(-1)
                nan_mask = ~torch.isnan(pc_data[..., 0])
                pc_data = pc_data[nan_mask,:]
                if pc_data.size(0) == 0:
                    continue
                pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), 1), device=pc_data.device)], dim=-1)
                
                v_mask = valid_mask(pc_data,region_pcd,0.05)
                # print(pc_data.shape,torch.nonzero(v_mask).shape)
                if torch.nonzero(v_mask).size(0) / (h*w) > 0.3:
                    f.write(f"{frame[0]} {frame[1]} {frame[2]}\n")
                    depth_new = torch.zeros_like(depth).view(-1)
                    depth = depth.view(-1)
                    inds = torch.arange(0,depth.size(0),device = "cuda:0",dtype = torch.long)
                    inds = inds[nan_mask][v_mask]
                    depth_new[inds] = depth[inds]
                    depth_new = depth_new.view(h,w) * 4000.0 
                    cv2.imwrite(os.path.join(depth_dir,f"{num}.png"),depth_new.cpu().numpy().astype(np.uint16))
                    num += 1
            f.close()

