from numpy.core.fromnumeric import size
from network import cnp_encoder
import cv2
import os
import yaml
import numpy as np
import open3d as o3d
from system.ext import unproject_depth, \
    estimate_normals, filter_depth
import time
import torch
import sys
import random
import time
from data.data_utils import voxel_down_sample,voxel_down_sample_with_rgb
from torch.utils import data

class MatterPort3dImageLoader(data.Dataset):
    def __init__(self,scene_path):
        
        self.depth_path = os.path.join(scene_path,"undistorted_depth_images")
        self.pose_path = os.path.join(scene_path,"matterport_camera_poses")
        self.intri_path = os.path.join(scene_path,"matterport_camera_intrinsics")
        tripod_numbers = [ins[:ins.find("_")] for ins in os.listdir(self.intri_path)]
        self.frames = []
        for tripod_number in tripod_numbers:
            for camera_id in range(3):
                for frame_id in range(6):
                    self.frames.append([tripod_number,camera_id,frame_id])
        # print(len(self.frames))
        self.select_ids = np.random.choice(a = len(self.frames),size = len(self.frames) // 2,replace=False)
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

class MatterPortImageLoader: 
    def __init__(self,sequence_path,tripod_number,max_frame = 6,w = 1280,h = 1024):
        
        self.depth_path = os.path.join(sequence_path,"undistorted_depth_images")
        pose_path = os.path.join(sequence_path,"matterport_camera_poses")
        intri_path = os.path.join(sequence_path,"matterport_camera_intrinsics")
        self.rgb_path = os.path.join(sequence_path,"undistorted_color_images")
        
        self.tripod_number=tripod_number
        
        self.intri=[]
        for i in range(0,3):
            f = open(os.path.join(intri_path,tripod_number+"_intrinsics_"+str(i)+".txt"))
            p = np.zeros((4))
            for idx,line in enumerate(f):
                ss = line.strip().split(" ")   
                for j in range(4):
                    p[j] = float(ss[j+2])
            f.close()
            K_depth = np.zeros((3,3))
            K_depth[0,0] = p[0]
            K_depth[1,1] = p[1]
            K_depth[2,2] = 1
            K_depth[0,2] = p[2]
            K_depth[1,2] = p[3]
            self.intri.append(K_depth)
        self.depthMapFactor = 4000
        self.img_size = (w,h)
        self.poses = []
        self.center = np.zeros((3))
        for i in range(0,3):
            for j in range(0,6):
                f = open(os.path.join(pose_path,tripod_number+"_pose_"+str(i)+"_"+str(j)+".txt"))
                p = np.zeros((4,4))
                for idx,line in enumerate(f):
                    ss = line.strip().split(" ")
                    for k in range(0,4):
                        p[idx,k] = float(ss[k])
                self.poses.append(p)
                # print(p[0:3,3].shape)
                self.center += p[0:3,3]
                # print(self.center)
        self.center /= 18
        self.max_frame=max_frame

    def getDepthIns(self,camera_id):
        return self.intri[camera_id]

    def getGtPose(self,frame,camera_id):
        pose_id=camera_id*self.max_frame+frame
        if pose_id >= len(self.poses):
            print("frame id > pose id!")
            return None
        return self.poses[pose_id]

    def getImage(self,frame_id,camera_id):
        depth_path = os.path.join(self.depth_path,self.tripod_number+"_d"+str(camera_id)+"_"+str(frame_id)+".png")
        rgb_path = os.path.join(self.rgb_path,self.tripod_number+"_i"+str(camera_id)+"_"+str(frame_id)+".jpg")
        depth =cv2.imread(depth_path,-1)
        if depth is None:
            print("get None image!")
            print(depth_path)
            return None
        depth = cv2.resize(depth,self.img_size,interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / self.depthMapFactor
        rgb = cv2.imread(rgb_path,-1)
        if rgb is None:
            print("get None image!")
            print(rgb_path)
            return None
        rgb = cv2.resize(rgb,self.img_size)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        # print(rgb.shape)
        return depth,rgb

def add_depth_noise(depthmaps, noise_sigma=0.005):

    # add noise
    if noise_sigma > 0:
        random.seed(time.perf_counter())
        np.random.seed(int(time.perf_counter()))
        sigma = noise_sigma
        noise = np.random.normal(0, 1, size=depthmaps.shape).astype(np.float32)
        depthmaps = depthmaps + noise * sigma * depthmaps

    return depthmaps


def add_kinect_noise(depth, sigma_fraction=0.05):
    r = np.random.uniform(0., 1., depth.shape)
    sign = np.ones(depth.shape)
    sign[r < 0.5] = -1.0
    sigma = sigma_fraction*depth
    magnitude = sigma*(1.0 - np.exp(-0.5*np.power(r, 2)))
    depth += sign*magnitude
    depth[depth < 0] = 0.
    return depth


    

class MatterPortPcdGenerator():
    def __init__(self,root,scene_number,tripodList):
        self.root = root
        self.scene_number=scene_number
        self.tripodList=tripodList
        self.min_depth = 0.1
        self.max_depth = 5.5
        self.save_root = "/home/chx/data_disk/MatterPort3d/v1"
        
    def generateOnePcd(self,whole = True):
        
        sequence_path = os.path.join(self.root,self.scene_number)
        final_pc_data = []
        final_normal_data = []
        self.depth_imgs = {}
        imgs = []
        
        for tripod_number in self.tripodList:
            dataLoader = MatterPortImageLoader(sequence_path,tripod_number,w = 1280,h = 1024)   
            c = dataLoader.center



        for tripod_number in self.tripodList:
            part_pc_data = []
            part_normal_data = []
            part_rgb_data = []
            try:
                dataLoader = MatterPortImageLoader(sequence_path,tripod_number,w = 1280,h = 1024)   
            except Exception as e:
                print("loading")
                print(e)
                sys.exit(-1)
                continue
            id_list = range(0,18)
            if whole:
                id_random_list = id_list
            else:
                id_random_list = random.sample(id_list, 14)
            for random_id in id_random_list:
                camera_id = random_id // 6
                frame_id = random_id % 6
                try:
                    depth,rgb = dataLoader.getImage(frame_id,camera_id)
                    if depth is None:
                        continue
                    depth = add_kinect_noise(depth,0.1)
                    depth = torch.from_numpy(depth).cuda()
                    rgb = torch.from_numpy(rgb).cuda().view(-1,3)
                    depth_data = depth.clone()
                    # filter_depth(depth,depth_data)
                    depth_data[torch.logical_or(depth_data < self.min_depth,depth_data > self.max_depth)] = np.nan
                    K_depth = dataLoader.getDepthIns(camera_id)
                    pc_data = unproject_depth(depth_data, K_depth[0,0], K_depth[1,1],
                                            K_depth[0,2], K_depth[1,2])
                except Exception as e:
                    print("unproject")
                    print(e)
                    continue
                pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), pc_data.size(1), 1), device=pc_data.device)], dim=-1)
                pc_data = pc_data.reshape(-1, 4)
                nan_mask = ~torch.isnan(pc_data[..., 0]) 
                pc_data = pc_data[nan_mask]
                rgb = rgb[nan_mask,:]
                if pc_data.size(0) <= 100:
                    continue
                try:
                    normal_data = estimate_normals(pc_data, 16, 0.1, [0.0, 0.0, 0.0])
                except Exception as e:
                    print("normals")
                    print(e)
                    continue
                pc_data = pc_data[:, :3]
                nan_mask = ~torch.isnan(normal_data[..., 0])
                
                pc_data = pc_data[nan_mask]
                normal_data = normal_data[nan_mask]
                rgb = rgb[nan_mask,:]

                voxel_down_sample_with_rgb(pc_data,normal_data,rgb,0.05)

                pc_data = pc_data.view(-1,3,1)
                normal_data = normal_data.view(-1,3,1)


                if pc_data.size(0) <= 50:
                    continue
                # try:
                #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_data.cpu().numpy()))
                #     pcd.normals = o3d.utility.Vector3dVector(normal_data.cpu().numpy())
                #     pcd = pcd.voxel_down_sample(0.03)
                # except Exception as e:
                #     print("pcd")
                #     print(e)
                #     continue

                cur_pose = dataLoader.getGtPose(frame_id,camera_id) 
                if np.any(np.isinf(cur_pose)) or np.any(np.isnan(cur_pose)):
                    continue
                try:
                    # pcd = pcd.transform(cur_pose)
                    # pc_data = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
                    # normal_data = torch.from_numpy(np.asarray(pcd.normals)).float().cuda()
                    # print(pc_data.shape)
                    cur_pose = torch.from_numpy(cur_pose).cuda().float()
                    pc_data = (torch.matmul(cur_pose[0:3,0:3],pc_data) + cur_pose[0:3,3:4]).squeeze(-1)
                    normal_data = torch.matmul(cur_pose[0:3,0:3],normal_data).squeeze(-1)

                except Exception as e:
                    print("transform")
                    print(e)
                    continue
                if pc_data.size(0) == 0:
                    continue
                self.depth_imgs[f"{camera_id}_{frame_id}"] = voxel_down_sample(pc_data.clone(),normal_data.clone(),0.05)[0]
                part_pc_data.append(pc_data)
                part_normal_data.append(normal_data)
                part_rgb_data.append(rgb)
            if len(part_pc_data) == 0:
                        continue
            center = dataLoader.center
            part_pc_data = torch.cat(part_pc_data,dim=0)
            part_normal_data = torch.cat(part_normal_data,dim=0)
            part_rgb_data = torch.cat(part_rgb_data,dim=0)

            voxel_down_sample_with_rgb(part_pc_data,part_normal_data,part_rgb_data,0.075)
            # part_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part_pc_data.cpu().numpy()))
            # part_pcd.normals = o3d.utility.Vector3dVector(part_normal_data.cpu().numpy())
            
            # o3d.io.write_point_cloud(os.path.join(save_part_path,tripod_number+"_partcloud.ply"),part_pcd)


        return part_pc_data,part_normal_data,part_rgb_data,center
    def get_depths(self):
        return self.depth_imgs

import threading
import multiprocessing
class myThread(multiprocessing.Process):
    def __init__(self,xyzs,mesh):
        multiprocessing.Process.__init__(self)
        self.xyzs = xyzs
        self.mesh = mesh
        self.mesh_small = []
    def run(self):
        # print(self.xyzs.shape)
        for i in range(self.xyzs.shape[1]):
            min_bound = self.xyzs[:,i:i+1]
            max_bound = min_bound + 0.1
            bound = o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
            self.mesh_small.append(self.mesh.crop(bound))
        self.result_mesh = self.mesh_small[0]
        for i in range(1,len(self.mesh_small)):
            self.result_mesh += self.mesh_small[i]

def crop_mesh(xyzs,mesh,return_dict):
    mesh_small = []
    for i in range(xyzs.shape[1]):
        min_bound = xyzs[:,i:i+1]
        max_bound = min_bound + 0.1
        bound = o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
        mesh_small.append(mesh.crop(bound))
    result_mesh = mesh_small[0]
    for i in range(1,len(mesh_small)):
        result_mesh += mesh_small[i]
    print("compute ok!")
    return_dict[os.getpid()] = os.getpid()
    o3d.io.write_triangle_mesh(f"./temp/{os.getpid()}.ply",result_mesh)
    

def get_small_mesh(mesh,center):
    voxel_size = 0.1
    mesh = mesh.compute_vertex_normals()
    mesh_pcd = mesh.sample_points_uniformly(3000000)
    pcd = torch.from_numpy(np.asarray(mesh_pcd.points)).float().cuda()
    normal = torch.from_numpy(np.asarray(mesh_pcd.normals)).float().cuda()
    min_bound = torch.min(pcd,dim = 0,keepdim = True)[0]
    center_t = torch.from_numpy(center).cuda()
    # processPcd = pcd - min_bound + 0.5 * voxel_size
    # center_t = center_t - min_bound + 0.5 * voxel_size
    # processPcd = (processPcd // voxel_size)
    # nx,ny,nz = (torch.max(processPcd,dim = 0)[0] + 1).cpu().numpy().tolist()
    # pc_index = processPcd[:,2] + processPcd[:,1] * nz + \
    #         processPcd[:,0] * ny * nz
    # unq = torch.unique(pc_index)
    # unique_xyz = torch.stack([unq // (ny * nz),(unq // nz) % ny, unq % nz],dim = -1) * voxel_size
    dist = (pcd - center_t).norm(dim = 1)
    mask = dist < 5.5
    
    # xyz =  (unique_xyz[mask] + min_bound - 0.5 * voxel_size).permute(1,0).cpu().numpy() 
    # xyzs = np.array_split(xyz,10,axis=1)
    # # pool = multiprocessing.Pool(10)
    # # ts = pool.map(crop_mesh,[(xyzs[i],mesh) for i in range(10)])
    # # q = multiprocessing.Queue()
    # if not os.path.exists("./temp"):
    #     os.makedirs("./temp")
    # manager = multiprocessing.Manager()
    # q = manager.dict()
    # ts = []
    # for i in range(10):
    #     t = multiprocessing.Process(target=crop_mesh,args=(xyzs[i],mesh,q))

    #     ts.append(t)
    #     ts[i].start()
    # ret = []
    # # for i in range(10):
    # #     ret.append(q.get())
    # for i in range(10):
    #     ts[i].join()
    # ret = q.values()
    # result_mesh = o3d.io.read_triangle_mesh(f"./temp/{ret[0]}.ply")
    # for i in range(1,10):
    #     result_mesh += o3d.io.read_triangle_mesh(f"./temp/{ret[i]}.ply")
    pcd = pcd[mask,:]
    normal = normal[mask,:]
    result_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.cpu().numpy()))
    result_pcd.normals = o3d.utility.Vector3dVector(normal.cpu().numpy())
    return result_pcd
    

