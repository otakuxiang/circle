import open3d as o3d
import os
import numpy as np
from tqdm import tqdm
import cv2,random
import sys,math
import torch
from utils.data_util import scene_cube

def get_random_Rotation():
    cita1 = random.randint(0,3) * np.pi / 2
    cita2 = random.randint(0,3) * np.pi / 2
    cita3 = random.randint(0,3) * np.pi / 2
    Rx = np.eye(3)
    Ry = np.eye(3)
    Rz = np.eye(3)
    Rx[1,1] = math.cos(cita1)
    Rx[1,2] = -math.sin(cita1)
    Rx[2,1] = math.sin(cita1)
    Rx[2,2] = math.cos(cita1)
    Ry[0,0] = math.cos(cita2)
    Ry[2,0] = -math.sin(cita2)
    Ry[0,2] = math.sin(cita2)
    Ry[2,2] = math.cos(cita2)
    Rz[0,0] = math.cos(cita3)
    Rz[0,1] = -math.sin(cita3)
    Rz[1,0] = math.sin(cita3)
    Rz[1,1] = math.cos(cita3)
    R = np.round(Rx.dot(Ry).dot(Rz))
    return R

def crop_small(scene,region):
    global count
    result_folder = os.path.join(result_root,scene)
    os.makedirs(result_folder,exist_ok=True)
    
    main_device = "cuda:0"
    oct = scene_cube(4,voxel_size = 3.2)
    layer_oct = scene_cube(5,voxel_size = 0.8)
    input_path = os.path.join(whole_pcd_root,scene,region)
    input_pcd = os.path.join(input_path,f"noisy_points_{rname}_{nname}.ply")
    input_pcd = o3d.io.read_point_cloud(input_pcd)
    sequence_path = os.path.join(src_root,"v1","scans",scene)
    region_mesh = o3d.io.read_triangle_mesh(os.path.join(sequence_path,"region_segmentations",f"{region}.ply"))
    gt_pcd = region_mesh.sample_points_uniformly(10000000)
    gt_pcd = gt_pcd.voxel_down_sample(0.01)
    points=torch.from_numpy(np.asarray(gt_pcd.points)).float().to(main_device)
    min_bound = torch.min(points,dim = 0)[0]
    oct.bound_min = min_bound - 0.025
    bound_group = oct.split_voxels(points)
    for bound in bound_group:
        small_pcd = input_pcd.crop(bound)
        if np.asarray(small_pcd.points).shape[0] == 0:
            continue
        small_gt = region_mesh.crop(bound)
        small_pcd = small_pcd.voxel_down_sample(0.005)
        small_points=torch.from_numpy(np.asarray(small_pcd.points)).to(main_device).float()
        min_bound = torch.min(small_points,dim = 0)[0]
        layer_oct.bound_min = min_bound - 0.025
        if not layer_oct.check_size(small_points):
            continue
        R = get_random_Rotation()
        small_pcd = small_pcd.rotate(R,center = small_gt.get_center())
        small_gt = small_gt.rotate(R,center = small_gt.get_center())
        o3d.io.write_triangle_mesh(os.path.join(result_folder,f"mesh_{count}.ply"),small_gt)
        o3d.io.write_point_cloud(os.path.join(result_folder,f"pcd_{count}.ply"),small_pcd)
        count += 1 

noise_sigma = float(sys.argv[3])
nname = int(noise_sigma * 1000)
src_root = "/home/chx/data_disk/MatterPort3D/"
whole_pcd_root = f"/home/chx/ssd/chx_data/MatterPort"
random_ratio = float(sys.argv[2])
rname = int(random_ratio * 100)
# the cropped rooms for training, used in train.py
result_root = f"/home/chx/data_disk/MatterPort3D/small_{rname}_{nname}/"
os.makedirs(result_root,exist_ok=True)
f = open(os.path.join(src_root,f"scenes_{sys.argv[1]}.txt"))
scene_dirs = [line.strip() for line in f]
scene_dirs = scene_dirs[:1]
f.close()
count = 0
for scene in scene_dirs:
    regions=os.listdir(os.path.join(whole_pcd_root,scene))
    count = 0
    for region in regions:  
        if region.find("region") == -1:
            continue
        crop_small(scene,region)