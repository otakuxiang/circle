import open3d as o3d
import os
import numpy as np
from tqdm import tqdm
import shutil,time
import cv2,random
import sys


def add_depth_noise(depthmaps, noise_sigma):

    # add noise
    if noise_sigma > 0:
        random.seed(time.time())
        np.random.seed(int(time.time()))
        sigma = noise_sigma
        noise = np.random.normal(0, 1, size=depthmaps.shape).astype(np.float32)
        depthmaps = depthmaps + noise * sigma * depthmaps

    return depthmaps

def add_noise(scene,region,noise_sigma):
    print(scene,region)    
    result_path = os.path.join(result_root,scene,region)
    os.makedirs(result_path,exist_ok=True)
    src_depth_path = os.path.join(tar_root,scene,region)
    sequence_path = os.path.join(src_root,"v1","scans",scene)
    region_mesh = o3d.io.read_triangle_mesh(os.path.join(sequence_path,"region_segmentations",f"{region}.ply"))
    region_pcd = region_mesh.sample_points_uniformly(10000000)
    region_pcd = region_pcd.voxel_down_sample(0.01)


    if not os.path.exists(os.path.join(src_depth_path,f"frame_list_{rname}.txt")):
        return 


    f_file = open(os.path.join(src_depth_path,f"frame_list_{rname}.txt"))
    frame_list = []
    for line in f_file:
        frame_list.append(line.strip().split(" "))
    if len(frame_list) == 0:
        return 
    depth_path = os.path.join(src_depth_path,f"depth_{rname}")
    result_depth = os.path.join(src_depth_path,f"depth_noise_{rname}_{nname}")
    if not os.path.exists(result_depth):
        os.makedirs(result_depth,exist_ok=True)
    else:
        shutil.rmtree(result_depth)
        os.makedirs(result_depth,exist_ok=True)
    pose_path = os.path.join(sequence_path,"matterport_camera_poses")
    intri_path = os.path.join(sequence_path,"matterport_camera_intrinsics")
    final_pcd = o3d.geometry.PointCloud()
    for frame_id,frames in enumerate(tqdm(frame_list)):
        tripod_number,camera_id,frame_idx = frames
        f = open(os.path.join(pose_path,tripod_number+"_pose_"+camera_id+"_"+frame_idx+".txt"))
        pose = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")
            for k in range(0,4):
                pose[idx,k] = float(ss[k])
        f.close()
        pose_inv = np.linalg.inv(pose)
        f = open(os.path.join(intri_path,tripod_number+"_intrinsics_"+camera_id+".txt"))
        p = np.zeros((4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")   
            for j in range(4):
                p[j] = float(ss[j+2])
        f.close()

        depth = os.path.join(depth_path,f"{frame_id}.png")
        depth = cv2.imread(depth,-1)
        depth = depth.astype(np.float32) / 4000
        depth = add_depth_noise(depth,noise_sigma)
        depth[depth<0.1] = np.nan
        depth[depth>4.0] = np.nan
        cv2.imwrite(os.path.join(result_depth,f"{frame_id}.png"),(depth * 4000).astype(np.uint16))
        depth = o3d.geometry.Image((depth * 4000).astype(np.uint16))

        ins = o3d.camera.PinholeCameraIntrinsic(1280,1024,p[0],p[1],p[2],p[3])
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth,ins,pose_inv,depth_scale = 4000)
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location(pose[0:3,3:4])
        pcd = pcd.normalize_normals()
        pcd = pcd.remove_non_finite_points()
        pcd = pcd.voxel_down_sample(0.01)
        final_pcd += pcd
    final_pcd = final_pcd.voxel_down_sample(0.01)
    o3d.io.write_point_cloud(os.path.join(result_path,f"noisy_points_{rname}_{nname}.ply"),final_pcd)

noise_sigma = float(sys.argv[3])
nname = int(noise_sigma * 1000)
# the original matterport3D dataset location, should cotains v1 folder
src_root = f"/home/chx/data_disk/MatterPort3D/"
# the sampled matterport3d location
tar_root = f"/home/chx/nas/disk_0/matterport_result/{sys.argv[1]}/"
random_ratio = float(sys.argv[2])
rname = int(random_ratio * 100)

f = open(os.path.join(src_root,f"scenes_{sys.argv[1]}.txt"))
scene_dirs = [line.strip() for line in f]
f.close()
# the folder which contains noisy pointclouds of the whole room
result_root = "/home/chx/ssd/chx_data/MatterPort/"
all_task = []
for scene in scene_dirs:
    regions=os.listdir(os.path.join(tar_root,scene))
    for region in regions:  
        if region.find("region") == -1:
            continue
        add_noise(scene,region,noise_sigma)
