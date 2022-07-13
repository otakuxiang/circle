import sys
import argparse,traceback
from utils import exp_util, vis_util
import open3d as o3d
import pycg
import structure.octree.unet_oct as oct
from data_proc.trainScannnetPcd import scene_cube
from data_proc.iclDataLoader import IclDataset
# import structure.octree.unet_oct_incremental as oct
import torch
import time
import numpy as np
from network import utility
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import random
import os
import random
from data_proc.trainSynthesisPcd import SynthesisDataset
from data_proc.trainDataLoader import ScannetSmallSceneHoleDataset,ScannetSmallSceneHoleSdfDataset,ScannetMeshDataset,MatterPortColorDataset
from network.diff_renderer import diff_renderer
from data_proc import data_utils
from tqdm import tqdm
if __name__ == '__main__':
    checkpoint = sys.argv[1]
    c_num = int(sys.argv[2])
    
    # model, args_model = utility.load_unet_model(f"pre-trained-weight/{checkpoint}/hyper_small.json",c_num,use_nerf= False,layer = 5)
    model, args_model = utility.load_origin_unet_model(f"pre-trained-weight/{checkpoint}/hyper_small.json",c_num,use_nerf= False,layer = 5)

    model.eval()


    
    main_device = torch.device("cuda:0")
    is_batch = False
    batch_size = 8
    voxel_size = 0.05
    h = 512
    w = 640
    # data_dir = "/home/chx/nas/disk_0/matterport_result/test"
    data_dir = "/home/chx/nas/other_disk/CVPR2022CIRCLE/noisy_depth/"

    

    result_root = "/home/chx/nas/disk_0/matterport_result/test"
    error_scenes = []
    # scenes = ["lr_kt0"
    scene = sys.argv[3]
    region = sys.argv[4]
    # r_name = int(sys.argv[5])
    # n_name = int(sys.argv[6])
    scene_path = os.path.join(data_dir,scene)
    # regions = os.listdir(scene_path)
    region_path = os.path.join(scene_path,region)
    result_path = os.path.join(result_root,scene,region,"result")
    os.makedirs(result_path,exist_ok=True)
    
    # if os.path.exists(os.path.join(region_path,f"noisy_points_{r_name}_{n_name}.ply")):   
        # if not os.path.exists(os.path.join(result_path,f"ours_{r_name}_{n_name}.ply")) :
    # input_points =  o3d.io.read_point_cloud(os.path.join(region_path,f"noisy_points_{r_name}_{n_name}.ply"))
    input_points = o3d.io.read_point_cloud(os.path.join(region_path,f"noisy_points.ply"))
    
    # input_points = input_points.voxel_down_sample(0.05)
    bound = (input_points.get_max_bound() - input_points.get_min_bound())
    # print(bound * 20)
    # o3d.io.write_point_cloud("kt0.ply",input_points)
    points=torch.from_numpy(np.asarray(input_points.points)).to(main_device).float()
    min_bound = torch.from_numpy(input_points.get_min_bound()).squeeze(-1).cuda().float() - 0.025
    
    normals = torch.from_numpy(np.asarray(input_points.normals)).to(main_device).float()
    # print(normals)
    octree = oct.unet_oct_cube(model,main_device,latent_dim=29,layer_num = 4,voxel_size = voxel_size,renderer = None,bound = bound)
    octree.bound_min = min_bound
    # print(min_bound)torch.cuda.synchronize()
    with torch.no_grad():
        # tt = 0
        # for i in (range(10)):
        octree.update_lowest(points,normals)
        octree.update_right_corner()
        # for i in (range(50)):
        #     start = time.time()
        #     octree.update_lowest(points,normals)
        #     octree.update_right_corner()
        #     end = time.time()
        #     tt += end - start
        # print(tt/50)
        mesh = octree.extract_whole_mesh_corner(4,use_rgb = False,max_n_triangles = int((2 ** 25)))

        # # mesh = mesh.merge_close_vertices(0.001)
        # # mesh = mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        # # pcd = pycg.vis.pointcloud(points,normal=normals) 
        pycg.vis.show_3d([mesh])
        # o3d.io.write_triangle_mesh("test.ply",mesh)
        # print(result_path,f"ours_{r_name}_{n_name}.ply")
        # o3d.io.write_triangle_mesh(os.path.join(result_path,f"ours_{r_name}_{n_name}.ply"),mesh)