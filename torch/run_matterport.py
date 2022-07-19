import sys
import argparse,traceback
import open3d as o3d
import structure.octree.unet_oct as oct
from utils.data_util import scene_cube
import torch
import time
import numpy as np
from utils import net_util
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import random
import os
import random
from network.diff_renderer import diff_renderer
from utils import data_util
from tqdm import tqdm
if __name__ == '__main__':
    checkpoint = sys.argv[1]
    c_num = int(sys.argv[2])
    model, args_model = net_util.load_origin_unet_model(f"pre-trained-weight/{checkpoint}/hyper_small.json",c_num,use_nerf= False,layer = 5)
    model.eval()
    main_device = torch.device("cuda:0")
    is_batch = False
    batch_size = 8
    voxel_size = 0.05
    h = 512
    w = 640
    data_dir = "/home/chx/nas/other_disk/CVPR2022CIRCLE/noisy_depth/"

    

    result_root = "/home/chx/nas/disk_0/matterport_result/test"
    scene = sys.argv[3]
    region = sys.argv[4]
    scene_path = os.path.join(data_dir,scene)
    region_path = os.path.join(scene_path,region)
    result_path = os.path.join(result_root,scene,region,"result")
    os.makedirs(result_path,exist_ok=True)
    
    input_points = o3d.io.read_point_cloud(os.path.join(region_path,f"noisy_points.ply"))
    
    bound = (input_points.get_max_bound() - input_points.get_min_bound())

    points=torch.from_numpy(np.asarray(input_points.points)).to(main_device).float()
    min_bound = torch.from_numpy(input_points.get_min_bound()).squeeze(-1).cuda().float() - voxel_size / 2
    
    normals = torch.from_numpy(np.asarray(input_points.normals)).to(main_device).float()

    octree = oct.unet_oct_cube(model,main_device,latent_dim=29,layer_num = 4,voxel_size = voxel_size,renderer = None,bound = bound)
    octree.bound_min = min_bound
    with torch.no_grad():

        octree.update_lowest(points,normals)
        octree.update_right_corner()
        mesh = octree.extract_whole_mesh_corner(4,use_rgb = False,max_n_triangles = int((2 ** 25)))
        mesh.compute_vertex_normals()

        # o3d.io.write_triangle_mesh(os.path.join(result_path,f"ours.ply"),mesh)
        o3d.io.write_triangle_mesh("temp.ply",mesh)
