import sys
import argparse
from utils import exp_util, vis_util
import open3d as o3d
import structure.octree.unet_oct as oct
import torch
import time
import numpy as np
from utils import net_util
from torch.utils.tensorboard import SummaryWriter
from utils.data_util import scene_cube
from pathlib import Path
import random
# from pytorch_memlab import profile,MemReporter
from utils.net_util import StepLearningRateSchedule,adjust_learning_rate
import math
import os
import random
from data.trainDataLoader import NoisyOtherMatterPortDataset,NoisyMatterPortDataset
import pdb,traceback
import shutil
from network.diff_renderer import diff_renderer
from system.ext import sdf_from_points

def preturb_points(points,normals,voxel_size,num = 1):
    sdf = torch.randn((points.shape[0],num,1)).cuda() * 0.05
    
    p = points.unsqueeze(1).repeat(1,num,1)
    n = normals.unsqueeze(1).repeat(1,num,1)
    p = p + n * sdf * voxel_size
    sdf = sdf.view(-1,1)
    p_points = p.view(-1,3)
    p_normals = n.view(-1,3)
    return p_points,sdf,p_normals

if __name__ == '__main__':
    layer = 5
    model, args_model = net_util.load_unet_model("config/hyper_small.json",-1,use_nerf = False,layer = layer)
    
    exp = sys.argv[1]
    main_device = torch.device("cuda:0")
    voxel_size = 0.05
    is_normal = True
    is_surface_normal = True
    voxel_resolution = int(10)
    h = 512
    w = 640
    normal_weight = 0.1
    xyz_num = 2500 if is_normal or is_surface_normal else 7500
    snum = 400000 if is_normal or is_surface_normal else 3000000
    checkpoint_dir = Path("./pre-trained-weight/"+exp)
    os.makedirs(checkpoint_dir,exist_ok = True)
    shutil.copy("./train.py",os.path.join(checkpoint_dir,"train.py"))
    f = open("/home/chx/chx_data/MatterPort3d/scenes_train.txt")
    scenes = [line.strip() for line in f]

    # train_dataset = NoisyOtherMatterPortDataset("/home/chx/ssd/MatterPort3d/",scenes,expand=True,voxel_size = voxel_size,snum = snum,layer = layer)
    train_dataset = NoisyMatterPortDataset("/home/chx/ssd/chx_data/MatterPort/",scenes,split = 50,expand=True,voxel_size = voxel_size,snum = snum,layer = layer)
   
    train_dataLoader = torch.utils.data.DataLoader(train_dataset,num_workers = 12,batch_size = 1,shuffle = True)

    dt_size = len(train_dataset)
    writer = SummaryWriter("./runs/"+exp)

    lr_schedules = []
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))
    optimizer_all = torch.optim.Adam([ 
        { "params": model.conv_kernels.parameters(), "lr": lr_schedules[0].get_learning_rate(0)},
        { "params": model.encoder.parameters(), "lr": lr_schedules[1].get_learning_rate(0)},
        { "params": model.decoder.parameters(), "lr": lr_schedules[2].get_learning_rate(0) },
    ])
    start = 0
    num_epoch = 200
    rgb_iter = 100000000
    renderer = diff_renderer(model,h,w,voxel_size,main_device,is_eval = False)
    sdf_octree = scene_cube(layer_num = 4,voxel_size = voxel_size,device = "cuda:0",lowest = 7)

    for epoch in range(start,num_epoch):
        for idx,[points,normals,structures,gt_points,gt_normals,min_bound,bound] in enumerate(train_dataLoader):

            if points.nelement() == 0 :
                continue
            
            it = idx + epoch * dt_size
            # clamp_dist = cd_schedule.get_learning_rate(it)
            clamp_dist = 1
            points = points.to(main_device).squeeze(0)
            normals = normals.to(main_device).squeeze(0)
            gt_points = gt_points.to(main_device).squeeze(0)
            gt_normals = gt_normals.to(main_device).squeeze(0)
            bound = bound.squeeze(0).numpy()

            octree = oct.unet_oct_cube(model,main_device,bound = bound,latent_dim=29,layer_num = 4,voxel_size = voxel_size,lowest = 7,renderer=renderer)
            octree.bound_min = min_bound.to(main_device).squeeze(0)
            sdf_octree.bound_min = octree.bound_min
            sdf_octree.set_bound(bound)
            
            with torch.no_grad():
                torch.cuda.empty_cache()
                rand_xyz,rand_sdf = sdf_octree.random_gt_sdf(gt_points,gt_normals,rand_num = 24,expand = True,xyz_num = xyz_num * 2)
                p_points,gt_sdf,p_normals = preturb_points(gt_points,gt_normals,voxel_size,5)
                end = min(snum,p_points.size(0))
                inds = torch.randperm(p_points.size(0),device="cuda:0")
                inds = inds[:end]
                surface_points = p_points[inds,:]
                surface_normals = p_normals[inds,:]
                gt_sdf = sdf_from_points(surface_points,gt_points,gt_normals,8,0.02)
                gt_sdf = gt_sdf / voxel_size

                inds = torch.randperm(gt_points.size(0),device="cuda:0")
                end = min(snum,gt_points.size(0))
                inds = inds[:end]
                gt_points = gt_points[inds,:]
                gt_normals = gt_normals[inds,:]
                surface_points = torch.cat([surface_points,gt_points])
                surface_normals = torch.cat([surface_normals,gt_normals])
                surface_sdfs = torch.cat([gt_sdf,torch.zeros(gt_points.size(0)).cuda()])

    
            try:
                adjust_learning_rate(lr_schedules, optimizer_all,it)
                optimizer_all.zero_grad()

                octree.update_lowest(points,normals,required_grad = True)
                loss4 = octree.update_right_corner(if_sloss=True,gt_s=structures,rgb = False)
                    
                loss2 = octree.compute_corner_reg_loss()
                if is_surface_normal:
                    sdf_loss,l1 = octree.compute_surface_loss(surface_points,surface_sdfs,cd = clamp_dist,clamp_mode="o")
                    normal_loss,normal_reg_loss,l2 = octree.compute_normal_loss(surface_points,surface_normals)
                    if sdf_loss is None:
                        del octree
                        optimizer_all.zero_grad()
                        continue
                    loss1,l0 = octree.compute_sdf_loss_corner(rand_xyz,rand_sdf)
                    if loss1 is None:
                        del octree
                        optimizer_all.zero_grad()
                        continue
                    normal_reg_loss = normal_reg_loss / l2
                    sdf_loss = (loss1 + sdf_loss) / (l1 + l0)
                    normal_loss = normal_loss / l2
                    loss = loss2 * 1e-4 + loss4 + sdf_loss + normal_weight * ( normal_loss + normal_reg_loss) 
 
                loss.backward()


                optimizer_all.step() 
                octree.intermediate_detach()
            except oct.BoundError as be:
                traceback.print_exc()
                continue
            except ValueError as ve:
                traceback.print_exc()
                continue
            except Exception as ex:
                traceback.print_exc()
                continue
                # pdb.post_mortem(ex.__traceback__)
            
            if it % 50 == 0:
                writer.add_scalar(f'train/Loss', loss, it)
                writer.add_scalar(f'train/StructureLoss', loss4, it)
                writer.add_scalar(f'train/SdfLoss', sdf_loss, it )
                if is_surface_normal:
                    writer.add_scalar(f'train/NormalLoss', normal_loss, it)
                    writer.add_scalar(f'train/NormalRegLoss', normal_reg_loss, it)
                print(f"[{idx}/{dt_size} Train Loss: {loss.detach().cpu().numpy()}]")
            if it % 1000 == 0:
                torch.save({
                        "encoder_state": model.encoder.state_dict(),
                        "decoder_state": model.decoder.state_dict(),
                        # "nerf_state": model.rgb_decoder.state_dict(),
                        "sparse_state": model.conv_kernels.state_dict(),
                        "optimizer_state": optimizer_all.state_dict(),
                    }, checkpoint_dir / f"model_{it}.pth.tar")

                    