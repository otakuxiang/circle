import sys
import argparse
from utils import exp_util, vis_util
import open3d as o3d
import structure.octree.unet_oct as oct
import torch
import time
import numpy as np
from network import utility
from torch.utils.tensorboard import SummaryWriter
from data_proc.trainScannnetPcd import scene_cube
from pathlib import Path
import random
# from pytorch_memlab import profile,MemReporter
from network.utility import StepLearningRateSchedule,adjust_learning_rate,CycleStepLearningRateSchedule,CycleStepDownLearningRateSchedule, \
    StepCycleStepLearningRateSchedule,StepCycleStepDownLearningRateSchedule,ConstantLearningRateSchedule,StepLearningRateMinSchedule
import math
import os
import random
from data_proc.trainSynthesisPcd import SynthesisDataset
from data_proc.trainDataLoader import OtherMatterPortDataset,NoisyOtherMatterPortDataset
import pdb,traceback
import shutil
from network.diff_renderer import diff_renderer


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
    # model, args_model = utility.load_unet_model("pre-trained-weight/CycleStepDown/hyper.json",10000,1)
    layer = 5
    # model, args_model = utility.load_unet_model("config/hyper_small.json",-1,use_nerf = False,layer = layer)
    model, args_model = utility.load_unet_model("pre-trained-weight/normal_1/hyper_small.json",32000,use_nerf= False,layer = layer)

    
    exp = sys.argv[1]
    main_device = torch.device("cuda:0")
    is_batch = False
    batch_size = 8
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
    shutil.copy("./train_other.py",os.path.join(checkpoint_dir,"train.py"))
    f = open("/home/chx/chx_data/MatterPort3d/scenes_train.txt")
    scenes = [line.strip() for line in f]

    train_dataset = NoisyOtherMatterPortDataset("/home/chx/ssd/MatterPort3d/",scenes,expand=True,voxel_size = voxel_size,snum = snum,layer = layer)
   
    train_dataLoader = torch.utils.data.DataLoader(train_dataset,num_workers = 12,batch_size = 1,shuffle = True)

    dt_size = len(train_dataset)
    writer = SummaryWriter("./runs/"+exp)

    lr_schedules = []
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))
    lr_schedules.append(StepLearningRateSchedule(2e-3,10000,0.9))

    # lr_schedules.append(StepLearningRateSchedule(5e-2,50,0.8))
    optimizer_all = torch.optim.Adam([ 
        { "params": model.conv_kernels.parameters(), "lr": lr_schedules[0].get_learning_rate(0)},
        { "params": model.encoder.parameters(), "lr": lr_schedules[1].get_learning_rate(0)},
        { "params": model.decoder.parameters(), "lr": lr_schedules[2].get_learning_rate(0) },
        # { "params": model.rgb_decoder.parameters(), "lr": lr_schedules[3].get_learning_rate(0) }
    ])
    # max_size = 900
    start = 0
    num_epoch = 200
    rgb_iter = 100000000
    renderer = diff_renderer(model,h,w,voxel_size,main_device,is_eval = False)
    octree = oct.unet_oct_cube(model,main_device,latent_dim=29,layer_num = 4,voxel_size = voxel_size,lowest = 7,renderer=renderer)
    sdf_octree = scene_cube(layer_num = 4,voxel_size = voxel_size,device = "cuda:0",lowest = 7)

    # cd_schedule = StepLearningRateMinSchedule(1,20000,0.9,0.2)
    for epoch in range(start,num_epoch):
        for idx,[points,normals,structures,gt_points,gt_normals,min_bound] in enumerate(train_dataLoader):

            if points.nelement() == 0 :
                continue
            
            it = idx + epoch * dt_size + start_idx
            # clamp_dist = cd_schedule.get_learning_rate(it)
            clamp_dist = 1
            points = points.to(main_device).squeeze(0)
            normals = normals.to(main_device).squeeze(0)
            gt_points = gt_points.to(main_device).squeeze(0)
            gt_normals = gt_normals.to(main_device).squeeze(0)

        
            octree.bound_min = min_bound.to(main_device).squeeze(0)
            sdf_octree.bound_min = octree.bound_min
            
            with torch.no_grad():
                
                torch.cuda.empty_cache()
                unique_xyz,sdf = sdf_octree.compute_gt_sdf(gt_points,gt_normals,voxel_resolution,expand = True)
                inds = np.array([i for i in range(unique_xyz.size(0))],dtype = np.int64)
                np.random.shuffle(inds)
                inds = torch.from_numpy(inds)
                end = min(xyz_num,unique_xyz.size(0))
                xyz_small = unique_xyz[inds,:][0:end,:]
                sdf_small = sdf[inds,:][0:end,:]
                # if is_surface_normal:
                p_points,gt_sdf,p_normals = preturb_points(gt_points,gt_normals,voxel_size,5)
                inds = np.array([i for i in range(p_points.size(0))],dtype = np.int64)
                end = min(snum,p_points.size(0))
                np.random.shuffle(inds)
                inds = torch.from_numpy(inds)
                surface_points = p_points[inds,:][:end,:]
                surface_normals = p_normals[inds,:][:end,:]
                surface_sdfs = gt_sdf[inds,:][:end,:]
                # print(sdf_small.shape)
            del unique_xyz,sdf,gt_points,gt_normals
            
            ep = 1
            octree.is_batch = is_batch
            octree.batch_size = batch_size

            try:
                adjust_learning_rate(lr_schedules, optimizer_all,it)
                optimizer_all.zero_grad()

                octree.update_lowest(points,normals,required_grad = True)
                loss4 = octree.update_right_corner(if_sloss=True,gt_s=structures,rgb = False)
                    
                # if it > rgb_iter:
                #     torch.cuda.empty_cache()
                    
                #     loss3,rnum = octree.compute_render_loss(targets_small[:,3],ray_ods_small,use_rgb = False)
                #     loss3 = loss3 / rnum
                loss2 = octree.compute_corner_reg_loss()
                if is_surface_normal:
                
                    sdf_loss,normal_loss,normal_reg_loss,l1 = octree.compute_surface_loss(surface_points,surface_sdfs,surface_normals,cd = clamp_dist,clamp_mode="o")
                    loss1,nrl,l0 = octree.compute_corner_sample_loss(xyz_small,sdf_small,voxel_resolution,clamp_dist,if_normal = is_normal,clamp_mode="o")
                    
                    normal_reg_loss = (nrl + normal_reg_loss) / (l1 + l0)  
                    sdf_loss = (loss1 + sdf_loss) / (l1 + l0)
                    normal_loss = normal_loss / l1
                    loss = loss2 * 1e-4 + loss4 + sdf_loss + normal_weight * (normal_loss + normal_reg_loss)
                else:
                    loss1,l0 = octree.compute_corner_sample_loss(xyz_small,sdf_small,voxel_resolution,clamp_dist,if_normal = False)
                    loss5,l1 = octree.compute_sdf_loss_corner(p_points,gt_sdf,clamp_dist,if_normal = False)
                    sdf_loss = (loss1+loss5) / (l1+l0)
                    loss = loss2 * 1e-4 + loss4 + sdf_loss

                # if it > rgb_iter:
                #     loss = loss + loss3        
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

                    