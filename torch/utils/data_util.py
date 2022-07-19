import math
import torch
import numpy as np
import open3d as o3d
from utils import net_util
from torch._C import dtype
from system.ext import unproject_depth, remove_radius_outlier, \
    estimate_normals, filter_depth, sdf_from_points
import cv2
import os
import random
from data.ScannetLoader import ScannetImageLoader
from torch_scatter import scatter

def clamp_select(tar,src,min,max):
    mask_max = torch.logical_and(src > max,tar > max)
    mask_min = torch.logical_and(tar < min,src < min)
    mask = ~torch.logical_or(mask_min,mask_max)
    return mask

def gram_schmidt_Rotation(Rs):
    # Rs (batch,3,2)
    b1 = Rs[:,:,0] / Rs[:,:,0].norm(dim = 1,keepdim = True)
    b2 = Rs[:,:,1] - (Rs[:,:,1] * b1).sum(dim = 1,keepdim = True) * b1
    b3 = b1.mm(b2)
    result = torch.stack([b1,b2,b3],dim = 2)

def gradient_clamp(a,min,max):
    min_mask = a < min
    inv = 1 / a[min_mask].detach()
    a[min_mask] = min * a[min_mask] * inv
    max_mask = a > max
    inv = 1 / a[max_mask].detach()
    a[max_mask] = max * a[max_mask] * inv
    return a

def gather_input(depth_imgs,directions,ray_xys,ins):
    # directions = directions / directions.norm(dim = 1,keepdim = True)
    fx = ins[0]
    fy = ins[1]
    cx = ins[2]
    cy = ins[3]
    targets = []
    ray_ods = []
    pose_ids = []
    weights = []
    for idx,depth_img in enumerate(depth_imgs):
        # if idx == 1:
        #     continue
        target = depth_img[ray_xys[:,1],ray_xys[:,0]].view(-1)
        u = ray_xys[:,0]
        v = ray_xys[:,1]
        dis = torch.sqrt( 1 + \
                ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2
            )
        depth_weight = 1 / target.clamp(0.5,2)
        target = target * dis
        ray_od = directions.unsqueeze(1)  # N x 1 x 3
        targets.append(target)
        ray_ods.append(ray_od)
        pose_id = torch.ones((directions.size(0),),dtype = torch.long,device= directions.device) * idx
        pose_ids.append(pose_id)
        weights.append(depth_weight)
    targets = torch.cat(targets,dim = 0)
    ray_ods = torch.cat(ray_ods,dim = 0)
    pose_ids = torch.cat(pose_ids,dim = 0)
    weights = torch.cat(weights,dim = 0)
    # ray_ods[:,0,:] = torch.matmul(Rs[0],ray_ods[:,0,:].unsqueeze(-1)).squeeze(-1)
    # print(ray_ods[:,0,:],compare)
    return ray_ods,targets,pose_ids,weights

import math

def load_depth_frames_dr_dt(data_dir,h,w,frame_num,start_frame = 0,device = "cuda:0"):
    config_path = os.path.join(data_dir,"intrinsic")
    pose_path = os.path.join(data_dir,"pose")
    dataLoader = ScannetImageLoader(data_dir,config_path,w = w,h = h)
    dataLoader.loadTraj(pose_path)
    depths = []
    colors = []
    Rs = []
    ts = []
    ins_K = dataLoader.getDepthIns()
    ins = [ins_K[0,0],ins_K[1,1],ins_K[0,2],ins_K[1,2]]

    for frame_id in range(start_frame,dataLoader.size()):
        if len(Rs) == frame_num:
            break
        cur_pose = dataLoader.getGtPose(frame_id) 
        if np.any(np.isinf(cur_pose)) or np.any(np.isnan(cur_pose)):
            continue
        if len(Rs) != 0:
            R = torch.from_numpy(cur_pose[0:3,0:3]).to(device).float()
            T = torch.from_numpy(cur_pose[0:3,3:]).float().permute(1,0).contiguous().to(device)
            dR = R.inverse() @ Rs[-1]
            dT = T - ts[-1]
            dr = torch.acos(((dR.trace() - 1) / 2)) / math.pi * 180
            dt = dT.view(-1).norm()
            # print(dt,dr)
            if dt < 0.3 and dr < 8:
                continue
        color,depth = dataLoader.getImage(frame_id)
        depth = torch.from_numpy(depth).to(device)
        depth_data = depth.clone()
        filter_depth(depth,depth_data)
        depth_data[torch.logical_or(depth_data < 0.1,depth_data > 4)] = np.nan
        depths.append(depth_data)
        Rs.append(torch.from_numpy(cur_pose[0:3,0:3]).to(device).float())
        ts.append(torch.from_numpy(cur_pose[0:3,3:]).float().permute(1,0).contiguous().to(device))
        color = torch.from_numpy(color).to(device).permute(2,0,1).contiguous()
        colors.append(color)

    return colors,depths,Rs,ts,ins


def load_depth_frames(data_dir,h,w,frame_list = None,start_frame = 0,skip_frame = 5,end_frame = -1,device = "cuda:0"):
    config_path = os.path.join(data_dir,"intrinsic")
    pose_path = os.path.join(data_dir,"pose")
    dataLoader = ScannetImageLoader(data_dir,config_path,w = w,h = h)
    dataLoader.loadTraj(pose_path)
    depths = []
    colors = []
    Rs = []
    ts = []
    ins_K = dataLoader.getDepthIns()
    ins = [ins_K[0,0],ins_K[1,1],ins_K[0,2],ins_K[1,2]]

    if frame_list is None:
        frame_id = start_frame
        frame_list = []
        while True:
            if end_frame != -1 and frame_id >= end_frame:
                break
            elif frame_id >= dataLoader.max_frame:
                break
        frame_list.append(frame_id)
        frame_id += skip_frame

    # print(dataLoader.max_frame)
    for frame_id in frame_list:
        color,depth = dataLoader.getImage(frame_id)
        depth = torch.from_numpy(depth).to(device)
        depth_data = depth.clone()
        filter_depth(depth,depth_data)
        depth_data[torch.logical_or(depth_data < 0.1,depth_data > 4)] = np.nan
        cur_pose = dataLoader.getGtPose(frame_id) 
        # print(frame_id)
        color = torch.from_numpy(color).to(device).permute(2,0,1).contiguous()

        if np.any(np.isinf(cur_pose)) or np.any(np.isnan(cur_pose)):
            continue
        depths.append(depth_data)
        Rs.append(torch.from_numpy(cur_pose[0:3,0:3]).to(device).float())
        ts.append(torch.from_numpy(cur_pose[0:3,3:]).float().permute(1,0).contiguous().to(device))
        colors.append(color)
        
        # print(frame_id)
    return colors,depths,Rs,ts,ins

def fuse_depth_to_pointcloud(depths,Rs,ts,ins):
    final_pc_data = []
    final_normal_data = []
    for idx,depth in enumerate(depths):
        R = Rs[idx].cpu().numpy()
        t = ts[idx].cpu().numpy()
        pose = np.ones((4,4))
        pose[0:3,0:3] = R
        pose[0:3,3:] = np.transpose(t)
        pc_data = unproject_depth(depth, ins[0], ins[1],ins[2], ins[3])
        pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), pc_data.size(1), 1), device=pc_data.device)], dim=-1)
        pc_data = pc_data.reshape(-1, 4)
        nan_mask = ~torch.isnan(pc_data[..., 0]) 
        pc_data = pc_data[nan_mask]
        # 
        pc_data_valid_mask = remove_radius_outlier(pc_data, 16, 0.05)
        pc_data = pc_data[pc_data_valid_mask]
        

        normal_data = estimate_normals(pc_data, 16, 0.1, [0.0, 0.0, 0.0])
        pc_data = pc_data[:, :3].float()
        nan_mask = ~torch.isnan(normal_data[..., 0])
        pc_data = pc_data[nan_mask].unsqueeze(-1)
        normal_data = normal_data[nan_mask].unsqueeze(-1)
        pc_data = (torch.matmul(Rs[idx],pc_data) + ts[idx].permute(1,0)).squeeze(-1)
        normal_data = torch.matmul(Rs[idx],normal_data).squeeze(-1)
        final_pc_data.append(pc_data)
        final_normal_data.append(normal_data)
    final_pc_data = torch.cat(final_pc_data,dim = 0)
    final_normal_data = torch.cat(final_normal_data,dim = 0)
        
    final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_pc_data.cpu().numpy()))
    final_pcd.normals = o3d.utility.Vector3dVector(final_normal_data.cpu().numpy())
    return final_pcd   

def convert_depth_to_pointcloud(depths,ins):
    pc_data_list = []
    normal_data_list = []
    # final_pose_id = []
    for idx,depth in enumerate(depths):
        pc_data = unproject_depth(depth, ins[0], ins[1],ins[2], ins[3])
        pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), pc_data.size(1), 1), device=pc_data.device)], dim=-1)
        pc_data = pc_data.reshape(-1, 4)
        nan_mask = ~torch.isnan(pc_data[..., 0]) 
        pc_data = pc_data[nan_mask]
        # 
        pc_data_valid_mask = remove_radius_outlier(pc_data, 16, 0.05)
        pc_data = pc_data[pc_data_valid_mask]
        

        normal_data = estimate_normals(pc_data, 16, 0.1, [0.0, 0.0, 0.0])
        pc_data = pc_data[:, :3].float()
        nan_mask = ~torch.isnan(normal_data[..., 0])
        pc_data = pc_data[nan_mask].unsqueeze(-1)
        normal_data = normal_data[nan_mask].unsqueeze(-1)
        # pc_data = (torch.matmul(Rs[idx],pc_data) + ts[idx].permute(1,0)).squeeze(-1)
        # normal_data = torch.matmul(Rs[idx],normal_data).squeeze(-1)
        pc_data_list.append(pc_data)
        normal_data_list.append(normal_data)
        # pose_id = torch.ones((pc_data.size(0)),device = depth.device,dtype=torch.int64) * idx
        # final_pose_id.append(pose_id)
    # final_pc_data = torch.cat(final_pc_data,dim = 0)
    # final_normal_data = torch.cat(final_normal_data,dim = 0)
        
    # final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_pc_data.cpu().numpy()))
    # final_pcd.normals = o3d.utility.Vector3dVector(final_normal_data.cpu().numpy())
    return pc_data_list,normal_data_list

def gather_point_cloud(pc_data_list,normal_data_list,Rs,ts):
    final_pc_data = []
    final_normal_data = []
    for idx,pc_data in enumerate(pc_data_list):
        pc_data_c = (torch.matmul(Rs[idx],pc_data) + ts[idx].permute(1,0)).squeeze(-1)
        normal_data_c = torch.matmul(Rs[idx],normal_data_list[idx]).squeeze(-1)
        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_data_c.cpu().numpy()))
        # pcd.normals = o3d.utility.Vector3dVector(normal_data_c.cpu().numpy())
        # o3d.io.write_point_cloud(f"test/input_cloud_{idx}.ply",pcd)
        
        final_pc_data.append(pc_data_c)
        final_normal_data.append(normal_data_c)
    final_pc_data = torch.cat(final_pc_data,dim = 0)
    final_normal_data = torch.cat(final_normal_data,dim = 0)
    return final_pc_data,final_normal_data

def voxel_down_sample(final_pc_data,final_normal_data,voxel_size):
    min_bound = torch.min(final_pc_data,dim = 0,keepdim = True)[0]
    with torch.no_grad():
        processPcd = final_pc_data - min_bound + 0.5 * voxel_size
        processPcd = (processPcd // voxel_size)
        # print(processPcd.shape)
        # print(torch.max(processPcd,dim = 0)[0])
        nx,ny,nz = (torch.max(processPcd,dim = 0)[0] + 1).cpu().numpy().tolist()
        pc_index = processPcd[:,2] + processPcd[:,1] * nz + \
            processPcd[:,0] * nz * ny
        unq ,inv_inds = torch.unique(pc_index,return_inverse = True)
    final_pc_data = scatter(final_pc_data,index=inv_inds,dim = 0,reduce="mean")
    final_normal_data = scatter(final_normal_data,index=inv_inds,dim = 0,reduce="mean")
    return final_pc_data,final_normal_data

def voxel_down_sample_with_rgb(final_pc_data,final_normal_data,final_rgb_data,voxel_size):
    min_bound = torch.min(final_pc_data,dim = 0,keepdim = True)[0]
    with torch.no_grad():
        processPcd = final_pc_data - min_bound + 0.5 * voxel_size
        processPcd = (processPcd // voxel_size)
        # print(processPcd.shape)
        # print(torch.max(processPcd,dim = 0)[0])
        nx,ny,nz = (torch.max(processPcd,dim = 0)[0] + 1).cpu().numpy().tolist()
        pc_index = processPcd[:,2] + processPcd[:,1] * nz + \
            processPcd[:,0] * nz * ny
        unq ,inv_inds = torch.unique(pc_index,return_inverse = True)
    final_pc_data = scatter(final_pc_data,index=inv_inds,dim = 0,reduce="mean")
    final_normal_data = scatter(final_normal_data,index=inv_inds,dim = 0,reduce="mean")
    final_rgb_data = scatter(final_rgb_data,index=inv_inds,dim = 0,reduce="mean")
    return final_pc_data,final_normal_data,final_rgb_data

def color_reproject(depth,colors,ray,Rs,ts,h,w,ins,frame_id):
    frame_mask = frame_id == colors.size(0) - 1
    target_frame_id = frame_id + 1
    target_frame_id[frame_mask] = frame_id - 1
    render_points = (ray[:,1,:] + depth * ray[:,0,:]).unsqueeze(-1)
    camera_points = render_points - ts[target_frame_id,:,:]
    R_inv =  Rs[target_frame_id,:,:].permute(0,2,1).contiguous()
    camera_points = torch.matmul(R_inv,camera_points)
    # camera_points = camera_points / camera_points[:,2,:]
    K = torch.eye(3,device=depth.device)
    K[0,0] = ins[0]
    K[0,2] = ins[2]
    K[1,1] = ins[1]
    K[1,2] = ins[3]
    camera_index = torch.matmul(K,camera_points).squeeze(-1)
    camera_index = camera_index / camera_index[:,2]
    camera_index = camera_index[:,0:2]
    valid_mask_h = torch.logical_and(camera_index[:,1] >= 0,camera_index[:,1] < h)
    valid_mask_w = torch.logical_and(camera_index[:,0] >= 0,camera_index[:,0] < w)
    valid_mask = torch.logical_and(valid_mask_h,valid_mask_w)
    valid_mask = torch.all(valid_mask,dim = 1)
    camera_index = camera_index[valid_mask]
    views = torch.unique(target_frame_id[valid_mask]).cpu().detach().numpy().tolist()
    r_colors = []
    for view in views:
        view_mask = target_frame_id[view_mask] == view
        index = camera_index[view_mask].view(1,1,-1,2)
        input_image = colors[view:view+1,:,:,:]
        index[:,:,:,0] = index[:,:,:,0] / w
        index[:,:,:,1] = index[:,:,:,1] / h
        index = (index - 0.5)
        reproject_color = torch.grid_sample(input_image,index).view(input_image.size(1),-1)
        r_colors.append(reproject_color)


# def render_clean_depth(pc_data_list,normal_data_list,octree,Rs,ts):
    
#     with torch.no_grad():
#         clean_depths = []
#         for idx,(point,normal) in enumerate(zip(pc_data_list,normal_data_list)):
#             points = (torch.matmul(Rs[idx],point) + ts[idx].permute(1,0)).squeeze(-1)
#             normals = torch.matmul(Rs[idx],normal).squeeze(-1)
#             points,normals = voxel_down_sample(points,normals,0.015)
#             min_bound = torch.min(points,dim = 0)[0].squeeze(0)
#             octree.bound_min = min_bound - octree.voxel_size / 2
#             octree.update_lowest(points,normals)
#             octree.update_right_corner()
#             R = Rs[idx]
#             t = ts[idx]
#             # R = torch.eye(3,device=point.device)
#             # t = torch.zeros((1,3),device=point.device)
#             clean_depth = octree.render_depth([R,t])
#             clean_depth[torch.logical_or(clean_depth < 0.1,clean_depth > 4)] = np.nan
#             clean_depths.append(clean_depth)

#     return clean_depths  

def compute_iou(src_pcd,tar_pcd):
    voxel_size = 0.1
    tt_pcd = torch.cat([src_pcd,tar_pcd],dim = 0)
    min_bound = torch.min(tt_pcd,dim = 0,keepdim = True)[0]
    process_src = src_pcd - min_bound + 0.5 * voxel_size
    process_tar = tar_pcd - min_bound + 0.5 * voxel_size
    process_src = process_src // voxel_size
    process_tar = process_tar // voxel_size
    process_tt = torch.cat([process_src,process_tar],dim = 0)
    nx,ny,nz = (torch.max(process_tt,dim = 0)[0] + 1).cpu().numpy().tolist()
    src_index = process_src[:,2] + process_src[:,1] * nz + \
            process_src[:,0] * nz * ny
    tar_index = process_tar[:,2] + process_tar[:,1] * nz + \
            process_tar[:,0] * nz * ny
    tt_index = torch.cat([src_index,tar_index],dim = 0)
    src_count = torch.unique(src_index).size(0)
    tar_count = torch.unique(tar_index).size(0)
    tt_count = torch.unique(tt_index).size(0)
    intersect_num = src_count + tar_count - tt_count
    return intersect_num / (src_count + 0.0)

def get_rays_from_depth(depths,colors,poses,camera_ins,h,w):
    with torch.no_grad():
        xs,ys = torch.meshgrid(torch.arange(w).cuda(),torch.arange(h).cuda())
        xs = xs.contiguous()
        ys = ys.contiguous()
        pixels = torch.stack([xs,ys,torch.ones_like(xs)],dim = 2).view(-1,3,1).contiguous().float()
        ray_ods = []
        dis_gts = []
        rgb_gts = []

        
        for camera_id in range(camera_ins.size(0)):
            camera_ins_inv = camera_ins[camera_id,:,:].inverse()
            c_rays = (camera_ins_inv @ pixels)
            ray_ori = poses[camera_id:camera_id+1,0:3,3:4].permute(0,2,1).repeat(c_rays.size(0),1,1).contiguous() # N,1,3
            ray_d = (poses[camera_id:camera_id+1,0:3,0:3] @ c_rays).permute(0,2,1).contiguous()
            ray_ods.append(torch.cat([ray_d,ray_ori],dim = 1))
            dis = c_rays.norm(dim = 1)
            depth_gt = depths[camera_id,ys.view(-1),xs.view(-1)].unsqueeze(-1)
            color_gt = colors[camera_id,ys.view(-1),xs.view(-1),:]
            # print(dis.shape,depth_gt.shape)
            dis_gt = dis * depth_gt
            dis_gts.append(dis_gt)
            rgb_gts.append(color_gt)
        ray_ods = torch.cat(ray_ods,dim = 0)
        dis_gts = torch.cat(dis_gts,dim = 0)
        rgb_gts = torch.cat(rgb_gts,dim = 0)
        return ray_ods,dis_gts,rgb_gts

class scene_cube:
    def __init__(self,layer_num,voxel_size = 0.1,device = "cuda:0",lowest = 7):
        self.layer_num=layer_num
        self.voxel_size=voxel_size
        self.bound_min = torch.tensor([0,0,0]).float().to(device)
        self.n_xyz_L=[]
        self.device = device
        for i in range(0,self.layer_num):
            self.n_xyz_L.append([2 ** (lowest - i),2 ** (lowest - i),2 ** (lowest - i)])
        self.integration_offsets = [torch.tensor(t, device=device, dtype=torch.float32) for t in [
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
        ]]
        self.relative_network_offset = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=torch.float32)

    def set_bound(self,bound):
        # print("meter",bound)
        bound = ((bound + self.voxel_size) / self.voxel_size // 32 + 1).astype(np.int32) * 32
        # print("voxel",bound)
        self.n_xyz_L=[]
        self.n_xyz_L.append(list(bound))
        for i in range(1,self.layer_num):
            self.n_xyz_L.append(list(bound // (2**i)))
            # print(self.n_xyz_L[i])


    def _linearize_id(self, xyz,L=0):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """

        scale=1
        if L>=1:
            scale = 2**L
        n_xyz=self.n_xyz_L[L]
        
        return xyz[:, 2]//scale + n_xyz[-1] * (xyz[:, 1]//scale) + (n_xyz[-1] * n_xyz[-2]) * (xyz[:, 0]//scale)

    def _unlinearize_id(self, idx: torch.Tensor, L=0):
        """
        :param idx: (N, ) linearized id for access in self.indexer
        :return: xyz (N, 3) id to be indexed in 3D
        """
        n_xyz=self.n_xyz_L[L]
        layer_id=idx

        return torch.stack([layer_id // (n_xyz[1] * n_xyz[2]),
                            (layer_id // n_xyz[2]) % n_xyz[1],
                            layer_id % n_xyz[2]], dim=-1).int().view(-1,3)
    
    def compute_gt_sdf(self,points,normals,voxel_resolution,expand = False,xyz_num = 2000):
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size
        if expand:
            gathered_points_latent_idx = []
            for offset in self.integration_offsets:
                points_offset_voxel_xyz = torch.ceil(points_xyz_aligned + offset) - 1
                #不要出界
                for dim in range(3):
                    points_offset_voxel_xyz[:, dim].clamp_(0, self.n_xyz_L[0][dim] - 1)
                #points相对对应voxel的单位长度位置
                points_offset_voxel_id= self._linearize_id(points_offset_voxel_xyz).long()
                gathered_points_latent_idx.append(points_offset_voxel_id)
            gathered_points_latent_idx = torch.cat(gathered_points_latent_idx)
        else:
            gathered_points_latent_idx= self._linearize_id(torch.floor(points_xyz_aligned)).long()
        
        # gathered_points_latent_idx = torch.cat(gathered_points_latent_idx)
        unique_ids = torch.unique(gathered_points_latent_idx)
        unique_xyz = self._unlinearize_id(unique_ids,L=0).long()
        # inds = np.array([i for i in range(unique_xyz.size(0))],dtype = np.int64)
        # np.random.shuffle(inds)
        # inds = torch.from_numpy(inds)
        # end = min(xyz_num,unique_xyz.size(0))
        inds = torch.randperm(unique_xyz.size(0),device="cuda:0")
        end = min(xyz_num,unique_xyz.size(0))         
        unique_xyz = unique_xyz[inds,:][0:end,:]
        B = unique_xyz.size(0)
        # sample_a = (-(voxel_resolution // 2) * (1. / voxel_resolution)) 
        # sample_b = (1. + (voxel_resolution - 1) // 2 * (1. / voxel_resolution)) 
        # low_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b) - \
        #                 self.relative_network_offset
        sample_a = 0
        sample_b = 1 - 1 / voxel_resolution
        # sample_a = - 0.5
        # sample_b = 1.5 - 1 / voxel_resolution
        
        low_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b)
        low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)
        # print(unique_xyz.shape)
        low_center = unique_xyz.unsqueeze(1).repeat(1,voxel_resolution ** 3,1)
        # print(low_center.shape)
        pc = (low_center + low_samples).view(-1,3).contiguous()
        pc = pc * self.voxel_size + self.bound_min.unsqueeze(0)
        pc = torch.cat([pc,torch.zeros((pc.size(0),1)).cuda()],dim = 1).contiguous()
        gt_pc = torch.cat([points,torch.zeros((points.size(0),1)).cuda()],dim = 1).contiguous()
        del low_center,low_samples
        torch.cuda.empty_cache()
        # gt_sdf = compute_sdf(pc,gt_pc,normals,self.voxel_size,self.voxel_size * 2)
        gt_sdf = sdf_from_points(pc,gt_pc,normals,8,0.02)
        gt_sdf = gt_sdf / self.voxel_size
        gt_sdf = gt_sdf.view(B,voxel_resolution,voxel_resolution,voxel_resolution)
        return unique_xyz,gt_sdf

    def get_big_voxels(self,points,normals):
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size
        points_voxel_xyz = torch.floor(points_xyz_aligned).long()
        points_voxel_id = self._linearize_id(points_voxel_xyz,L = 0)
        points_voxel_ids, groups, group_num= torch.unique(points_voxel_id, return_inverse=True,
                                                                return_counts=True)
        points_group = []
        normals_group = []
        # print(points_voxel_ids.shape,groups.shape,group_num.shape)
        for i in range(group_num.size(0)):
            if group_num[i] > 10000:
                g_id = points_voxel_ids[i]
                mask = points_voxel_id == g_id
                # print(points[mask,:].shape)
                points_group.append(points[mask,:])
                normals_group.append(normals[mask,:])
        return points_group,normals_group

    def split_voxels(self,points):
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size
        points_voxel_xyz = torch.floor(points_xyz_aligned).long()
        points_voxel_id = self._linearize_id(points_voxel_xyz,L = 0)
        points_voxel_ids, groups, group_num= torch.unique(points_voxel_id, return_inverse=True,
                                                                return_counts=True)
        bound_group = []
        min_xyzs = self._unlinearize_id(points_voxel_ids).to(torch.float64)
        max_xyzs = min_xyzs + 1
        # print(points_voxel_ids.shape,groups.shape,group_num.shape)
        for i in range(group_num.size(0)):
            if group_num[i] > 50000:
                min_xyz = min_xyzs[i,:] * self.voxel_size + self.bound_min.unsqueeze(0)
                max_xyz = max_xyzs[i,:] * self.voxel_size + self.bound_min.unsqueeze(0)
                bound = o3d.geometry.AxisAlignedBoundingBox(min_xyz.cpu().permute(1,0).numpy(),max_xyz.cpu().permute(1,0).numpy())
                bound_group.append(bound)
        return bound_group
    
    def random_delete(self,points,normals,delete_num = 50,layer = 2):
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size
        points_voxel_xyz = torch.ceil(points_xyz_aligned).long()-1
        points_voxel_id = self._linearize_id(points_voxel_xyz,L = layer)
        points_voxel_ids, groups, group_num= torch.unique(points_voxel_id, return_inverse=True,
                                                                return_counts=True)
        d_id = []
        for i in range(delete_num):
            idx = random.randint(0,points_voxel_ids.size(0)-1)
            if idx not in d_id:
                d_id.append(idx)
        mask = torch.zeros_like(groups).bool()
        for id in d_id:
            mask = torch.logical_or(mask,groups == id)
        mask = ~mask

        # print(points.size(0))
        points_new = points[mask,:]
        # print(points_new.size(0))
        normals_new = normals[mask,:]
        return points_new,normals_new

    def get_structure(self,points,L,expand = False):
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size
        if expand:
            gathered_points_latent_idx = []
            # points_offset_voxel_xyz = torch.floor(points_xyz_aligned)
            for offset in self.integration_offsets:
                points_offset_voxel_xyz = torch.floor(points_xyz_aligned + offset)
                #不要出界
                for dim in range(3):
                    points_offset_voxel_xyz[:, dim].clamp_(0, self.n_xyz_L[0][dim] - 1)
                #points相对对应voxel的单位长度位置
                points_offset_voxel_id= self._linearize_id(points_offset_voxel_xyz).long()
                gathered_points_latent_idx.append(points_offset_voxel_id)    
            gathered_points_latent_idx = torch.cat(gathered_points_latent_idx)
            unique_ids = torch.unique(gathered_points_latent_idx)
        else:
            points_offset_voxel_xyz = torch.floor(points_xyz_aligned)
            points_offset_voxel_id = self._linearize_id(points_offset_voxel_xyz).long()
            unique_ids = torch.unique(points_offset_voxel_id)
        unique_xyz = self._unlinearize_id(unique_ids,L=0).long()
        strutures = []
        s = torch.zeros(self.n_xyz_L[0]).to(self.device).int()
        s[unique_xyz[:,0],unique_xyz[:,1],unique_xyz[:,2]] = 1
        strutures.append(s)
        for layer in range(1,L):
            layer_ids = self._linearize_id(unique_xyz,L=layer)
            layer_unique_ids = torch.unique(layer_ids)
            layer_unique_xyz = self._unlinearize_id(layer_unique_ids,L=layer).long()
            s = torch.zeros(self.n_xyz_L[layer]).to(self.device).int()
            s[layer_unique_xyz[:,0],layer_unique_xyz[:,1],layer_unique_xyz[:,2]] = 1
            strutures.append(s)
        return strutures
    
    def random_gt_sdf(self,points,normals,rand_num,expand = False,xyz_num = None):
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size
        if expand:
            gathered_points_latent_idx = []
            for offset in self.integration_offsets:
                points_offset_voxel_xyz = torch.ceil(points_xyz_aligned + offset) - 1
                #不要出界
                for dim in range(3):
                    points_offset_voxel_xyz[:, dim].clamp_(0, self.n_xyz_L[0][dim] - 1)
                #points相对对应voxel的单位长度位置
                points_offset_voxel_id= self._linearize_id(points_offset_voxel_xyz).long()
                gathered_points_latent_idx.append(points_offset_voxel_id)
            gathered_points_latent_idx = torch.cat(gathered_points_latent_idx)
        else:
            gathered_points_latent_idx= self._linearize_id(torch.floor(points_xyz_aligned)).long()
        
        # gathered_points_latent_idx = torch.cat(gathered_points_latent_idx)
        unique_ids = torch.unique(gathered_points_latent_idx)
        unique_xyz = self._unlinearize_id(unique_ids,L=0).long()
        if xyz_num is not None:
            inds = torch.randperm(unique_xyz.size(0),device="cuda:0")
            end = min(xyz_num,unique_xyz.size(0))        
            inds = inds[:end] 
            unique_xyz = unique_xyz[inds,:]
        low_samples = torch.rand(unique_xyz.size(0),rand_num,3).cuda()
        # print(unique_xyz.shape)
        low_center = unique_xyz.unsqueeze(1).repeat(1,rand_num,1)
        # print(low_center.shape)
        pc = (low_center + low_samples).view(-1,3).contiguous()
        pc = pc * self.voxel_size + self.bound_min.unsqueeze(0)
        gt_sdf = sdf_from_points(pc,points,normals,8,0.02)
        gt_sdf = gt_sdf / self.voxel_size
        gt_sdf = gt_sdf.view(-1)
        return pc, gt_sdf
        
def splitScene(data_path,scan_name):
    pcd_path = os.path.join(data_path+scan_name,scan_name+"_mesh_pc_clean.ply")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points=torch.from_numpy(np.asarray(pcd.points)).float().cuda()
    normals=torch.from_numpy(np.asarray(pcd.normals)).float().cuda()
    octree = scene_cube(layer_num = 5,voxel_size = 0.1,device = "cuda:0")
    return octree.get_big_voxels(points,normals)
    