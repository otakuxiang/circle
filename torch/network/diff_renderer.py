import math
from IPython.core.display import Latex
import torch
import torch.nn as nn
from system.ext import sparse_ray_intersection,generate_rays
from torch.autograd import Function
import numpy as np
import utils.net_util as net_util
import time
class diff_renderer:
    def __init__(self,model,img_h : int,img_w : int,
            voxel_size,main_device,march_step = 50,
            ray_marching_ratio=1., threshold=1e-3,is_eval = True,mode="idr"):
        self.ray_marching_ratio = ray_marching_ratio
        self.march_step = march_step
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.is_eval = is_eval
        self.threshold = threshold
        self.voxel_size = voxel_size
        self.device = main_device
        self.model = model
        self.is_eval = is_eval
        self.mode = mode
    def set_voxel_latents(self,voxel_latents):
        self.voxel_latents = voxel_latents

    def set_camera(self,fx,fy,cx,cy):
        self.camera_ins = [fx,fy,cx,cy]
        self.directions , self.ray_xys = self.get_rays()

    def set_bound_min(self,bound_min):
        self.bound_min = bound_min

    def set_octree(self,octree):
        self.octree = octree

    def get_rays(self):
        fx = self.camera_ins[0]
        fy = self.camera_ins[1]
        cx = self.camera_ins[2]
        cy = self.camera_ins[3]
        directions,ray_xys = generate_rays([self.img_h,self.img_w],
            fx,fy,cx,cy    
        )
        return directions,ray_xys

    

    def compute_final_pairs(self,ray_voxel_pair,ray_voxel_d,rv_pointer,
            ray_ori_align,directions,voxel_id_to_xyz):
        vaild_rays = rv_pointer[:,0] != -1

        current_pair_id = rv_pointer[vaild_rays,0]
        marching_steps = torch.zeros(current_pair_id.size(0)).to(self.device)
        init_d = ray_voxel_d.clone()
        current_ds = ray_voxel_d[current_pair_id].unsqueeze(-1)
        hitted_pairs_id = []

        
        # run until all pair converge
        while current_pair_id.size(0) != 0:
            # for each ray get query point

            ray_ids = ray_voxel_pair[current_pair_id,1]
            ray_ori = ray_ori_align[ray_ids,:]
            voxel_ids = ray_voxel_pair[current_pair_id,0]
            points = ray_ori + current_ds * directions[ray_ids]
            voxels_xyz = voxel_id_to_xyz[voxel_ids]
            # judge if the points out of voxel
            is_out = torch.any((points - voxels_xyz).abs() > (1.0 + 1e-5) , dim = 1)
            # if points out of voxel then into next ray_voxel_pair and update ray_ids voxel_ids
            if torch.any(is_out):

                current_pair_id[is_out] = current_pair_id[is_out] + 1
                not_over = current_pair_id != rv_pointer[ray_ids,1]

                current_ds = current_ds[not_over]
                is_out = is_out[not_over]
                current_pair_id = current_pair_id[not_over]
                marching_steps = marching_steps[not_over]
                current_ds[is_out] = ray_voxel_d[current_pair_id][is_out].unsqueeze(-1)
                
                ray_ids = ray_voxel_pair[current_pair_id,1]
                ray_ori = ray_ori_align[ray_ids,:]
                voxel_ids = ray_voxel_pair[current_pair_id,0]
                points = ray_ori + current_ds * directions[ray_ids]
                voxels_xyz = voxel_id_to_xyz[voxel_ids]
            if current_pair_id.size(0) == 0:
                break
            # if ray overhit ,then finish it, update all tensors
            
            # compute sdf
            points_relative = points - voxels_xyz
            latent_input,_ = self.octree.trilinear_interpolate(points,voxels_xyz)
                        
            sdfs = net_util.forward_model(self.model.decoder,max_sample = 2 ** 25,
                latent_input = latent_input,
                xyz_input = points_relative)[0]
            
            # update current_ds

            sdfs = torch.clamp(sdfs,-0.2,0.2)
            is_hit = sdfs.view(-1).abs() <= self.threshold

            # update all tensors
            if torch.any(is_hit):
                hitted_pairs_id.append(current_pair_id[is_hit])
                points = points[is_hit]
                voxels_xyz = voxels_xyz[is_hit]
                


                not_over = ~is_hit
                current_pair_id = current_pair_id[not_over]
                marching_steps = marching_steps[not_over]
                current_ds = current_ds[not_over,:]
                sdfs = sdfs[not_over,:]
                


            if current_pair_id.size(0) == 0:
                break
            # step in

            current_ds = current_ds + self.ray_marching_ratio * sdfs
            marching_steps = marching_steps + 1
            # judge if the step is over
            is_end = marching_steps>=self.march_step
            if torch.any(is_end):
                not_over = ~is_end
                current_pair_id = current_pair_id[not_over]
                marching_steps = marching_steps[not_over]
                current_ds = current_ds[not_over,:]
            
        if len(hitted_pairs_id) == 0:
            return None,None
        if hitted_pairs_id[0].size(0) == 0:
            return None,None
        hitted_pairs_id = torch.cat(hitted_pairs_id,dim = 0)

        hitted_ray_voxel_pair = ray_voxel_pair[hitted_pairs_id,:]
        hitted_ray_voxel_d = init_d[hitted_pairs_id]

        return hitted_ray_voxel_pair,hitted_ray_voxel_d

    def compute_intersect(self,ray_voxel_pair,ray_voxel_d,rv_pointer,
            ray_ori_align,directions,voxel_id_to_xyz):
        vaild_rays = rv_pointer[:,0] != -1

        current_pair_id = rv_pointer[vaild_rays,0]
        marching_steps = torch.zeros(current_pair_id.size(0)).to(self.device)
        current_ds = ray_voxel_d[current_pair_id].unsqueeze(-1)
        hitted_pairs_id = []
        ray_d = []


        # run until all pair converge
        while current_pair_id.size(0) != 0:
            # for each ray get query point

            ray_ids = ray_voxel_pair[current_pair_id,1]
            ray_ori = ray_ori_align[ray_ids,:]
            voxel_ids = ray_voxel_pair[current_pair_id,0]
            points = ray_ori + current_ds * directions[ray_ids]
            voxels_xyz = voxel_id_to_xyz[voxel_ids]
            # judge if the points out of voxel
            is_out = torch.any((points - voxels_xyz).abs() > (1.0 + 1e-5) , dim = 1)
            # if points out of voxel then into next ray_voxel_pair and update ray_ids voxel_ids
            if torch.any(is_out):

                current_pair_id[is_out] = current_pair_id[is_out] + 1

                not_over = current_pair_id != rv_pointer[ray_ids,1]
                
                

                current_ds = current_ds[not_over]
                is_out = is_out[not_over]
                current_pair_id = current_pair_id[not_over]
                marching_steps = marching_steps[not_over]
                current_ds[is_out] = ray_voxel_d[current_pair_id][is_out].unsqueeze(-1)
                

                ray_ids = ray_voxel_pair[current_pair_id,1]
                ray_ori = ray_ori_align[ray_ids,:]
                voxel_ids = ray_voxel_pair[current_pair_id,0]
                points = ray_ori + current_ds * directions[ray_ids]
                voxels_xyz = voxel_id_to_xyz[voxel_ids]
            if current_pair_id.size(0) == 0:
                break
            # if ray overhit ,then finish it, update all tensors
            
            
            # compute sdf
            points_relative = points - voxels_xyz
            latent_input,_ = self.octree.trilinear_interpolate(points,voxels_xyz)
                        
            sdfs = net_util.forward_model(self.model.decoder,max_sample = 2 ** 25,
                latent_input = latent_input,
                xyz_input = points_relative)[0]
            
            # update current_ds

            is_hit = sdfs.view(-1).abs() <= self.threshold

            # update all tensors
            if torch.any(is_hit):
                hitted_pairs_id.append(current_pair_id[is_hit])
                result_d = current_ds[is_hit] + (1 - self.ray_marching_ratio) * sdfs[is_hit]
                ray_d.append(result_d)

                not_over = ~is_hit
                current_pair_id = current_pair_id[not_over]
                marching_steps = marching_steps[not_over]
                current_ds = current_ds[not_over,:]
                sdfs = sdfs[not_over,:]
                


            if current_pair_id.size(0) == 0:
                break
            # step in

            current_ds = current_ds + self.ray_marching_ratio * sdfs
            marching_steps = marching_steps + 1
            # judge if the step is over
            is_end = marching_steps>=self.march_step
            if torch.any(is_end):
                not_over = ~is_end
                current_pair_id = current_pair_id[not_over]
                marching_steps = marching_steps[not_over]
                current_ds = current_ds[not_over,:]
                

        if len(hitted_pairs_id) == 0:
            return None,None
        if hitted_pairs_id[0].size(0) == 0:
            return None,None
        hitted_pairs_id = torch.cat(hitted_pairs_id,dim = 0)
        ray_d = torch.cat(ray_d,dim = 0)

        hitted_ray_voxel_pair = ray_voxel_pair[hitted_pairs_id,:]
        return hitted_ray_voxel_pair,ray_d



    def diff_intersection(self,ray_voxel_pair,ray_voxel_d,ray_ori_align,voxel_id_to_xyz,directions):
        ray_d = ray_voxel_d
        voxel_ids = ray_voxel_pair[:,0]
        ray_id = ray_voxel_pair[:,1]
        ray_ori = ray_ori_align[ray_id,:]


        points = ray_ori + ray_d * directions[ray_id]
        voxels_xyz = voxel_id_to_xyz[voxel_ids]
        points_relative_0 = points.detach() - voxels_xyz
        points_relative_0.requires_grad_()

        latent_input,_ = self.octree.trilinear_interpolate(voxels_xyz + points_relative_0,voxels_xyz)

        sdf_detach = net_util.forward_model(self.model.decoder,max_sample = 2 ** 23,
            latent_input = latent_input,no_detach=True,
            xyz_input = points_relative_0)[0]
        surface_grad = torch.autograd.grad(sdf_detach, [points_relative_0], grad_outputs=torch.ones_like(sdf_detach),
                                        retain_graph=True, create_graph=True)[0]
        
        points_relative = points - voxels_xyz
        latent_input,_ = self.octree.trilinear_interpolate(points,voxels_xyz)
        sdf_no_detach = net_util.forward_model(self.model.decoder,max_sample = 2 ** 23,
            latent_input = latent_input,no_detach=True,
            xyz_input = points_relative)[0]
        directions_0 = directions[ray_id].detach()
        
        surface_points_dot = torch.bmm(surface_grad.view(-1,1,3),directions_0.view(-1,3,1)).squeeze(-1)
        surface_points_dot[torch.logical_and((surface_points_dot >= 0),(surface_points_dot < 1e-7))] = 1e-7
        surface_points_dot[torch.logical_and((surface_points_dot < 0),(surface_points_dot > -1e-7))] = -1e-7
        dist = ray_d - (sdf_no_detach - sdf_detach.detach()) / surface_points_dot.detach()
        return dist,ray_id

    # the input is hitted ray_voxel_pair
    def ray_marching(self,ray_voxel_pair,ray_voxel_d,ray_ori_align,voxel_id_to_xyz,directions):
        ray_d = ray_voxel_d.unsqueeze(-1)
        unfinished_mask = torch.ones_like(ray_d.view(-1)).bool()
        all_sdfs = torch.ones_like(ray_d)

        for i in range(self.march_step):
            voxel_ids = ray_voxel_pair[unfinished_mask,0]
            ray_id = ray_voxel_pair[unfinished_mask,1]
            ray_ori = ray_ori_align[ray_id,:]

            current_d = ray_d[unfinished_mask,:]
            if ray_id.size(0) == 0:
                break
            points = ray_ori + current_d * directions[ray_id]
            voxels_xyz = voxel_id_to_xyz[voxel_ids]
            points_relative = points - voxels_xyz
            latent_input,_ = self.octree.trilinear_interpolate(points,voxels_xyz)
            sdfs = net_util.forward_model(self.model.decoder,max_sample = 2 ** 20,
                latent_input = latent_input,no_detach=True,
                xyz_input = points_relative)[0]
            
            sdf_mask = (sdfs.abs() > self.threshold).squeeze(-1)
            all_sdfs[unfinished_mask,:] = sdfs
            unfinished_mask = all_sdfs[:,0].abs() > self.threshold

            ray_d[unfinished_mask,:] = ray_d[unfinished_mask,:] + sdfs[sdf_mask,:]
                        
            
        ray_id = ray_voxel_pair[~unfinished_mask,1]
        ray_d = ray_d[~unfinished_mask]
        
        return ray_d,ray_id

    

    def render_depth(self,ray_xys,ray_d,ray_ids):
        fx = self.camera_ins[0]
        fy = self.camera_ins[1]
        cx = self.camera_ins[2]
        cy = self.camera_ins[3]
        ray_d = (ray_d * self.voxel_size).view(-1)
        u = ray_xys[ray_ids,0]
        v = ray_xys[ray_ids,1]
        ray_depth = ray_d / (
            torch.sqrt( 1 + \
                ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2
            )
        )
        depth = torch.zeros((self.img_h,self.img_w)).to(self.device)
        
        depth[v,u] = ray_depth
        return depth


    def render(self,octree_xyzs,octree_sons,pose):

        R,t = pose
        ray_od = torch.matmul(R,self.directions.unsqueeze(-1)).squeeze(-1)
        ray_ori = t.repeat(ray_od.size(0),1)
        ray_od = torch.cat([ray_od.unsqueeze(1),ray_ori.unsqueeze(1)],dim = 1)  # N x 2 x 3
        if self.is_eval:
            with torch.no_grad():
                directions = ray_od[:,0,:].squeeze(1).contiguous()
                directions = directions / directions.norm(dim = 1,keepdim=True)
                ray_ori_align = (ray_od[:,1,:] / self.voxel_size)
                directions_inv = 1.0 / directions
                rv_pointer = torch.ones((directions.size(0),2)).to(self.device).long() * -1
                ray_voxel_pair,ray_voxel_d,rv_pointer = sparse_ray_intersection(
                    octree_xyzs,octree_sons,ray_ori_align,
                    directions,directions_inv,rv_pointer
                )


                hitted_ray_voxel_pair,hitted_ray_voxel_d = self.compute_final_pairs(ray_voxel_pair,ray_voxel_d,rv_pointer,
                    ray_ori_align,directions,octree_xyzs[-1])

                ray_d,ray_ids = self.ray_marching(hitted_ray_voxel_pair,hitted_ray_voxel_d,ray_ori_align,
                    octree_xyzs[-1],directions)

                
                depth_img = self.render_depth(self.ray_xys,ray_d,ray_ids)
        else:
            directions = ray_od[:,0,:].squeeze(1).contiguous()
            directions = directions / directions.norm(dim = 1,keepdim=True)
            ray_ori_align = (ray_od[:,1,:] / self.voxel_size)
            directions_inv = 1.0 / directions
            rv_pointer = torch.ones((directions.size(0),2)).to(self.device).long() * -1
                
            with torch.no_grad():

                ray_voxel_pair,ray_voxel_d,rv_pointer = sparse_ray_intersection(
                    octree_xyzs,octree_sons,ray_ori_align,
                    directions,directions_inv,rv_pointer
                )

                hitted_ray_voxel_pair,hitted_ray_voxel_d = self.compute_final_pairs(ray_voxel_pair,ray_voxel_d,rv_pointer,
                    ray_ori_align,directions,octree_xyzs[-1])

            ray_d,ray_ids = self.ray_marching(hitted_ray_voxel_pair,hitted_ray_voxel_d,ray_ori_align,
                octree_xyzs[-1],directions)
            depth_img = self.render_depth(self.ray_xys,ray_d,ray_ids)

        return depth_img
    
    def train_render(self,octree_xyzs,octree_sons,ray_od):
        directions = ray_od[:,0,:].squeeze(1).contiguous()

        ray_ori_align = (ray_od[:,1,:] / self.voxel_size)
        directions_inv = 1.0 / directions
        rv_pointer = torch.ones((directions.size(0),2)).to(self.device).long() * -1
            
        with torch.no_grad():
            ray_voxel_pair,ray_voxel_d,rv_pointer = sparse_ray_intersection(
                octree_xyzs,octree_sons,ray_ori_align,
                directions,directions_inv,rv_pointer
            )
            if len(ray_voxel_pair.shape) == 1:
                return None
            if self.mode == "dist":
                hitted_ray_voxel_pair,hitted_ray_voxel_d = self.compute_final_pairs(ray_voxel_pair,ray_voxel_d,rv_pointer,
                    ray_ori_align,directions,octree_xyzs[-1])
            if self.mode == "idr":
                hitted_ray_voxel_pair,hitted_ray_voxel_d = self.compute_intersect(ray_voxel_pair,ray_voxel_d,rv_pointer,
                    ray_ori_align,directions,octree_xyzs[-1])
            
            if hitted_ray_voxel_d is None:
                print("None!")
                return None

        if self.mode == "dist":
            ray_d,ray_ids = self.ray_marching(hitted_ray_voxel_pair,hitted_ray_voxel_d,ray_ori_align,
                octree_xyzs[-1],directions)
        if self.mode == "idr":
            ray_d,ray_ids = self.diff_intersection(hitted_ray_voxel_pair,hitted_ray_voxel_d,ray_ori_align,
                octree_xyzs[-1],directions)
        ray_d = (ray_d * self.voxel_size).view(-1)
        depth_img = torch.zeros((ray_od.size(0),),device = self.device) 
        depth_img[ray_ids] = ray_d
        return depth_img

        