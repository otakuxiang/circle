
import utils.net_util as net_util
from utils.data_util import gradient_clamp,clamp_select
import numpy as np
import torch
import logging
import system.ext
import os
import math
import sys,pdb,traceback
import threading
from system.ext import set_empty_voxels,build_octree
import open3d as o3d
import time
import matplotlib
from sklearn.decomposition import PCA
import torch.nn.functional as F



class MeshExtractCache:
    def __init__(self, device):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.triangles = None
        self.updated_vec_id = None
        self.device = device
        self.colors = None
        self.clear_updated_vec()

    def clear_updated_vec(self):
        self.updated_vec_id = torch.empty((0, ), device=self.device, dtype=torch.long)

    def clear_all(self):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.updated_vec_id = None
        self.clear_updated_vec()

class tensor_vector():
    def __init__(self,device,latent_dim,tt_size,if_reverse = False,if_conf = False):
        self.latent_dim = latent_dim
        self.device = device
        self.lock = threading.Lock()
        self.indexer = torch.ones(tt_size, device=device, dtype=torch.long) * -1
        self.latent_vecs=torch.empty((1, self.latent_dim), dtype=torch.float32, device=device)
        self.latent_vecs.zero_()
        self.reverse = if_reverse
        self.occupied_voxel_num = 0
        self.if_conf = if_conf
        if if_reverse:
            self.latent_vecs_id=torch.ones((1, ), dtype=torch.long, device=device) * -1
        if if_conf:
            self.voxel_confs = torch.zeros((1, ), dtype=torch.float32, device=device)
    def _inflate_latent_buffer(self,count):
        target_n_occupied = self.occupied_voxel_num + count
        # print(target_n_occupied,self.latent_vecs.size(0))
        if self.latent_vecs.size(0) < target_n_occupied:
            new_size = self.latent_vecs.size(0)
            while new_size < target_n_occupied:
                new_size *= 2
            new_vec = torch.empty((new_size, self.latent_dim), dtype=torch.float32, device=self.device)
            new_vec[:self.latent_vecs.size(0)] = self.latent_vecs
            new_vec[self.latent_vecs.size(0):].zero_()
            
            if self.reverse:
                new_vec_pos = torch.ones((new_size, ), dtype=torch.long, device=self.device) * -1
                new_vec_pos[:self.latent_vecs.size(0)] = self.latent_vecs_id
                self.latent_vecs_id = new_vec_pos
            if self.if_conf:
                new_voxel_confs = torch.zeros((new_size, ), dtype=torch.float32, device=self.device)
                new_voxel_confs[:self.latent_vecs.size(0)] = self.voxel_confs
                self.voxel_confs = new_voxel_confs
            self.latent_vecs = new_vec

        new_inds = torch.arange(self.occupied_voxel_num, target_n_occupied, device=self.device, dtype=torch.long)
        self.occupied_voxel_num = target_n_occupied
        return new_inds

    def size(self):
        return self.occupied_voxel_num

    def allocated_mask(self,ids):
        self.lock.acquire()
        mask = self.indexer[ids] == -1
        self.lock.release()
        return mask
    def allocated_mask_inv(self,ids):
        self.lock.acquire()
        mask = self.indexer[ids] != -1
        self.lock.release()
        return mask

    def allocate_block(self,ids):
        self.lock.acquire()
        # print(torch.max(ids),self.indexer.size(0))
        mask = self.indexer[ids] == -1
        idx = ids[mask]
        idx = torch.unique(idx)
        if idx.size(0) == 0:
            self.lock.release()
            return 
        new_id = self._inflate_latent_buffer(idx.size(0))
        if self.reverse:
            self.latent_vecs_id[new_id] = idx
        self.indexer[idx] = new_id
        self.lock.release()

    def __setitem__(self,index,value):
        self.lock.acquire()
        self.latent_vecs[index] = value
        self.lock.release()

    def __getitem__(self,index):
        return self.latent_vecs[index]

    def set_confidence(self,index,value):
        self.lock.acquire()
        self.voxel_confs[index] = value
        self.lock.release()

    def get_confidence(self,index):
        self.lock.acquire()
        v = self.voxel_confs[index]
        self.lock.release()
        return v
    
    def get_idx(self,ids):
        self.lock.acquire()
        idx = self.indexer[ids]
        self.lock.release()
        return idx

    def get_reverse_id(self,idx):
        self.lock.acquire()
        ids = self.latent_vecs_id[idx]
        self.lock.release()
        return ids

    def detach(self):
        self.indexer = self.indexer.detach()
        self.latent_vecs = self.latent_vecs.detach()

    def print_memory(self):
        print(f"indexer :{self.indexer.numel() * 8 / 1024 / 1024} M")
        print(f"latent_vecs :{self.latent_vecs.numel() * 4 / 1024 / 1024} M")
        if self.reverse:
            print(f"latent_vecs_id :{self.latent_vecs_id.numel() * 8 / 1024 / 1024} M ")
        if self.if_conf:
            print(f"voxel_confs :{self.voxel_confs.numel() * 4 / 1024 / 1024} M ")
    
    def requires_grad_(self):
        self.latent_vecs.requires_grad_()

    def reset(self):
        self.occupied_voxel_num = 0
        self.indexer = torch.ones(self.indexer.size(0), device=self.device, dtype=torch.long) * -1
        self.latent_vecs.zero_()
        if self.reverse:
            self.latent_vecs_id=torch.ones((self.latent_vecs_id.size(0), ), dtype=torch.long, device=self.device) * -1
        if self.if_conf:
            self.voxel_confs = torch.zeros((self.voxel_confs.size(0), ), dtype=torch.float32, device=self.device)


class BoundError(Exception):
    def __init__(self):
        super().__init__()



class unet_oct_cube():
    def __init__(self,model:net_util.Networks,device: torch.device,latent_dim,renderer,layer_num = 4,voxel_size = 0.1,bound = None,lowest = 7):
        #cm
        self.model = model
        self.model.encoder.to(device)
        self.model.decoder.to(device)
        self.model.conv_kernels.to(device)
        if not self.model.rgb_decoder is None:
            self.model.rgb_decoder.to(device)
        
        # self.model.eval()
        self.layer_num=layer_num
        if bound is None:
            bound = [2 ** lowest,2 ** lowest,2 ** lowest]   
            bound = np.array(bound)     
            self.lowest = lowest
        else:
            bound = (np.array(bound + voxel_size) / voxel_size // 32 + 1).astype(np.int32) * 32
            self.lowest = 5
            

        self.n_xyz_L=[]
        self.n_xyz_L.append(bound)
        self.corner_xyz = bound + 1
        
        self.model.conv_kernels.set_input_dim(bound.tolist())
        
        self.latent_vecs_left = tensor_vector(device,self.model.encoder.latent_size,np.product(self.n_xyz_L[0]),True,True)   

        self.latent_vecs_right_corner = tensor_vector(device,self.model.encoder.latent_size,np.product(self.corner_xyz),True,True)        
        self.latent_vecs_right_corner_rgb = tensor_vector(device,self.model.encoder.latent_size,np.product(self.corner_xyz),True,True)        
        
        self.occ_right_flag = torch.zeros(np.product(self.n_xyz_L[0]),device=device).bool()

        self.voxel_size=voxel_size
        self.bound_min = torch.tensor([0,0,0], device=device).float()
        self.bound_max = torch.tensor([self.n_xyz_L[0][0] * voxel_size,self.n_xyz_L[0][1] * voxel_size,self.n_xyz_L[0][2] * voxel_size], device=device).float()
        self.latent_dim = latent_dim
        self.dense_thresold=5
        self.igonore_thresold=5
        self.integrate_require_thresold=10000
        self.trustworth_thresold=10000
        self.device = device
        # self.xyz = torch.ones(self.n_xyz_L[0]).cuda()
        
        self.integration_offsets = [torch.tensor(t, device=self.device, dtype=torch.float32) for t in [
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [ 0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
        ]]
        self.mesh_update_affected = [torch.tensor([t], device=self.device)
                                     for t in [[-1, 0, 0], [1, 0, 0],
                                               [0, -1, 0], [0, 1, 0],
                                               [0, 0, -1], [0, 0, 1]]]
        self.relative_network_offset = torch.tensor([[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float32)
        self.meshing_thread = None
        self.extract_mesh_std_range = None
        self.meshing_thread_id = -1
        self.meshing_stream = torch.cuda.Stream()
        self.modifying_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.mesh_cache = MeshExtractCache(self.device)
        self.used_latent_vecs_id=None
        self.offset = torch.tensor([
            [0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]
        ],dtype = torch.int32,device = device)
        self.optimize_stream = torch.cuda.Stream()
        
        self.is_busy = False
        self.now_layer = 0
        self.is_batch = False
        self.batch_size = 16
        self.renderer = renderer
        if renderer != None:
            self.renderer.set_octree(self)
        
        

    STATUS_CONF_BIT = 1 << 0

    def set_bound(self,bound):
        bound += self.voxel_size 
        self.model.conv_kernels.set_input_dim(bound.tolist())
        bound = (bound / self.voxel_size // 32 + 1).astype(np.int32) * 32
        self.n_xyz_L=[]
        self.n_xyz_L.append(bound)
        for i in range(1,self.layer_num):
            self.n_xyz_L.append(bound // (2**i))


    def _linearize_id(self, xyz,L=0,dim = None):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """
        scale=1
        if L>=1:
            scale = 2**L
        
        if dim is not None:
            n_xyz = dim
        else:
            n_xyz=self.n_xyz_L[L]
        return xyz[:, 2]//scale + n_xyz[-1] * (xyz[:, 1]//scale) + (n_xyz[-1] * n_xyz[-2]) * (xyz[:, 0]//scale)

    def _unlinearize_id(self, idx: torch.Tensor, L=0,dim = None):
        """ 
        :param idx: (N, ) linearized id for access in self.indexer
        :return: xyz (N, 3) id to be indexed in 3D
        """
        
        if dim is not None:
            n_xyz = dim
        else:
            n_xyz=self.n_xyz_L[L]
        scale=1
        layer_id=idx
        if L>=1:
            scale = 2**L
        return torch.stack([layer_id // (n_xyz[1] * n_xyz[2])*scale,
                            (layer_id // n_xyz[2]) % n_xyz[1]*scale,
                            layer_id % n_xyz[2]*scale], dim=-1).int().view(-1,3)
    def _lower_voxel_xyz(self, xyz,L=0):
        """
        :param xyz (N, 3) long id
        :return: (N*8,3) lineraized id to be accessed in self.indexer
        """
        
        if L>=1:
            scale = 2**L
            scale_lower=2**(L-1)
            son = xyz.unsqueeze(1).repeat(1,8,1) + self.offset * scale_lower
            return son.reshape((-1,3))

    
    def _mark_updated_vec_id(self, new_vec_id: torch.Tensor):
        """
        :param new_vec_id: (B,) updated id (indexed in latent vectors)
        """
        self.mesh_cache.updated_vec_id = torch.cat([self.mesh_cache.updated_vec_id, new_vec_id])
        self.mesh_cache.updated_vec_id = torch.unique(self.mesh_cache.updated_vec_id)
    
    def _make_mesh_from_cache(self):
        # print("make mesh")
        vertices = self.mesh_cache.vertices
        triangles = self.mesh_cache.triangles
        # print("reshape")

        final_mesh = o3d.geometry.TriangleMesh()
        # The pre-conversion is saving tons of time
        final_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(float))
        # print("v ok")
        
        final_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        # print("copy")
        
        if not self.mesh_cache.colors is None:
            final_mesh.vertex_colors = o3d.utility.Vector3dVector(self.mesh_cache.colors)
        # Assign color:
        # if vertices.shape[0] > 0:
        #     import matplotlib.cm
        #     vert_color = self.mesh_cache.vertices_std.reshape((-1, )).astype(float)
        #     if self.extract_mesh_std_range is not None:
        #         vcolor_min, vcolor_max = self.extract_mesh_std_range
        #         vert_color = np.clip(vert_color, vcolor_min, vcolor_max)
        #     else:
        #         vcolor_min, vcolor_max = vert_color.min(), vert_color.max()
        #     vert_color = (vert_color - vcolor_min) / (vcolor_max - vcolor_min)
        #     vert_color = matplotlib.cm.jet(vert_color)[:, :3]
        #     final_mesh.vertex_colors = o3d.utility.Vector3dVector(vert_color)
        #     # print(f"Vert range {vcolor_min} to {vcolor_max}.")

        return final_mesh

    def trilinear_interpolate(self,xyz,base_xyz = None,detach = False,if_normal = False):
        latent_vecs = self.latent_vecs_right_corner
        # if detach:
        #     latent_vecs = latent_vecs.detach()
        mask = torch.ones((xyz.size(0)),dtype = torch.bool,device= self.device)
        
        with torch.no_grad():
            corner_index = []
            corner_xyz = []
            if base_xyz is None:
                base_xyz = torch.floor(xyz)

            # print(base_xyz)
            # get 8 corner of xyz
            for offset in self.offset:
                points_offset_voxel_xyz = base_xyz + offset
                # print(self.n_xyz_L[0])

                #不要出界
                
                points_offset_voxel_id= self._linearize_id(points_offset_voxel_xyz,dim = self.corner_xyz).long()
                points_voxel_id_valid = latent_vecs.indexer[points_offset_voxel_id] != -1
                mask = torch.logical_and(mask,points_voxel_id_valid)
                points_offset_voxel_xyz = points_offset_voxel_xyz.unsqueeze(-1)
                corner_index.append(latent_vecs.indexer[points_offset_voxel_id].unsqueeze(-1))
                corner_xyz.append(points_offset_voxel_xyz)

        corner_xyz = torch.cat(corner_xyz,dim = -1)
        corner_index = torch.cat(corner_index,dim = -1).long()
        if if_normal:
            mask = mask.view(-1,6)
            n_mask = torch.all(mask,dim=1)
            mask[~n_mask,:] = torch.logical_and(mask[~n_mask,:],torch.zeros(mask[~n_mask,:].shape,device=self.device,dtype=torch.bool))
            mask = mask.view(-1)
        corner_xyz = corner_xyz[mask,:,:]
        corner_index = corner_index[mask,:]
        xyz = xyz[mask,:]
        # print(xyz.shape)
        # tmp = tmp[tmp!=1]
        
        
        # internel is 1 so ignore /
        bb = xyz.size(0) // 10000000 + 1
        inter = xyz.size(0) // bb
        
        # print(self.latent_vecs_right_corner[corner_index[:,0],:].shape,xd.shape)
        # l(x0,y0,z0) * (1 - xd) + l(x1,y0,z0) * xd
        c = []
        for i in range(bb):
            start = i * inter
            end = (i + 1) * inter if i != (bb - 1) else xyz.size(0)
            # print(start,end)
            xd = (xyz[start:end,0] - corner_xyz[start:end,0,0]) 
            yd = (xyz[start:end,1] - corner_xyz[start:end,1,0]) 
            zd = (xyz[start:end,2] - corner_xyz[start:end,2,0]) 
            xd = xd.view(-1,1)
            yd = yd.view(-1,1)
            zd = zd.view(-1,1)
            c_b = (1 - xd) * (1 - yd) * (1 - zd) * latent_vecs[corner_index[start:end,0],:]
            c_b = c_b +  xd * (1 - yd) * (1 - zd) * latent_vecs[corner_index[start:end,4],:]
            c_b = c_b +  (1 - xd) * (1 - yd) * (zd) * latent_vecs[corner_index[start:end,1],:]
            c_b = c_b +  xd * (1 - yd) * (zd) * latent_vecs[corner_index[start:end,5],:]
            c_b = c_b +  (1 - xd) * (yd) * (1 - zd) * latent_vecs[corner_index[start:end,2],:]
            c_b = c_b +  xd * (yd) * (1 - zd) * latent_vecs[corner_index[start:end,6],:]
            c_b = c_b +  (1 - xd) * (yd) * (zd) * latent_vecs[corner_index[start:end,3],:]
            c_b = c_b +  xd * (yd) * (zd) * latent_vecs[corner_index[start:end,7],:]
            c.append(c_b)
        c = torch.cat(c,dim = 0)
        return c,mask

    def trilinear_interpolate_rgb_and_geo(self,xyz,base_xyz = None,rgb = False):
        latent_vecs = self.latent_vecs_right_corner
        latent_vecs_rgb = self.latent_vecs_right_corner_rgb

        mask = torch.ones((xyz.size(0)),dtype = torch.bool,device= self.device)
        corner_index = []
        corner_xyz = []
        if base_xyz is None:
            base_xyz = torch.floor(xyz)

        # print(base_xyz)
        # get 8 corner of xyz
        for offset in self.offset:
            points_offset_voxel_xyz = base_xyz + offset
            # print(self.n_xyz_L[0])

            #不要出界
            
            points_offset_voxel_id= self._linearize_id(points_offset_voxel_xyz,dim = self.corner_xyz).long()
            points_voxel_id_valid = latent_vecs.indexer[points_offset_voxel_id] != -1
            mask = torch.logical_and(mask,points_voxel_id_valid)
            points_offset_voxel_xyz = points_offset_voxel_xyz.unsqueeze(-1)
            corner_index.append(latent_vecs.indexer[points_offset_voxel_id].unsqueeze(-1))
            corner_xyz.append(points_offset_voxel_xyz)

        corner_xyz = torch.cat(corner_xyz,dim = -1)
        corner_index = torch.cat(corner_index,dim = -1).long()
        corner_xyz = corner_xyz[mask,:,:]
        corner_index = corner_index[mask,:]
        xyz = xyz[mask,:]
        # print(xyz.shape)
        # tmp = tmp[tmp!=1]
        
    
        bb = xyz.size(0) // 5000000 + 1
        inter = xyz.size(0) // bb

        # l(x0,y0,z0) * (1 - xd) + l(x1,y0,z0) * xd
        c = []
        c_rgb = []
        for i in range(bb):
            start = i * inter
            end = (i + 1) * inter if i != (bb - 1) else xyz.size(0)
            # print(start,end)
            xd = (xyz[start:end,0] - corner_xyz[start:end,0,0]) 
            yd = (xyz[start:end,1] - corner_xyz[start:end,1,0]) 
            zd = (xyz[start:end,2] - corner_xyz[start:end,2,0]) 
            xd = xd.view(-1,1)
            yd = yd.view(-1,1)
            zd = zd.view(-1,1)

            c00 = latent_vecs[corner_index[start:end,0],:] * (1 - xd) +\
                latent_vecs[corner_index[start:end,4],:] * xd
            c01 = latent_vecs[corner_index[start:end,1],:] * (1 - xd) +\
                latent_vecs[corner_index[start:end,5],:] * xd
            c10 = latent_vecs[corner_index[start:end,2],:] * (1 - xd) +\
                latent_vecs[corner_index[start:end,6],:] * xd
            c11 = latent_vecs[corner_index[start:end,3],:] * (1 - xd) +\
                latent_vecs[corner_index[start:end,7],:] * xd
            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd
            c_b = c0 * (1 - zd) + c1 * zd

            c00_r = latent_vecs_rgb[corner_index[start:end,0],:] * (1 - xd) +\
                latent_vecs_rgb[corner_index[start:end,4],:] * xd
            c01_r = latent_vecs_rgb[corner_index[start:end,1],:] * (1 - xd) +\
                latent_vecs_rgb[corner_index[start:end,5],:] * xd
            c10_r = latent_vecs_rgb[corner_index[start:end,2],:] * (1 - xd) +\
                latent_vecs_rgb[corner_index[start:end,6],:] * xd
            c11_r = latent_vecs_rgb[corner_index[start:end,3],:] * (1 - xd) +\
                latent_vecs_rgb[corner_index[start:end,7],:] * xd
            c0_r = c00_r * (1 - yd) + c10_r * yd
            c1_r = c01_r * (1 - yd) + c11_r * yd
            c_b_r = c0_r * (1 - zd) + c1_r * zd

            c.append(c_b)
            c_rgb.append(c_b_r)
        c = torch.cat(c,dim = 0)
        c_rgb = torch.cat(c_rgb,dim = 0)
        return c,c_rgb,mask

    def update_lowest(self, points_xyz: torch.Tensor, points_normal = None,points_color = None, required_grad = False):
        # assert points_xyz.device == points_normal.device == self.device, \
        #     f"Device of map {self.device} and input observation " \
        #     f"{points_xyz.device, points_normal.device} must be the same."
    
        use_color = not points_color is None
        use_normal = not points_normal is None
        # self.modifying_lock.acquire()
        self.latent_vecs_left.reset()
        points_xyz_aligned = points_xyz - self.bound_min.unsqueeze(0)
        voxel_size_lowest= self.voxel_size
        points_xyz_aligned = points_xyz_aligned / voxel_size_lowest
        points_voxel_xyz = torch.ceil(points_xyz_aligned).long()-1
        flag = torch.max(points_voxel_xyz[:,0]) >= self.n_xyz_L[0][0] or \
            torch.max(points_voxel_xyz[:,1]) >= self.n_xyz_L[0][1] or \
            torch.max(points_voxel_xyz[:,2]) >= self.n_xyz_L[0][2] 
        if torch.min(points_voxel_xyz) < 0 or flag:
            print(self.n_xyz_L[0])
            print(torch.max(points_voxel_xyz[:,0]),torch.max(points_voxel_xyz[:,1]),torch.max(points_voxel_xyz[:,2]),torch.min(points_voxel_xyz))
            raise BoundError()
        points_voxel_id = self._linearize_id(points_voxel_xyz)

        #1.初始化空间
        #得到那些所在voxel点数较为密集的observation点
        # dense_mask=None
        # _, unq_inv, unq_count = torch.unique(points_voxel_id, return_counts=True, return_inverse=True)
        # dense_mask = (unq_count > self.dense_thresold)[unq_inv]
        # points_xyz_aligned = points_xyz_aligned[dense_mask]
        # points_voxel_id = points_voxel_id[dense_mask]
        # points_normal = points_normal[dense_mask]
        
        # 添加空间给invalid的voxel
        # points_voxel_id_invalid = self.indexer[points_voxel_id] == -1
        points_voxel_id = torch.unique(points_voxel_id)
        self.latent_vecs_left.allocate_block(points_voxel_id)
        
        # def get_pruned_surface(enabled=True, lin_pos=None):
            # # Prune useless surface points for quicker gathering (set to True to enable)
            # if enabled:
            #     encoder_voxel_pos_exp = self._expand_flatten_id(lin_pos, False)
            #     # encoder_voxel_pos_exp = lin_pos
            #     exp_indexer = torch.zeros_like(self.indexer)
            #     exp_indexer[encoder_voxel_pos_exp] = 1
            #     focus_mask = exp_indexer[points_voxel_id] == 1
            #     return points_xyz_aligned[focus_mask], points_normal[focus_mask]
            # else:
            #     return points_xyz_aligned , points_normal

        #2.更新voxel的latent vector：
        #  1）valid
        #  2）有新加入的点
         
        # pruned_points_xyz_aligned, pruned_points_normal = points_xyz_aligned , points_normal

        # 得到每个point对应的latent vector，1对8
        # gathered_points_latent_indexes = []
        # gathered_points_xyzn = []
        # for offset in self.integration_offsets:
        #     points_offset_voxel_xyz = torch.ceil(points_xyz_aligned + offset) - 1
            
        #     #不要出界
        #     for dim in range(3):
        #         points_offset_voxel_xyz[:, dim].clamp_(0, self.n_xyz_L[0][dim] - 1)
        #     #points相对对应voxel的单位长度位置
        #     points_relative_xyz = points_xyz_aligned - (points_offset_voxel_xyz + self.relative_network_offset)

            

        #     points_offset_voxel_id= self._linearize_id(points_offset_voxel_xyz).long()
            
             
        #     self.latent_vecs_left.allocate_block(points_offset_voxel_id)
           
        #     points_offset_latent_index = self.latent_vecs_left.get_idx(points_offset_voxel_id)
            
            
            
        #     gathered_points_latent_indexes.append(points_offset_latent_index)
        #     if use_color:
        #         gathered_points_xyzn.append(torch.cat(
        #             [points_relative_xyz,
        #                 points_normal,points_color], dim=-1))
        #     else:
        #         gathered_points_xyzn.append(torch.cat(
        #             [points_relative_xyz,
        #                 points_normal], dim=-1))
            
        # gathered_points_xyzn = torch.cat(gathered_points_xyzn)
        

        # gathered_points_latent_indexes = torch.cat(gathered_points_latent_indexes)
        # #一共有这么多latent vector要被更新
        points_offset_voxel_xyz = torch.ceil(points_xyz_aligned) - 1
        points_relative_xyz = points_xyz_aligned - (points_offset_voxel_xyz)
        points_offset_voxel_id= self._linearize_id(points_offset_voxel_xyz).long()
        self.latent_vecs_left.allocate_block(torch.unique(points_offset_voxel_id))
        gathered_points_latent_indexes = self.latent_vecs_left.get_idx(points_offset_voxel_id)
        if use_normal:
            gathered_points_xyzn = torch.cat([points_relative_xyz,
                            points_normal], dim=-1)
        else:
            gathered_points_xyzn = points_relative_xyz
        if use_color:
            gathered_points_xyzn = torch.cat([gathered_points_xyzn,
                            points_color], dim=-1)
        # print(gathered_points_xyzn.shape)
        updated_latent_vecs, latent_vecs_points_id, latent_vecs_points_num= torch.unique(gathered_points_latent_indexes, return_inverse=True,
                                                                return_counts=True)
        updated_latent_vecs = updated_latent_vecs.detach()
        latent_vecs_points_num = latent_vecs_points_num.float()
        # print(f"Points/Voxel: avg = {latent_vecs_points_num.mean().item()}, "
        #                 f"min = {latent_vecs_points_num.min().item()}, "
        #                 f"max = {latent_vecs_points_num.max().item()}")
        # logging.info(f"{updated_latent_vecs.size(0)} voxels will be updated by the encoder. "
        #                 f"Points/Voxel: avg = {latent_vecs_points_num.mean().item()}, "
        #                 f"min = {latent_vecs_points_num.min().item()}, "
        #                 f"max = {latent_vecs_points_num.max().item()}")
        if required_grad:
            # points_encoded_latent_vecs = self.model.encoder(gathered_points_xyzn)
            points_encoded_latent_vecs = net_util.forward_model(self.model.encoder,max_sample = 2 ** 23,
                                                    network_input=gathered_points_xyzn,
                                                    no_detach = True)[0]
            encoder_latent_sum = net_util.groupby_reduce(latent_vecs_points_id, points_encoded_latent_vecs, op="sum")   # (C, L)
            self.latent_vecs_left.set_confidence(updated_latent_vecs,latent_vecs_points_num)
            self.latent_vecs_left[updated_latent_vecs] = encoder_latent_sum / \
                self.latent_vecs_left.get_confidence(updated_latent_vecs).unsqueeze(-1)
        else:
            with torch.no_grad():
                # points_encoded_latent_vecs = self.model.encoder(gathered_points_xyzn)
                # print(gathered_points_xyzn.shape)
                points_encoded_latent_vecs = net_util.forward_model(self.model.encoder,max_sample = 2 ** 23,
                                                    network_input=gathered_points_xyzn,
                                                    no_detach = True)[0]
                encoder_latent_sum = net_util.groupby_reduce(latent_vecs_points_id, points_encoded_latent_vecs, op="sum")   # (C, L)
                 
                self.latent_vecs_left.set_confidence(updated_latent_vecs,latent_vecs_points_num)
                self.latent_vecs_left[updated_latent_vecs] = encoder_latent_sum / \
                    self.latent_vecs_left.get_confidence(updated_latent_vecs).unsqueeze(-1)
                
        # self._mark_updated_vec_id(updated_latent_vecs)

        # if self.used_latent_vecs_id == None:
        self.used_latent_vecs_id = updated_latent_vecs
        # self.used_latent_vecs_id = torch.unique(torch.cat([self.used_latent_vecs_id,updated_latent_vecs],dim = 0))
        
        # self.modifying_lock.release()

    

    def update_right(self,rgb = False,if_sloss = False,gt_s = None):
       
        lower_layer_voxel_id = self.latent_vecs_left.get_reverse_id(self.used_latent_vecs_id)
        lower_voxel_input_id_valid = lower_layer_voxel_id
        #3.快速对每一层进行“卷积”
        #找到本次更新了的voxel上一层的voxel

        #得到每个下层voxel对应的上层voxel
        lower_layer_voxel_xyz = self._unlinearize_id(lower_voxel_input_id_valid)

        lower_voxel_input_index = self.latent_vecs_left.get_idx(lower_voxel_input_id_valid)
        lower_voxel_input_latent = self.latent_vecs_left[lower_voxel_input_index]
        # lower_layer_voxel_xyz = torch.cat(
        #     [torch.ones(lower_layer_voxel_xyz.size(0),1).cuda(),lower_layer_voxel_xyz],dim = 1
        # )
        #input valid voxels location and features into sparse convolution
        dense_voxels , logits ,xyzs = self.model.conv_kernels([lower_layer_voxel_xyz,lower_voxel_input_latent])
        if if_sloss:
            loss_func = torch.nn.CrossEntropyLoss()
            s_losses = []
            for idx,logit in enumerate(logits):
                # print(logit.shape,gt_s[idx].shape)
                if gt_s[idx].dim() == 3:
                    gt = gt_s[idx].unsqueeze(0)
                # print(logit.shape,gt_s[idx].shape)
                else:
                    gt = gt_s[idx]
                xyz_mask = xyzs[idx]
                gt = gt[:,xyz_mask[:,0],xyz_mask[:,1],xyz_mask[:,2]].cuda().long()
                # print(logit.shape,gt.shape)
                # if idx == 0:
                #     mask = gt.squeeze(0)
                    # print(torch.nonzero(gt == 1).size(0))
                    # print(torch.nonzero(mask == 1).size(0) / mask.size(0))
                    # print("acc: ",torch.nonzero(mask == gt).size(0)/xyz_mask.size(0))
                s_losses.append(loss_func(logit,gt))
            s_losses = sum(s_losses)
        mask = torch.argmax(logits[0],dim = 1).squeeze(0)
        # mask = gt_s[0][:,xyz_mask[:,0],xyz_mask[:,1],xyz_mask[:,2]].cuda().long()
        updated_upper_voxel_xyz = xyzs[0][mask == 1,0:3]
        
        # print(updated_upper_voxel_xyz.shape)
        # updated_upper_voxel_xyz = dense_voxels.get_spatial_locations().cuda()[:,0:3]
        # updated_upper_voxel_latent = dense_voxels.features
        updated_upper_voxel_latent = dense_voxels[:,:,updated_upper_voxel_xyz[:,0],updated_upper_voxel_xyz[:,1],updated_upper_voxel_xyz[:,2]].squeeze(0)
        updated_upper_voxel_id = self._linearize_id(updated_upper_voxel_xyz).long()
        # print(updated_upper_voxel_id.size(),lower_layer_voxel_id.size())
        self.latent_vecs_right.reset()
        self.latent_vecs_right.allocate_block(updated_upper_voxel_id)
        idx = self.latent_vecs_right.get_idx(updated_upper_voxel_id)
        self.latent_vecs_right[idx] = updated_upper_voxel_latent.permute(1,0).contiguous()
        if if_sloss:
            return s_losses
        else:
            return  

    def update_right_corner_with_structure(self):
        lower_layer_voxel_id = self.latent_vecs_left.get_reverse_id(self.used_latent_vecs_id)
        lower_voxel_input_id_valid = lower_layer_voxel_id
        lower_layer_voxel_xyz = self._unlinearize_id(lower_voxel_input_id_valid)

        lower_voxel_input_index = self.latent_vecs_left.get_idx(lower_voxel_input_id_valid)
        lower_voxel_input_latent = self.latent_vecs_left[lower_voxel_input_index]
        

        sparse_corners , logits ,xyzs,cords = self.model.conv_kernels([lower_layer_voxel_xyz,lower_voxel_input_latent])
        
        occ_right_ids = self._linearize_id(cords).long()
        self.occ_right_flag[:] = 0
        self.occ_right_flag[occ_right_ids] = 1
        
        updated_upper_voxel_xyz = sparse_corners.get_spatial_locations().cuda()
        updated_upper_voxel_latent = sparse_corners.features
        updated_upper_voxel_id = self._linearize_id(updated_upper_voxel_xyz,dim = self.corner_xyz).long()
        self.latent_vecs_right_corner.allocate_block(updated_upper_voxel_id)
        idx = self.latent_vecs_right_corner.get_idx(updated_upper_voxel_id)
        self.latent_vecs_right_corner[idx] = updated_upper_voxel_latent
        return logits,xyzs

    def update_right_corner(self,rgb = False,if_sloss = False,gt_s = None):
       
        lower_layer_voxel_id = self.latent_vecs_left.get_reverse_id(self.used_latent_vecs_id)
        lower_voxel_input_id_valid = lower_layer_voxel_id
        #3.快速对每一层进行“卷积”
        #找到本次更新了的voxel上一层的voxel

        #得到每个下层voxel对应的上层voxel
        lower_layer_voxel_xyz = self._unlinearize_id(lower_voxel_input_id_valid)

        lower_voxel_input_index = self.latent_vecs_left.get_idx(lower_voxel_input_id_valid)
        lower_voxel_input_latent = self.latent_vecs_left[lower_voxel_input_index]
        # pdb.set_trace()
        # lower_layer_voxel_xyz = torch.cat(
        #     [torch.ones(lower_layer_voxel_xyz.size(0),1).cuda(),lower_layer_voxel_xyz],dim = 1
        # )
        #input valid voxels location and features into sparse convolution
        if if_sloss:
            for idx,gt in enumerate(gt_s):
                if gt.dim() == 3:
                    gt_s[idx] = gt.unsqueeze(0).cuda()
                else:
                    gt_s[idx] = gt.cuda()    
            self.model.conv_kernels.structure = gt_s
        # print(lower_layer_voxel_xyz,lower_voxel_input_latent)
        
        if rgb:
            # s = time.time()
            sparse_corners ,sparse_corners_rgb, logits ,xyzs,cords,c_cords = self.model.conv_kernels([lower_layer_voxel_xyz,lower_voxel_input_latent])
            # print(f"compute latent use {float(time.time() - s) * 1000} ms")
        else:
            sparse_corners , logits ,xyzs,cords,c_cords = self.model.conv_kernels([lower_layer_voxel_xyz,lower_voxel_input_latent])
           
        if if_sloss:
            loss_func = torch.nn.CrossEntropyLoss()
            s_losses = []
            ttnum = 0
            for idx,logit in enumerate(logits):
                xyz_mask = xyzs[idx]
                gt = gt_s[idx][:,xyz_mask[:,0],xyz_mask[:,1],xyz_mask[:,2]].long()
                s_losses.append(loss_func(logit,gt) * xyz_mask.size(0))
                ttnum += xyz_mask.size(0) 
            s_losses = sum(s_losses) / ttnum

        # mask = gt_s[0][:,xyz_mask[:,0],xyz_mask[:,1],xyz_mask[:,2]].cuda().long()
        occ_right_ids = self._linearize_id(cords).long()
        # print(cords)
        self.occ_right_flag[:] = 0
        self.occ_right_flag[occ_right_ids] = 1
        
        # print(updated_upper_voxel_xyz.shape)
        
        # print(occ_right_ids.shape,updated_upper_voxel_xyz.shape)
        # print(updated_upper_voxel_xyz.shape)
        # updated_upper_voxel_xyz = dense_voxels.get_spatial_locations().cuda()[:,0:3]
        # updated_upper_voxel_latent = dense_voxels.features
        

        updated_upper_voxel_id = self._linearize_id(c_cords,dim = self.corner_xyz).long()
        # print(updated_upper_voxel_id.size(),lower_layer_voxel_id.size())
        self.latent_vecs_right_corner.reset()
        self.latent_vecs_right_corner.allocate_block(updated_upper_voxel_id)
        idx = self.latent_vecs_right_corner.get_idx(updated_upper_voxel_id)
        self.latent_vecs_right_corner[idx] = sparse_corners
        if rgb:
            self.latent_vecs_right_corner_rgb.reset()
            self.latent_vecs_right_corner_rgb.allocate_block(updated_upper_voxel_id)
            idx = self.latent_vecs_right_corner_rgb.get_idx(updated_upper_voxel_id)
            self.latent_vecs_right_corner_rgb[idx] = sparse_corners_rgb
        if if_sloss:
            return s_losses
        else:
            return  


        
    def compute_sdf_loss_corner(self,surface_xyz,gt_sdf,cd = 1,if_normal=True):
        with torch.no_grad():
            points_xyz_aligned = surface_xyz - self.bound_min.unsqueeze(0)
            points_xyz_aligned = points_xyz_aligned / self.voxel_size 
            voxel_xyzs = torch.floor(points_xyz_aligned)

            mask = torch.logical_and(voxel_xyzs > 0,voxel_xyzs < self.n_xyz_L[0][0])
            mask = torch.all(mask,dim = 1)
            points_xyz_aligned = points_xyz_aligned[mask,:]
            voxel_xyzs = voxel_xyzs[mask,:]
            gt_sdf = gt_sdf[mask]
            
            points_xyz_relative = points_xyz_aligned - voxel_xyzs
            points_voxel_id = self._linearize_id(voxel_xyzs).long()
            points_voxel_id_valid = self.occ_right_flag[points_voxel_id] == 1
            voxel_xyzs = voxel_xyzs[points_voxel_id_valid,:]
            points_xyz_aligned = points_xyz_aligned[points_voxel_id_valid,:]
            points_xyz_relative = points_xyz_relative[points_voxel_id_valid]
            gt_sdf = gt_sdf[points_voxel_id_valid].view(-1)
            if len(points_xyz_relative) == 0:
                return None
        #found latent in range [-1,1]
        latent_input,mask = self.trilinear_interpolate(points_xyz_relative + voxel_xyzs)
        
        points_xyz_relative = points_xyz_relative[mask]
        gt_sdf = gt_sdf[mask]
        
        sdf_b = net_util.forward_model(self.model.decoder,max_sample = 2 ** 22,
            latent_input = latent_input,no_detach= True,
            xyz_input = points_xyz_relative,layer = 0)[0]
        
        src = sdf_b.view(-1)
        sdf_loss = (src - gt_sdf).abs().sum()

        return sdf_loss,gt_sdf.size(0)



    def compute_corner_reg_loss(self):
        if self.latent_vecs_right_corner.size() == 0:
            return 0
        
        occupied_latent_vecs_1 = self.latent_vecs_right_corner[0:self.latent_vecs_right_corner.size()]
        occupied_latent_vecs_2 = self.latent_vecs_left[0:self.latent_vecs_left.size()]
        # occupied_latent_vecs_3 = self.latent_vecs_right_corner_rgb[0:self.latent_vecs_right_corner_rgb.size()]
        # occupied_latent_vecs = torch.cat([occupied_latent_vecs_1,occupied_latent_vecs_2,occupied_latent_vecs_3],dim = 0)
        loss = occupied_latent_vecs_1.norm(dim=1).sum() +\
            occupied_latent_vecs_2.norm(dim=1).sum()
            # occupied_latent_vecs_3.norm(dim=1).sum()
        loss = loss / self.latent_vecs_right_corner.size()
        # print(loss)
        return loss

    #TODO: use global xyz to encoder and decoder adding position embedding
    def compute_corner_sample_loss(self,xyz,sdf,voxel_resolution,cd = 1,if_normal=True,clamp_mode="o"):
        # print(indexer.shape,sdf.shape)
        with torch.no_grad():
            ids = self._linearize_id(xyz).long()
            mask = self.occ_right_flag[ids] == 1
            sdf = sdf[mask,:,:,:].view(-1)
            xyz = xyz[mask,:]
            B = ids[mask].size(0)
            if B == 0:
                return None,None,None
            sample_a = 0
            sample_b = 1 - 1 / voxel_resolution
            # sample_a = - 0.5
            # sample_b = 1.5 - 1 / voxel_resolution
            low_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b)
            low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)
            low_samples = low_samples.view(-1, low_samples.size(-1))

        low_center = xyz.unsqueeze(1).repeat(1, voxel_resolution ** 3, 1).view(-1,3)
        # print(low_center.shape,low_samples.shape)
        xyz_samples = low_center + low_samples 
        low_latents,_ = self.trilinear_interpolate(xyz_samples,base_xyz = low_center)
        # print(torch.any(torch.isinf(low_latents)))
        # print(torch.any(torch.isnan(low_samples)))

        # for p in model.decoder.parameters():

        low_sdf = net_util.forward_model(self.model.decoder,max_sample = 2 ** 22,no_detach = True,
                                            latent_input=low_latents,
                                            xyz_input=low_samples,layer = 0)[0]


        
        mask = ~torch.isnan(sdf)
        src = low_sdf.view(-1)[mask]
        tar = sdf[mask].clamp(-1,1)
        if clamp_mode == 'o':
            mask = clamp_select(tar,src,-cd,cd)
            loss = (src[mask] - tar[mask]).abs().sum()
        else:
            tar = sdf[mask].clamp(-cd,cd)
            src = src.clamp(-cd,cd)
            loss = (src - tar).abs().sum()


        return loss,src.size(0)

    
    
    def compute_surface_loss(self,surface_xyz,gt_sdf,cd = 1,clamp_mode="o"):
        with torch.no_grad():
            
            points_xyz_aligned = surface_xyz - self.bound_min.unsqueeze(0)
            points_xyz_aligned = points_xyz_aligned / self.voxel_size 
            voxel_xyzs = torch.floor(points_xyz_aligned)
            mask = torch.logical_and(voxel_xyzs > 0,voxel_xyzs < self.n_xyz_L[0][0])
            mask = torch.all(mask,dim = 1)
            points_xyz_aligned = points_xyz_aligned[mask,:]
            voxel_xyzs = voxel_xyzs[mask,:]
            gt_sdf = gt_sdf[mask]

            points_xyz_relative = points_xyz_aligned - voxel_xyzs

            points_voxel_id = self._linearize_id(voxel_xyzs).long()
            points_voxel_id_valid = self.occ_right_flag[points_voxel_id] == 1

            
            voxel_xyzs = voxel_xyzs[points_voxel_id_valid,:]
            points_xyz_aligned = points_xyz_aligned[points_voxel_id_valid,:]
            
            points_xyz_relative = points_xyz_relative[points_voxel_id_valid]
            gt_sdf = gt_sdf[points_voxel_id_valid].view(-1)
            if len(points_xyz_relative) == 0:
                return None,None,None,None
        #found latent in range [-1,1]


        latent_input,mask = self.trilinear_interpolate(points_xyz_relative + voxel_xyzs,voxel_xyzs)
        points_xyz_relative = points_xyz_relative[mask,:]
        gt_sdf = gt_sdf[mask]
        sdf_b = net_util.forward_model(self.model.decoder,max_sample = 2 ** 22,
            latent_input = latent_input,no_detach= True,
            xyz_input = points_xyz_relative,layer = 0)[0]
        tar_sdf = gt_sdf.clamp(-1,1)
        # src_sdf = gradient_clamp(sdf_b.view(-1),-cd,cd)
        src_sdf = sdf_b.view(-1)
        if clamp_mode == "o":
            mask = clamp_select(tar_sdf,src_sdf,-cd,cd)
            sdf_loss = (src_sdf[mask] - tar_sdf[mask]).abs().sum()
        else:
            tar_sdf = tar_sdf.clamp(-cd,cd)
            src_sdf = src_sdf.clamp(-cd,cd)
            sdf_loss = (src_sdf - tar_sdf).abs().sum()


        return sdf_loss,src_sdf.size(0)

    def compute_normal_loss(self,points,normals):
        interval = 0.0025 / self.voxel_size
        with torch.no_grad():
            normal_offset = torch.tensor([[-interval,0,0], [interval,0,0],[0,-interval,0],[0,interval,0],[0,0,-interval],[0,0,interval]]).cuda()
            points_xyz_aligned = points - self.bound_min.unsqueeze(0)
            points_xyz_aligned = points_xyz_aligned / self.voxel_size 
            points_xyz_perturb = points_xyz_aligned.unsqueeze(1).repeat(1,6,1) + normal_offset
            
            voxel_xyzs_perturb = torch.floor(points_xyz_perturb)
            mask_perturb = torch.logical_and(voxel_xyzs_perturb > 0,voxel_xyzs_perturb < self.n_xyz_L[0][0])
            mask_perturb = torch.all(mask_perturb,dim=2)
            n_mask = torch.all(mask_perturb,dim=1)
    
            points_xyz_perturb = points_xyz_perturb[n_mask,:,:]
            voxel_xyzs_perturb = voxel_xyzs_perturb[n_mask,:,:]
            normals = normals[n_mask,:]
            points_xyz_relative_perturb = points_xyz_perturb - voxel_xyzs_perturb
        points_xyz_relative_perturb = points_xyz_relative_perturb.view(-1,3)
        voxel_xyzs_perturb = voxel_xyzs_perturb.view(-1,3)
        latent_input,mask = self.trilinear_interpolate(points_xyz_relative_perturb + voxel_xyzs_perturb,if_normal = True)
        mask = mask.view(-1,6)
        points_xyz_relative_perturb = points_xyz_relative_perturb.view(-1,6,3)
        mask = torch.all(mask,dim=1)
        points_xyz_relative_perturb = points_xyz_relative_perturb[mask,:,:].view(-1,3)
        gt_normals = normals[mask,:]
        sdf_b = net_util.forward_model(self.model.decoder,max_sample = 2 ** 22,
            latent_input = latent_input,no_detach= True,
            xyz_input = points_xyz_relative_perturb,layer = 0)[0]
        sdf_b = sdf_b.view(-1,6)

        
        normal_x = (sdf_b[:,1] - sdf_b[:,0]) / interval * 2
        normal_y = (sdf_b[:,3] - sdf_b[:,2]) / interval * 2
        normal_z = (sdf_b[:,5] - sdf_b[:,4]) / interval * 2
        normal = torch.stack([normal_x,normal_y,normal_z],dim = 1)
        normal_loss = (1 - F.cosine_similarity(gt_normals,normal,dim = 1)).sum()
        normal_reg_loss = ((normal.norm(dim=-1) - 1).abs()).sum(dim = 0)
        
        return normal_loss,normal_reg_loss,normal.size(0)

    def intermediate_detach(self):
        self.latent_vecs_left.detach()
        # self.latent_vecs_right.detach()
        self.latent_vecs_right_corner.detach()
        # self.latent_vecs_right_corner_rgb.detach()

    def build_octree(self): 
        # voxel_ids = self.latent_vecs_right.latent_vecs_id[0 : self.latent_vecs_right.size()]
        layer_id_valid = self.occ_right_flag == 1    # mask of layer_id  
        voxel_ids = torch.arange(0, self.occ_right_flag.size(0), device=self.device,
                                                            dtype=torch.long)[layer_id_valid]
        voxel_id_to_xyz = self._unlinearize_id(voxel_ids) 
        # print(voxel_id_to_xyz.shape)
        self.octree_xyzs = []
        self.octree_xyzs.insert(0,voxel_id_to_xyz.contiguous())
        self.octree_sons = []
        father_ids = []
        for i in range(self.lowest - 1):
            # print(self.octree_xyzs[0])
            num = 2 ** (i+1)
            father_xyzs = self.octree_xyzs[0] // num * num
            father_id = self._linearize_id(father_xyzs) 
            father_id,father_id_reverse = torch.unique(father_id,return_inverse = True)
            father_xyzs = self._unlinearize_id(father_id) 
            # print(father_id_reverse.shape)
            father_ids.insert(0,father_id_reverse.int())
            self.octree_xyzs.insert(0,father_xyzs)
            son_array = torch.ones((father_xyzs.size(0),8),dtype = torch.int32,device = self.device) * -1
            self.octree_sons.insert(0,son_array)
        self.octree_sons = build_octree(self.octree_xyzs,father_ids,self.octree_sons)
        
        self.renderer.set_bound_min(self.bound_min)
        # self.renderer.set_voxel_latents(self.latent_vecs_right_corner.latent_vecs)
        
    def render_depth(self,pose):
        self.build_octree()
        
        pose[1] = pose[1] - self.bound_min.unsqueeze(0)
        # print(pose[1])
        # start = time.time()
        depth_img = self.renderer.render(self.octree_xyzs,self.octree_sons,pose)

        return depth_img

    def train_render_rgb_and_depth(self,ray_od,use_depth):
        self.build_octree()
        ray_od[:,1,:] = ray_od[:,1,:] - self.bound_min.unsqueeze(0)
        depth_img,rgb_img,ray_ids = self.renderer.train_render_rgb_and_depth(self.octree_xyzs,self.octree_sons,ray_od,use_depth)
        return depth_img,rgb_img,ray_ids

    def compute_render_loss(self,targets,ray,use_rgb = False,use_depth = True):
        
        if not use_rgb:
            depth_img = self.train_render_depth(ray)
            if depth_img is None:
                return 2,1
            zero_mask = torch.logical_and(depth_img >= 0.1,targets >= 0.1)
            src = depth_img[zero_mask] 
            depth_target = targets[zero_mask]
            loss = ((src - depth_target) ** 2).sum()
        else:
            depth_img,rgb_img,ray_ids = self.train_render_rgb_and_depth(ray,use_depth)
            if depth_img is None:
                return 2,1
            rgb_target,depth_target = torch.split(targets,[3,1],dim = 1)
            rgb_target = rgb_target[ray_ids,:]
            depth_target = depth_target[ray_ids,:].squeeze(-1)
            mask = torch.any(torch.isnan(rgb_img),dim = 1)
            # zero_mask = torch.logical_and(depth_img >= 0.1,depth_target >= 0.1)
            if use_depth:
                loss = ((rgb_target[~mask,:] - rgb_img[~mask,:]) ** 2).sum() + ((depth_img - depth_target) ** 2).sum()
            else:
                loss = ((rgb_target[~mask,:] - rgb_img[~mask,:]) ** 2).sum()
            # loss = ((depth_img - depth_target) ** 2).sum()
        return loss,depth_target.size(0)
        
    def train_render_depth(self,ray_od):
        self.build_octree()
        ray_od[:,1,:] = ray_od[:,1,:] - self.bound_min.unsqueeze(0)
        depth_img = self.renderer.train_render(self.octree_xyzs,self.octree_sons,ray_od)
        return depth_img

    def check_points(self,points):
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size 
        points_voxel_xyz = torch.ceil(points_xyz_aligned).long()-1
        
        lower_layer_voxel_id = self._linearize_id(points_voxel_xyz).long()
        points_xyz_upper = points_xyz_aligned 
        points_xyz_relative = points_xyz_upper - torch.ceil(points_xyz_upper) + 0.5
        lower_voxel_input_index = self.latent_vecs_right.get_idx(lower_layer_voxel_id)
        # print(torch.nonzero(lower_voxel_input_index == -1).size(0))
        lower_voxel_input_latent = self.latent_vecs_right[lower_voxel_input_index]
        sdf_b = net_util.forward_model(self.model.decoder,max_sample = 2 ** 20,
                latent_input = lower_voxel_input_latent,no_detach= True,
                xyz_input = points_xyz_relative,layer = 0)[0]
        
        # cord = torch.nonzero(sdf_b > 1e-5)

    def mask_rays(self,ray_ods,targets):
        points = ray_ods[:,1,:] + ray_ods[:,0,:] * targets.unsqueeze(-1)
        points_xyz_aligned = points - self.bound_min.unsqueeze(0)
        points_xyz_aligned = points_xyz_aligned / self.voxel_size 
        voxel_xyzs = torch.floor(points_xyz_aligned)
        mask_x = voxel_xyzs[:,0] < self.n_xyz_L[0][0]
        mask_y = voxel_xyzs[:,1] < self.n_xyz_L[0][1]
        mask_z = voxel_xyzs[:,2] < self.n_xyz_L[0][2]
        mask_0 = voxel_xyzs >= 0
        # mask = torch.logical_and(voxel_xyzs > 0,voxel_xyzs < torch.from_numpy(self.n_xyz_L[0]).cuda())
        mask = torch.all(mask_0,dim = 1)
        mask = torch.logical_and(mask,mask_x)
        mask = torch.logical_and(mask,mask_y)
        mask = torch.logical_and(mask,mask_z)
        ray_ods = ray_ods[mask,:,:]
        targets = targets[mask]
        voxel_xyzs = voxel_xyzs[mask,:]
        points_voxel_id = self._linearize_id(voxel_xyzs).long()
        mask = self.occ_right_flag[points_voxel_id] == 1
        return ray_ods[mask,:,:],targets[mask]
    
    def get_fast_preview_visuals(self):
        # xyz = torch.ones((128,128,128)).cuda()
        # total_xyz = torch.nonzero(xyz)
        # layer_id = self._linearize_id(total_xyz).long() # layer_id id in L
        # layer_id_valid = self.indexer[layer_id] != -1
        # occupied_flatten_id = layer_id[layer_id_valid]
        occupied_flatten_id = self.occ_right_flag[:] == 1
        occupied_flatten_id = torch.arange(0,self.occ_right_flag.size(0),device = self.device)[occupied_flatten_id]
        # occupied_vec_id = self.latent_vecs_right.indexer[ids]
        blk_verts = [self._unlinearize_id(occupied_flatten_id) * self.voxel_size + self.bound_min]
        n_block = blk_verts[0].size(0)
        blk_edges = []
        for vert_offset in [[0.0, 0.0, self.voxel_size], [0.0, self.voxel_size, 0.0],
                            [0.0, self.voxel_size, self.voxel_size], [self.voxel_size, 0.0, 0.0],
                            [self.voxel_size, 0.0, self.voxel_size], [self.voxel_size, self.voxel_size, 0.0],
                            [self.voxel_size, self.voxel_size, self.voxel_size]]:
            blk_verts.append(
                blk_verts[0] + torch.tensor(vert_offset, dtype=torch.float32, device=blk_verts[0].device).unsqueeze(0)
            )
        for vert_edge in [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]:
            blk_edges.append(np.stack([np.arange(n_block, dtype=np.int32) + vert_edge[0] * n_block,
                                       np.arange(n_block, dtype=np.int32) + vert_edge[1] * n_block], axis=1))
        blk_verts = torch.cat(blk_verts, dim=0).cpu().numpy().astype(float)
        blk_wireframe = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(blk_verts),
            lines=o3d.utility.Vector2iVector(np.concatenate(blk_edges, axis=0)))
        from utils import vis_util
        return [
            blk_wireframe,
            vis_util.wireframe_bbox(self.bound_min.cpu().numpy(),
                                    self.bound_max.cpu().numpy(), color_id=4)
        ]

    def extract_whole_mesh(self, voxel_resolution: int, max_n_triangles: int,
                    max_std: float = 2000.0,extract_async: bool = False,L = 0):
        def do_meshing(voxel_resolution):
            
            total_xyz = torch.nonzero(self.xyz)
            # del xyz
            # print(total_xyz)
            # for layer in range(0,self.layer):
            print(self.n_xyz_L)
            
            ll = self.xyz.shape[0] / self.n_xyz_L[0][0]
            if self.xyz.size(0) < self.n_xyz_L[0][0]:
                layer_id = self._linearize_id(total_xyz // (self.xyz.size(0)/self.n_xyz_L[0][0])).long()
            else:
                layer_id = self._linearize_id(total_xyz).long() # layer_id id in L
            layer_id_valid = self.latent_vecs_right.indexer[layer_id] != -1    # mask of layer_id  
            layer_id_invalid = self.latent_vecs_right.indexer[layer_id] == -1  # inv mask of layer_id  
            occupied_xyz_valid = total_xyz[layer_id_valid]   # get every valid xyz range in (128,128,128)
            total_xyz = total_xyz[layer_id_invalid]          # get rest of xyz range in (128,128,128)


            layer_id_valid = layer_id[layer_id_valid]        # apply mask on layer_id
            occupied_vec_id = self.latent_vecs_right.indexer[layer_id_valid]

            # get every valid id in layer 0 and map into batch id
            
            # layer_id_valid_l0 = self._linearize_id(occupied_xyz_valid,L=0).long()
            layer_id_valid_l0 = occupied_xyz_valid[:, 2] + self.xyz.shape[0] * occupied_xyz_valid[:, 1] + (self.xyz.shape[0] * self.xyz.shape[0]) * occupied_xyz_valid[:, 0]


            # build xyz -> batch id
            indexer = torch.ones(self.xyz.size(0)*self.xyz.size(0)*self.xyz.size(0) , device=self.device, dtype=torch.long) * -1 
            indexer[layer_id_valid_l0] = torch.arange(0, layer_id_valid.size(0), device=self.device,
                                                                dtype=torch.long)
            
            # get each pos of voxel xyz in range(128,128,128) in bigger layer voxel 
            upper_xyz = occupied_xyz_valid // int(ll) * ll
            voxel_xyz_relative = occupied_xyz_valid - upper_xyz
            diff_center = (voxel_xyz_relative - ll / 2 + 0.5) / ll

            occupied_latent_vecs = self.latent_vecs_right[occupied_vec_id]
            # xyz_input = occupied_xyz_valid / self.n_xyz_L[0][0] - 0.5
            # occupied_latent_vecs,_ = self.get_latent_sum(xyz_input,layer,mode ='direct')
            
            # print(occupied_latent_vecs.shape)
            B = occupied_latent_vecs.size(0)
            sample_a = -(voxel_resolution // 2) * (1. / voxel_resolution)
            sample_b = 1. + (voxel_resolution - 1) // 2 * (1. / voxel_resolution)
            
            low_resolution = voxel_resolution
            low_samples = net_util.get_samples(low_resolution, self.device, a=sample_a, b=sample_b) - \
                            self.relative_network_offset # (l**3, 3)
            
            
            xyz_expand =  occupied_xyz_valid.unsqueeze(1).repeat(1, low_samples.size(0), 1) + 0.5
            # print(xyz_expand.shape)
            low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)  # (B, l**3, 3)
            # low_samples = (low_samples + xyz_expand) / xyz.size(0) - 0.5
            # print(torch.max(low_samples),torch.min(low_samples))
            low_latents = occupied_latent_vecs.unsqueeze(1).repeat(1, low_samples.size(1), 1)  # (B, l**3, 3)

            with torch.no_grad():
                
                low_sdf = net_util.forward_model(self.model.decoder,max_sample = 2 ** 20,
                                                    latent_input=low_latents.view(-1, low_latents.size(-1)),
                                                    xyz_input=low_samples.view(-1, low_samples.size(-1)),layer = 0)[0]
                high_sdf = low_sdf.reshape(B, low_resolution, low_resolution, low_resolution)
                high_std = torch.ones_like(high_sdf) * 0.1
                # print(torch.max(high_std))
            batch_num = int(math.ceil(low_latents.size(0)/(2**20)))
            bs = int(low_latents.size(0)/batch_num)
            for i in range(0,batch_num):
                start = bs * i
                end = bs * (i+1) if i < batch_num - 1 else low_latents.size(0) 
                high_sdf[start:end,:,:,:] = -high_sdf[start:end,:,:,:]
            # n_xyz = np.ceil((np.asarray(self.bound_max.cpu()) - np.asarray(self.bound_min.cpu())) / self.voxel_size).astype(int).tolist()

            n_xyz = [self.xyz.size(0),self.xyz.size(0),self.xyz.size(0)]
            # indexer , involved xyz, latent index batch id
            print('before marching')
            vertices, vertices_flatten_id,vertices_std = system.ext.marching_cubes_interp(
                                                                    layer_id_valid_l0, indexer.view(n_xyz),
                                                                    high_sdf, high_std,max_n_triangles, n_xyz,
                                                                    max_std)  # (T, 3, 3), (T, ), (T, 3)
            print('after marching')
            print(vertices.shape)
            vertices = vertices * self.voxel_size  + self.bound_min
            vertices = vertices.cpu().numpy()
            print(vertices.shape)
            
            # Remove relevant cached vertices and append updated/new ones.
            vertices_flatten_id = vertices_flatten_id.cpu().numpy()
            vertices_std = vertices_std.cpu().numpy()
            if self.mesh_cache.vertices is None:
                self.mesh_cache.vertices = vertices
                self.mesh_cache.vertices_flatten_id = vertices_flatten_id
                self.mesh_cache.vertices_std = vertices_std
            else:
                p = np.sort(np.unique(vertices_flatten_id))
                valid_verts_idx = _get_valid_idx(self.mesh_cache.vertices_flatten_id, p)
                self.mesh_cache.vertices = np.concatenate([self.mesh_cache.vertices[valid_verts_idx], vertices], axis=0)
                self.mesh_cache.vertices_flatten_id = np.concatenate([
                    self.mesh_cache.vertices_flatten_id[valid_verts_idx], vertices_flatten_id
                ], axis=0)
                    
            
        do_meshing(voxel_resolution)
        return self._make_mesh_from_cache()

    def extract_mesh_from_point(self, voxel_resolution: int, xyz, sdf,max_n_triangles: int,
                    max_std: float = 2000.0):
        def do_meshing(voxel_resolution,xyz,sdf):
            
            layer_id_valid = self.occ_right_flag == 1    # mask of layer_id  
            layer_id_valid = torch.arange(0, self.occ_right_flag.size(0), device=self.device,
                                                                dtype=torch.long)[layer_id_valid]
            layer_id_valid = self._linearize_id(xyz).long()

            # get every valid id in layer 0 and map into batch id
            
            # build xyz -> batch id
            indexer = torch.ones(self.occ_right_flag.size(0), device=self.device, dtype=torch.long) * -1 
            indexer[layer_id_valid] = torch.arange(0, layer_id_valid.size(0), device=self.device,
                                                                dtype=torch.long)
            
            # get each pos of voxel xyz in range(128,128,128) in bigger layer voxel 
            
            
            
            
            
            B = layer_id_valid.size(0)
            sample_a = 0
            sample_b = 1
            low_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b)
            low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)
        
            high_sdf = sdf
            high_sdf = high_sdf.view(B,voxel_resolution,voxel_resolution,voxel_resolution)
            high_std = torch.ones_like(high_sdf) * 0.1
            high_sdf = - high_sdf
            # n_xyz = np.ceil((np.asarray(self.bound_max.cpu()) - np.asarray(self.bound_min.cpu())) / self.voxel_size).astype(int).tolist()

            n_xyz = [self.xyz.size(0),self.xyz.size(0),self.xyz.size(0)]
            # indexer , involved xyz, latent index batch id
            print('before marching')
            # print(layer_id_valid_l0)
            # sys.exit(0)
            vertices, vertices_flatten_id = system.ext.marching_cubes(
                                                                    layer_id_valid,indexer.view(n_xyz),
                                                                    high_sdf,max_n_triangles, n_xyz
                                                                    )  # (T, 3, 3), (T, ), (T, 3)
            print('after marching')
            print(vertices.shape)
            vertices = vertices * self.voxel_size  + self.bound_min
            vertices = vertices.cpu().numpy()
            print(vertices.shape)
            
            # Remove relevant cached vertices and append updated/new ones.
            vertices_flatten_id = vertices_flatten_id.cpu().numpy()
            # vertices_std = vertices_std.cpu().numpy()
            if self.mesh_cache.vertices is None:
                self.mesh_cache.vertices = vertices
                self.mesh_cache.vertices_flatten_id = vertices_flatten_id
                # self.mesh_cache.vertices_std = vertices_std
            # else:
            #     p = np.sort(np.unique(vertices_flatten_id))
            #     valid_verts_idx = _get_valid_idx(self.mesh_cache.vertices_flatten_id, p)
            #     self.mesh_cache.vertices = np.concatenate([self.mesh_cache.vertices[valid_verts_idx], vertices], axis=0)
            #     self.mesh_cache.vertices_flatten_id = np.concatenate([
            #         self.mesh_cache.vertices_flatten_id[valid_verts_idx], vertices_flatten_id
            #     ], axis=0)
                    
            
        do_meshing(voxel_resolution,xyz,sdf)
        return self._make_mesh_from_cache()

    def extract_whole_mesh_corner(self, voxel_resolution: int, max_n_triangles: int,
                    max_std: float = 2000.0,extract_async: bool = False,use_rgb = False):
        def do_meshing(voxel_resolution):
            
            layer_id_valid = self.occ_right_flag == 1    # mask of layer_id  
            layer_id_valid = torch.arange(0, self.occ_right_flag.size(0), device=self.device,
                                                                dtype=torch.long)[layer_id_valid]
            
    
            # get every valid id in layer 0 and map into batch id
            
            # build xyz -> batch id
            indexer = torch.ones(self.occ_right_flag.size(0), device=self.device, dtype=torch.long) * -1 
            indexer[layer_id_valid] = torch.arange(0, layer_id_valid.size(0), device=self.device,
                                                                dtype=torch.long)
            
            # get each pos of voxel xyz in range(128,128,128) in bigger layer voxel 
            upper_xyz = self._unlinearize_id(layer_id_valid)
            
            
            B = upper_xyz.size(0)
            sample_a = 0
            sample_b = 1 - 1/ voxel_resolution
            low_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b)
            low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)
            
            low_samples = low_samples.view(-1, low_samples.size(-1))
            low_center = upper_xyz.unsqueeze(1).repeat(1, voxel_resolution ** 3, 1).view(-1,3)

            xyz_samples = low_center + low_samples 
            batch_num = int(math.ceil(xyz_samples.size(0)/(2**23)))
            bs = int(xyz_samples.size(0)/batch_num)
            start = 0
            end = min(bs,xyz_samples.size(0))
            low_sdf = []
            for i in range(0,batch_num):
                start = bs * i
                end = bs * (i+1) if i < batch_num - 1 else xyz_samples.size(0) 
                low_latents_b,_ = self.trilinear_interpolate(xyz_samples[start:end,:],low_center[start:end,:])

                low_sdf_b = net_util.forward_model(self.model.decoder,max_sample = 2 ** 25,
                                                    latent_input=low_latents_b,
                                                    xyz_input=low_samples[start:end,:])[0]
                low_sdf.append(-low_sdf_b)
            low_sdf = torch.cat(low_sdf,dim = 0)
            high_sdf = low_sdf.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
            
            n_xyz = [self.n_xyz_L[0][0],self.n_xyz_L[0][1],self.n_xyz_L[0][2]]

            torch.cuda.synchronize()
            self.end = time.time()
            # print("d ok!")
            s = time.time()
            vertices, vertices_flatten_id = system.ext.marching_cubes(
                                                                    layer_id_valid,indexer.view(n_xyz),
                                                                    high_sdf,max_n_triangles, n_xyz
                                                                    )  # (T, 3, 3), (T, ), (T, 3)
            # vertices, vertices_flatten_id = system.ext.marching_cubes_interp(
            #                                                         layer_id_valid,indexer.view(n_xyz),
            #                                                         high_sdf,max_n_triangles, n_xyz
            #                                                         )  # (T, 3, 3), (T, ), (T, 3)
            t = time.time() - s
            # print(f"marching cubes {float(t) * 1000}")
            if use_rgb:
                center = torch.floor(vertices.view(-1,3))
                relative_points = vertices.view(-1,3) - center
                
                rgb_latents,mask = self.trilinear_interpolate(vertices.view(-1,3),center,rgb=True)
                # print(rgb_latents.shape,relative_points.shape)
                rgbs = net_util.forward_model(self.model.rgb_decoder,max_sample = 2 ** 20,
                    latent_input = rgb_latents,
                    xyz_input = relative_points[mask],layer = None)[0]
                cs = torch.zeros_like(vertices.view(-1,3),device=vertices.device)
                cs[mask] = rgbs
                self.mesh_cache.colors = cs.cpu().numpy()
            # print('after marching')
            # print(vertices.shape)
            vertices = vertices * self.voxel_size  + self.bound_min
            vertices = vertices.view(-1,3).to(torch.float64)
            triangles = torch.arange(vertices.shape[0]).reshape((-1, 3)).int()
            vertices = vertices.cpu().numpy()
            # print(vertices.shape)
            # print('after copy')
            
            # Remove relevant cached vertices and append updated/new ones.
            vertices_flatten_id = vertices_flatten_id.cpu().numpy()

            self.mesh_cache.vertices = vertices
            self.mesh_cache.triangles = triangles.cpu().numpy()
            self.mesh_cache.vertices_flatten_id = vertices_flatten_id

            
                    
            
        do_meshing(voxel_resolution)
        return self._make_mesh_from_cache()

    def extract_whole_mesh_corner_from_point(self, voxel_resolution: int, max_n_triangles: int,
                    points,normals):
        def do_meshing(voxel_resolution,points,normals):
            points_xyz_aligned = points - self.bound_min.unsqueeze(0)
            points_xyz_aligned = points_xyz_aligned / self.voxel_size
            gathered_points_latent_idx= self._linearize_id(torch.floor(points_xyz_aligned)).long()
            unique_ids = torch.unique(gathered_points_latent_idx)
            unique_xyz = self._unlinearize_id(unique_ids,L=0).long()
            B = unique_ids.size(0)
            # build xyz -> batch id
            indexer = torch.ones(self.occ_right_flag.size(0), device=self.device, dtype=torch.long) * -1 
            indexer[unique_ids] = torch.arange(0, B, device=self.device,
                                                                dtype=torch.long)
            
            # get each pos of voxel xyz in range(128,128,128) in bigger layer voxel 
           
            sample_a = 0
            sample_b = 0.9999
            low_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b) - 0.5
            
            low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)
            low_center = unique_xyz.unsqueeze(1).repeat(1,voxel_resolution ** 3,1) + 0.5
            # print(low_center.shape)
            pc = (low_center + low_samples).view(-1,3).contiguous()
            # print(pc)
            
            pc = pc * self.voxel_size + self.bound_min.unsqueeze(0)

            pc = torch.cat([pc,torch.zeros((pc.size(0),1)).cuda()],dim = 1).contiguous()
            gt_pc = torch.cat([points,torch.zeros((points.size(0),1)).cuda()],dim = 1).contiguous()
            
            gt_sdf = system.ext.compute_sdf(pc,gt_pc,normals,self.voxel_size,self.voxel_size * 2)
            gt_sdf = gt_sdf.view(B,voxel_resolution,voxel_resolution,voxel_resolution)
            # print(gt_sdf[~torch.isnan(gt_sdf)])
            
            gt_sdf = - gt_sdf
            # n_xyz = np.ceil((np.asarray(self.bound_max.cpu()) - np.asarray(self.bound_min.cpu())) / self.voxel_size).astype(int).tolist()

            n_xyz = [self.xyz.size(0),self.xyz.size(0),self.xyz.size(0)]
            # indexer , involved xyz, latent index batch id
            print('before marching')
            vertices, vertices_flatten_id = system.ext.marching_cubes(
                                                                    unique_ids,indexer.view(n_xyz),
                                                                    gt_sdf,max_n_triangles, n_xyz
                                                                    )  # (T, 3, 3), (T, ), (T, 3)
            print('after marching')
            print(vertices.shape)
            vertices = vertices * self.voxel_size  + self.bound_min
            
            print(vertices.shape)
            
            # Remove relevant cached vertices and append updated/new ones.
            vertices_flatten_id = vertices_flatten_id.cpu().numpy()

            self.mesh_cache.vertices = vertices
            self.mesh_cache.vertices_flatten_id = vertices_flatten_id

            
                    
            
        do_meshing(voxel_resolution,points,normals)
        return self._make_mesh_from_cache()
    
    def vis_latent(self,voxel_resolution):
        with torch.no_grad():
            layer_id_valid = self.occ_right_flag == 1    # mask of layer_id  
            layer_id_valid = torch.arange(0, self.occ_right_flag.size(0), device=self.device,
                                                                dtype=torch.long)[layer_id_valid]
            

            # get every valid id in layer 0 and map into batch id
            
            # build xyz -> batch id
            indexer = torch.ones(self.occ_right_flag.size(0), device=self.device, dtype=torch.long) * -1 
            indexer[layer_id_valid] = torch.arange(0, layer_id_valid.size(0), device=self.device,
                                                                dtype=torch.long)
            
            # get each pos of voxel xyz in range(128,128,128) in bigger layer voxel 
            upper_xyz = self._unlinearize_id(layer_id_valid)
            
            
            B = upper_xyz.size(0)
            sample_a = 0
            sample_b = 1
            low_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b)
            low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)
            
            low_samples = low_samples.view(-1, low_samples.size(-1))
            low_center = upper_xyz.unsqueeze(1).repeat(1, voxel_resolution ** 3, 1).view(-1,3)

            xyz_samples = low_center + low_samples 
            low_latents,_ = self.trilinear_interpolate(xyz_samples,low_center)
            X = low_latents.cpu().numpy()
            pca = PCA(n_components=1)
            vert_color = pca.fit_transform(X)
            
            vcolor_min = np.min(vert_color)
            vcolor_max = np.max(vert_color)
            vert_color = (vert_color - vcolor_min) / (vcolor_max - vcolor_min)
            print(vert_color.shape)
            vert_color = matplotlib.cm.jet(vert_color.reshape(-1))[:, :3]
            xyz_samples = xyz_samples * self.voxel_size + self.bound_min
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_samples.cpu().numpy()))
            pcd.colors = o3d.utility.Vector3dVector(vert_color)
        return pcd 
    