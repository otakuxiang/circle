import open3d as o3d
import os,torch,pdb,traceback,sys
import numpy as np
from tqdm import tqdm
import shutil,cv2,geoopt
from system.ext import generate_rays
from data_proc import data_utils
from network.utility import StepLearningRateSchedule,adjust_learning_rate
from network.diff_renderer import diff_renderer
import structure.octree.unet_oct as oct
from network import utility
import math

data_dir = "/home/chx/data_disk/MatterPort3d"

def add_translate_noise(pose,std = 0.02):
    t_rand = np.random.normal(0, 1, size=[3]).astype(np.float32) * std
    pose[0:3,2] = pose[0:3,2] + t_rand
    return pose

def add_rotate_noise(pose,std = 2):
    rand_v = np.random.normal(0, 1, size=[3,1]).astype(np.float32)
    rand_v = rand_v / np.linalg.norm(rand_v)
    rand_r = np.random.normal(0, 1, size=[1]).astype(np.float32) * std * math.pi / 180
    temp_R = np.array([[0,-rand_v[2,0],rand_v[1,0]],[rand_v[2,0],0,-rand_v[0,0]],[-rand_v[1,0],rand_v[0,0],0]])
    rand_R = math.cos(rand_r) * np.eye(3) + (1 - math.cos(rand_r)) * np.dot(rand_v,rand_v.T) + math.sin(rand_r) * temp_R
    pose[0:3,0:3] = rand_R.dot(pose[0:3,0:3])
    return pose


def get_iter(tensor_list,fixed = -1):
    for idx,t in enumerate(tensor_list):
        if idx != fixed:
            yield t

def get_input_cloud(partial_points,dRs,dts,Rs,ts):
    final_pcd = o3d.geometry.PointCloud()
    for idx,pcd in enumerate(partial_points):

        # R = (dRs[idx] @ Rs[idx]).detach()
        # t = (dts[idx].permute(1,0) + ts[idx]).detach()
        T = np.eye(4)
        R = (dRs[idx] @ Rs[idx]).detach()
        t = (dts[idx].permute(1,0) + ts[idx]).detach()
        # R = Rs[idx]
        # t = ts[idx]
        T[0:3,0:3] = R.cpu().numpy()
        T[0:3,3:] = t.cpu().numpy()
        # pcd = pcd.translate(-ts[idx].cpu().numpy()
        # ).rotate(dRs[idx].detach().cpu().numpy(),center=np.zeros([3,]))
        # pcd = pcd.translate(t.cpu().numpy())
        # pcd = pcd.rotate(R.cpu().numpy(),center = np.zeros([3,]))
        # pcd = pcd.translate(t.cpu().numpy())
        final_pcd += o3d.geometry.PointCloud(pcd).transform(T)
        final_pcd = final_pcd.voxel_down_sample(0.01)    
    return final_pcd


def get_optimizer(opt_left,opt_pose,octree,dRs,dts):
    lr_schedules = []
    params_group = []
    if not opt_left:
        slr = StepLearningRateSchedule(1e-3,150,0.9)
        lr_schedules.append(slr)
        params_group.append({ "params": octree.latent_vecs_right_corner.latent_vecs, "lr": slr.get_learning_rate(0)})
        octree.latent_vecs_right_corner.requires_grad_()
    else:
        slr = StepLearningRateSchedule(1e-2,200,0.9)
        lr_schedules.append(slr)
        params_group.append({ "params": octree.latent_vecs_left.latent_vecs, "lr": slr.get_learning_rate(0)})
        
        octree.latent_vecs_left.requires_grad_()
    if opt_pose:
        # slr0 = StepLearningRateSchedule(5e-4,200,0.9)
        slr = StepLearningRateSchedule(5e-4,200,0.9)
        # lr_schedules.append(slr0)
        lr_schedules.append(slr)
        # params_group.append({"params": get_iter(dRs,fixed=-1), "lr": slr0.get_learning_rate(0)})
        params_group.append({"params": get_iter(dts,fixed=-1), "lr": slr.get_learning_rate(0)})
        optimizer_all = geoopt.optim.RiemannianAdam(params_group)
    else:
        optimizer_all = torch.optim.Adam(params_group)
    return optimizer_all,lr_schedules


def get_inputs(scene,region):
    
    sequence_path = os.path.join(data_dir,"v1/scans",scene)
    region_path = os.path.join(data_dir,"v1/data_pro",scene,region)
    f_file = open(os.path.join(region_path,"frame_list.txt"))
    frame_list = [line.strip().split(" ") for line in f_file]
    depth_path = os.path.join(region_path,"depth")
    pose_path = os.path.join(sequence_path,"matterport_camera_poses")
    intri_path = os.path.join(sequence_path,"matterport_camera_intrinsics")

    pose_ids = []
    Rs = []
    ts = []
    ray_ods = []
    targets = []
    final_pcd = o3d.geometry.PointCloud()
    point_parts = []
    poses = []
    for frame_id,frames in enumerate(tqdm(frame_list)):
        depth = os.path.join(depth_path,f"{frame_id}.png")
        depth = cv2.imread(depth,-1)
        depth = cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / 4000
        depth = torch.from_numpy(depth).cuda()
        depth[torch.logical_or(depth>4.0,depth<0.1)] = np.nan
        tripod_number,camera_id,frame_idx = frames
        f = open(os.path.join(pose_path,tripod_number+"_pose_"+camera_id+"_"+frame_idx+".txt"))
        pose = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")
            for k in range(0,4):
                pose[idx,k] = float(ss[k])
        f.close()
        # pose = np.linalg.inv(pose)
        pose = add_translate_noise(pose,0.03)
        # pose = add_rotate_noise(pose)
        f = open(os.path.join(intri_path,tripod_number+"_intrinsics_"+camera_id+".txt"))
        p = np.zeros((4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")   
            for j in range(4):
                p[j] = float(ss[j+2])
        f.close()
        p /= 2 
        pose = torch.from_numpy(pose).float().unsqueeze(0)
        Rs.append(pose[0:,0:3,0:3])
        ts.append(pose[0:,0:3,3:4])
        directions,ray_xys = generate_rays([h,w],
            p[0],p[1],p[2],p[3]    
        )
        ray_od,target,_,_ = data_utils.gather_input([depth],directions,ray_xys,[p[0],p[1],p[2],p[3]])
        nan_mask = ~torch.isnan(target)
        ray_od = ray_od[nan_mask,:]
        target = target[nan_mask]
        ray_od[:,0,:] = ray_od[:,0,:] / ray_od[:,0,:].norm(dim = 1,keepdim = True)
        ray_ods.append(ray_od.cpu())
        targets.append(target.cpu())
        pose_ids.append(torch.ones(ray_od.size(0),dtype=torch.long) * frame_id)
        part_pcd = ray_od[:,0,:] * target.unsqueeze(-1)
        part_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part_pcd.cpu().numpy()))
        part_pcd.remove_radius_outlier(16,0.05)
        part_pcd.estimate_normals()
        part_pcd.orient_normals_towards_camera_location(np.zeros([3,1]))
        part_pcd = part_pcd.remove_none_finite_points()
        point_parts.append(o3d.geometry.PointCloud(part_pcd))
        poses.append(pose.squeeze(0).cpu().numpy())
        pcd = part_pcd.transform(pose.squeeze(0).cpu().numpy())
        final_pcd += pcd
        final_pcd = final_pcd.voxel_down_sample(0.01)  
    final_pcd = final_pcd.remove_none_finite_points()
    o3d.io.write_point_cloud(os.path.join(result_path,"temp.ply"),final_pcd)
    ray_ods = torch.cat(ray_ods)
    targets = torch.cat(targets)
    pose_ids = torch.cat(pose_ids)
    Rs = torch.cat(Rs)
    ts = torch.cat(ts)
    return point_parts,final_pcd,ray_ods,targets,pose_ids,Rs,ts


def shuffle_tensors(tensor_list):
    inds = np.arange(0,tensor_list[0].size(0),dtype=np.int64)
    np.random.shuffle(inds)
    inds = torch.from_numpy(inds)
    for idx,tensor in enumerate(tensor_list):
        tensor_list[idx] = tensor[inds]
    return tensor_list

count = 1

def update_octree(octree,pcd,result_path):
    global count
    main_device = torch.device("cuda:0")
    points=torch.from_numpy(np.asarray(pcd.points)).to(main_device).float()
    min_bound = torch.min(points,dim = 0)[0] - 0.5 * voxel_size
    bound = (pcd.get_max_bound() - pcd.get_min_bound())
    normals = torch.from_numpy(np.asarray(pcd.normals)).to(main_device).float()
    octree = oct.unet_oct_cube(model,octree.device,latent_dim=29,layer_num = 4,voxel_size = octree.voxel_size,renderer = octree.renderer,bound = bound)
    with torch.no_grad():
        octree.bound_min = min_bound
        octree.update_lowest(points,normals)
        octree.update_right_corner()
        mesh = octree.extract_whole_mesh_corner(6,use_rgb = False,max_n_triangles = 2 ** 23)
        mesh = mesh.merge_close_vertices(0.001)
        mesh = mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(result_path,f"our_mesh_{count}.ply"),mesh)
        count += 1
    return octree


scene = "5ZKStnWn8Zo"
region = "region6"
result_path = os.path.join(data_dir,"result",scene,region)
os.makedirs(result_path,exist_ok=True)

h,w = 512,640
voxel_size = 0.05
# model,_ = utility.load_unet_model("pre-trained-weight/noisy_clamp_descend/hyper_small.json",100000,use_nerf= False,layer=5)
model,_ = utility.load_unet_model("pre-trained-weight/layer_5_small_decoder/hyper_small.json",100000,use_nerf= False,layer = 5)

opt_left = False
opt_pose = True
num_epoch = 20 if opt_pose else 1

# model.eval()
main_device = torch.device("cuda:0")
renderer = diff_renderer(model,h,w,voxel_size,main_device,is_eval = False)

point_parts,final_pcd,ray_ods,targets,pose_ids,Rs,ts = get_inputs(scene,region)
Rs = Rs.cuda()
ts = ts.cuda()
dRs = []
dts = []
for i in range(len(point_parts)):
    dRs.append(geoopt.ManifoldTensor(torch.eye(3,device = main_device),manifold=geoopt.Stiefel()).requires_grad_())
    dts.append(torch.zeros(1,3).to(main_device).requires_grad_())
points=torch.from_numpy(np.asarray(final_pcd.points)).to(main_device).float()
min_bound = torch.min(points,dim = 0)[0] - 0.5 * voxel_size
bound = (final_pcd.get_max_bound() - final_pcd.get_min_bound())
normals = torch.from_numpy(np.asarray(final_pcd.normals)).to(main_device).float()
octree = oct.unet_oct_cube(model,main_device,latent_dim=29,layer_num = 4,voxel_size = voxel_size,renderer = renderer,bound = bound)
with torch.no_grad():
    octree.bound_min = min_bound
    octree.update_lowest(points,normals)
    octree.update_right_corner()
    mesh = octree.extract_whole_mesh_corner(6,use_rgb = False,max_n_triangles = 2 ** 23)
    mesh = mesh.merge_close_vertices(0.001)
    mesh = mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(result_path,"our_mesh_init.ply"),mesh)

optimizer_all,lr_schedules = get_optimizer(opt_left,opt_pose,octree,dRs,dts)

ray_ods,targets,pose_ids = shuffle_tensors([ray_ods,targets,pose_ids])
num = ray_ods.shape[0] // 120000
ray_od_batch = torch.chunk(ray_ods,num)
target_batch = torch.chunk(targets,num)
pose_id_batch = torch.chunk(pose_ids,num)
final_pcd = o3d.geometry.PointCloud()

for epoch in range(num_epoch):
    retain_ray = None
    retain_target = None
    dRs0 = torch.cat([R.unsqueeze(0) for R in dRs],dim=0).detach()
    dts0 = torch.cat(dts,dim=0).unsqueeze(-1).detach()
    trace = dRs0[:,0,0] + dRs0[:,1,1] + dRs0[:,2,2]
    # final_pcd = get_input_cloud(point_parts,dRs,dts,Rs,ts,poses)
    # o3d.io.write_point_cloud(os.path.join(result_path,f"input_opt_points_{epoch}.ply"),final_pcd)
    # # print(dts0.view(-1).cpu().numpy().tolist(),(((trace - 1)/2).acos() * 180 / math.pi).view(-1).cpu().numpy().tolist())
    if torch.any(dts0 > 0.07) or torch.any(((trace - 1)/2).acos() * 180 / math.pi > 5):
        final_pcd = get_input_cloud(point_parts,dRs,dts,Rs,ts)
        o3d.io.write_point_cloud(os.path.join(result_path,f"opt_points_{count}.ply"),final_pcd)
        octree = update_octree(octree,final_pcd,result_path)
        Rs = dRs0 @ Rs
        ts = dts0 + ts
        dRs = []
        dts = []
        for idx in range(dRs0.size(0)):
            dRs.append(geoopt.ManifoldTensor(torch.eye(3,device = main_device),manifold=geoopt.Stiefel()).requires_grad_())
            dts.append(torch.zeros(1,3).to(main_device).requires_grad_())
            optimizer_all,lr_schedules = get_optimizer(opt_left,opt_pose,octree,dRs,dts)

    # final_pcd = o3d.geometry.PointCloud()

    for idx,[ray_od,target,pose_id] in enumerate(zip(ray_od_batch,target_batch,pose_id_batch)):
        try:
            pose_id = pose_id.cuda()
            ray_od = ray_od.cuda()
            target = target.cuda()

            it = idx + len(ray_od_batch) * epoch
            optimizer_all.zero_grad()
            adjust_learning_rate(lr_schedules, optimizer_all,it )
            dR = torch.stack(dRs) # N X 3 X 3
            dt = torch.cat(dts).unsqueeze(1) # N X 1 X 3
            r = dR[pose_id] @ Rs[pose_id]
            t = dt[pose_id] + ts[pose_id].permute(0,2,1)
            # r = Rs[pose_id]
            # t = ts[pose_id].permute(0,2,1)
            # print(ray_od.shape)
            ray = torch.cat([ray_od,t],dim = 1)
            ray[:,0,:] = torch.matmul(r,ray[:,0,:].unsqueeze((-1))).squeeze(-1).contiguous()
            
            # if retain_ray is not None:
            #     ray = torch.cat([ray,retain_ray],dim = 0)
            #     targets = torch.cat([targets,retain_target],dim = 0)
            # sys.exit(0)
            # if epoch == 0 :
            #     points = ray[:,1,:] + ray[:,0,:] * target.unsqueeze(-1)
            #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.detach().cpu().numpy()))
            #     final_pcd += pcd
            #     continue

            ray,target = octree.mask_rays(ray,target)
            
            if ray.size(0) == 0:
                # print("0!")
                continue
            if opt_left:
                octree.update_right_corner()
            depth_img = octree.train_render_depth(ray.clone())
            
            if depth_img is None:
                continue
            zero_mask = torch.logical_and(depth_img >= 0.05,target >= 0.05)
            
            ray = ray[zero_mask,:,:] 
            src = depth_img[zero_mask] 
            tar = target[zero_mask]
            # points = ray[:,1,:] + ray[:,0,:] * src.unsqueeze(-1)
            # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.detach().cpu().numpy()))
            # final_pcd += pcd
            
            res = src - tar
            loss = res.abs().mean()
            loss.backward()
            # print(dts[0].grad,dts[0].requires_grad)
            optimizer_all.step()
            # print(f"{epoch}\{idx}: {loss.detach().cpu().numpy()}")

            # retain_mask = res.abs() > 0.01
            # retain_ray =  ray[retain_mask,:,:]
            # retain_target = tar[retain_mask]
            if opt_left:
                octree.latent_vecs_right_corner.detach()
        except Exception as ex:
            traceback.print_exc()
            pdb.post_mortem(ex.__traceback__)
            sys.exit(0)
    # o3d.io.write_point_cloud(os.path.join(result_path,f"opt_points.ply"),final_pcd)

            
    if epoch % 1 == 0:
        print(f"{epoch}\{20}: {loss.detach().cpu().numpy()}")
            # mesh = octree.extract_whole_mesh_corner(6,use_rgb = False,max_n_triangles = 2 ** 22)
            # mesh = mesh.merge_close_vertices(0.01)
            # mesh = mesh.remove_degenerate_triangles()
            # mesh.compute_vertex_normals()
            # # o3d.io.write_triangle_mesh(os.path.join(scene_path,"our_mesh_opt.ply"),mesh)
            # o3d.io.write_triangle_mesh(f"test/our_mesh_opt_{idx}.ply",mesh)
# final_pcd = final_pcd.voxel_down_sample(0.01)   
# o3d.io.write_point_cloud(os.path.join(result_path,"temp.ply"),final_pcd)

mesh = octree.extract_whole_mesh_corner(6,use_rgb = False,max_n_triangles = 2 ** 22)
mesh = mesh.merge_close_vertices(0.001)
mesh = mesh.remove_degenerate_triangles()
mesh.compute_vertex_normals()
if not opt_pose:
    o3d.io.write_triangle_mesh(os.path.join(result_path,"our_mesh_opt_no_pose.ply"),mesh)
else:
    o3d.io.write_triangle_mesh(os.path.join(result_path,"our_mesh_opt_with_pose.ply"),mesh)
