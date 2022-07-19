import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait,ALL_COMPLETED, ProcessPoolExecutor
import open3d as o3d
import numpy as np
import shutil
import ast
from evaluation import EvaluateHisto
from joblib import parallel_backend

result_path="/home/chx/data_disk/MatterPort3D/test/"
gt_path="/home/chx/data_disk/MatterPort3D/v1/scans/"

def run_evaluation(input_ply, gt_ply,dTau):
    '''
    print("")
    print("===========================")
    print("Evaluating")
    print("===========================")
    '''
    pcd = input_ply.sample_points_uniformly(500000)
    gt_pcd = gt_ply.sample_points_uniformly(500000)
    dist_threshold = dTau
    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        gt_pcd,
        dist_threshold / 2.0,
        dist_threshold,
        plot_stretch,
    )
    eva = [precision, recall, fscore]
    '''
    print("==============================")
    print("evaluation result " )
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")
    '''
    return eva

def compute_fscore(scene,region,ablate_name,names,mesh_names):
    print(scene+":"+region)
    local_gt_path=os.path.join(gt_path,scene,"region_segmentations")
    gt_mesh=os.path.join(local_gt_path,region+".ply")
    local_result_path=os.path.join(result_path,scene,region,"result")
    if len(os.listdir(local_result_path)) == 0:
        return
    meshs = []
    for mesh_name in mesh_names:
        meshs.append(os.path.join(local_result_path,mesh_name))

    scene_names = names
    gt_ply = o3d.io.read_triangle_mesh(gt_mesh)
    gt_ply = gt_ply.compute_vertex_normals()
    result_dict = {}
    for idx,mesh in enumerate(meshs): 
        input_ply = o3d.io.read_triangle_mesh(mesh)
        if np.asarray(input_ply.vertices).shape[0] == 0:
            continue
        input_ply = input_ply.compute_vertex_normals()
        # print("compute_f_score")
        with parallel_backend("loky"):
            eva = run_evaluation(input_ply, gt_ply,0.02)
        # print("compute_f_score_ok")

        result_dict[scene_names[idx]] = eva 
    f = open(os.path.join(local_result_path,f"fscore.txt"),"w+")
    f.write(str(result_dict))
    f.close()

def compute_rmse(scene,region,ablate_name,names,mesh_names):
    print(scene+":"+region)
    local_gt_path=os.path.join(gt_path,scene,"region_segmentations")
    gt_mesh=os.path.join(local_gt_path,region+".ply")
    local_result_path=os.path.join(result_path,scene,region,ablate_name)

    meshs = []
    for mesh_name in mesh_names:
        meshs.append(os.path.join(local_result_path,mesh_name))

    scene_names = names
    
    gt_ply = o3d.io.read_triangle_mesh(gt_mesh)
    gt_ply = gt_ply.compute_vertex_normals()
    result_dict = {}
    for idx,mesh in enumerate(meshs): 
        input_ply = o3d.io.read_triangle_mesh(mesh)
        
        if np.asarray(input_ply.vertices).shape[0] == 0:
            continue
        input_ply = input_ply.compute_vertex_normals()
        pcd = input_ply.sample_points_uniformly(2000000)
        gt_pcd = gt_ply.sample_points_uniformly(2000000)
        s = pcd.voxel_down_sample(0.005)
        t = gt_pcd.voxel_down_sample(0.005)
        with parallel_backend("loky"):
            dist = np.asarray(s.compute_point_cloud_distance(t))
            dist = dist[dist < 0.1]
            rmse = np.sqrt(np.average(dist*dist))
        result_dict[scene_names[idx]] = rmse 
    f = open(os.path.join(local_result_path,f"{ablate_name}_rmse.txt"),"w+")
    f.write(str(result_dict))
    f.close()

def compute_chamfer(scene,region,ablate_name,names,mesh_names):
    print(scene+":"+region)
    local_gt_path=os.path.join(gt_path,scene,"region_segmentations")
    gt_mesh=os.path.join(local_gt_path,region+".ply")
    local_result_path=os.path.join(result_path,scene,region,ablate_name)

    meshs = []
    for mesh_name in mesh_names:
        meshs.append(os.path.join(local_result_path,mesh_name))

    scene_names = names
    gt_ply = o3d.io.read_triangle_mesh(gt_mesh)
    gt_ply = gt_ply.compute_vertex_normals()
    result_dict = {}
    for idx,mesh in enumerate(meshs): 
        input_ply = o3d.io.read_triangle_mesh(mesh)
        input_ply = input_ply.compute_vertex_normals()
        if np.asarray(input_ply.vertices).shape[0] == 0:
            continue
        pcd = input_ply.sample_points_uniformly(2000000)
        gt_pcd = gt_ply.sample_points_uniformly(2000000)
        s = pcd.voxel_down_sample(0.005)
        t = gt_pcd.voxel_down_sample(0.005)
        with parallel_backend("loky"):
            dist = np.asarray(s.compute_point_cloud_distance(t))
            dist = dist[np.where(dist<0.1)]
            s2t = np.average(dist*dist)
            dist = np.asarray(t.compute_point_cloud_distance(s))
            dist = dist[np.where(dist<0.1)]
            t2s = np.average(dist*dist)
        chamfer=s2t+t2s      
        result_dict[scene_names[idx]] = chamfer
    f = open(os.path.join(local_result_path,f"{ablate_name}_chamfer.txt"),"w+")
    f.write(str(result_dict))
    f.close()

def processCutPcd(scene,region):
    local_result_path=os.path.join(result_path,scene,region)
    input_mesh = os.path.join(local_result_path,"input_mesh.ply")
    spsg_mesh=os.path.join(local_result_path,"spsg_mesh.ply")
    onet_mesh=os.path.join(local_result_path,"conv_onet.off")
    routed_mesh=os.path.join(local_result_path,"routed_mesh.ply")
    spsg_mesh = o3d.io.read_triangle_mesh(spsg_mesh)
    onet_mesh = o3d.io.read_triangle_mesh(onet_mesh)    
    routed_mesh = o3d.io.read_triangle_mesh(routed_mesh)    
    # print("read ok")
    max_height = spsg_mesh.get_max_bound()[2]
    min_b = onet_mesh.get_min_bound()
    max_b = onet_mesh.get_max_bound()
    max_b[2] = max_height
    bound = o3d.geometry.AxisAlignedBoundingBox(min_b,max_b)
    onet_mesh = onet_mesh.crop(bound)
    o3d.io.write_triangle_mesh(os.path.join(local_result_path,"conv_onet.off"),onet_mesh)
    min_b = routed_mesh.get_min_bound()
    max_b = routed_mesh.get_max_bound()
    max_b[2] = max_height
    bound = o3d.geometry.AxisAlignedBoundingBox(min_b,max_b)
    routed_mesh = routed_mesh.crop(bound)
    o3d.io.write_triangle_mesh(os.path.join(local_result_path,"routed_mesh.ply"),routed_mesh)

def compute_one_fscore(scene,region,approach,mesh_name,ablate_name):
    local_gt_path=os.path.join(gt_path,scene,"region_segmentations")
    gt_mesh=os.path.join(local_gt_path,region+".ply")
    local_result_path=os.path.join(result_path,scene,region,ablate_name)
    our_mesh = os.path.join(local_result_path,mesh_name)
    
    meshs = []

    meshs.append(our_mesh)
    scene_names = [approach]
    gt_ply = o3d.io.read_triangle_mesh(gt_mesh)
    f = open(os.path.join(local_result_path,f"{ablate_name}_fscore.txt"))

    result_dict = eval(f.read())
    f.close()
    for idx,mesh in enumerate(meshs): 
        input_ply = o3d.io.read_triangle_mesh(mesh)
        if np.asarray(input_ply.vertices).shape[0] == 0:
            continue
        eva = run_evaluation(input_ply, gt_ply,0.02)
        result_dict[scene_names[idx]] = eva 
    f = open(os.path.join(local_result_path,f"{ablate_name}_fscore.txt"),"w+")
    f.write(str(result_dict))
    f.close()

def compute_one_rmse(scene,region,approach,mesh_name,ablate_name):
    local_gt_path=os.path.join(gt_path,scene,"region_segmentations")
    gt_mesh=os.path.join(local_gt_path,region+".ply")
    local_result_path=os.path.join(result_path,scene,region,ablate_name)
    our_mesh = os.path.join(local_result_path,mesh_name)
    
    meshs = []

    meshs.append(our_mesh)
    scene_names = [approach]
    gt_ply = o3d.io.read_triangle_mesh(gt_mesh)
    f = open(os.path.join(local_result_path,"rmse.txt"))
    result_dict = eval(f.read())
    f.close()
    for idx,mesh in enumerate(meshs): 
        input_ply = o3d.io.read_triangle_mesh(mesh)
        if np.asarray(input_ply.vertices).shape[0] == 0:
            continue
        pcd = input_ply.sample_points_uniformly(2000000)
        gt_pcd = gt_ply
        s = pcd.voxel_down_sample(0.005)
        t = gt_pcd.voxel_down_sample(0.005)
        dist = np.asarray(s.compute_point_cloud_distance(t))
        dist = dist[np.where(dist<0.1)]
        rmse = np.sqrt(np.average(dist*dist))
        result_dict[scene_names[idx]] = rmse 
    f = open(os.path.join(local_result_path,"rmse.txt"),"w")
    f.write(str(result_dict))
    f.close()

def compute_one_chamfer(scene,region,approach,mesh_name,ablate_name):
    local_gt_path=os.path.join(gt_path,scene,"region_segmentations")
    gt_mesh=os.path.join(local_gt_path,region+".ply")
    local_result_path=os.path.join(result_path,scene,region,ablate_name)
    our_mesh = os.path.join(local_result_path,mesh_name)
    
    meshs = []

    meshs.append(our_mesh)
    scene_names = [approach]
    gt_ply = o3d.io.read_triangle_mesh(gt_mesh)
    f = open(os.path.join(local_result_path,f"{ablate_name}_rmse.txt"))
    result_dict = eval(f.read())
    f.close()
    for idx,mesh in enumerate(meshs): 
        input_ply = o3d.io.read_triangle_mesh(mesh)
        if np.asarray(input_ply.vertices).shape[0] == 0:
            continue
        pcd = input_ply.sample_points_uniformly(2000000)
        gt_pcd = gt_ply
        s = pcd.voxel_down_sample(0.005)
        t = gt_pcd.voxel_down_sample(0.005)

        dist = np.asarray(s.compute_point_cloud_distance(t))
        dist = dist[dist<0.1]
        s2t = np.average(dist*dist)
        
        dist = np.asarray(t.compute_point_cloud_distance(s))
        dist = dist[dist<0.1]
        t2s = np.average(dist*dist)
        chamfer=s2t+t2s      
        result_dict[scene_names[idx]] = chamfer
    f = open(os.path.join(local_result_path,f"{ablate_name}_rmse.txt"),"w+")
    f.write(str(result_dict))
    f.close()

from tqdm import tqdm
def compare_scenenn():
    data_dir = "/home/chx/data_disk/scenenn_processed/"
    scenes = os.listdir(data_dir)
    names = ['con','di','our_opt','our','routed','spsg']
    mesh_names = ['conv_onet.ply','difusion.ply','our_mesh_opt_no_pose.ply','ours.ply','routed.ply','spsg.ply']
    gt_dir = "/home/chx/data_disk/scenenn_data/"
    # scenes = scenes[:len(scenes) // 2]
    scenes = scenes[len(scenes) // 2:]

    for scene in tqdm(scenes):
        tqdm.write(scene)
        result_path = os.path.join("/home/chx/data_disk/scenenn_result/",scene)
        gt_path = os.path.join(gt_dir,scene,f"{scene}.ply")
        gt_ply = o3d.io.read_triangle_mesh(gt_path)
        # gt_ply = gt_ply.compute_vertex_normals()
        gt_pcd = gt_ply.sample_points_uniformly(500000)
        fscore_dict = {}
        chamfer_dict = {}
        rmse_dict = {}
        for name,mesh in zip(names,mesh_names): 
            input_ply = o3d.io.read_triangle_mesh(os.path.join(result_path,mesh))
            if np.asarray(input_ply.vertices).shape[0] == 0:
                continue
            # input_ply = input_ply.compute_vertex_normals()
            pcd = input_ply.sample_points_uniformly(500000)
            fscore_eva = run_evaluation(input_ply, gt_ply,0.02)
            fscore_dict[name] = fscore_eva
            s = pcd.voxel_down_sample(0.005)
            t = gt_pcd.voxel_down_sample(0.005)
            dist = np.asarray(s.compute_point_cloud_distance(t))
            dist = dist[dist < 0.1]
            rmse = np.sqrt(np.average(dist*dist))
            rmse_dict[name] = rmse 
            dist = np.asarray(s.compute_point_cloud_distance(t))
            dist = dist[np.where(dist<0.1)]
            s2t = np.average(dist*dist)
            dist = np.asarray(t.compute_point_cloud_distance(s))
            dist = dist[np.where(dist<0.1)]
            t2s = np.average(dist*dist)
            chamfer=s2t+t2s      
            chamfer_dict[name] = chamfer
        with open(os.path.join(result_path,"rmse.txt"),"w+") as f:
            f.write(str(rmse_dict))
        with open(os.path.join(result_path,"fscore.txt"),"w+") as f:
            f.write(str(fscore_dict))
        with open(os.path.join(result_path,"chamfer.txt"),"w+") as f:
            f.write(str(chamfer_dict))            
compare_scenenn()

# f = open("/home/chx/data_disk/MatterPort3D/scenes_test.txt")
# scenes = [line.strip() for line in f]
# # print(scenes)
# # f.closcenesse()
# scenes = scenes[11:14]
# # names = ["v50","v75","v100","v75_w50","v75_w100"]
# # meshes = ["our_v50","our_v75","our_v100","our_v75_w50","our_v75_w100"]
# # ablate_name = "voxel"

# names = ["10_5","10_10","10_15","10_20","25_5","25_10","25_15","25_20","50_5","50_10","50_15","50_20"]
# meshes = [f"ours_{name}.ply" for name in names]
# names += ["tsdf_10","tsdf_25","tsdf_50"]
# meshes += ["tsdf_10.ply","tsdf_25.ply","tsdf_50.ply"]
# ablate_name = "split"

# all_tasks=[]
# executor = ProcessPoolExecutor(max_workers=8)
# for scene in scenes:
#     regions=os.listdir(os.path.join("/home/chx/data_disk/MatterPort3D/test/",scene))

#     # all_task=[executor.submit(compute_fscore,scene,region,ablate_name,names,meshes) for region in regions]
#     # all_tasks.extend(all_task)
#     # all_task=[executor.submit(compute_rmse,scene,region,ablate_name,names,meshes) for region in regions]
#     # all_tasks.extend(all_task)
#     # all_task=[executor.submit(compute_chamfer,scene,region,ablate_name,names,meshes) for region in regions]
#     # all_tasks.extend(all_task)
#     # all_tasks.extend([executor.submit(processCutPcd,scene,region) for region in regions])
#     for region in regions:
#         compute_fscore(scene,region,ablate_name,names,meshes)
#     # all_tasks.extend([executor.submit(compute_one_fscore,scene,region,"ours","our_mesh.ply") for region in regions])
#     # all_tasks.extend([executor.submit(compute_one_rmse,scene,region,"ours","our_mesh.ply") for region in regions])
#     # all_tasks.extend([executor.submit(compute_one_chamfer,scene,region,"ours","our_mesh.ply") for region in regions])

# # wait(all_tasks,return_when=ALL_COMPLETED)
# # print('hi')
# # compute_fscore("2t7WUuJeko7","region3")