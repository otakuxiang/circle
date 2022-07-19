#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include "cutil_math.h"
#include "render.h"

bool intersection(int3 xyz, float3 r_inv,float3 ray_ori) {
    float3 lb = make_float3(xyz.x,xyz.y,xyz.z);
    float3 rt = lb + 1;

    double t1 = (lb.x - ray_ori.x)*r_inv.x;
    double t2 = (rt.x - ray_ori.x)*r_inv.x;

    double tmin = min(t1, t2);
    double tmax = max(t1, t2);
    t1 = (lb.y - ray_ori.y)*r_inv.y;
    t2 = (rt.y - ray_ori.y)*r_inv.y;
    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
    t1 = (lb.z - ray_ori.z)*r_inv.z;
    t2 = (rt.z - ray_ori.z)*r_inv.z;
    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
    
    return tmax > max(tmin, 0.0);
}

int sparse_ray_intersection_cpu(
    std::vector<torch::Tensor> octree_xyzs, // voxel_xyzs for each node in layer
    std::vector<float> origin_point, // camera t
    torch::Tensor directions,
    torch::Tensor directions_inv
){
    // printf("input ok!\n");

    CHECK_INPUT(directions);
    CHECK_INPUT(directions_inv);
    int layer_num = octree_xyzs.size();
    for (int i = 0;i < layer_num;++i){
        CHECK_INPUT(octree_xyzs[i]);
        // if(i != layer_num - 1){
        //     CHECK_INPUT(octree_sons[i]);
        // } 
    }
    // printf("check ok!\n");
    auto r_invs = directions_inv.to(device(torch::kCPU));
    torch::Tensor xyzs = octree_xyzs[layer_num - 1].to(device(torch::kCPU));
    float3 ray_origin = make_float3(origin_point[0],origin_point[1],origin_point[2]);
    int ray_num = directions.size(0);
    int voxel_num = octree_xyzs[layer_num - 1].size(0);
    // thrust::device_vector<RVPair> parent_rvpairs(voxel_num * ray_num);
    // printf("array ok!\n");
    auto xyz = xyzs.accessor<int, 2>();
    auto d_inv = r_invs.accessor<float, 2>();
    thrust::host_vector<int> is_hit(ray_num);
    
    for(int i = 0;i < ray_num;++i){
        float3 r_inv = make_float3(d_inv[i][0],d_inv[i][1],d_inv[i][2]);  
        for (int j = 0;j < voxel_num;++j){
            int3 b = make_int3(xyz[j][0],xyz[j][1],xyz[j][2]);
            if(intersection(b,r_inv,ray_origin)){
                is_hit[i] = 1;
                break;
            }
        }
    } 
    thrust::device_vector<int> hit_d = is_hit;
    int sum = thrust::reduce(hit_d.begin(),hit_d.end());
    return sum;
    // dim3 dimBlock = dim3(512);
    // dim3 dimGrid = dim3(div_up(ray_num,dimBlock.x));
    // RVPair *parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
    // rvpair_init_kernel<<<dimGrid,dimBlock>>>(
    //     parent_rvpairs_array,voxel_num,ray_num
    // );
    // // ORcudaKernelCheck
    // // printf("init ok!\n");
    // thrust::device_vector<RVPair> son_rvpairs;
    // RVPair *son_rvpairs_array = thrust::raw_pointer_cast(son_rvpairs.data());
    // for(int i = 0;i < layer_num; ++i){
    //     // decide if the rvpair is hitted
    //     float r = 1 << (layer_num - i - 1);
        
    //     dimGrid = dim3(div_up(parent_rvpairs.size(),dimBlock.x));
    //     parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
    //     // printf("r : %f\n",r);
    //     ray_aabb_kernel<<<dimGrid,dimBlock>>>(
    //         parent_rvpairs_array,
    //         directions_inv.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
    //         octree_xyzs[i].packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
    //         r,ray_origin,parent_rvpairs.size()
    //     );
    //     ORcudaKernelCheck
    //     // printf("ray aabb ok!\n");

    //     // remove unhitted rvpairs
    //     // printf("before remove %d pairs\n",parent_rvpairs.size());
    //     parent_rvpairs.erase(thrust::remove_if(parent_rvpairs.begin(),parent_rvpairs.end(),is_unhitted()),parent_rvpairs.end());
    //     // printf("remove ok! remain %d pairs\n",parent_rvpairs.size());
        
    //     parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
    //     // if current layer is not lowest layer subdivide it 
    //     if(i != layer_num - 1){
    //         son_rvpairs.resize(parent_rvpairs.size() * 8);
    //         dimGrid = dim3(div_up(son_rvpairs.size(),dimBlock.x));
    //         son_rvpairs_array = thrust::raw_pointer_cast(son_rvpairs.data());
    //         subdivide_kernel<<<dimGrid,dimBlock>>>(
    //             parent_rvpairs_array,
    //             octree_sons[i].packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
    //             parent_rvpairs.size(),
    //             son_rvpairs_array
    //         );
    //         ORcudaKernelCheck
    //         // printf("subdivide ok!\n");
    //         // remove the empty pairs
    //         // printf("before remove %d pairs\n",son_rvpairs.size());
    //         son_rvpairs.erase(thrust::remove_if(son_rvpairs.begin(),son_rvpairs.end(),is_unhitted()),son_rvpairs.end());
    //         // printf("remove ok! remain %d pairs\n",son_rvpairs.size());
            
    //         parent_rvpairs.swap(son_rvpairs);
    //     }
    //     // else{
    //     //     // else we convert voxel_id to the index of voxel_latent
    //     //     dimGrid = dim3(div_up(parent_rvpairs.size(),dimBlock.x));
    //     //     convert_voxel_id_kernel<<<dimGrid,dimBlock>>>(
    //     //         parent_rvpairs_array,
    //     //         octree_id.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
    //     //         parent_rvpairs.size()
    //     //     );
    //     // }   
    // } 
    
    // // // finally we sort the rvpairs and get 3 tensor ray_voxel_pair 
    // // // ray_voxel_d and rv_pointer
    // thrust::sort(parent_rvpairs.begin(),parent_rvpairs.end());
    // parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
    // int pair_num = parent_rvpairs.size();
    // torch::Tensor ray_voxel_pair = torch::empty({(long)pair_num,2}, torch::dtype(torch::kLong).device(torch::kCUDA));
    // torch::Tensor ray_voxel_d = torch::empty({(long)pair_num}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    // dimGrid = dim3(div_up(pair_num,dimBlock.x));
    // rvpair_to_tensor_kernel<<<dimGrid,dimBlock>>>(
    //     parent_rvpairs_array,
    //     ray_voxel_pair.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
    //     ray_voxel_d.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
    //     pair_num
    // );
    // ORcudaKernelCheck
    // // // get rv_pointer
    // scan_to_get_pointer_kernel<<<dimGrid,dimBlock>>>(
    //     parent_rvpairs_array,pair_num,
    //     rv_pointer.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
    // );
    // ORcudaKernelCheck
    // return {ray_voxel_pair,ray_voxel_d,rv_pointer};
}