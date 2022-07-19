#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include "cutil_math.h"
#include "render.h"

using DAccessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using CAccessor = torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>;
using IDAccessor = torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>;
using Long2Accessor = torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits>;
using Float1Accessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;



__global__ void rvpair_init_kernel(
    RVPair* rvpairs,int voxel_num,int ray_num
){
    const uint ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(ray_id >= ray_num) return;
    for(int i = 0;i < voxel_num;++i){
        int pair_id = ray_id * voxel_num + i;
        rvpairs[pair_id].ray_id = ray_id;
        rvpairs[pair_id].voxel_id = i;
    }
}

__global__ void ray_aabb_kernel(
    RVPair* rvpairs,const DAccessor directions_inv,
    const CAccessor octree_xyz,float r,
    DAccessor ray_ori_tensor,int pair_num
){
    const uint pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair_id >= pair_num) return;
    const uint voxel_id = rvpairs[pair_id].voxel_id;
    const uint ray_id = rvpairs[pair_id].ray_id;
    float3 ray_ori = make_float3(ray_ori_tensor[ray_id][0],
        ray_ori_tensor[ray_id][1],ray_ori_tensor[ray_id][2]);
    float3 lb = make_float3(octree_xyz[voxel_id][0],octree_xyz[voxel_id][1],octree_xyz[voxel_id][2]);
    float3 rt = lb + r;
    float t1 = (lb.x - ray_ori.x)*directions_inv[ray_id][0];
    float t2 = (rt.x - ray_ori.x)*directions_inv[ray_id][0];
    float t3 = (lb.y - ray_ori.y)*directions_inv[ray_id][1];
    float t4 = (rt.y - ray_ori.y)*directions_inv[ray_id][1];
    float t5 = (lb.z - ray_ori.z)*directions_inv[ray_id][2];
    float t6 = (rt.z - ray_ori.z)*directions_inv[ray_id][2];
    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    // if (tmax < 0)
    // {
    //     rvpairs[pair_id].voxel_id = -1;
    //     // printf("%d\n",pair_id);
    //     return ;
    // }
    // if (tmin > tmax)
    // {
    //     rvpairs[pair_id].voxel_id = -1;
    //     // printf("%d\n",pair_id);
    //     return ;
    // }
    if (tmax > max(tmin, 0.0)){
        rvpairs[pair_id].t = tmin;
    }
    else{
        rvpairs[pair_id].voxel_id = -1;
    }
    
}

__global__ void subdivide_kernel(
    RVPair* parent_rvpairs,
    const CAccessor sons,
    int pair_num,
    RVPair* son_rvpairs
){
    const uint pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair_id >= pair_num) return;
    const uint voxel_id = parent_rvpairs[pair_id].voxel_id;
    const uint ray_id = parent_rvpairs[pair_id].ray_id;
    for (int i = 0; i < 8; ++i){
        const uint s_pair_id = pair_id * 8 + i;
        const uint s_voxel_id = sons[voxel_id][i];
        son_rvpairs[s_pair_id].ray_id = ray_id;
        son_rvpairs[s_pair_id].voxel_id = s_voxel_id;
    }
}

// __global__ void convert_voxel_id_kernel(
//     RVPair* rvpairs,
//     const IDAccessor octree_id,
//     int pair_num
// ){
//     const uint pair_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if(pair_id >= pair_num) return;
//     const uint voxel_id = rvpairs[pair_id].voxel_id;
//     rvpairs[pair_id].voxel_id = octree_id[voxel_id];
// }

__global__ void rvpair_to_tensor_kernel(
    RVPair* rvpairs,
    Long2Accessor ray_voxel_pair,
    Float1Accessor ray_voxel_d,
    int pair_num
){
    const uint pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair_id >= pair_num) return;
    const uint voxel_id = rvpairs[pair_id].voxel_id;
    const uint ray_id = rvpairs[pair_id].ray_id;
    const float t = rvpairs[pair_id].t;
    ray_voxel_pair[pair_id][0] = voxel_id;
    ray_voxel_pair[pair_id][1] = ray_id;
    ray_voxel_d[pair_id] = t;
}

// get a tensor with (n,2) point the start and end index of each ray 
__global__ void scan_to_get_pointer_kernel(
    RVPair* rvpairs,
    int pair_num,
    Long2Accessor rv_pointer
){  
    const uint pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair_id >= pair_num) return;
    const uint ray_id = rvpairs[pair_id].ray_id;
    if(pair_id == pair_num - 1){
        rv_pointer[ray_id][1] = pair_id + 1;
        
    }
    else if(pair_id == 0){
        rv_pointer[ray_id][0] = pair_id;
    }
    else{
        const uint ray_id_next = rvpairs[pair_id + 1].ray_id;
        if(ray_id_next != ray_id){
            rv_pointer[ray_id][1] = pair_id + 1;
            rv_pointer[ray_id_next][0] = pair_id+1;
        }
    }
}
__global__ void generate_ray_kernel(float fx_inv,float fy_inv,float cx,float cy,
    DAccessor directions,Long2Accessor ray_xys
){
    uint u = blockIdx.x,v = threadIdx.x;
    float3 direction = make_float3(0.0f,0.0f,1.0f);
    direction.x = (u - cx) * fx_inv;
    direction.y = (v - cy) * fy_inv;
    // direction = normalize(direction);
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    ray_xys[ray_id][0] = u;
    ray_xys[ray_id][1] = v;
    directions[ray_id][0] = direction.x;
    directions[ray_id][1] = direction.y;
    directions[ray_id][2] = direction.z;
}

__global__ void build_son_array_kernel(
    const CAccessor octree_xyz,
    const IDAccessor father_id,
    int son_num,
    int r,
    CAccessor son_array
){
    const uint son_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (son_id >= son_num) return;
    const uint f_id = father_id[son_id];
    const uint x = octree_xyz[son_id][0] / r,y = octree_xyz[son_id][1] / r
        ,z = octree_xyz[son_id][2] / r;
    int3 father_xyz = make_int3(x,y,z);
    father_xyz.x = father_xyz.x / 2 * 2;
    father_xyz.y = father_xyz.y / 2 * 2;
    father_xyz.z = father_xyz.z / 2 * 2;

    uint offset = (x - father_xyz.x) * 4 + (y - father_xyz.y) * 2 +
        (z - father_xyz.z);
    son_array[f_id][offset] = son_id;
}

std::vector<torch::Tensor> generate_rays(
    std::vector<int> img_size,
    float fx,float fy,
    float cx,float cy

){
    int h = img_size[0];
    int w = img_size[1];

    torch::Tensor ray_xys = torch::empty({(long) h*w,2}, torch::dtype(torch::kLong).device(torch::kCUDA));
    torch::Tensor directions = torch::empty({(long) h*w,3}, torch::dtype(torch::kFloat).device(torch::kCUDA));torch::empty({(long) h*w}, torch::dtype(torch::kInt).device(torch::kCUDA));
    
    // there should be h*w / dimblock but for 320*240 or 640*480 it's ok
    
    dim3 dimBlock = dim3(h);
    dim3 dimGrid = dim3(w);
    float fx_inv = 1.0 / fx,fy_inv = 1.0 / fy;
    generate_ray_kernel<<<dimGrid,dimBlock>>>(fx_inv,fy_inv,cx,cy,
        directions.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_xys.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
    );
    return {directions,ray_xys};
}


std::vector<torch::Tensor> sparse_ray_intersection(
    std::vector<torch::Tensor> octree_xyzs, // voxel_xyzs for each node in layer
    std::vector<torch::Tensor> octree_sons, // voxel_sons for each node in layer
    torch::Tensor origin_point,             // camera t
    torch::Tensor directions,
    torch::Tensor directions_inv,
    torch::Tensor& rv_pointer
){  
    // printf("input ok!\n");

    CHECK_INPUT(directions);
    CHECK_INPUT(directions_inv);
    int layer_num = octree_xyzs.size();
    // printf("layer num %d\n",layer_num);
    for (int i = 0;i < layer_num;++i){
        CHECK_INPUT(octree_xyzs[i]);
        // printf("%d\n",i);
        if(i != layer_num - 1){
            CHECK_INPUT(octree_sons[i]);
        } 
    }
    // printf("check ok!\n");

    int ray_num = directions.size(0);
    int voxel_num = octree_xyzs[0].size(0);
    thrust::device_vector<RVPair> parent_rvpairs(voxel_num * ray_num);
    // printf("array ok!\n");
    
    dim3 dimBlock = dim3(512);
    dim3 dimGrid = dim3(div_up(ray_num,dimBlock.x));
    RVPair *parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
    rvpair_init_kernel<<<dimGrid,dimBlock>>>(
        parent_rvpairs_array,voxel_num,ray_num
    );
    // ORcudaKernelCheck
    // printf("init ok!\n");
    thrust::device_vector<RVPair> son_rvpairs;
    RVPair *son_rvpairs_array = thrust::raw_pointer_cast(son_rvpairs.data());
    bool zero_flag = false; 
    for(int i = 0;i < layer_num; ++i){
        // decide if the rvpair is hitted
        float r = 1 << (layer_num - i - 1);
        if (parent_rvpairs.size() == 0){
            zero_flag = true;
            break;
        }
        dimGrid = dim3(div_up(parent_rvpairs.size(),dimBlock.x));
        parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
        // printf("r : %f\n",r);
        
        ray_aabb_kernel<<<dimGrid,dimBlock>>>(
            parent_rvpairs_array,
            directions_inv.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            octree_xyzs[i].packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            r,origin_point.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            parent_rvpairs.size()
        );
        ORcudaKernelCheck
        // printf("ray aabb ok!\n");

        // remove unhitted rvpairs
        // printf("before remove %d pairs\n",parent_rvpairs.size());
        parent_rvpairs.erase(thrust::remove_if(parent_rvpairs.begin(),parent_rvpairs.end(),is_unhitted()),parent_rvpairs.end());
        // printf("remove ok! remain %d pairs\n",parent_rvpairs.size());
        
        parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
        // if current layer is not lowest layer subdivide it 
        if(i != layer_num - 1){
            son_rvpairs.resize(parent_rvpairs.size() * 8);
            dimGrid = dim3(div_up(son_rvpairs.size(),dimBlock.x));
            
            son_rvpairs_array = thrust::raw_pointer_cast(son_rvpairs.data());
            subdivide_kernel<<<dimGrid,dimBlock>>>(
                parent_rvpairs_array,
                octree_sons[i].packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                parent_rvpairs.size(),
                son_rvpairs_array
            );
            ORcudaKernelCheck
            // printf("subdivide ok!\n");
            // remove the empty pairs
            // printf("before remove %d pairs\n",son_rvpairs.size());
            son_rvpairs.erase(thrust::remove_if(son_rvpairs.begin(),son_rvpairs.end(),is_unhitted()),son_rvpairs.end());
            // printf("remove ok! remain %d pairs\n",son_rvpairs.size());
            
            parent_rvpairs.swap(son_rvpairs);
        }
        // else{
        //     // else we convert voxel_id to the index of voxel_latent
        //     dimGrid = dim3(div_up(parent_rvpairs.size(),dimBlock.x));
        //     convert_voxel_id_kernel<<<dimGrid,dimBlock>>>(
        //         parent_rvpairs_array,
        //         octree_id.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        //         parent_rvpairs.size()
        //     );
        // }   
    } 
    if (zero_flag){
        return{ torch::zeros({1}),torch::zeros({1}),rv_pointer};
    }
    // // finally we sort the rvpairs and get 3 tensor ray_voxel_pair 
    // // ray_voxel_d and rv_pointer
    thrust::sort(parent_rvpairs.begin(),parent_rvpairs.end());
    parent_rvpairs_array = thrust::raw_pointer_cast(parent_rvpairs.data());
    int pair_num = parent_rvpairs.size();
    torch::Tensor ray_voxel_pair = torch::empty({(long)pair_num,2}, torch::dtype(torch::kLong).device(torch::kCUDA));
    torch::Tensor ray_voxel_d = torch::empty({(long)pair_num}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    dimGrid = dim3(div_up(pair_num,dimBlock.x));
    rvpair_to_tensor_kernel<<<dimGrid,dimBlock>>>(
        parent_rvpairs_array,
        ray_voxel_pair.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        ray_voxel_d.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        pair_num
    );
    ORcudaKernelCheck
    // // get rv_pointer
    scan_to_get_pointer_kernel<<<dimGrid,dimBlock>>>(
        parent_rvpairs_array,pair_num,
        rv_pointer.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
    );
    ORcudaKernelCheck
    return {ray_voxel_pair,ray_voxel_d,rv_pointer};
}

std::vector<torch::Tensor> build_octree(
    std::vector<torch::Tensor> octree_xyzs,
    std::vector<torch::Tensor> father_ids,
    std::vector<torch::Tensor> octree_sons
){
    int layer_num = octree_xyzs.size();
    for (int i = 0;i < layer_num;++i){
        CHECK_INPUT(octree_xyzs[i]);
        // printf("%d\n",i);
        if(i != layer_num - 1){
            CHECK_INPUT(octree_sons[i]);
            CHECK_INPUT(father_ids[i]);
        } 
    }
    dim3 dimBlock = dim3(512);
    int rr = 1;
    for(int i = octree_xyzs.size() - 1; i > 0; --i){
        
        dim3 dimGrid = dim3(div_up(octree_xyzs[i].size(0),dimBlock.x));
        build_son_array_kernel<<<dimGrid,dimBlock>>>(
            octree_xyzs[i].packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            father_ids[i -1].packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            octree_xyzs[i].size(0),
            rr,
            octree_sons[i - 1].packed_accessor32<int, 2, torch::RestrictPtrTraits>()
        );
        rr *= 2;
    }
    return octree_sons;
} 