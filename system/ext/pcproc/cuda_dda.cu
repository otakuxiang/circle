#include <cuda.h>
#include <limits>
#include "cutil_math.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
using IndexAccesser = torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>;
using PCAccessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;


static uint div_up(const uint a, const uint b) {
    return (a + b - 1) / b;
}

__device__ u_int32_t inline linearize_id(u_int32_t x,u_int32_t y,u_int32_t z,const IndexAccesser layer_offsets,u_char layer){
    u_int32_t offset = 0;
    u_int32_t dim = 128;
    if(layer > 0){
        u_int32_t scale = pow(2,layer);
        x /= scale;
        y /= scale;
        z /= scale;
        offset = layer_offsets[layer - 1];
        dim /= pow(2,layer);
    }
    return z + y * dim + x * dim * dim + offset;
}

// __device__ void 

__global__ void dda_kernel(const PCAccessor input_pc,float3 cam_pos,IndexAccesser indexer,const IndexAccesser layer_offsets){
    float dx,dy,dz,step,x,y,z;
    const uint pc_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pc_id >= input_pc.size(0)) {
        return;
    }
    dx = input_pc[pc_id][0] - cam_pos.x;
    dy = input_pc[pc_id][1] - cam_pos.y;
    dz = input_pc[pc_id][2] - cam_pos.z;
    step = max(max(fabs(dx),fabs(dy)),fabs(dz));
    int i = 1;
    dx /= step;
    dy /= step;
    dz /= step;
    x = cam_pos.x;
    y = cam_pos.y;
    z = cam_pos.z;

    while(i <= step){
        x += dx;
        y += dy;
        z += dz;
        u_int32_t tx = ceil(x) - 1,ty = ceil(y) - 1,tz = ceil(z) - 1;
        u_int32_t id = linearize_id(tx,ty,tz,layer_offsets,0);
        if (indexer[id] == -1){
            indexer[id] = -2;
        }
        // if(i <= step - 3 && indexer[id] >= 0){
        //     indexer[id] = -2;
        // }
        i++;    
    }    
    return;
}   

torch::Tensor setEmptyVoxels(
        torch::Tensor input_pc,
        const std::vector<float>& cur_pos,
        torch::Tensor indexer,
        torch::Tensor layer_offsets
){
    CHECK_INPUT(input_pc);
    CHECK_INPUT(indexer);
    CHECK_INPUT(layer_offsets);
    size_t n_point = input_pc.size(0);
    
    dim3 dimBlock = dim3(128);
    dim3 dimGrid = dim3(div_up(n_point, dimBlock.x));

    dda_kernel<<<dimGrid, dimBlock>>>(input_pc.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        ,make_float3(cur_pos[0],cur_pos[1],cur_pos[2]),
        indexer.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>()
        ,layer_offsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>());

    return indexer;
}