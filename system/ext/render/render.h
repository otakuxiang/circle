#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString(err) );
        exit(-1);
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) {CHECK_CUDA(x); CHECK_CONTIGUOUS(x)}
#define ORcudaKernelCheck { cudaDeviceSynchronize(); __cudaSafeCall(cudaPeekAtLastError(), __FILE__, __LINE__); }

struct RVPair{
    float t; // length of ray
    int ray_id;
    int voxel_id;
    __host__ __device__
    RVPair(){
        t = -1.0;
        ray_id = -1;
        voxel_id = -1;
    }
    __host__ __device__ 
    inline bool operator <(const RVPair &a) const{
        return ray_id == a.ray_id ? t < a.t : ray_id < a.ray_id;
    }
};

struct is_unhitted{
    __host__ __device__
    bool operator()(const RVPair &a){
        return a.voxel_id == -1;
    }
};

static uint div_up(const uint a, const uint b) {
    return (a + b - 1) / b;
}