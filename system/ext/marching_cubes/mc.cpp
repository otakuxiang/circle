#include <torch/extension.h>

std::vector<torch::Tensor> marching_cubes_sparse_cuda(
        torch::Tensor valid_blocks,         // (K, )     
        torch::Tensor batch_indexer,    // (nx,ny,nz) -> batch id
        torch::Tensor cube_sdf,             // (M, rx, ry, rz)
        int max_n_triangles,                // Maximum number of triangle buffer.
        const std::vector<int>& n_xyz      // [nx, ny, nz]
);

std::vector<torch::Tensor> marching_cubes_sparse_interp_cuda(
        torch::Tensor valid_blocks,         // (K, )     
        torch::Tensor batch_indexer,    // (nx,ny,nz) -> batch id
        torch::Tensor cube_sdf,             // (M, rx, ry, rz)
        int max_n_triangles,                // Maximum number of triangle buffer.
        const std::vector<int>& n_xyz      // [nx, ny, nz]
);

torch::Tensor marching_cubes_dense_cuda(
        torch::Tensor valid_cords,
        torch::Tensor dense_sdf,
        torch::Tensor mask,
        int max_n_triangles
);

std::vector<torch::Tensor> marching_cubes_dense_cpu(
        torch::Tensor valid_cords,
        torch::Tensor dense_sdf,
        torch::Tensor mask,
        int max_n_triangles
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cubes_sparse", &marching_cubes_sparse_cuda, "Marching Cubes Sparse (CUDA)");
    m.def("marching_cubes_sparse_interp", &marching_cubes_sparse_interp_cuda, "Marching Cubes with Interpolation (CUDA)");
    m.def("marching_cubes_dense", &marching_cubes_dense_cuda, "Dense Marching Cubes with mask ");
    
}
