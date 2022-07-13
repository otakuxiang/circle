#include <torch/extension.h>

std::vector<torch::Tensor> generate_rays(
    std::vector<int> img_size,
    float fx,float fy,
    float cx,float cy
);

std::vector<torch::Tensor> sparse_ray_intersection(
    std::vector<torch::Tensor> octree_xyzs, // voxel_xyzs for each node in layer
    std::vector<torch::Tensor> octree_sons, // voxel_sons for each node in layer
    torch::Tensor origin_point, // camera t
    torch::Tensor directions,
    torch::Tensor directions_inv,
    torch::Tensor& rv_pointer
);

std::vector<torch::Tensor> build_octree(
    std::vector<torch::Tensor> octree_xyzs,
    std::vector<torch::Tensor> father_ids,
    std::vector<torch::Tensor> octree_sons
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("marching_cubes_sparse", &marching_cubes_sparse_cuda, "Marching Cubes Sparse (CUDA)");
    m.def("generate_rays", &generate_rays, "generate rays use CUDA");
    m.def("sparse_ray_intersection", &sparse_ray_intersection," use octree to do fast ray intersection");
    m.def("build_octree",&build_octree,"build ray_intersection needed octree");
    
}
