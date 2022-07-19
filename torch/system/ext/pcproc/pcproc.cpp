#include <torch/extension.h>

torch::Tensor setEmptyVoxels(
        torch::Tensor input_pc,
        const std::vector<float>& cur_pos,
        torch::Tensor indexer,
        torch::Tensor layer_offsets
);

torch::Tensor remove_radius_outlier(
        torch::Tensor input_pc,
        int nb_points,
        float radius
);

torch::Tensor estimate_normals(
        torch::Tensor input_pc,
        int max_nn,
        float radius,
        const std::vector<float>& cam_xyz
);

torch::Tensor compute_sdf(torch::Tensor input_pc, 
        torch::Tensor gt_pc, 
        torch::Tensor gt_normal
        ,float voxel_size,float radius);
        
torch::Tensor valid_mask(torch::Tensor input_pc, 
        torch::Tensor gt_pc, 
        float radius);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("remove_radius_outlier", &remove_radius_outlier, "Remove point outliers by radius (CUDA)");
    m.def("estimate_normals", &estimate_normals, "Estimate point cloud normals (CUDA)");
    m.def("setEmptyVoxels", &setEmptyVoxels, "using ray tracing to set empty voxels");
    m.def("compute_sdf", &compute_sdf,"compute sdf of gt");
    m.def("valid_mask", &valid_mask,"find valid points");
}