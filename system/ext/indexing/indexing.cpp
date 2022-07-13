#include <torch/extension.h>

torch::Tensor pack_batch(torch::Tensor indices, uint n_batch, uint n_point);
torch::Tensor groupby_max(torch::Tensor values, torch::Tensor indices, uint C);
std::vector<torch::Tensor> groupby_sum(torch::Tensor values, torch::Tensor indices, uint C);
torch::Tensor groupby_sum_backward(torch::Tensor grad_out, torch::Tensor grad_in,torch::Tensor sample_indexer);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_batch", &pack_batch, "Pack Batch (CUDA)");
    m.def("groupby_max", &groupby_max, "GroupBy Max (CUDA)");
    m.def("groupby_sum", &groupby_sum, "GroupBy Sum (CUDA)");
    m.def("groupby_sum_backward",&groupby_sum_backward,"backward of groupBy sum");
}
