#include <torch/extension.h>
#include <vector>

// Declare the definitions of the CUDA kernel function calls
void linear_transform(torch::Tensor h, torch::Tensor W, torch::Tensor h_prime, int N, int F, int F_prime);
void compute_attention_coeff(torch::Tensor Wh, torch::Tensor a, torch::Tensor e, int N, int F_prime);
void softmax_kernel(torch::Tensor e, torch::Tensor alpha, int N);
void aggregate_features(torch::Tensor alpha, torch::Tensor Wh, torch::Tensor h_prime, int N, int F_prime);

// Define the PyTorch interface for the custom CUDA operations
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_transform", &linear_transform, "Linear Transform");
  m.def("compute_attention_coeff", &compute_attention_coeff, "Attention Coefficients");
  m.def("softmax_kernel", &softmax_kernel, "Softmax");
  m.def("aggregate_features", &aggregate_features, "Feature Aggregation");
}
