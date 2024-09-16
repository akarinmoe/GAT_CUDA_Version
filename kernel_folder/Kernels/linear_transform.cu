#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

// Define the CUDA kernel function
__global__ void linear_transform_kernel(float* h, float* W, float* h_prime, int N, int F, int F_prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * F_prime) {
        int i = idx / F_prime;
        int j = idx % F_prime;
        float sum = 0.0f;
        for (int k = 0; k < F; ++k) {
            sum += h[i * F + k] * W[k * F_prime + j];
        }
        h_prime[idx] = sum;
    }
}

// Function exposed to the C++ side
void linear_transform(torch::Tensor h, torch::Tensor W, torch::Tensor h_prime, int N, int F, int F_prime) {
    const int threads = 1024;
    const int blocks = (N * F_prime + threads - 1) / threads;

    // Launch the CUDA kernel
    linear_transform_kernel<<<blocks, threads>>>(
        h.data_ptr<float>(),
        W.data_ptr<float>(),
        h_prime.data_ptr<float>(),
        N, F, F_prime
    );
}
