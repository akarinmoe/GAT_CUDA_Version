#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

// Define the kernel function
__global__ void aggregate_features_kernel(float* alpha, float* Wh, float* h_prime, int N, int F_prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * F_prime) {
        int i = idx / F_prime;
        int j = idx % F_prime;
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += alpha[i * N + k] * Wh[k * F_prime + j];
        }
        h_prime[idx] = sum;
    }
}

// Define the function called from C++
// This function will be exposed to the C++ side
void aggregate_features(torch::Tensor alpha, torch::Tensor Wh, torch::Tensor h_prime, int N, int F_prime) {
    const int threads = 1024;
    const int blocks = (N * F_prime + threads - 1) / threads;

    // Launch the CUDA kernel
    aggregate_features_kernel<<<blocks, threads>>>(
        alpha.data_ptr<float>(),
        Wh.data_ptr<float>(),
        h_prime.data_ptr<float>(),
        N, F_prime
    );
}
