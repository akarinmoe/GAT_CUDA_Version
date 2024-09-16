#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

// Define the kernel function
__global__ void compute_attention_coeff_kernel(float* Wh, float* a, float* e, int N, int F_prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int i = idx / N;
        int j = idx % N;
        float e_ij = 0.0f;
        for (int k = 0; k < F_prime; ++k) {
            e_ij += Wh[i * F_prime + k] * a[k] + Wh[j * F_prime + k] * a[F_prime + k];
        }
        e[idx] = e_ij > 0 ? e_ij : 0.2f * e_ij;  // Leaky ReLU
    }
}

// Define the function called from C++
// Expose it to the C++ side
void compute_attention_coeff(torch::Tensor Wh, torch::Tensor a, torch::Tensor e, int N, int F_prime) {
    const int threads = 1024;
    const int blocks = (N * N + threads - 1) / threads;

    // Launch the CUDA kernel
    compute_attention_coeff_kernel<<<blocks, threads>>>(
        Wh.data_ptr<float>(),
        a.data_ptr<float>(),
        e.data_ptr<float>(),
        N, F_prime
    );
}
