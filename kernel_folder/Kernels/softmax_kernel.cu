#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>
#include <torch/extension.h>

// Define the CUDA kernel function
__global__ void softmax_kernel_kernel(float* e, float* alpha, int N) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int i = idx / N;
    int j = idx % N;

    // 1. Load the current thread's `e` value into shared memory
    shared_data[tid] = e[idx];
    __syncthreads();

    // 2. Compute the maximum value (using reduction)
    float max_val = -FLT_MAX;
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && shared_data[tid + offset] > shared_data[tid]) {
            shared_data[tid] = shared_data[tid + offset];
        }
        __syncthreads();
    }
    max_val = shared_data[0];  // The maximum value is stored in the first position of shared memory

    // 3. Compute the denominator of softmax (using reduction to calculate the sum)
    float sum = 0.0f;
    shared_data[tid] = expf(e[idx] - max_val);
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_data[tid] += shared_data[tid + offset];
        }
        __syncthreads();
    }
    sum = shared_data[0];  // The denominator is stored in the first position of shared memory

    // 4. Compute softmax and write the result back
    alpha[idx] = expf(e[idx] - max_val) / sum;
}

// Function exposed to the C++ side
void softmax_kernel(torch::Tensor e, torch::Tensor alpha, int N) {
    const int threads = 1024;
    const int blocks = (N * N + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    // Launch the CUDA kernel
    softmax_kernel_kernel<<<blocks, threads, shared_mem_size>>>(
        e.data_ptr<float>(),
        alpha.data_ptr<float>(),
        N
    );
}
