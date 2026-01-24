// task1_reduction.cu
// Практическая работа №7 — Задание 1: редукция суммы на GPU

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << std::endl;                                \
        std::exit(1);                                          \
    }                                                          \
} while(0)

__global__ void reduce_sum_kernel(const float* input,
                                  float* partial,
                                  int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int n = 1024;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> h(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; ++i)
        h[i] = dist(gen);

    float cpu_sum = 0.0f;
    for (float x : h) cpu_sum += x;

    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_in, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partial(blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial,
                          blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float gpu_sum = 0.0f;
    for (float x : h_partial) gpu_sum += x;

    std::cout << "TASK 1 — Reduction\n";
    std::cout << "CPU sum = " << cpu_sum << "\n";
    std::cout << "GPU sum = " << gpu_sum << "\n";
    std::cout << "Diff    = " << std::abs(cpu_sum - gpu_sum) << "\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    return 0;
}
