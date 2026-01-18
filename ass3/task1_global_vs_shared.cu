// Assignment 3 — Task 1
// Element-wise multiply: global memory vs shared memory

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n"; \
        exit(1); \
    } \
} while (0)

__global__ void mul_global(float* a, float k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= k;
}

__global__ void mul_shared(float* a, float k, int n) {
    __shared__ float buf[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    if (i < n) buf[t] = a[i];
    __syncthreads();

    if (i < n) a[i] = buf[t] * k;
}

float measure(void (*kernel)(float*, float, int),
              float* d, float k, int n, int block) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    int grid = (n + block - 1) / block;
    cudaEventRecord(s);
    kernel<<<grid, block>>>(d, k, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

int main() {
    const int N = 1'000'000;
    const int BLOCK = 256;
    std::vector<float> h(N, 1.0f);

    float* d;
    CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float t1 = measure(mul_global, d, 2.0f, N, BLOCK);
    float t2 = measure(mul_shared, d, 2.0f, N, BLOCK);

    std::cout << "Global memory time: " << t1 << " ms\n";
    std::cout << "Shared memory time: " << t2 << " ms\n";

    cudaFree(d);
    return 0;
}
