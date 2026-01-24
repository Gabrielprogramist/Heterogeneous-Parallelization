// task1_cuda_sum.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void sum_kernel(const float* data, float* partial, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        atomicAdd(partial, data[idx]);
}

int main() {
    const int N = 100000;
    std::vector<float> h(N, 1.0f);

    // CPU
    auto c1 = std::chrono::high_resolution_clock::now();
    float cpu_sum = 0.0f;
    for (float x : h) cpu_sum += x;
    auto c2 = std::chrono::high_resolution_clock::now();
    double cpu_time =
        std::chrono::duration<double, std::milli>(c2 - c1).count();

    // GPU
    float *d_data, *d_sum;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_data, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);
    sum_kernel<<<(N+255)/256, 256>>>(d_data, d_sum, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, s, e);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "TASK 1 — CUDA sum\n";
    std::cout << "CPU sum = " << cpu_sum << ", time = " << cpu_time << " ms\n";
    std::cout << "GPU sum = " << gpu_sum << ", time = " << gpu_time << " ms\n";

    cudaFree(d_data);
    cudaFree(d_sum);
    return 0;
}
