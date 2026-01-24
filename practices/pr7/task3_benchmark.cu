// task3_benchmark.cu
// Практическая работа №7 — Задание 3: анализ производительности

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void dummy_kernel(float* data)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < 1024) data[i] += 1.0f;
}

int main()
{
    const int n = 1024;
    float* d;
    cudaMalloc(&d, n * sizeof(float));

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dummy_kernel<<<4, 256>>>(d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // CPU timing
    auto t1 = std::chrono::high_resolution_clock::now();
    volatile float x = 0.0f;
    for (int i = 0; i < n; ++i) x += 1.0f;
    auto t2 = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "TASK 3 — Benchmark\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    std::cout << "CPU time: " << cpu_ms << " ms\n";

    cudaFree(d);
    return 0;
}
