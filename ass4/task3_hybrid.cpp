// task3_hybrid.cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void gpu_part(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= 2.0f;
}

int main() {
    const int N = 1'000'000;
    const int HALF = N / 2;

    std::vector<float> data(N, 1.0f);

    // CPU only
    auto t1 = std::chrono::high_resolution_clock::now();
    for (float& x : data) x *= 2.0f;
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    // GPU only
    float* d;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    gpu_part<<<(N+255)/256, 256>>>(d, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, s, e);

    // Hybrid
    auto h1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < HALF; ++i)
        data[i] *= 2.0f;

    cudaMemcpy(d, data.data() + HALF,
               HALF * sizeof(float), cudaMemcpyHostToDevice);

    gpu_part<<<(HALF+255)/256, 256>>>(d, HALF);
    cudaMemcpy(data.data() + HALF, d,
               HALF * sizeof(float), cudaMemcpyDeviceToHost);

    auto h2 = std::chrono::high_resolution_clock::now();
    double hybrid_time =
        std::chrono::duration<double, std::milli>(h2 - h1).count();

    std::cout << "TASK 3 — Hybrid\n";
    std::cout << "CPU time = " << cpu_time << " ms\n";
    std::cout << "GPU time = " << gpu_time << " ms\n";
    std::cout << "Hybrid time = " << hybrid_time << " ms\n";

    cudaFree(d);
    return 0;
}
