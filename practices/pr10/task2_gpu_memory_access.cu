// task2_gpu_memory_access.cu
// Практическая работа №10 — Задание 2
// Сравнение коалесцированного и некоалесцированного доступа к памяти

#include <cuda_runtime.h>
#include <iostream>

__global__ void coalesced(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] *= 2.0f;
}

__global__ void non_coalesced(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (i * 32) % n;  // намеренно плохой паттерн
    if (i < n)
        data[idx] *= 2.0f;
}

int main() {
    const int N = 1 << 24;
    float* d;
    cudaMalloc(&d, N * sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(s);
    coalesced<<<blocks, threads>>>(d, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float t1;
    cudaEventElapsedTime(&t1, s, e);

    cudaEventRecord(s);
    non_coalesced<<<blocks, threads>>>(d, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float t2;
    cudaEventElapsedTime(&t2, s, e);

    std::cout << "TASK 2 — GPU memory access\n";
    std::cout << "Coalesced time:     " << t1 << " ms\n";
    std::cout << "Non-coalesced time: " << t2 << " ms\n";

    cudaFree(d);
    return 0;
}
