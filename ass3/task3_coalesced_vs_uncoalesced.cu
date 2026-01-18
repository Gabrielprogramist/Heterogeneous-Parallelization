// Assignment 3 — Task 3
// Coalesced vs uncoalesced memory access

#include <cuda_runtime.h>
#include <iostream>

__global__ void coalesced(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += 1.0f;
}

__global__ void uncoalesced(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (i * 32) % n;
    a[idx] += 1.0f;
}

float run(void (*k)(float*, int), float* d, int n) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int block = 256, grid = (n + block - 1) / block;
    cudaEventRecord(s);
    k<<<grid, block>>>(d, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

int main() {
    const int N = 1'000'000;
    float* d;
    cudaMalloc(&d, N * sizeof(float));

    float t1 = run(coalesced, d, N);
    float t2 = run(uncoalesced, d, N);

    std::cout << "Coalesced: " << t1 << " ms\n";
    std::cout << "Uncoalesced: " << t2 << " ms\n";

    cudaFree(d);
    return 0;
}
