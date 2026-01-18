// Assignment 3 — Task 2
// Vector addition with different block sizes

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

float run(int block, float* a, float* b, float* c, int n) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    int grid = (n + block - 1) / block;
    cudaEventRecord(s);
    add<<<grid, block>>>(a, b, c, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

int main() {
    const int N = 1'000'000;
    std::vector<float> h(N, 1.0f);

    float *a, *b, *c;
    cudaMalloc(&a, N * sizeof(float));
    cudaMalloc(&b, N * sizeof(float));
    cudaMalloc(&c, N * sizeof(float));

    cudaMemcpy(a, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    for (int block : {128, 256, 512}) {
        float t = run(block, a, b, c, N);
        std::cout << "Block size " << block << ": " << t << " ms\n";
    }

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
