// task3_hybrid_async.cu
// Практическая работа №10 — Задание 3
// Асинхронный гибрид CPU + GPU

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void gpu_kernel(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= 2.0f;
}

int main() {
    const int N = 1'000'000;
    std::vector<float> h(N, 1.0f);

    float* d;
    cudaMalloc(&d, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d, h.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    gpu_kernel<<<(N+255)/256, 256, 0, stream>>>(d, N);

    cudaMemcpyAsync(h.data(), d, N * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    std::cout << "TASK 3 — Hybrid async completed\n";

    cudaFree(d);
    cudaStreamDestroy(stream);
    return 0;
}
