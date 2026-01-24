// task2_scan_shared.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void scan_kernel(float* data, float* out) {
    __shared__ float temp[1024];
    int tid = threadIdx.x;
    temp[tid] = data[tid];
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = (tid >= offset) ? temp[tid - offset] : 0.0f;
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
    out[tid] = temp[tid];
}

int main() {
    const int N = 1024; // учебный scan внутри блока
    std::vector<float> h(N, 1.0f);

    // CPU
    auto c1 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < N; ++i)
        h[i] += h[i - 1];
    auto c2 = std::chrono::high_resolution_clock::now();
    double cpu_time =
        std::chrono::duration<double, std::milli>(c2 - c1).count();

    // GPU
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);
    scan_kernel<<<1, N>>>(d_in, d_out);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, s, e);

    cudaMemcpy(h.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "TASK 2 — Prefix sum (shared memory)\n";
    std::cout << "CPU time = " << cpu_time << " ms\n";
    std::cout << "GPU time = " << gpu_time << " ms\n";
    std::cout << "Last element = " << h[N - 1] << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
