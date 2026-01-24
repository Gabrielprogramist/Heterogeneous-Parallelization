// task2_scan.cu
// Практическая работа №7 — Задание 2: префиксная сумма (scan)

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void scan_kernel(float* data, float* out, int n)
{
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int gid = tid;

    if (gid < n)
        temp[tid] = data[gid];
    else
        temp[tid] = 0.0f;

    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0.0f;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (gid < n)
        out[gid] = temp[tid];
}

int main()
{
    const int n = 1024;
    const int threads = 1024;

    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_in, h.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);

    scan_kernel<<<1, threads, threads * sizeof(float)>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h.data(), d_out, n * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "TASK 2 — Prefix sum\n";
    std::cout << "Last element = " << h[n - 1] << " (expected " << n << ")\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
