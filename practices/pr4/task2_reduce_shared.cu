// task2_reduce_shared.cu
// Практическая №4, Задание 2(b):
// - Реализовать редукцию суммы элементов массива с использованием комбинации глобальной и разделяемой памяти.
// Идея: каждый блок грузит свою порцию в shared memory и делает редукцию внутри блока.
// Затем один поток блока делает atomicAdd в глобальный результат.
// Это уменьшает число атомиков примерно в BLOCK_SIZE раз (по одному атомику на блок),
// и часть работы делается в быстрой shared memory.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

static inline void cudaCheck(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << ctx << "): " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void reduceSharedBlock(const float* input, float* out, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1) Читаем из глобальной памяти в shared (быстрый доступ внутри блока)
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 2) Редукция в shared memory
    // stride = половина блока, затем 1/2, 1/4, ...
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 3) Один поток на блок добавляет сумму блока в глобальный результат
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

static float cpuSum(const std::vector<float>& a) {
    double s = 0.0;
    for (float x : a) s += x;
    return static_cast<float>(s);
}

int main() {
    const int N = 1'000'000;
    const int BLOCK = BLOCK_SIZE;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) h[i] = static_cast<float>(std::rand()) / RAND_MAX;

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)), "cudaMalloc d_in");
    cudaCheck(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc d_out");

    cudaCheck(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D copy");
    cudaCheck(cudaMemset(d_out, 0, sizeof(float)), "cudaMemset d_out");

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start), "event create start");
    cudaCheck(cudaEventCreate(&stop), "event create stop");

    cudaCheck(cudaEventRecord(start), "event record start");
    reduceSharedBlock<<<GRID, BLOCK>>>(d_in, d_out, N);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaEventRecord(stop), "event record stop");
    cudaCheck(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

    float gpuSumVal = 0.0f;
    cudaCheck(cudaMemcpy(&gpuSumVal, d_out, sizeof(float), cudaMemcpyDeviceToHost), "D2H sum");

    float cpuSumVal = cpuSum(h);

    std::cout << "Reduce (shared+global) N=" << N << "\n";
    std::cout << "GPU sum=" << gpuSumVal << " | CPU sum=" << cpuSumVal << "\n";
    std::cout << "Time: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
