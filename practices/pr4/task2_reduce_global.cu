// task2_reduce_global.cu
// Практическая №4, Задание 2(a):
// - Реализовать редукцию суммы элементов массива с использованием только глобальной памяти.
// Этот вариант нарочно "плохой" для сравнения: каждый поток делает atomicAdd в глобальную память.
// Из-за высокой конкуренции атомиков и медленного доступа к глобальной памяти это обычно медленнее.

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

__global__ void reduceGlobalAtomic(const float* input, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Все потоки напрямую атомарно добавляют в один глобальный аккумулятор.
    // Это гарантирует корректность, но убивает производительность.
    if (idx < n) {
        atomicAdd(out, input[idx]);
    }
}

static float cpuSum(const std::vector<float>& a) {
    double s = 0.0;
    for (float x : a) s += x;
    return static_cast<float>(s);
}

int main() {
    const int N = 1'000'000;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Данные на CPU
    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) h[i] = static_cast<float>(std::rand()) / RAND_MAX;

    // Данные на GPU (глобальная память)
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)), "cudaMalloc d_in");
    cudaCheck(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc d_out");

    cudaCheck(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D copy");
    cudaCheck(cudaMemset(d_out, 0, sizeof(float)), "cudaMemset d_out");

    // Тайминг через cudaEvent
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start), "event create start");
    cudaCheck(cudaEventCreate(&stop), "event create stop");

    cudaCheck(cudaEventRecord(start), "event record start");
    reduceGlobalAtomic<<<GRID, BLOCK>>>(d_in, d_out, N);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaEventRecord(stop), "event record stop");
    cudaCheck(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

    float gpuSumVal = 0.0f;
    cudaCheck(cudaMemcpy(&gpuSumVal, d_out, sizeof(float), cudaMemcpyDeviceToHost), "D2H sum");

    float cpuSumVal = cpuSum(h);

    std::cout << "Reduce (global atomic) N=" << N << "\n";
    std::cout << "GPU sum=" << gpuSumVal << " | CPU sum=" << cpuSumVal << "\n";
    std::cout << "Time: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
