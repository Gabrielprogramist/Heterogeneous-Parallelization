// task4_optimized_config.cu
// Assignment 3 — Task 4
// Подбор оптимальных параметров конфигурации сетки/блоков для CUDA-ядра.
// Выбрана программа из Task 2: поэлементное сложение двух массивов.
//
// Что делает:
// 1) Создаёт два массива A и B длиной N=1_000_000, заполняет данными.
// 2) Запускает kernel vector_add с разными block sizes.
// 3) Для каждого block size измеряет время через cudaEvent.
// 4) Находит лучший block size (минимальное время).
// 5) Сравнивает "неоптимальную" конфигурацию (например, block=64) и "оптимальную" (best).
//
// Сборка (Colab / Tesla T4):
// nvcc -O2 -arch=sm_75 task4_optimized_config.cu -o task4
//
// Запуск:
// ./task4
// (Можно передать N как аргумент: ./task4 1000000)

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t e = (call);                                    \
    if (e != cudaSuccess) {                                    \
        std::cerr << "CUDA error: " << cudaGetErrorString(e)   \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << "\n";                                     \
        std::exit(1);                                          \
    }                                                          \
} while (0)

// Простое memory-bound ядро: C[i] = A[i] + B[i]
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// Измерение времени запуска одного kernel (один прогон)
static float time_once(const float* dA, const float* dB, float* dC, int n, int block) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int grid = (n + block - 1) / block;

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// Для более стабильных результатов обычно делают несколько прогонов и усреднение
static float time_avg(const float* dA, const float* dB, float* dC, int n, int block, int iters) {
    // warmup
    (void)time_once(dA, dB, dC, n, block);

    float sum = 0.0f;
    for (int i = 0; i < iters; ++i) sum += time_once(dA, dB, dC, n, block);
    return sum / iters;
}

static void fill_data(std::vector<float>& v, float base) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = base + 0.001f * (float)(i % 1000);
}

int main(int argc, char** argv) {
    int N = 1'000'000;
    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));

    const int iters = 10; // усреднение

    std::vector<float> hA(N), hB(N), hC(N);
    fill_data(hA, 1.0f);
    fill_data(hB, 2.0f);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // Набор кандидатов. (T4 часто хорош на 256/512; 1024 иногда хуже из-за регистров/occupancy.)
    int candidates[] = {64, 128, 256, 512, 1024};

    std::cout << "Task4: optimizing launch configuration for vector_add\n";
    std::cout << "N=" << N << ", averaging iters=" << iters << "\n\n";
    std::cout << "block\tgrid\tavg_time_ms\n";

    int bestBlock = candidates[0];
    float bestTime = std::numeric_limits<float>::infinity();

    // Замер всех кандидатов
    for (int block : candidates) {
        int grid = (N + block - 1) / block;
        float t = time_avg(dA, dB, dC, N, block, iters);
        std::cout << block << "\t" << grid << "\t" << t << "\n";

        if (t < bestTime) {
            bestTime = t;
            bestBlock = block;
        }
    }

    // “Неоптимальная” конфигурация — возьмём намеренно маленький block
    int badBlock = 64;
    float badTime = time_avg(dA, dB, dC, N, badBlock, iters);

    std::cout << "\nBest block size: " << bestBlock << " (avg " << bestTime << " ms)\n";
    std::cout << "Bad  block size: " << badBlock  << " (avg " << badTime  << " ms)\n";

    double speedup = (bestTime > 0.0f) ? (badTime / bestTime) : 0.0;
    std::cout << "Speedup (bad / best): " << speedup << "x\n";

    // Проверка корректности: скопируем пару элементов
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i : {0, N/2, N-1}) {
        float ref = hA[i] + hB[i];
        if (std::fabs(hC[i] - ref) > 1e-5f) ok = false;
    }
    std::cout << "Correctness check: " << (ok ? "OK" : "FAILED") << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
