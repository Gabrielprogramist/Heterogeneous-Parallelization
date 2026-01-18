// task3_bubble_sort_local.cu
// Практическая №4, Задание 3 (часть 1):
// - Реализовать сортировку пузырьком для небольших подмассивов с использованием локальной памяти.
// Локальная память/регистры: переменные, доступные одному потоку.
// Важно: пузырьковая сортировка медленная, но для маленьких подмассивов (например 32 элемента)
// можно показать принцип "локального" вычисления, чтобы минимизировать обращения к глобальной памяти.
//
// Стратегия: один блок/один поток сортирует один подмассив длины SUB.
// Это проще и стабильно для учебной демонстрации локальной памяти.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

static inline void cudaCheck(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << ctx << "): " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

constexpr int SUB = 32; // размер подмассива (небольшой!)

__global__ void bubbleSortSubarraysLocal(float* data, int n) {
    int subIdx = blockIdx.x;               // номер подмассива
    int start = subIdx * SUB;              // начало подмассива
    if (start + SUB > n) return;           // хвост не сортируем для простоты

    // Локальный массив: на практике может уйти в регистры/локальную память.
    // Доступ к нему быстрый и только внутри потока.
    float local[SUB];

    // Загружаем подмассив из глобальной памяти
    #pragma unroll
    for (int i = 0; i < SUB; ++i) {
        local[i] = data[start + i];
    }

    // Пузырьковая сортировка локально (внутри одного потока)
    for (int i = 0; i < SUB - 1; ++i) {
        for (int j = 0; j < SUB - i - 1; ++j) {
            if (local[j] > local[j + 1]) {
                float t = local[j];
                local[j] = local[j + 1];
                local[j + 1] = t;
            }
        }
    }

    // Записываем обратно в глобальную память
    #pragma unroll
    for (int i = 0; i < SUB; ++i) {
        data[start + i] = local[i];
    }
}

static bool isSortedChunk(const std::vector<float>& a, int start, int len) {
    for (int i = start + 1; i < start + len; ++i) {
        if (a[i - 1] > a[i]) return false;
    }
    return true;
}

int main() {
    const int N = 100000; // можно менять; для демонстрации сортируем кусками по 32
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) h[i] = static_cast<float>(std::rand()) / RAND_MAX;

    float* d = nullptr;
    cudaCheck(cudaMalloc(&d, N * sizeof(float)), "cudaMalloc d");
    cudaCheck(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D copy");

    int numSubs = N / SUB; // сортируем только полные подмассивы
    // 1 поток на подмассив: blockDim=1, gridDim=numSubs
    bubbleSortSubarraysLocal<<<numSubs, 1>>>(d, N);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "device sync");

    cudaCheck(cudaMemcpy(h.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H copy");

    // Проверим первые несколько подмассивов на отсортированность
    bool ok = true;
    for (int s = 0; s < std::min(numSubs, 10); ++s) {
        if (!isSortedChunk(h, s * SUB, SUB)) {
            ok = false; break;
        }
    }

    std::cout << "Bubble sort local subarrays: N=" << N << ", SUB=" << SUB << "\n";
    std::cout << "First 10 subarrays sorted? " << (ok ? "YES" : "NO") << "\n";

    cudaFree(d);
    return 0;
}
