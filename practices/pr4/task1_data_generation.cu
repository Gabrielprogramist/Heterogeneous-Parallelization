// task1_data_generation.cu
// Практическая №4, Задание 1:
// - Сгенерировать массив случайных чисел размером 1,000,000 элементов.
// - Показать, что данные реально созданы (например, вывести первые N чисел).
// Здесь генерация делается на CPU (это нормально для подготовки данных),
// затем массив копируется на GPU (глобальная память) и обратно для проверки.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>

static inline void cudaCheck(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << ctx << "): " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

int main() {
    const int N = 1'000'000;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // 1) Генерация случайных чисел на CPU
    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) {
        // [0,1)
        h[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    // 2) Копирование на GPU (глобальная память)
    float* d = nullptr;
    cudaCheck(cudaMalloc(&d, N * sizeof(float)), "cudaMalloc d");
    cudaCheck(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D copy");

    // 3) Скопируем обратно небольшой фрагмент для проверки
    std::vector<float> h_back(10);
    cudaCheck(cudaMemcpy(h_back.data(), d, h_back.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H copy");

    std::cout << "Generated N=" << N << " random floats. First 10 values:\n";
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < h_back.size(); ++i) {
        std::cout << "h[" << i << "]=" << h_back[i] << "\n";
    }

    cudaCheck(cudaFree(d), "cudaFree d");
    return 0;
}
