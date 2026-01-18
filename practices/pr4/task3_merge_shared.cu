
// task3_merge_shared.cu
// Практическая №4, Задание 3 (часть 2):
// - Глобальная память: общий массив
// - Разделяемая память: буферизация двух отсортированных подмассивов
// - Слияние (merge) соседних отсортированных чанков
//
// Важно: мы делаем один проход merge: width + width -> 2*width.
// Для полной сортировки нужно делать несколько проходов (width *= 2).

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

// Кол-во потоков в блоке. width должен быть <= BLOCK, чтобы загрузка в shared была удобной.
constexpr int BLOCK = 256;

// Один блок обрабатывает один сегмент длиной 2*width:
// [start .. start+width-1] и [start+width .. start+2*width-1] считаются уже отсортированными.
// Затем блок делает merge в out.
__global__ void mergePassShared(const float* in, float* out, int n, int width) {
    extern __shared__ float sh[]; // размер = 2*width * sizeof(float)
    float* left  = sh;            // [0..width-1]
    float* right = sh + width;    // [0..width-1] (логически)

    int seg   = blockIdx.x;
    int start = seg * 2 * width;
    int tid   = threadIdx.x;

    // 1) Грузим левую половину в shared (если выходим за n — подставляем большое число)
    // Здесь можно использовать просто большое число, но мы всё равно ниже merge делаем с проверкой границ,
    // поэтому значение вне диапазона не будет участвовать (мы ограничим lenL/lenR).
    if (tid < width) {
        int idxL = start + tid;
        left[tid] = (idxL < n) ? in[idxL] : 0.0f;
    }

    // 2) Грузим правую половину в shared
    if (tid < width) {
        int idxR = start + width + tid;
        right[tid] = (idxR < n) ? in[idxR] : 0.0f;
    }

    __syncthreads();

    // Реальные длины половин (на последнем сегменте массив может быть "обрезан")
    int lenL = 0;
    int lenR = 0;
    if (start < n) {
        lenL = min(width, n - start);
        int startR = start + width;
        if (startR < n) lenR = min(width, n - startR);
        else lenR = 0;
    }

    // 3) Слияние делаем одним потоком (для простоты учебной работы)
    if (tid == 0) {
        int i = 0, j = 0;
        int outLen = lenL + lenR;

        for (int k = 0; k < outLen; ++k) {
            float take;

            // Если правая половина закончилась — берём из левой
            if (j >= lenR) {
                take = left[i++];
            }
            // Если левая закончилась — берём из правой
            else if (i >= lenL) {
                take = right[j++];
            }
            // Иначе берём минимальный из двух текущих
            else {
                float a = left[i];
                float b = right[j];
                if (a <= b) { take = a; i++; }
                else        { take = b; j++; }
            }

            int outIdx = start + k;
            if (outIdx < n) out[outIdx] = take;
        }
    }
}

// Для демонстрации: сначала делаем "сортированные чанки" на CPU (как будто bubbleSortLocal уже сделал),
// затем запускаем merge на GPU.
int main() {
    const int N = 100000;
    const int width = 128; // width <= BLOCK
    if (width > BLOCK) {
        std::cerr << "width must be <= " << BLOCK << "\n";
        return 1;
    }

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) h[i] = static_cast<float>(std::rand()) / RAND_MAX;

    // Готовим "отсортированные чанки" на CPU (имитация результата пузырька на GPU)
    for (int start = 0; start < N; start += width) {
        int end = std::min(start + width, N);
        std::sort(h.begin() + start, h.begin() + end);
    }

    float *d_in = nullptr, *d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in,  N * sizeof(float)), "cudaMalloc d_in");
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)), "cudaMalloc d_out");
    cudaCheck(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D copy");

    int numSegs = (N + 2 * width - 1) / (2 * width);
    size_t shmemBytes = 2ULL * width * sizeof(float);

    mergePassShared<<<numSegs, BLOCK, shmemBytes>>>(d_in, d_out, N, width);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "device sync");

    cudaCheck(cudaMemcpy(h.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H copy");

    // Проверка: каждый сегмент длиной до 2*width после merge должен быть отсортирован
    bool ok = true;
    for (int start = 0; start < N; start += 2 * width) {
        int end = std::min(start + 2 * width, N);
        for (int i = start + 1; i < end; ++i) {
            if (h[i - 1] > h[i]) { ok = false; break; }
        }
        if (!ok) break;
    }

    std::cout << "Merge shared pass: N=" << N << ", width=" << width << "\n";
    std::cout << "Each merged segment sorted? " << (ok ? "YES" : "NO") << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
