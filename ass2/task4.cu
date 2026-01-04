// task4_cuda_mergesort.cu
// Задача 4 (CUDA): параллельная сортировка слиянием на GPU (bottom-up mergesort на device)
// Требования из PDF: массивы 10 000 и 100 000, замер производительности. :contentReference[oaicite:1]{index=1}
//
// Сборка (Linux):
// nvcc -O2 -std=c++17 task4_cuda_mergesort.cu -o task4
// Запуск:
// ./task4
//
// Примечание по реализации:
// - Массив делится на подмассивы (runs) ширины width; на каждом проходе width удваивается.
// - Каждый merge-проход выполняется на GPU kernel'ом, где каждый поток вычисляет один элемент результата
//   в своём сегменте через бинарный поиск (merge path/partition).
// - Это корректная GPU-версия merge sort (итеративная), хотя не самая оптимальная по памяти/кэшу.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstdint>

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// Вспомогательная функция: бинарный поиск partition i (сколько взять из левого массива),
// чтобы определить k-й элемент в слиянии двух отсортированных отрезков
__device__ __forceinline__ int merge_partition(
    const int* left, int left_len,
    const int* right, int right_len,
    int k
) {
    // i in [max(0, k-right_len), min(k, left_len)]
    int i_min = (k - right_len) > 0 ? (k - right_len) : 0;
    int i_max = (k < left_len) ? k : left_len;

    while (i_min < i_max) {
        int i = (i_min + i_max) >> 1;
        int j = k - i;

        // хотим left[i] >= right[j-1] и right[j] >= left[i-1]
        int left_i   = (i < left_len) ? left[i] : INT32_MAX;
        int left_im1 = (i > 0) ? left[i - 1] : INT32_MIN;

        int right_j   = (j < right_len) ? right[j] : INT32_MAX;
        int right_jm1 = (j > 0) ? right[j - 1] : INT32_MIN;

        if (left_i < right_jm1) {
            // мало взяли из левого
            i_min = i + 1;
        } else if (right_j < left_im1) {
            // много взяли из левого
            i_max = i;
        } else {
            return i;
        }
    }
    return i_min;
}

__global__ void merge_pass_kernel(
    const int* __restrict__ in,
    int* __restrict__ out,
    int n,
    int width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // каждый поток обрабатывает один глобальный индекс
    if (tid >= n) return;

    // определяем, в каком сегменте (пара run'ов) находится tid
    int seg_size = 2 * width;
    int seg_id = tid / seg_size;
    int seg_start = seg_id * seg_size;
    int offset_in_seg = tid - seg_start;

    int left_start = seg_start;
    int right_start = seg_start + width;

    int left_len = width;
    int right_len = width;

    // границы массива
    if (left_start >= n) return;

    int left_end = left_start + left_len;
    int right_end = right_start + right_len;

    if (left_end > n) left_end = n;
    if (right_start > n) right_start = n;
    if (right_end > n) right_end = n;

    left_len = left_end - left_start;
    right_len = right_end - right_start;

    // Если правого отрезка нет, просто копируем левый
    if (right_len <= 0) {
        out[tid] = in[tid];
        return;
    }

    const int* left = in + left_start;
    const int* right = in + right_start;

    int k = offset_in_seg;

    int i = merge_partition(left, left_len, right, right_len, k);
    int j = k - i;

    int left_i = (i < left_len) ? left[i] : INT32_MAX;
    int right_j = (j < right_len) ? right[j] : INT32_MAX;

    out[tid] = (left_i <= right_j) ? left_i : right_j;
}

static bool is_sorted_host(const std::vector<int>& v) {
    for (size_t i = 1; i < v.size(); ++i) if (v[i-1] > v[i]) return false;
    return true;
}

static void gpu_merge_sort(std::vector<int>& h) {
    const int n = static_cast<int>(h.size());
    if (n <= 1) return;

    int* d_in = nullptr;
    int* d_out = nullptr;

    cuda_check(cudaMalloc(&d_in, n * sizeof(int)), "cudaMalloc d_in");
    cuda_check(cudaMalloc(&d_out, n * sizeof(int)), "cudaMalloc d_out");

    cuda_check(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice), "H2D");

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    int width = 1;
    while (width < n) {
        merge_pass_kernel<<<blocks, threads>>>(d_in, d_out, n, width);
        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaDeviceSynchronize(), "kernel sync");

        // swap buffers
        int* tmp = d_in;
        d_in = d_out;
        d_out = tmp;

        width *= 2;
    }

    cuda_check(cudaMemcpy(h.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost), "D2H");

    cudaFree(d_in);
    cudaFree(d_out);
}

static void run_case(int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(1, 1'000'000);

    std::vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = dist(rng);

    std::vector<int> cpu = a;
    std::vector<int> gpu = a;

    // CPU baseline: std::stable_sort (хороший ориентир)
    auto c0 = std::chrono::high_resolution_clock::now();
    std::stable_sort(cpu.begin(), cpu.end());
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    // GPU time: cudaEvent (корректный способ измерять именно время на GPU)
    cudaEvent_t e0, e1;
    cuda_check(cudaEventCreate(&e0), "event create e0");
    cuda_check(cudaEventCreate(&e1), "event create e1");

    cuda_check(cudaEventRecord(e0), "event record e0");
    gpu_merge_sort(gpu);
    cuda_check(cudaEventRecord(e1), "event record e1");
    cuda_check(cudaEventSynchronize(e1), "event sync e1");

    float gpu_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&gpu_ms, e0, e1), "event elapsed");

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    bool ok_cpu = is_sorted_host(cpu);
    bool ok_gpu = is_sorted_host(gpu);
    bool same = (cpu == gpu); // для int сортировка детерминирована, должно совпасть

    std::cout << "N = " << n << "\n";
    std::cout << "CPU std::stable_sort: " << cpu_ms << " ms, sorted=" << (ok_cpu ? "yes" : "no") << "\n";
    std::cout << "GPU merge sort (CUDA): " << gpu_ms << " ms, sorted=" << (ok_gpu ? "yes" : "no") << "\n";
    std::cout << "Same result: " << (same ? "yes" : "no") << "\n";
    if (gpu_ms > 0.0f) std::cout << "Speedup (CPU/GPU) = " << (cpu_ms / gpu_ms) << "x\n";
    std::cout << "Note: на малых размерах массивов выигрыш может съедаться накладными расходами копирования/запуска kernel.\n\n";
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Чтобы программа не падала на машинах без GPU:
    int deviceCount = 0;
    cudaError_t ce = cudaGetDeviceCount(&deviceCount);
    if (ce != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA device found. Please run on a machine with NVIDIA GPU + CUDA.\n";
        return 1;
    }
    cuda_check(cudaSetDevice(0), "set device");

    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout << "Task 4 (CUDA merge sort)\n";
    run_case(10'000, rng);
    run_case(100'000, rng);

    return 0;
}
