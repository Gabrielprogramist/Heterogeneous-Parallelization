#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cassert>

// -------------------- CUDA check --------------------
// Русский комментарий: проверка ошибок CUDA, чтобы не ловить "тихие" падения.
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << "\n";                                     \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// -------------------- CPU merge sort --------------------
// Русский комментарий: обычная рекурсивная сортировка слиянием (последовательная).

static void cpu_merge(std::vector<int>& a, std::vector<int>& tmp, int L, int M, int R) {
    int i = L, j = M, k = L;
    while (i < M && j < R) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else              tmp[k++] = a[j++];
    }
    while (i < M) tmp[k++] = a[i++];
    while (j < R) tmp[k++] = a[j++];
    for (int p = L; p < R; ++p) a[p] = tmp[p];
}

static void cpu_merge_sort_rec(std::vector<int>& a, std::vector<int>& tmp, int L, int R) {
    if (R - L <= 1) return;
    int M = L + (R - L) / 2;
    cpu_merge_sort_rec(a, tmp, L, M);
    cpu_merge_sort_rec(a, tmp, M, R);
    cpu_merge(a, tmp, L, M, R);
}

static void cpu_merge_sort(std::vector<int>& a) {
    std::vector<int> tmp(a.size());
    cpu_merge_sort_rec(a, tmp, 0, (int)a.size());
}

// -------------------- GPU merge pass --------------------
// Русский комментарий:
// Делаем bottom-up merge sort: сначала width=1 (пары), потом 2,4,8...
// На каждом проходе сливаем пары отсортированных блоков ширины width.
// Каждый поток пишет один элемент результата, используя "merge path" через бинарный поиск.

__device__ __forceinline__ int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__global__ void merge_pass_kernel(const int* in, int* out, int n, int width) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // Находим, в какой паре сегментов находится gid
    int pairSize = 2 * width;
    int pairStart = (gid / pairSize) * pairSize;

    int L0 = pairStart;
    int L1 = clamp_int(pairStart + width, 0, n);
    int R0 = L1;
    int R1 = clamp_int(pairStart + 2 * width, 0, n);

    // Если правой части нет, просто копируем
    if (R0 >= R1) {
        out[gid] = in[gid];
        return;
    }

    // gid внутри итогового диапазона [L0, R1)
    int k = gid - L0;                 // индекс в merged
    int leftLen  = R0 - L0;
    int rightLen = R1 - R0;

    // Русский комментарий:
    // Ищем сколько взять из левой части (i), остальное из правой (j=k-i).
    // Условие корректного разбиения:
    // left[i-1] <= right[j] и right[j-1] < left[i]
    int iMin = max(0, k - rightLen);
    int iMax = min(k, leftLen);

    while (iMin < iMax) {
        int i = (iMin + iMax) / 2;
        int j = k - i;

        int left_im1  = (i > 0) ? in[L0 + i - 1] : INT_MIN;
        int left_i    = (i < leftLen) ? in[L0 + i] : INT_MAX;
        int right_jm1 = (j > 0) ? in[R0 + j - 1] : INT_MIN;
        int right_j   = (j < rightLen) ? in[R0 + j] : INT_MAX;

        if (left_im1 <= right_j) {
            if (right_jm1 < left_i) {
                // нашли i
                iMin = i;
                iMax = i;
                break;
            } else {
                // берём меньше из левой
                iMin = i + 1;
            }
        } else {
            // берём меньше из левой
            iMax = i;
        }
    }

    int i = iMin;
    int j = k - i;

    int left_im1  = (i > 0) ? in[L0 + i - 1] : INT_MIN;
    int left_i    = (i < leftLen) ? in[L0 + i] : INT_MAX;
    int right_jm1 = (j > 0) ? in[R0 + j - 1] : INT_MIN;
    int right_j   = (j < rightLen) ? in[R0 + j] : INT_MAX;

    // Русский комментарий: следующий элемент — минимум из "кандидатов"
    // (на границе разбиения).
    int val = min(left_i, right_j);
    // Но если один из сегментов "кончился", min всё равно корректен из-за INT_MAX.
    out[gid] = val;
}

static float gpu_merge_sort(std::vector<int>& a) {
    int n = (int)a.size();
    int *d_in = nullptr, *d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid  = (n + block - 1) / block;

    CUDA_CHECK(cudaEventRecord(start));

    // Русский комментарий: bottom-up проходы
    for (int width = 1; width < n; width <<= 1) {
        merge_pass_kernel<<<grid, block>>>(d_in, d_out, n, width);
        CUDA_CHECK(cudaGetLastError());
        // меняем местами буферы
        std::swap(d_in, d_out);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // d_in содержит последний результат (из-за swap)
    CUDA_CHECK(cudaMemcpy(a.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ms;
}

static bool is_sorted_non_decreasing(const std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

int main() {
    int N;
    std::cout << "Enter N (e.g., 10000 / 100000 / 1000000): ";
    std::cin >> N;

    std::vector<int> a(N), b(N);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < N; ++i) a[i] = dist(gen);
    b = a;

    // CPU timing
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_merge_sort(a);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // GPU timing
    float gpu_ms = gpu_merge_sort(b);

    std::cout << "CPU merge sort time: " << cpu_ms << " ms\n";
    std::cout << "GPU merge sort time: " << gpu_ms << " ms\n";

    bool ok1 = is_sorted_non_decreasing(a);
    bool ok2 = is_sorted_non_decreasing(b);

    // Русский комментарий: для честности сравним с CPU результатом
    bool same = (a == b);

    std::cout << "CPU sorted: " << (ok1 ? "true" : "false") << "\n";
    std::cout << "GPU sorted: " << (ok2 ? "true" : "false") << "\n";
    std::cout << "CPU == GPU: " << (same ? "true" : "false") << "\n";

    return 0;
}
