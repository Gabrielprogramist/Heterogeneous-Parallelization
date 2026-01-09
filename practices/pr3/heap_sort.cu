#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << "\n";                                     \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// -------------------- CPU heap sort --------------------
// Русский комментарий: стандартная heap sort (последовательная).

static void cpu_sift_down(std::vector<int>& a, int n, int i) {
    while (true) {
        int l = 2*i + 1;
        int r = 2*i + 2;
        int mx = i;
        if (l < n && a[l] > a[mx]) mx = l;
        if (r < n && a[r] > a[mx]) mx = r;
        if (mx == i) break;
        std::swap(a[i], a[mx]);
        i = mx;
    }
}

static void cpu_heap_sort(std::vector<int>& a) {
    int n = (int)a.size();
    for (int i = n/2 - 1; i >= 0; --i) cpu_sift_down(a, n, i);
    for (int end = n - 1; end > 0; --end) {
        std::swap(a[0], a[end]);
        cpu_sift_down(a, end, 0);
    }
}

// -------------------- GPU heapify --------------------
// Русский комментарий: heapify по уровню.
// Берём набор узлов (индексы i), каждый поток делает sift-down для своего узла.
// Это не идеальный parallel heap, но для учебной демонстрации нормально.

__device__ void device_sift_down(int* a, int n, int i) {
    while (true) {
        int l = 2*i + 1;
        int r = 2*i + 2;
        int mx = i;
        if (l < n && a[l] > a[mx]) mx = l;
        if (r < n && a[r] > a[mx]) mx = r;
        if (mx == i) break;
        int t = a[i]; a[i] = a[mx]; a[mx] = t;
        i = mx;
    }
}

__global__ void heapify_level_kernel(int* a, int n, int start, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    int i = start + tid;
    if (i < n) device_sift_down(a, n, i);
}

// Русский комментарий: sift-down для корня после swap (один поток).
__global__ void sift_root_kernel(int* a, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        device_sift_down(a, n, 0);
    }
}

static bool is_sorted_non_decreasing(const std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

static float gpu_heap_sort(std::vector<int>& a) {
    int n = (int)a.size();
    int* d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Русский комментарий: heapify bottom-up.
    // Индексы внутренних узлов: 0 .. n/2-1. Идём "уровнями" с конца.
    int lastInternal = n/2 - 1;
    int block = 256;

    // Пример: берём уровни по диапазонам индексов.
    // Это грубо, но работает: от lastInternal до 0 чанками.
    for (int start = lastInternal; start >= 0; ) {
        int chunk = 4096; // сколько узлов "одновременно"
        int s = std::max(0, start - chunk + 1);
        int count = start - s + 1;
        int grid = (count + block - 1) / block;
        heapify_level_kernel<<<grid, block>>>(d_a, n, s, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        start = s - 1;
    }

    // Русский комментарий: извлечение максимума.
    // Swap(0,end) делаем на хосте через копирование одного элемента — да, не идеально,
    // но корректно и достаточно для учебной работы.
    for (int end = n - 1; end > 0; --end) {
        int root, last;
        CUDA_CHECK(cudaMemcpy(&root, d_a + 0,   sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, d_a + end, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_a + 0,   &last, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a + end, &root, sizeof(int), cudaMemcpyHostToDevice));

        // Просеиваем корень на GPU
        sift_root_kernel<<<1, 1>>>(d_a, end);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(a.data(), d_a, n*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    return ms;
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

    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_heap_sort(a);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    float gpu_ms = gpu_heap_sort(b);

    std::cout << "CPU heap sort time: " << cpu_ms << " ms\n";
    std::cout << "GPU heap sort time: " << gpu_ms << " ms\n";

    bool ok1 = is_sorted_non_decreasing(a);
    bool ok2 = is_sorted_non_decreasing(b);
    bool same = (a == b);

    std::cout << "CPU sorted: " << (ok1 ? "true" : "false") << "\n";
    std::cout << "GPU sorted: " << (ok2 ? "true" : "false") << "\n";
    std::cout << "CPU == GPU: " << (same ? "true" : "false") << "\n";

    return 0;
}
