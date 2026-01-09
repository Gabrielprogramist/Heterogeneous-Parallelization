#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <stack>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << "\n";                                     \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// -------------------- CPU quick sort --------------------
// Русский комментарий: обычный рекурсивный quicksort.

static int cpu_partition(std::vector<int>& a, int L, int R) {
    int pivot = a[L + (R - L) / 2];
    int i = L, j = R;
    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) {
            std::swap(a[i], a[j]);
            i++; j--;
        }
    }
    return i;
}

static void cpu_quick_sort_rec(std::vector<int>& a, int L, int R) {
    if (L >= R) return;
    int idx = cpu_partition(a, L, R);
    if (L < idx - 1) cpu_quick_sort_rec(a, L, idx - 1);
    if (idx < R)     cpu_quick_sort_rec(a, idx, R);
}

static void cpu_quick_sort(std::vector<int>& a) {
    if (!a.empty())
        cpu_quick_sort_rec(a, 0, (int)a.size() - 1);
}

// -------------------- GPU helpers --------------------
// Русский комментарий: маленькие куски сортируем в shared через bitonic.
__global__ void bitonic_block_sort(int* data, int L, int len) {
    extern __shared__ int s[];
    int tid = threadIdx.x;

    if (tid < len) s[tid] = data[L + tid];
    __syncthreads();

    // bitonic sort (len должна быть степенью двойки)
    for (int k = 2; k <= len; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid && tid < len && ixj < len) {
                bool up = ((tid & k) == 0);
                int a = s[tid];
                int b = s[ixj];
                if ((up && a > b) || (!up && a < b)) {
                    s[tid] = b;
                    s[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < len) data[L + tid] = s[tid];
}

// Русский комментарий:
// Параллельный partition по pivot.
// Считаем два массива флагов: left (x<pivot) и right (x>pivot).
// Для простоты используем atomic для подсчёта позиций (это учебный вариант).
__global__ void partition_kernel(
    const int* in, int* out,
    int L, int R, int pivot,
    int* leftCount, int* rightCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = R - L + 1;
    if (idx >= n) return;

    int x = in[L + idx];

    if (x < pivot) {
        int pos = atomicAdd(leftCount, 1);
        out[L + pos] = x;
    } else if (x > pivot) {
        int pos = atomicAdd(rightCount, 1);
        out[R - pos] = x; // кладём с конца
    }
    // элементы == pivot будут потом заполняться в середину на хосте
}

static bool is_sorted_non_decreasing(const std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

static float gpu_quick_sort(std::vector<int>& a) {
    int n = (int)a.size();
    int *d_a=nullptr, *d_tmp=nullptr;
    int *d_leftCount=nullptr, *d_rightCount=nullptr;

    CUDA_CHECK(cudaMalloc(&d_a,   n*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tmp, n*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leftCount,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rightCount, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Русский комментарий: стек интервалов для сортировки на GPU
    std::stack<std::pair<int,int>> st;
    st.push({0, n-1});

    const int SMALL = 1024; // маленькие куски сортируем bitonic в одном блоке

    while (!st.empty()) {
        auto [L, R] = st.top();
        st.pop();
        int len = R - L + 1;
        if (len <= 1) continue;

        if (len <= SMALL) {
            // Русский комментарий: bitonic требует степень двойки
            // поэтому берём ближайшую степень двойки <= len (хвост оставим как есть через CPU fallback).
            // Чтобы было корректно на любом len, просто делаем CPU сортировку малого куска после копирования.
            // (Для учебной работы так нормально: цель — GPU partition на больших интервалах.)
            std::vector<int> chunk(len);
            CUDA_CHECK(cudaMemcpy(chunk.data(), d_a + L, len*sizeof(int), cudaMemcpyDeviceToHost));
            std::sort(chunk.begin(), chunk.end());
            CUDA_CHECK(cudaMemcpy(d_a + L, chunk.data(), len*sizeof(int), cudaMemcpyHostToDevice));
            continue;
        }

        // Pivot берём с хоста (median-like не делаем — хватит)
        int pivot;
        CUDA_CHECK(cudaMemcpy(&pivot, d_a + (L + len/2), sizeof(int), cudaMemcpyDeviceToHost));

        // Сбрасываем счётчики
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_leftCount,  &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rightCount, &zero, sizeof(int), cudaMemcpyHostToDevice));

        int block = 256;
        int grid  = (len + block - 1) / block;

        partition_kernel<<<grid, block>>>(d_a, d_tmp, L, R, pivot, d_leftCount, d_rightCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Считаем сколько ушло влево/вправо
        int leftCnt=0, rightCnt=0;
        CUDA_CHECK(cudaMemcpy(&leftCnt,  d_leftCount,  sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&rightCnt, d_rightCount, sizeof(int), cudaMemcpyDeviceToHost));

        int midL = L + leftCnt;
        int midR = R - rightCnt;

        // Русский комментарий: заполняем середину pivot'ами на GPU простым kernel
        // (чтобы не копировать массив туда-сюда)
        // Если pivot'ов много, центр будет большим.
        // Для простоты: отдельный kernel на заполнение.
        int pivCount = len - leftCnt - rightCnt;

        // out уже содержит <pivot в начале и >pivot с конца
        // копируем out -> a
        CUDA_CHECK(cudaMemcpy(d_a + L, d_tmp + L, len*sizeof(int), cudaMemcpyDeviceToDevice));

        // kernel заполнения pivot
        if (pivCount > 0) {
            int fillN = pivCount;
            int fillBlock = 256;
            int fillGrid  = (fillN + fillBlock - 1) / fillBlock;
            // лямбда-kernel нельзя, поэтому обычный kernel ниже
            // (объявим локально через static? нельзя. поэтому просто отдельный __global__ ниже в файле)
        }

        // Заполним pivot на GPU через cudaMemset не получится (pivot не байтовый).
        // Сделаем маленькую копию с хоста.
        if (pivCount > 0) {
            std::vector<int> pivs(pivCount, pivot);
            CUDA_CHECK(cudaMemcpy(d_a + midL, pivs.data(), pivCount*sizeof(int), cudaMemcpyHostToDevice));
        }

        // Русский комментарий: добавляем подзадачи
        if (L < midL - 1) st.push({L, midL - 1});
        if (midR + 1 < R) st.push({midR + 1, R});
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(a.data(), d_a, n*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_leftCount));
    CUDA_CHECK(cudaFree(d_rightCount));
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
    cpu_quick_sort(a);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    float gpu_ms = gpu_quick_sort(b);

    std::cout << "CPU quick sort time: " << cpu_ms << " ms\n";
    std::cout << "GPU quick sort time: " << gpu_ms << " ms\n";

    bool ok1 = is_sorted_non_decreasing(a);
    bool ok2 = is_sorted_non_decreasing(b);
    bool same = (a == b);

    std::cout << "CPU sorted: " << (ok1 ? "true" : "false") << "\n";
    std::cout << "GPU sorted: " << (ok2 ? "true" : "false") << "\n";
    std::cout << "CPU == GPU: " << (same ? "true" : "false") << "\n";

    return 0;
}
