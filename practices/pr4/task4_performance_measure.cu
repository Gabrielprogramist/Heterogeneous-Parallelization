
// task4_performance_measure.cu
// Практическая работа №4 (CUDA)
// Задача 4: замер времени выполнения для разных типов памяти
// N = 10,000; 100,000; 1,000,000
//
// Измеряем:
// - reduce_global_atomic: только глобальная память (наивно, atomicAdd по элементам)
// - reduce_shared_block: глобальная + shared (редукция внутри блока, atomicAdd по блоку)
// - bubble_sort_subarrays: пузырёк малых подмассивов с локальной памятью потока
// - merge_pairs_shared: слияние двух отсортированных подмассивов через shared
//
// Пишем таблицу и сохраняем performance.csv

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <algorithm>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err__ = (call);                               \
    if (err__ != cudaSuccess) {                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1);                                         \
    }                                                         \
} while (0)

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef SUB_SIZE
#define SUB_SIZE 32
#endif

// --------- Kernels for reduction ---------

__global__ void reduce_global_atomic(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(out, in[idx]);
}

__global__ void reduce_shared_block(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    s[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, s[0]);
}

// --------- Kernels for sort/merge ---------

__global__ void bubble_sort_subarrays(float* data, int n) {
    // Один подмассив SUB_SIZE сортируем одним потоком (tid==0) для простоты.
    if (threadIdx.x != 0) return;

    int blockStart = blockIdx.x * SUB_SIZE;
    if (blockStart >= n) return;

    float local[SUB_SIZE];

    // Загружаем подмассив в локальную память потока
    for (int i = 0; i < SUB_SIZE; ++i) {
        int idx = blockStart + i;
        local[i] = (idx < n) ? data[idx] : 1e20f;
    }

    // Пузырёк
    for (int i = 0; i < SUB_SIZE - 1; ++i) {
        for (int j = 0; j < SUB_SIZE - i - 1; ++j) {
            if (local[j] > local[j + 1]) {
                float t = local[j];
                local[j] = local[j + 1];
                local[j + 1] = t;
            }
        }
    }

    // Пишем обратно
    for (int i = 0; i < SUB_SIZE; ++i) {
        int idx = blockStart + i;
        if (idx < n) data[idx] = local[i];
    }
}

// Безопасное слияние двух отсортированных подмассивов длиной SUB_SIZE.
// ВАЖНО: в конце массива реальная длина может быть меньше SUB_SIZE, поэтому считаем lenL/lenR.
__global__ void merge_pairs_shared(const float* in, float* out, int n) {
    __shared__ float left[SUB_SIZE];
    __shared__ float right[SUB_SIZE];

    int pairId = blockIdx.x;
    int start  = pairId * (2 * SUB_SIZE);
    int tid    = threadIdx.x;

    // Загружаем обе половины в shared
    if (tid < SUB_SIZE) {
        int li = start + tid;
        int ri = start + SUB_SIZE + tid;
        left[tid]  = (li < n) ? in[li] : 0.0f;   // значение не важно, т.к. ниже есть lenL/lenR
        right[tid] = (ri < n) ? in[ri] : 0.0f;
    }
    __syncthreads();

    // Реальные длины левой/правой половины для этого сегмента
    int lenL = 0, lenR = 0;
    if (start < n) {
        lenL = min(SUB_SIZE, n - start);
        int startR = start + SUB_SIZE;
        if (startR < n) lenR = min(SUB_SIZE, n - startR);
        else lenR = 0;
    }

    // Слияние делаем одним потоком (tid==0) — так проще и достаточно для практики по памяти.
    if (tid == 0) {
        int i = 0, j = 0;
        int outLen = lenL + lenR;

        for (int k = 0; k < outLen; ++k) {
            float take;

            if (j >= lenR) {
                take = left[i++];
            } else if (i >= lenL) {
                take = right[j++];
            } else {
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

// --------- Helpers ---------

static void generate_random(std::vector<float>& a, uint32_t seed) {
    std::srand(seed);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
}

// Замер времени одного GPU-участка через cudaEvent.
// Важно: fn() должна запускать kernel(ы). Ошибки ядра "всплывут" при синхронизации event/Device.
template <typename Fn>
static float time_gpu(Fn&& fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    const int sizes[] = {10'000, 100'000, 1'000'000};

    std::ofstream csv("performance.csv", std::ios::out);
    csv << "N,reduce_global_ms,reduce_shared_ms,bubble_subarrays_ms,merge_pairs_ms\n";

    std::cout << "Task4 performance\n";
    std::cout << "BLOCK_SIZE=" << BLOCK_SIZE << ", SUB_SIZE=" << SUB_SIZE << "\n\n";
    std::cout << "N\treduce_global(ms)\treduce_shared(ms)\tbubble(ms)\tmerge(ms)\n";

    for (int N : sizes) {
        std::vector<float> h(N);
        generate_random(h, 12345u);

        float* d_data = nullptr;
        float* d_tmp  = nullptr;
        float* d_sum  = nullptr;

        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sum,  sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_data, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // --- reduce global ---
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
        float t_reduce_global = time_gpu([&]() {
            int block = 256;
            int grid  = (N + block - 1) / block;
            reduce_global_atomic<<<grid, block>>>(d_data, d_sum, N);
            CUDA_CHECK(cudaGetLastError());
        });

        // --- reduce shared ---
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
        float t_reduce_shared = time_gpu([&]() {
            int block = BLOCK_SIZE;
            int grid  = (N + block - 1) / block;
            reduce_shared_block<<<grid, block>>>(d_data, d_sum, N);
            CUDA_CHECK(cudaGetLastError());
        });

        // --- bubble sort subarrays (local memory) ---
        float t_bubble = time_gpu([&]() {
            int grid = (N + SUB_SIZE - 1) / SUB_SIZE;
            dim3 block(32); // достаточно, т.к. реально работает только threadIdx.x==0
            bubble_sort_subarrays<<<grid, block>>>(d_data, N);
            CUDA_CHECK(cudaGetLastError());
        });

        // --- merge pairs (shared memory) ---
        // Перед merge логично иметь отсортированные подмассивы.
        // Поэтому здесь bubble уже отработал и d_data содержит отсортированные чанки SUB_SIZE.
        float t_merge = time_gpu([&]() {
            int pairs = (N + (2 * SUB_SIZE) - 1) / (2 * SUB_SIZE);
            dim3 block(32);
            merge_pairs_shared<<<pairs, block>>>(d_data, d_tmp, N);
            CUDA_CHECK(cudaGetLastError());
        });

        std::cout << N << "\t"
                  << t_reduce_global << "\t\t"
                  << t_reduce_shared << "\t\t"
                  << t_bubble << "\t\t"
                  << t_merge << "\n";

        csv << N << ","
            << t_reduce_global << ","
            << t_reduce_shared << ","
            << t_bubble << ","
            << t_merge << "\n";

        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_tmp));
        CUDA_CHECK(cudaFree(d_sum));
    }

    csv.close();
    std::cout << "\nSaved CSV: performance.csv\n";
    return 0;
}
