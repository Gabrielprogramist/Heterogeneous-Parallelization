// task1_parallel_stack.cu
// Практическая работа №5 — Часть 1
// Реализация параллельного стека (LIFO) на CUDA
// Используются атомарные операции для безопасного доступа из нескольких потоков.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error: "                              \
                  << cudaGetErrorString(err) << "\n";           \
        std::exit(1);                                           \
    }                                                           \
} while (0)

struct Stack {
    int* data;
    int* top;
    int capacity;

    __device__ void init(int* buffer, int* topPtr, int size) {
        data = buffer;
        top = topPtr;
        capacity = size;
        if (threadIdx.x == 0) *top = -1;
    }

    __device__ bool push(int value) {
        int pos = atomicAdd(top, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    __device__ bool pop(int* value) {
        int pos = atomicSub(top, 1);
        if (pos >= 0) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

__global__ void stackKernel(int* buffer, int* top, int capacity, int* out) {
    Stack stack;
    stack.init(buffer, top, capacity);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток кладёт своё значение
    stack.push(tid);

    __syncthreads();

    int val;
    if (stack.pop(&val)) {
        out[tid] = val;
    }
}

int main() {
    const int N = 256;

    int *d_buffer, *d_top, *d_out;
    CUDA_CHECK(cudaMalloc(&d_buffer, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_top, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    stackKernel<<<1, N>>>(d_buffer, d_top, N, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Stack output (pop results):\n";
    for (int i = 0; i < 10; ++i)
        std::cout << h_out[i] << " ";
    std::cout << "\n";

    cudaFree(d_buffer);
    cudaFree(d_top);
    cudaFree(d_out);
    return 0;
}
