// task2_parallel_queue.cu
// Практическая работа №5 — Часть 2
// Реализация параллельной очереди (FIFO) на CUDA

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

struct Queue {
    int* data;
    int* head;
    int* tail;
    int capacity;

    __device__ void init(int* buffer, int* h, int* t, int size) {
        data = buffer;
        head = h;
        tail = t;
        capacity = size;
        if (threadIdx.x == 0) {
            *head = 0;
            *tail = 0;
        }
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(head, 1);
        if (pos < *tail) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

__global__ void queueKernel(int* buffer, int* head, int* tail, int capacity, int* out) {
    Queue q;
    q.init(buffer, head, tail, capacity);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    q.enqueue(tid);
    __syncthreads();

    int val;
    if (q.dequeue(&val)) {
        out[tid] = val;
    }
}

int main() {
    const int N = 256;

    int *d_buffer, *d_head, *d_tail, *d_out;
    CUDA_CHECK(cudaMalloc(&d_buffer, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_head, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tail, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    queueKernel<<<1, N>>>(d_buffer, d_head, d_tail, N, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Queue output (dequeue results):\n";
    for (int i = 0; i < 10; ++i)
        std::cout << h_out[i] << " ";
    std::cout << "\n";

    cudaFree(d_buffer);
    cudaFree(d_head);
    cudaFree(d_tail);
    cudaFree(d_out);
    return 0;
}
