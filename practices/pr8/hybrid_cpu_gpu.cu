// hybrid_cpu_gpu.cu
// Практическая работа №8
// Гибридная обработка массива: CPU (OpenMP) + GPU (CUDA)
//
// Компиляция:
// nvcc -O3 -Xcompiler -fopenmp hybrid_cpu_gpu.cu -o hybrid
//
// Запуск:
// ./hybrid

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << std::endl;                                \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// ---------------- GPU KERNEL ----------------
__global__ void gpu_process(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= 2.0f;
}

// ---------------- CPU (OpenMP) ----------------
void cpu_process(float* data, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        data[i] *= 2.0f;
    }
}

// ---------------- MAIN ----------------
int main()
{
    const int N = 1'000'000;
    const int HALF = N / 2;

    std::vector<float> data(N, 1.0f);

    // ================= CPU ONLY =================
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_process(data.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ================= GPU ONLY =================
    std::vector<float> data_gpu(N, 1.0f);

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data_gpu.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t gstart, gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);

    cudaEventRecord(gstart);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gpu_process<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(gstop);
    cudaEventSynchronize(gstop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, gstart, gstop);

    CUDA_CHECK(cudaMemcpy(data_gpu.data(), d_data,
                          N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    // ================= HYBRID =================
    std::vector<float> data_hybrid(N, 1.0f);

    float* d_half = nullptr;
    CUDA_CHECK(cudaMalloc(&d_half, HALF * sizeof(float)));

    auto hybrid_start = std::chrono::high_resolution_clock::now();

    // CPU half (OpenMP)
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            cpu_process(data_hybrid.data(), HALF);
        }

        #pragma omp section
        {
            // GPU half
            CUDA_CHECK(cudaMemcpy(d_half, data_hybrid.data() + HALF,
                                  HALF * sizeof(float), cudaMemcpyHostToDevice));

            int b = (HALF + threads - 1) / threads;
            gpu_process<<<b, threads>>>(d_half, HALF);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(data_hybrid.data() + HALF, d_half,
                                  HALF * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    auto hybrid_end = std::chrono::high_resolution_clock::now();

    double hybrid_time =
        std::chrono::duration<double, std::milli>(hybrid_end - hybrid_start).count();

    CUDA_CHECK(cudaFree(d_half));

    // ================= RESULTS =================
    std::cout << "Практическая работа №8 — Гибридные вычисления\n";
    std::cout << "CPU time:     " << cpu_time << " ms\n";
    std::cout << "GPU time:     " << gpu_time << " ms\n";
    std::cout << "Hybrid time:  " << hybrid_time << " ms\n";

    return 0;
}
