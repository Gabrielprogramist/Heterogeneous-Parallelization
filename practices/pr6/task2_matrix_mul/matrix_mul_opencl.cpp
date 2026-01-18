// matrix_mul_opencl.cpp
// Практическая работа №6 — Задача 2
// Умножение матриц на OpenCL + проверка с CPU

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK(err) if (err != CL_SUCCESS) { \
    std::cerr << "OpenCL error: " << err << std::endl; exit(1); }

int main() {
    const int N = 256, M = 256, K = 256;

    std::vector<float> A(N * M), B(M * K), C(N * K), C_cpu(N * K);

    for (int i = 0; i < N * M; i++) A[i] = 1.0f;
    for (int i = 0; i < M * K; i++) B[i] = 1.0f;

    // CPU reference
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            for (int t = 0; t < M; t++)
                C_cpu[i * K + j] += A[i * M + t] * B[t * K + j];

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK(err);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK(err);

    const char* src =
        "__kernel void matrix_mul(__global const float* A, __global const float* B, "
        "__global float* C, int N, int M, int K) {"
        "int row = get_global_id(0);"
        "int col = get_global_id(1);"
        "float sum = 0.0f;"
        "for (int i = 0; i < M; i++) sum += A[row*M+i] * B[i*K+col];"
        "C[row*K+col] = sum;}";

    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
    CHECK(err);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
    CHECK(err);

    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              A.size() * sizeof(float), A.data(), &err);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              B.size() * sizeof(float), B.data(), &err);
    cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                              C.size() * sizeof(float), nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &M);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t global[2] = { (size_t)N, (size_t)K };
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, C.size() * sizeof(float), C.data(), 0, nullptr, nullptr);

    bool correct = true;
    for (int i = 0; i < N * K; i++) {
        if (std::fabs(C[i] - C_cpu[i]) > 1e-3) {
            correct = false;
            break;
        }
    }

    std::cout << "Matrix multiplication correct: " << (correct ? "YES" : "NO") << "\n";

    return 0;
}
