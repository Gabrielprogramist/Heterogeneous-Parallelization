// vector_add_opencl.cpp
// Практическая работа №6 — Задача 1
// Сложение двух векторов с использованием OpenCL (CPU / GPU)

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>

#define CHECK(err) if (err != CL_SUCCESS) { \
    std::cerr << "OpenCL error: " << err << std::endl; exit(1); }

std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }
    return std::string(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

int main() {
    const int N = 1'000'000;
    size_t bytes = N * sizeof(float);

    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; i++) {
        A[i] = i * 0.5f;
        B[i] = i * 0.25f;
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    CHECK(clGetPlatformIDs(1, &platform, nullptr));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr));

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK(err);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK(err);

    // --- ВАЖНО: корректная загрузка kernel ---
    std::string kernelSource = loadKernel("kernel_vector_add.cl");
    const char* sourcePtr = kernelSource.c_str();
    size_t sourceSize = kernelSource.size();

    cl_program program = clCreateProgramWithSource(
        context, 1, &sourcePtr, &sourceSize, &err
    );
    CHECK(err);

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CHECK(err);

    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data(), &err);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B.data(), &err);
    cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);

    size_t globalSize = N;

    auto start = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr);

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Vector add time: " << elapsed.count() << " seconds\n";
    std::cout << "C[0] = " << C[0] << ", C[N-1] = " << C[N-1] << "\n";

    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
