// kernel_matrix_mul.cl
// OpenCL kernel: умножение матриц C = A * B

__kernel void matrix_mul(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         int N, int M, int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += A[row * M + i] * B[i * K + col];
    }

    C[row * K + col] = sum;
}
