// task4_mpi.cpp
#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1'000'000;
    int local_n = N / size;

    std::vector<float> local(local_n, 1.0f);

    double t0 = MPI_Wtime();
    float local_sum = 0.0f;
    for (float x : local) local_sum += x;

    float global_sum = 0.0f;
    MPI_Reduce(&local_sum, &global_sum, 1,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "TASK 4 — MPI\n";
        std::cout << "Processes: " << size << "\n";
        std::cout << "Sum = " << global_sum << "\n";
        std::cout << "Time = " << (t1 - t0) << " s\n";
    }

    MPI_Finalize();
    return 0;
}
