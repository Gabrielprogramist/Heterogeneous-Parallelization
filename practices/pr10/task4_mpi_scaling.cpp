// task4_mpi_scaling.cpp
// Практическая работа №10 — Задание 4
// Strong / Weak scaling

#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 10'000'000;
    int local_n = N / size;

    std::vector<double> local(local_n, 1.0);

    double t0 = MPI_Wtime();
    double local_sum = 0.0;
    for (double x : local) local_sum += x;

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "TASK 4 — MPI scaling\n";
        std::cout << "Processes: " << size << "\n";
        std::cout << "Time: " << (t1 - t0) << " s\n";
        std::cout << "Sum: " << global_sum << "\n";
    }

    MPI_Finalize();
    return 0;
}
