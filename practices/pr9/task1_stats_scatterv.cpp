// task1_stats_scatterv.cpp
// Практическая работа №9 — Задание 1
// MPI_Scatterv + MPI_Reduce: среднее и стандартное отклонение
//
// Компиляция: mpic++ -O3 -std=c++17 task1_stats_scatterv.cpp -o task1
// Запуск:     mpirun -np 4 ./task1 1000000

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

static long long parse_N(int argc, char** argv) {
    if (argc >= 2) return std::atoll(argv[1]);
    return 1'000'000LL;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = parse_N(argc, argv);

    double start = MPI_Wtime();

    // ----------- Prepare counts/displs for Scatterv -----------
    std::vector<int> counts(size, 0), displs(size, 0);
    long long base = N / size;
    long long rem  = N % size;

    for (int i = 0; i < size; ++i) {
        long long cnt = base + (i < rem ? 1 : 0);
        if (cnt > INT32_MAX) {
            if (rank == 0) std::cerr << "N too large for int counts in Scatterv.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[i] = (int)cnt;
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + counts[i-1];

    // ----------- Root creates full array -----------
    std::vector<double> full;
    if (rank == 0) {
        full.resize((size_t)N);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (long long i = 0; i < N; ++i) full[(size_t)i] = dist(rng);
    }

    // ----------- Each rank receives its chunk -----------
    int local_n = counts[rank];
    std::vector<double> local((size_t)local_n);

    MPI_Scatterv(rank == 0 ? full.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 local.data(),
                 local_n,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    // ----------- Local sums -----------
    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    for (int i = 0; i < local_n; ++i) {
        double x = local[(size_t)i];
        local_sum += x;
        local_sum_sq += x * x;
    }

    // ----------- Reduce to root -----------
    double global_sum = 0.0;
    double global_sum_sq = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        double mean = global_sum / (double)N;
        double ex2  = global_sum_sq / (double)N;
        double var  = ex2 - mean * mean;
        if (var < 0) var = 0; // на случай численной погрешности
        double stddev = std::sqrt(var);

        std::cout << "TASK 1 — Stats (Scatterv + Reduce)\n";
        std::cout << "N = " << N << ", processes = " << size << "\n";
        std::cout << "Mean   = " << mean << "\n";
        std::cout << "Stddev = " << stddev << "\n";
        std::cout << "Execution time: " << (end - start) << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
