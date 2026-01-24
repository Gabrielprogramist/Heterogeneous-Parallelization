// task3_floyd_allgather.cpp
// Практическая работа №9 — Задание 3
// Флойд–Уоршелл с MPI_Scatter + MPI_Allgather
//
// Компиляция: mpic++ -O3 -std=c++17 task3_floyd_allgather.cpp -o task3
// Запуск:     mpirun -np 4 ./task3 256

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cstdlib>
#include <algorithm>

static int parse_N(int argc, char** argv) {
    if (argc >= 2) return std::atoi(argv[1]);
    return 256;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = parse_N(argc, argv);
    if (N <= 0) {
        if (rank == 0) std::cerr << "N must be positive.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Чтобы Scatter работал просто, сделаем padding по строкам как в task2
    int N_pad = N;
    if (N % size != 0) {
        N_pad = ((N / size) + 1) * size;
        if (rank == 0) {
            std::cout << "Note: N=" << N << " not divisible by processes=" << size
                      << ", padding to N_pad=" << N_pad << " for MPI_Scatter.\n";
        }
    }
    int rows_per_proc = N_pad / size;

    const int INF = 1e9;

    double start = MPI_Wtime();

    // Root creates adjacency matrix
    std::vector<int> G_full;
    if (rank == 0) {
        G_full.assign((size_t)N_pad * N_pad, INF);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> wdist(1, 20);
        std::uniform_real_distribution<double> p(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            G_full[(size_t)i * N_pad + i] = 0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                // С вероятностью 0.2 есть ребро
                if (p(rng) < 0.2) {
                    G_full[(size_t)i * N_pad + j] = wdist(rng);
                }
            }
        }
        // padding rows/cols:
        for (int i = N; i < N_pad; ++i) {
            G_full[(size_t)i * N_pad + i] = 0;
        }
    }

    // local rows
    std::vector<int> G_local((size_t)rows_per_proc * N_pad, INF);

    // Scatter rows
    MPI_Scatter(rank==0 ? G_full.data() : nullptr,
                rows_per_proc * N_pad,
                MPI_INT,
                G_local.data(),
                rows_per_proc * N_pad,
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    // We'll maintain a global matrix copy on each rank via allgather
    std::vector<int> G_global((size_t)N_pad * N_pad, INF);

    // initial allgather to fill global
    MPI_Allgather(G_local.data(), rows_per_proc * N_pad, MPI_INT,
                  G_global.data(), rows_per_proc * N_pad, MPI_INT,
                  MPI_COMM_WORLD);

    // Floyd–Warshall
    for (int k = 0; k < N; ++k) {
        // row k is in G_global at offset k*N_pad
        const int* row_k = &G_global[(size_t)k * N_pad];

        // update local rows
        for (int li = 0; li < rows_per_proc; ++li) {
            int gi = rank * rows_per_proc + li;
            if (gi >= N) continue;

            int* row_i = &G_local[(size_t)li * N_pad];
            int dik = row_i[k];
            if (dik >= INF) continue;

            for (int j = 0; j < N; ++j) {
                int via = dik + row_k[j];
                if (via < row_i[j]) row_i[j] = via;
            }
        }

        // sync local updates into global for next k
        MPI_Allgather(G_local.data(), rows_per_proc * N_pad, MPI_INT,
                      G_global.data(), rows_per_proc * N_pad, MPI_INT,
                      MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "TASK 3 — Floyd–Warshall (Scatter + Allgather)\n";
        std::cout << "N = " << N << ", processes = " << size << "\n";
        std::cout << "Execution time: " << (end - start) << " seconds\n";

        // Печать маленького фрагмента, чтобы не засорять вывод
        int show = std::min(N, 10);
        std::cout << "Top-left " << show << "x" << show << " of distance matrix:\n";
        for (int i = 0; i < show; ++i) {
            for (int j = 0; j < show; ++j) {
                int v = G_global[(size_t)i * N_pad + j];
                if (v >= INF/2) std::cout << "INF ";
                else std::cout << v << " ";
            }
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
