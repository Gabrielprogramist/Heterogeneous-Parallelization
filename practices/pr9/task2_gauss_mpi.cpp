// task2_gauss_mpi.cpp
// Практическая работа №9 — Задание 2
// Распределённый метод Гаусса: Scatter + Bcast + Gather
//
// Компиляция: mpic++ -O3 -std=c++17 task2_gauss_mpi.cpp -o task2
// Запуск:     mpirun -np 4 ./task2 512

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

static int parse_N(int argc, char** argv) {
    if (argc >= 2) return std::atoi(argv[1]);
    return 512;
}

static void make_diag_dominant_system(std::vector<double>& A, std::vector<double>& b, int N) {
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // A in row-major: A[i*N + j]
    for (int i = 0; i < N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            double v = dist(rng);
            A[(size_t)i * N + j] = v;
            row_sum += std::abs(v);
        }
        // Сделаем диагональ доминирующей
        A[(size_t)i * N + i] = row_sum + 1.0;
        b[i] = dist(rng);
    }
}

// кому принадлежит глобальная строка r при блоковом разбиении по counts/displs
static int owner_of_row(int r, const std::vector<int>& row_counts, const std::vector<int>& row_displs) {
    int p = (int)row_counts.size();
    for (int i = 0; i < p; ++i) {
        int start = row_displs[i];
        int end = start + row_counts[i];
        if (r >= start && r < end) return i;
    }
    return -1;
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

    double start = MPI_Wtime();

    // ----------- Row distribution (Scatter) -----------
    // Scatter по строкам. Если N не делится, используем Scatterv,
    // но в задании написано Scatter — поэтому делаем ровно Scatter,
    // а если не делится — дополняем нулями до ближайшего кратного.
    int N_pad = N;
    if (N % size != 0) {
        N_pad = ((N / size) + 1) * size;
        if (rank == 0) {
            std::cout << "Note: N=" << N << " not divisible by processes=" << size
                      << ", padding to N_pad=" << N_pad << " for MPI_Scatter.\n";
        }
    }
    int rows_per_proc = N_pad / size;

    // Root full system (padded)
    std::vector<double> A_full, b_full;
    if (rank == 0) {
        A_full.assign((size_t)N_pad * N_pad, 0.0);
        b_full.assign((size_t)N_pad, 0.0);

        // заполняем только NxN, остальное 0
        std::vector<double> A_tmp((size_t)N * N);
        std::vector<double> b_tmp((size_t)N);
        make_diag_dominant_system(A_tmp, b_tmp, N);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A_full[(size_t)i * N_pad + j] = A_tmp[(size_t)i * N + j];
            }
            b_full[i] = b_tmp[i];
        }
        // для дополненных строк поставим единичную диагональ, чтобы не ломать исключение
        for (int i = N; i < N_pad; ++i) {
            A_full[(size_t)i * N_pad + i] = 1.0;
            b_full[i] = 0.0;
        }
    }

    // локальные блоки строк: rows_per_proc x N_pad
    std::vector<double> A_local((size_t)rows_per_proc * N_pad, 0.0);
    std::vector<double> b_local((size_t)rows_per_proc, 0.0);

    // Scatter матрицы по строкам
    MPI_Scatter(rank==0 ? A_full.data() : nullptr,
                rows_per_proc * N_pad,
                MPI_DOUBLE,
                A_local.data(),
                rows_per_proc * N_pad,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    // Scatter вектора b по строкам
    MPI_Scatter(rank==0 ? b_full.data() : nullptr,
                rows_per_proc,
                MPI_DOUBLE,
                b_local.data(),
                rows_per_proc,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    // Для удобства создадим counts/displs по строкам (для определения владельца pivot)
    std::vector<int> row_counts(size, rows_per_proc);
    std::vector<int> row_displs(size, 0);
    for (int i = 1; i < size; ++i) row_displs[i] = row_displs[i-1] + row_counts[i-1];

    // pivot row buffer (размер N_pad + 1 для b)
    std::vector<double> pivot_row((size_t)N_pad + 1, 0.0);

    // ----------- Forward elimination -----------
    for (int k = 0; k < N; ++k) {
        int owner = owner_of_row(k, row_counts, row_displs);
        int local_k = k - row_displs[owner];

        if (rank == owner) {
            // Нормализуем pivot-строку
            double pivot = A_local[(size_t)local_k * N_pad + k];
            if (std::abs(pivot) < 1e-12) {
                std::cerr << "Pivot too small at row " << k << " (no pivoting implemented)\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }

            // Запишем нормализованную pivot-строку в pivot_row
            for (int j = k; j < N; ++j) {
                pivot_row[(size_t)j] = A_local[(size_t)local_k * N_pad + j] / pivot;
                A_local[(size_t)local_k * N_pad + j] = pivot_row[(size_t)j];
            }
            pivot_row[(size_t)N_pad] = b_local[(size_t)local_k] / pivot;
            b_local[(size_t)local_k] = pivot_row[(size_t)N_pad];

            // элементы <k не нужны (можно оставить)
            for (int j = 0; j < k; ++j) pivot_row[(size_t)j] = 0.0;
        }

        // Рассылаем pivot-строку всем
        MPI_Bcast(pivot_row.data(), N_pad + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Обновляем локальные строки i>k
        for (int li = 0; li < rows_per_proc; ++li) {
            int gi = row_displs[rank] + li;
            if (gi <= k || gi >= N) continue; // только реальные строки N и ниже pivot

            double factor = A_local[(size_t)li * N_pad + k];
            if (factor == 0.0) continue;

            // A[gi, j] -= factor * pivot_row[j]
            for (int j = k; j < N; ++j) {
                A_local[(size_t)li * N_pad + j] -= factor * pivot_row[(size_t)j];
            }
            b_local[(size_t)li] -= factor * pivot_row[(size_t)N_pad];
            A_local[(size_t)li * N_pad + k] = 0.0;
        }
    }

    // ----------- Gather to root -----------
    if (rank == 0) {
        A_full.assign((size_t)N_pad * N_pad, 0.0);
        b_full.assign((size_t)N_pad, 0.0);
    }

    MPI_Gather(A_local.data(), rows_per_proc * N_pad, MPI_DOUBLE,
               rank==0 ? A_full.data() : nullptr, rows_per_proc * N_pad, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Gather(b_local.data(), rows_per_proc, MPI_DOUBLE,
               rank==0 ? b_full.data() : nullptr, rows_per_proc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // ----------- Back substitution on root (only first N rows/cols) -----------
    if (rank == 0) {
        std::vector<double> x((size_t)N, 0.0);

        for (int i = N - 1; i >= 0; --i) {
            double sum = b_full[(size_t)i];
            for (int j = i + 1; j < N; ++j) {
                sum -= A_full[(size_t)i * N_pad + j] * x[(size_t)j];
            }
            double diag = A_full[(size_t)i * N_pad + i];
            if (std::abs(diag) < 1e-12) {
                std::cerr << "Zero diagonal at back-sub row " << i << "\n";
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            x[(size_t)i] = sum / diag;
        }

        double end = MPI_Wtime();
        std::cout << "TASK 2 — Gaussian elimination (Scatter + Bcast + Gather)\n";
        std::cout << "N = " << N << ", processes = " << size << "\n";
        std::cout << "Execution time: " << (end - start) << " seconds\n";

        // Для больших N печатать всё — бессмысленно. Покажем первые 10.
        int show = std::min(N, 10);
        std::cout << "x[0.." << show-1 << "]:\n";
        for (int i = 0; i < show; ++i) {
            std::cout << "x[" << i << "] = " << x[(size_t)i] << "\n";
        }
    } else {
        double end = MPI_Wtime();
        (void)end;
    }

    MPI_Finalize();
    return 0;
}
