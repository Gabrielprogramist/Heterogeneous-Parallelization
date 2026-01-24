// task1_openmp_performance.cpp
// Практическая работа №10 — Задание 1
// Анализ производительности OpenMP-программы

#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <cmath>

int main() {
    const int N = 10'000'000;
    std::vector<double> data(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; ++i)
        data[i] = dist(gen);

    // ----- Последовательная часть -----
    double t_start = omp_get_wtime();
    double seq_sum = 0.0;
    for (int i = 0; i < N / 10; ++i)   // искусственно последовательная часть
        seq_sum += data[i];
    double t_seq = omp_get_wtime() - t_start;

    // ----- Параллельная часть -----
    t_start = omp_get_wtime();
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = N / 10; i < N; ++i)
        sum += data[i];
    double t_par = omp_get_wtime() - t_start;

    std::cout << "TASK 1 — OpenMP performance\n";
    std::cout << "Threads: " << omp_get_max_threads() << "\n";
    std::cout << "Sequential time: " << t_seq << " s\n";
    std::cout << "Parallel time:   " << t_par << " s\n";
    std::cout << "Total sum:       " << (seq_sum + sum) << "\n";

    return 0;
}
