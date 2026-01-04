// task2_openmp_minmax.cpp
// Задача 2 (OpenMP): массив 10 000, min/max последовательно и параллельно, сравнение времени
// Компиляция:
// g++ -O2 -std=c++17 -fopenmp task2_openmp_minmax.cpp -o task2
// Запуск:
// ./task2

#include <iostream>
#include <random>
#include <chrono>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;

static inline double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const std::size_t N = 10'000;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(1, 1000000);

    int* a = new int[N];
    for (std::size_t i = 0; i < N; ++i) a[i] = dist(rng);

    // Последовательно
    int mn_seq = std::numeric_limits<int>::max();
    int mx_seq = std::numeric_limits<int>::min();

    auto t0 = Clock::now();
    for (std::size_t i = 0; i < N; ++i) {
        int v = a[i];
        if (v < mn_seq) mn_seq = v;
        if (v > mx_seq) mx_seq = v;
    }
    auto t1 = Clock::now();
    double t_seq = ms_since(t0, t1);

    // Параллельно OpenMP (переносимый вариант: локальные min/max + объединение)
    int mn_par = std::numeric_limits<int>::max();
    int mx_par = std::numeric_limits<int>::min();

    auto p0 = Clock::now();
    #pragma omp parallel
    {
        int local_min = std::numeric_limits<int>::max();
        int local_max = std::numeric_limits<int>::min();

        #pragma omp for nowait
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
            int v = a[i];
            if (v < local_min) local_min = v;
            if (v > local_max) local_max = v;
        }

        #pragma omp critical
        {
            if (local_min < mn_par) mn_par = local_min;
            if (local_max > mx_par) mx_par = local_max;
        }
    }
    auto p1 = Clock::now();
    double t_par = ms_since(p0, p1);

    std::cout << "Task 2 (OpenMP min/max)\n";
    std::cout << "N = " << N << "\n";
    std::cout << "seq: min=" << mn_seq << " max=" << mx_seq << " time=" << t_seq << " ms\n";
    std::cout << "par: min=" << mn_par << " max=" << mx_par << " time=" << t_par << " ms\n";

    if (mn_seq != mn_par || mx_seq != mx_par) {
        std::cout << "WARNING: results differ (should not happen)\n";
    } else {
        std::cout << "OK: results match\n";
    }

    if (t_par > 0.0) std::cout << "speedup = " << (t_seq / t_par) << "x\n";

    std::cout << "Conclusion: на маленьких N ускорение может быть слабым из-за накладных расходов OpenMP.\n";

    delete[] a;
    return 0;
}
