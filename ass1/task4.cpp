// task4.cpp
// Компиляция:
// g++ -O2 -std=c++17 -fopenmp task4.cpp -o task4
// Запуск:
// ./task4

#include <iostream>
#include <random>
#include <chrono>
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

    const std::size_t N = 5'000'000;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    int* a = new int[N];
    for (std::size_t i = 0; i < N; ++i) a[i] = dist(rng);

    // Последовательно
    auto t_seq0 = Clock::now();
    long long sum_seq = 0;
    for (std::size_t i = 0; i < N; ++i) sum_seq += a[i];
    double avg_seq = static_cast<double>(sum_seq) / static_cast<double>(N);
    auto t_seq1 = Clock::now();
    double t_seq_ms = ms_since(t_seq0, t_seq1);

    // Параллельно с reduction
    auto t_par0 = Clock::now();
    long long sum_par = 0;

    #pragma omp parallel for reduction(+:sum_par)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        sum_par += a[i];
    }

    double avg_par = static_cast<double>(sum_par) / static_cast<double>(N);
    auto t_par1 = Clock::now();
    double t_par_ms = ms_since(t_par0, t_par1);

    std::cout << "Task 4\n";
    std::cout << "N = " << N << "\n";
    std::cout << "seq: average = " << avg_seq << ", time = " << t_seq_ms << " ms\n";
    std::cout << "par: average = " << avg_par << ", time = " << t_par_ms << " ms\n";
    std::cout << "speedup = " << (t_seq_ms / t_par_ms) << "x\n";

    delete[] a;
    return 0;
}
