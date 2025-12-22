// task2.cpp
// Компиляция:
// g++ -O2 -std=c++17 task2.cpp -o task2
// Запуск:
// ./task2

#include <iostream>
#include <random>
#include <chrono>
#include <limits>
#include <cstddef>

using Clock = std::chrono::high_resolution_clock;

static inline double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const std::size_t N = 1'000'000;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    int* a = new int[N];
    for (std::size_t i = 0; i < N; ++i) a[i] = dist(rng);

    int mn = std::numeric_limits<int>::max();
    int mx = std::numeric_limits<int>::min();

    auto t0 = Clock::now();
    for (std::size_t i = 0; i < N; ++i) {
        int v = a[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    auto t1 = Clock::now();

    std::cout << "Task 2\n";
    std::cout << "N = " << N << "\n";
    std::cout << "min = " << mn << ", max = " << mx << "\n";
    std::cout << "time(seq) = " << ms_since(t0, t1) << " ms\n";

    delete[] a;
    return 0;
}
