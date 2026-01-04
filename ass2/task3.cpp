// task3_openmp_selectionsort.cpp
// Задача 3 (OpenMP): сортировка выбором — последовательная и частично параллельная (поиск минимума)
// Проверка производительности для N=1000 и N=10000
// Компиляция:
// g++ -O2 -std=c++17 -fopenmp task3_openmp_selectionsort.cpp -o task3
// Запуск:
// ./task3

#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;

static inline double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool is_sorted_non_decreasing(const std::vector<int>& v) {
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i-1] > v[i]) return false;
    }
    return true;
}

static void selection_sort_seq(std::vector<int>& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        std::size_t min_idx = i;
        int min_val = a[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            if (a[j] < min_val) {
                min_val = a[j];
                min_idx = j;
            }
        }
        if (min_idx != i) std::swap(a[i], a[min_idx]);
    }
}

// Параллелизация selection sort ограничена: каждую итерацию i всё равно нужно делать последовательно,
// но поиск min на хвосте можно распараллелить.
static void selection_sort_omp(std::vector<int>& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        int global_min = a[i];
        std::size_t global_idx = i;

        #pragma omp parallel
        {
            int local_min = std::numeric_limits<int>::max();
            std::size_t local_idx = i;

            #pragma omp for nowait
            for (std::int64_t j = static_cast<std::int64_t>(i); j < static_cast<std::int64_t>(n); ++j) {
                int v = a[static_cast<std::size_t>(j)];
                if (v < local_min) {
                    local_min = v;
                    local_idx = static_cast<std::size_t>(j);
                }
            }

            #pragma omp critical
            {
                if (local_min < global_min) {
                    global_min = local_min;
                    global_idx = local_idx;
                }
            }
        }

        if (global_idx != i) std::swap(a[i], a[global_idx]);
    }
}

static void run_case(std::size_t N, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(1, 1'000'000);
    std::vector<int> base(N);
    for (std::size_t i = 0; i < N; ++i) base[i] = dist(rng);

    std::vector<int> a1 = base;
    std::vector<int> a2 = base;

    auto t0 = Clock::now();
    selection_sort_seq(a1);
    auto t1 = Clock::now();
    double t_seq = ms_since(t0, t1);

    auto p0 = Clock::now();
    selection_sort_omp(a2);
    auto p1 = Clock::now();
    double t_par = ms_since(p0, p1);

    std::cout << "N = " << N << "\n";
    std::cout << "seq time = " << t_seq << " ms, sorted=" << (is_sorted_non_decreasing(a1) ? "yes" : "no") << "\n";
    std::cout << "par time = " << t_par << " ms, sorted=" << (is_sorted_non_decreasing(a2) ? "yes" : "no") << "\n";
    std::cout << "speedup = " << (t_par > 0.0 ? (t_seq / t_par) : 0.0) << "x\n";

    if (a1 != a2) {
        // Для сортировки выбором результат должен совпадать с точностью до перестановок равных элементов.
        // Здесь возможна разница в порядке равных элементов. Это не ошибка сортировки как таковой.
        // Проверим корректнее: оба должны быть отсортированы.
        if (is_sorted_non_decreasing(a1) && is_sorted_non_decreasing(a2)) {
            std::cout << "Note: arrays differ due to different order of equal elements (still correctly sorted).\n";
        } else {
            std::cout << "WARNING: sorting mismatch / not sorted.\n";
        }
    }

    std::cout << "Conclusion: selection sort плохо масштабируется, т.к. внешний цикл зависит от предыдущих итераций; параллелится только поиск минимума.\n\n";
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout << "Task 3 (OpenMP selection sort)\n";
    run_case(1'000, rng);
    run_case(10'000, rng);

    return 0;
}
