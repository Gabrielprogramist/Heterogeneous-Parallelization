// task1.cpp
// Компиляция:
// g++ -O2 -std=c++17 task1.cpp -o task1
// Запуск:
// ./task1

#include <iostream>
#include <random>
#include <cstddef>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const std::size_t N = 50'000;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    int* a = new int[N];

    long long sum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        sum += a[i];
    }

    double avg = static_cast<double>(sum) / static_cast<double>(N);

    std::cout << "Task 1\n";
    std::cout << "N = " << N << "\n";
    std::cout << "average = " << avg << "\n";

    delete[] a;
    return 0;
}
