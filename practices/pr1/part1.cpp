#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

int main() {
    int N;
    std::cout << "Enter N: ";
    std::cin >> N;

    std::vector<int> a(N);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < N; i++)
        a[i] = dist(gen);

    int min1 = a[0], max1 = a[0];

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        if (a[i] < min1) min1 = a[i];
        if (a[i] > max1) max1 = a[i];
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    int min2 = a[0], max2 = a[0];

    auto t3 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(min:min2) reduction(max:max2)
    for (int i = 0; i < N; i++) {
        if (a[i] < min2) min2 = a[i];
        if (a[i] > max2) max2 = a[i];
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    auto seq = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto par = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "Sequential min = " << min1 << " max = " << max1 << "\n";
    std::cout << "Parallel   min = " << min2 << " max = " << max2 << "\n";
    std::cout << "Sequential time = " << seq << " us\n";
    std::cout << "Parallel time   = " << par << " us\n";

    return 0;
}
