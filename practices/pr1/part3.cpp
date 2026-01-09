#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>

int main() {
    int N;
    std::cout << "Enter N: ";
    std::cin >> N;

    int* a = new int[N];

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < N; i++)
        a[i] = dist(gen);

    long long sum1 = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        sum1 += a[i];
    auto t2 = std::chrono::high_resolution_clock::now();

    long long sum2 = 0;

    auto t3 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum2)
    for (int i = 0; i < N; i++)
        sum2 += a[i];
    auto t4 = std::chrono::high_resolution_clock::now();

    double avg1 = (double)sum1 / N;
    double avg2 = (double)sum2 / N;

    auto seq = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto par = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "Sequential average = " << avg1 << " time = " << seq << " us\n";
    std::cout << "Parallel   average = " << avg2 << " time = " << par << " us\n";

    delete[] a;
    return 0;
}
