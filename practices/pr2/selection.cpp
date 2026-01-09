#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

void selection_sort_seq(std::vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int min_i = i;
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[min_i]) min_i = j;
        std::swap(a[i], a[min_i]);
    }
}

void selection_sort_par(std::vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int min_i = i;

        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[min_i]) {
                #pragma omp critical
                if (a[j] < a[min_i])
                    min_i = j;
            }
        }
        std::swap(a[i], a[min_i]);
    }
}

int main() {
    int N;
    std::cout << "Enter N: ";
    std::cin >> N;

    std::vector<int> a(N), b;

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 100000);

    for (int i = 0; i < N; i++) a[i] = dist(gen);
    b = a;

    auto t1 = std::chrono::high_resolution_clock::now();
    selection_sort_seq(a);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto t3 = std::chrono::high_resolution_clock::now();
    selection_sort_par(b);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto seq = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto par = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    std::cout << "Sequential time: " << seq << " ms\n";
    std::cout << "Parallel time:   " << par << " ms\n";

    return 0;
}
