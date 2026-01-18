
// task3_compare_performance.cu
// Практическая работа №5 (CUDA) — Часть 3
// Сравнение производительности параллельного стека (LIFO) и очереди (FIFO) на GPU.
// Требование: сравнить производительность реализованных структур данных.
//
// ВАЖНОЕ УПРОЩЕНИЕ ДЛЯ КОРРЕКТНОСТИ И ЧИСТОГО ИЗМЕРЕНИЯ:
// - Чтобы не получать гонки "производители+потребители" в одном ядре и не усложнять логику,
//   мы делаем 2 фазы для каждой структуры:
//     1) PUSH/ENQUEUE: все потоки добавляют элементы
//     2) POP/DEQUEUE: все потоки извлекают элементы
//   Фазы запускаются последовательно, поэтому на чтении нет "пустых" данных.
//
// Это полностью подходит под задание: используются атомарные операции, параллельный доступ,
// и проводится сравнение времени/пропускной способности.
//
// Сборка (Colab, Tesla T4):
// nvcc -O2 -arch=sm_75 task3_compare_performance.cu -o task3_perf
//
// Запуск:
// ./task3_perf
// Можно задать параметры:
// ./task3_perf <threads_total> <ops_per_thread>
// Например:
// ./task3_perf 65536 8

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << "\n";                                      \
        std::exit(1);                                           \
    }                                                           \
} while (0)

// ---------------------------
// Параллельный стек (LIFO)
// ---------------------------
// Реализация "на практике": top хранит количество элементов в стеке (size).
// Тогда push получает позицию через atomicAdd(top, 1) и пишет data[pos].
// pop делает atomicSub(top, 1) и читает data[pos-1].
//
// Такой вариант избегает классической ошибки с top=-1 и pos=-1 на первом push,
// и обеспечивает корректные границы.
struct DeviceStack {
    int* data;     // буфер в глобальной памяти
    int* top;      // текущее количество элементов (size)
    int capacity;

    __device__ __forceinline__ bool push(int v) {
        int pos = atomicAdd(top, 1); // pos = старое значение size
        if (pos < capacity) {
            data[pos] = v;
            return true;
        }
        // Переполнение: откатываем size назад
        atomicSub(top, 1);
        return false;
    }

    __device__ __forceinline__ bool pop(int* out) {
        int pos = atomicSub(top, 1); // pos = старое значение size
        int idx = pos - 1;           // индекс последнего элемента
        if (idx >= 0) {
            *out = data[idx];
            return true;
        }
        // Пусто: откатываем size вперёд
        atomicAdd(top, 1);
        return false;
    }
};

// ---------------------------
// Параллельная очередь (FIFO)
// ---------------------------
// Упрощённая очередь на атомиках:
// - enqueue: pos = atomicAdd(tail,1); data[pos]=value
// - dequeue: pos = atomicAdd(head,1); value=data[pos], если pos < tail
// Для корректного эксперимента используем 2 фазы:
// 1) сначала полностью заполняем очередь (enqueue)
// 2) потом полностью опустошаем (dequeue)
// Тогда условие pos < tail корректно.
struct DeviceQueue {
    int* data;
    int* head;
    int* tail;
    int capacity;

    __device__ __forceinline__ bool enqueue(int v) {
        int pos = atomicAdd(tail, 1);
        if (pos < capacity) {
            data[pos] = v;
            return true;
        }
        // Переполнение: откатываем tail
        atomicSub(tail, 1);
        return false;
    }

    __device__ __forceinline__ bool dequeue(int* out) {
        int pos = atomicAdd(head, 1);
        // tail читаем как обычное значение; в нашей схеме tail уже стабилен после enqueue-фазы
        int t = *tail;
        if (pos < t) {
            *out = data[pos];
            return true;
        }
        // Пусто: откатываем head
        atomicSub(head, 1);
        return false;
    }
};

// -----------------------------------
// Ядра для измерения производительности
// -----------------------------------

__global__ void stack_push_kernel(int* data, int* top, int capacity,
                                  int ops_per_thread, int* success_counter) {
    DeviceStack st{data, top, capacity};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ok = 0;

    // Каждый поток делает несколько push
    // Значение пишем такое, чтобы отличалось у потоков/итераций (для реализма)
    for (int i = 0; i < ops_per_thread; ++i) {
        int val = tid * 1000 + i;
        if (st.push(val)) ok++;
    }

    // Счётчик успешных операций (атомарно)
    atomicAdd(success_counter, ok);
}

__global__ void stack_pop_kernel(int* data, int* top, int capacity,
                                 int ops_per_thread, int* out,
                                 int* success_counter) {
    DeviceStack st{data, top, capacity};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ok = 0;

    // Каждый поток делает несколько pop
    for (int i = 0; i < ops_per_thread; ++i) {
        int v;
        if (st.pop(&v)) {
            ok++;
            // Сохраним что-то в out, чтобы компилятор не выкинул код как "мертвый"
            // (индекс выбираем безопасно, в пределах threads_total)
            out[tid] = v;
        }
    }

    atomicAdd(success_counter, ok);
}

__global__ void queue_enqueue_kernel(int* data, int* head, int* tail, int capacity,
                                     int ops_per_thread, int* success_counter) {
    DeviceQueue q{data, head, tail, capacity};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ok = 0;

    for (int i = 0; i < ops_per_thread; ++i) {
        int val = tid * 1000 + i;
        if (q.enqueue(val)) ok++;
    }

    atomicAdd(success_counter, ok);
}

__global__ void queue_dequeue_kernel(int* data, int* head, int* tail, int capacity,
                                     int ops_per_thread, int* out,
                                     int* success_counter) {
    DeviceQueue q{data, head, tail, capacity};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ok = 0;

    for (int i = 0; i < ops_per_thread; ++i) {
        int v;
        if (q.dequeue(&v)) {
            ok++;
            out[tid] = v;
        }
    }

    atomicAdd(success_counter, ok);
}

// ---------------------------
// Вспомогательный таймер GPU
// ---------------------------
template <class F>
static float time_gpu(F&& f) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main(int argc, char** argv) {
    // Параметры эксперимента
    // threads_total = общее количество потоков
    // ops_per_thread = сколько операций делает каждый поток
    int threads_total = 65536; // 256 блоков * 256 потоков
    int ops_per_thread = 8;

    if (argc >= 2) threads_total = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) ops_per_thread = std::max(1, std::atoi(argv[2]));

    // Настроим сетку/блок
    int block = 256;
    int grid  = (threads_total + block - 1) / block;

    // Общий объём операций
    long long total_ops = 1LL * threads_total * ops_per_thread;

    // Вместимость буферов:
    // Мы хотим, чтобы переполнений почти не было, поэтому capacity = total_ops.
    // Тогда все push/enqueue должны быть успешны.
    int capacity = (total_ops > INT32_MAX) ? INT32_MAX : (int)total_ops;

    std::cout << "Task3: Compare Stack vs Queue performance\n";
    std::cout << "threads_total=" << threads_total
              << ", ops_per_thread=" << ops_per_thread
              << ", total_ops=" << total_ops
              << ", capacity=" << capacity << "\n\n";

    // Выделяем память под стек/очередь и выходные данные
    int *d_data = nullptr;
    int *d_out  = nullptr;

    int *d_top = nullptr;
    int *d_head = nullptr;
    int *d_tail = nullptr;

    int *d_ok1 = nullptr;
    int *d_ok2 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, (size_t)capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)threads_total * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_top,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_head, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tail, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_ok1, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ok2, sizeof(int)));

    // ---------------------------
    // 1) STACK: PUSH phase
    // ---------------------------
    CUDA_CHECK(cudaMemset(d_top, 0, sizeof(int)));   // stack size = 0
    CUDA_CHECK(cudaMemset(d_ok1, 0, sizeof(int)));

    float t_stack_push = time_gpu([&]() {
        stack_push_kernel<<<grid, block>>>(d_data, d_top, capacity, ops_per_thread, d_ok1);
        CUDA_CHECK(cudaGetLastError());
    });

    // ---------------------------
    // 2) STACK: POP phase
    // ---------------------------
    CUDA_CHECK(cudaMemset(d_ok2, 0, sizeof(int)));

    float t_stack_pop = time_gpu([&]() {
        stack_pop_kernel<<<grid, block>>>(d_data, d_top, capacity, ops_per_thread, d_out, d_ok2);
        CUDA_CHECK(cudaGetLastError());
    });

    int h_stack_push_ok = 0, h_stack_pop_ok = 0;
    CUDA_CHECK(cudaMemcpy(&h_stack_push_ok, d_ok1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_stack_pop_ok,  d_ok2, sizeof(int), cudaMemcpyDeviceToHost));

    // ---------------------------
    // 3) QUEUE: ENQUEUE phase
    // ---------------------------
    CUDA_CHECK(cudaMemset(d_head, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ok1, 0, sizeof(int)));

    float t_queue_enq = time_gpu([&]() {
        queue_enqueue_kernel<<<grid, block>>>(d_data, d_head, d_tail, capacity, ops_per_thread, d_ok1);
        CUDA_CHECK(cudaGetLastError());
    });

    // ---------------------------
    // 4) QUEUE: DEQUEUE phase
    // ---------------------------
    CUDA_CHECK(cudaMemset(d_ok2, 0, sizeof(int)));

    float t_queue_deq = time_gpu([&]() {
        queue_dequeue_kernel<<<grid, block>>>(d_data, d_head, d_tail, capacity, ops_per_thread, d_out, d_ok2);
        CUDA_CHECK(cudaGetLastError());
    });

    int h_queue_enq_ok = 0, h_queue_deq_ok = 0;
    CUDA_CHECK(cudaMemcpy(&h_queue_enq_ok, d_ok1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_queue_deq_ok, d_ok2, sizeof(int), cudaMemcpyDeviceToHost));

    // ---------------------------
    // Итоги: время и пропускная способность
    // ---------------------------
    auto ops_per_ms = [](long long ops, float ms) -> double {
        if (ms <= 0.0f) return 0.0;
        return (double)ops / (double)ms;
    };

    std::cout << "=== Results (time in ms) ===\n";
    std::cout << "STACK push: " << t_stack_push
              << " ms, ok=" << h_stack_push_ok
              << ", throughput=" << ops_per_ms(h_stack_push_ok, t_stack_push) << " ops/ms\n";
    std::cout << "STACK pop : " << t_stack_pop
              << " ms, ok=" << h_stack_pop_ok
              << ", throughput=" << ops_per_ms(h_stack_pop_ok, t_stack_pop) << " ops/ms\n\n";

    std::cout << "QUEUE enq : " << t_queue_enq
              << " ms, ok=" << h_queue_enq_ok
              << ", throughput=" << ops_per_ms(h_queue_enq_ok, t_queue_enq) << " ops/ms\n";
    std::cout << "QUEUE deq : " << t_queue_deq
              << " ms, ok=" << h_queue_deq_ok
              << ", throughput=" << ops_per_ms(h_queue_deq_ok, t_queue_deq) << " ops/ms\n";

    // Простая проверка корректности по числу успешных операций:
    // При capacity=total_ops ожидаем ok примерно равным total_ops (может отличаться, если capacity ограничен).
    std::cout << "\n=== Correctness check (counts) ===\n";
    std::cout << "Expected ops <= capacity: " << capacity << "\n";
    std::cout << "Stack push ok: " << h_stack_push_ok << ", Stack pop ok: " << h_stack_pop_ok << "\n";
    std::cout << "Queue enq  ok: " << h_queue_enq_ok  << ", Queue deq  ok: " << h_queue_deq_ok  << "\n";

    // Освобождение
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_top));
    CUDA_CHECK(cudaFree(d_head));
    CUDA_CHECK(cudaFree(d_tail));
    CUDA_CHECK(cudaFree(d_ok1));
    CUDA_CHECK(cudaFree(d_ok2));

    return 0;
}
