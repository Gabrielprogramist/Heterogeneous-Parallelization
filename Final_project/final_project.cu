// Build: nvcc -O3 -std=c++17 main.cu -o gpu_pipeline
// Run example:
//   ./gpu_pipeline --n_mb 512 --chunk_mb 32 --streams 4 --iters 5

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) do {                                  \
  cudaError_t err = (call);                                    \
  if (err != cudaSuccess) {                                    \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";\
    std::exit(1);                                              \
  }                                                            \
} while(0)

// ------------------------- Kernel -------------------------
__global__ void saxpy_kernel(const float* x, float* y, float a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

// ------------------------- Utils -------------------------
static size_t mb_to_elems(size_t mb) {
  // mb = megabytes of float buffer size
  // 1 MB = 1024*1024 bytes, float = 4 bytes
  return (mb * 1024ull * 1024ull) / sizeof(float);
}

static double max_abs_diff(const float* a, const float* b, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(double(a[i]) - double(b[i])));
  }
  return m;
}

struct Args {
  size_t n_mb = 256;        // total size of arrays (x and y), in MB each
  size_t chunk_mb = 32;     // chunk size in MB
  int streams = 4;          // number of streams
  int iters = 5;            // measurement iterations (after warmup)
  float a = 2.0f;           // saxpy coefficient
};

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need_val = [&](const std::string& key) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << key << "\n";
        std::exit(1);
      }
      return std::string(argv[++i]);
    };

    if (k == "--n_mb") args.n_mb = std::stoull(need_val(k));
    else if (k == "--chunk_mb") args.chunk_mb = std::stoull(need_val(k));
    else if (k == "--streams") args.streams = std::stoi(need_val(k));
    else if (k == "--iters") args.iters = std::stoi(need_val(k));
    else if (k == "--a") args.a = std::stof(need_val(k));
    else if (k == "--help") {
      std::cout <<
        "Usage: ./gpu_pipeline [options]\n"
        "Options:\n"
        "  --n_mb <MB>        Total array size per buffer (x and y), default 256\n"
        "  --chunk_mb <MB>    Chunk size for async pipeline, default 32\n"
        "  --streams <N>      Number of CUDA streams, default 4\n"
        "  --iters <N>        Measurement iterations (after warmup), default 5\n"
        "  --a <float>        SAXPY coefficient, default 2.0\n";
      std::exit(0);
    }
  }
  if (args.streams <= 0) {
    std::cerr << "--streams must be > 0\n";
    std::exit(1);
  }
  if (args.chunk_mb == 0) {
    std::cerr << "--chunk_mb must be > 0\n";
    std::exit(1);
  }
  return args;
}

static float time_event_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

// ------------------------- SYNC version -------------------------
static float run_sync(const float* h_x, float* h_y, size_t n, float a) {
  float *d_x = nullptr, *d_y = nullptr;
  size_t bytes = n * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = int((n + threads - 1) / threads);
  saxpy_kernel<<<blocks, threads>>>(d_x, d_y, a, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = time_event_ms(start, stop);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return ms;
}

// ------------------------- ASYNC chunked pipeline -------------------------
static float run_async_chunked(const float* h_x_pinned, float* h_y_pinned,
                               size_t n, size_t chunk_elems,
                               int numStreams, float a) {
  float *d_x = nullptr, *d_y = nullptr;
  size_t bytes = n * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));

  std::vector<cudaStream_t> streams(numStreams);
  for (int s = 0; s < numStreams; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  size_t chunk_idx = 0;
  for (size_t offset = 0; offset < n; offset += chunk_elems, ++chunk_idx) {
    size_t this_chunk = std::min(chunk_elems, n - offset);
    size_t this_bytes = this_chunk * sizeof(float);
    cudaStream_t st = streams[chunk_idx % (size_t)numStreams];

    // H2D async
    CUDA_CHECK(cudaMemcpyAsync(d_x + offset, h_x_pinned + offset, this_bytes,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_y + offset, h_y_pinned + offset, this_bytes,
                               cudaMemcpyHostToDevice, st));

    // Kernel in same stream: respects order H2D -> kernel -> D2H
    int threads = 256;
    int blocks = int((this_chunk + threads - 1) / threads);
    saxpy_kernel<<<blocks, threads, 0, st>>>(d_x + offset, d_y + offset, a, this_chunk);
    CUDA_CHECK(cudaGetLastError());

    // D2H async (result y chunk)
    CUDA_CHECK(cudaMemcpyAsync(h_y_pinned + offset, d_y + offset, this_bytes,
                               cudaMemcpyDeviceToHost, st));
  }

  // End-to-end requires that all D2H are completed
  for (int s = 0; s < numStreams; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = time_event_ms(start, stop);

  for (int s = 0; s < numStreams; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return ms;
}

// ------------------------- Main -------------------------
int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "GPU: " << prop.name << "\n";

  // Total elements per array
  size_t n = mb_to_elems(args.n_mb);
  size_t chunk_elems = mb_to_elems(args.chunk_mb);

  if (chunk_elems == 0) {
    std::cerr << "Chunk size too small\n";
    return 1;
  }
  if (chunk_elems > n) chunk_elems = n;

  std::cout << "n_mb=" << args.n_mb
            << " (n=" << n << " floats ~ " << (n*sizeof(float))/ (1024.0*1024.0) << " MB)\n"
            << "chunk_mb=" << args.chunk_mb
            << " (chunk_elems=" << chunk_elems << ")\n"
            << "streams=" << args.streams
            << " iters=" << args.iters
            << " a=" << args.a << "\n\n";

  // Host pageable buffers (for sync)
  std::vector<float> h_x(n), h_y(n), h_y_sync(n);

  for (size_t i = 0; i < n; ++i) {
    h_x[i] = float(i % 100) * 0.01f;
    h_y[i] = 1.0f;
  }

  // Warm-up (important)
  h_y_sync = h_y;
  (void)run_sync(h_x.data(), h_y_sync.data(), n, args.a);

  // Measure SYNC
  std::vector<float> sync_times;
  sync_times.reserve(args.iters);

  for (int it = 0; it < args.iters; ++it) {
    h_y_sync = h_y;
    float ms = run_sync(h_x.data(), h_y_sync.data(), n, args.a);
    sync_times.push_back(ms);
  }

  auto avg = [](const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += x;
    return float(s / std::max<size_t>(1, v.size()));
  };

  float sync_avg = avg(sync_times);

  // Pinned host buffers (for async)
  float *h_x_pin = nullptr, *h_y_pin = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_x_pin, n * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_y_pin, n * sizeof(float)));

  for (size_t i = 0; i < n; ++i) {
    h_x_pin[i] = h_x[i];
    h_y_pin[i] = h_y[i];
  }

  // Warm-up async
  (void)run_async_chunked(h_x_pin, h_y_pin, n, chunk_elems, args.streams, args.a);

  // Measure ASYNC
  std::vector<float> async_times;
  async_times.reserve(args.iters);

  for (int it = 0; it < args.iters; ++it) {
    // reset input y
    for (size_t i = 0; i < n; ++i) h_y_pin[i] = h_y[i];

    float ms = run_async_chunked(h_x_pin, h_y_pin, n, chunk_elems, args.streams, args.a);
    async_times.push_back(ms);
  }

  float async_avg = avg(async_times);

  // Validate correctness against last sync result (h_y_sync holds last sync run)
  // Make sure pinned result corresponds to last async run already in h_y_pin
  // For correctness: compare async output with sync output computed for same input
  // We'll compute one more sync for strict match:
  std::vector<float> h_y_check = h_y;
  (void)run_sync(h_x.data(), h_y_check.data(), n, args.a);

  double err = max_abs_diff(h_y_pin, h_y_check.data(), n);

  std::cout << "SYNC avg ms  : " << sync_avg << "\n";
  std::cout << "ASYNC avg ms : " << async_avg << "\n";
  std::cout << "Speedup      : " << (sync_avg / async_avg) << "x\n";
  std::cout << "Max abs err  : " << err << "\n";

  CUDA_CHECK(cudaFreeHost(h_x_pin));
  CUDA_CHECK(cudaFreeHost(h_y_pin));

  return 0;
}
