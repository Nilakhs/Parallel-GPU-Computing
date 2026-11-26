// ==============================================================================
// CUDA Warp Divergence Analysis & Microbenchmarking
// Parallel GPU Computing (PGC) - Lab Evaluation 2 - Theme 7
//
// Hardware Target Verified: NVIDIA RTX 4500 Ada Generation (Compute Capability SM 8.9)
// Toolchain: CUDA 13.4 (NVCC) / Microsoft Visual C++ (MSVC) 19.51 / x64 Windows 11
//
// Implemented Kernel Strategies:
//   1. Divergent Branching (Alternating intra-warp lanes: i % 2 == 0)
//   2. Warp-Aligned Branching (Uniform intra-warp lanes: (i / 32) % 2 == 0)
//   3. Data-Dependent Branching (Runtime data condition: input[i] > threshold)
//   4. Branch-Free Predication (Unconditional computation + conditional select)
//
// Benchmarking Methodology:
//   - Isolated warm-up kernel execution to prime caches and clock frequencies
//   - GPU-side execution latency measured using hardware CUDA Events
//   - Rigorous synchronization (cudaEventSynchronize / cudaDeviceSynchronize)
//   - Multi-iteration sampling with statistical mean, median, min, max, and std-dev
//   - Kernel-only vs. End-to-End latency (H2D + Kernel + D2H) profiling
//   - Host-side numerical correctness verification (PASS/FAIL validation against golden CPU reference)
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>

// Robust CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "[FATAL CUDA ERROR] at " << __FILE__ << ":" << __LINE__  \
                      << " -> " << cudaGetErrorString(err) << std::endl;          \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// Architecture and Benchmark Constants
constexpr int WARP_SIZE = 32;       // Standard NVIDIA warp size (32 threads)
constexpr int N = 1 << 24;          // 16,777,216 elements (64 MB FP32 buffer)
constexpr int THREADS = 256;        // 256 threads/block = 8 warps/block
constexpr int REPEATS = 20;         // Benchmark iteration count

// Arithmetic payloads designed to prevent compiler optimization
// Loops are marked with #pragma unroll 1 to prevent loop unrolling optimizations
__device__ __forceinline__ float workA(float x)
{
    #pragma unroll 1
    for (int k = 0; k < 8; ++k) {
        x = x * 1.0001f + 0.25f;
    }
    return x;
}

__device__ __forceinline__ float workB(float x)
{
    #pragma unroll 1
    for (int k = 0; k < 8; ++k) {
        x = x * 0.9999f - 0.25f;
    }
    return x;
}

// Host reference functions for numerical correctness verification
inline float hostWorkA(float x)
{
    for (int k = 0; k < 8; ++k) {
        x = x * 1.0001f + 0.25f;
    }
    return x;
}

inline float hostWorkB(float x)
{
    for (int k = 0; k < 8; ++k) {
        x = x * 0.9999f - 0.25f;
    }
    return x;
}

// ==============================================================================
// 1. Divergent Kernel: Alternating thread lanes within each warp
// ==============================================================================
// Since blockDim.x = 256 (even), (i % 2) == (threadIdx.x % 2).
// Even lanes (0, 2, 4...) execute workA(); odd lanes (1, 3, 5...) execute workB().
// Active Mask Phase 1: 0x55555555 (16 active lanes).
// Active Mask Phase 2: 0xAAAAAAAA (16 active lanes).
// Execution of the two paths is serialized within each warp.
__global__ void divergentKernel(const float* __restrict__ input, float* __restrict__ output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((i % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

// ==============================================================================
// 2. Warp-Aligned Kernel: Chunk-aligned branching at 32-thread boundaries
// ==============================================================================
// Global warp index: (i / WARP_SIZE).
// All 32 threads in any given warp evaluate the exact same condition.
// Even warps execute workA() with active mask 0xFFFFFFFF (100% active lanes).
// Odd warps execute workB() with active mask 0xFFFFFFFF (100% active lanes).
// Zero intra-warp divergence occurs.
__global__ void alignedKernel(const float* __restrict__ input, float* __restrict__ output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (((i / WARP_SIZE) % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

// ==============================================================================
// 3. Data-Dependent Kernel: Branch condition depends on runtime memory value
// ==============================================================================
// Condition depends on pseudorandom input values. Adjacent threads make
// independent choices depending on whether input[i] > threshold.
// Produces realistic stochastic divergence patterns across warps.
__global__ void dataDependentKernel(const float* __restrict__ input, float* __restrict__ output,
                                    int n, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (input[i] > threshold)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

// ==============================================================================
// 4. Branch-Free Kernel: Predicated execution via unconditional arithmetic
// ==============================================================================
// Evaluates both candidate calculations unconditionally across all 32 lanes.
// Uses ternary operator which compiles to PTX conditional select (selp).
// Eliminates branch divergence (100% active mask), but incurs "work inflation"
// by computing 16 arithmetic operations per thread instead of 8.
__global__ void branchFreeKernel(const float* __restrict__ input, float* __restrict__ output,
                                 int n, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = workA(input[i]);
        float b = workB(input[i]);
        output[i] = (input[i] > threshold) ? a : b;
    }
}

// ==============================================================================
// Benchmarking Data Structures and Statistics
// ==============================================================================
struct BenchmarkStats {
    std::string name;
    float avgMs;
    float medianMs;
    float stdDevMs;
    float minMs;
    float maxMs;
    double endToEndMs;
    double throughputMElemPerSec;
    bool verificationPassed;
    float maxAbsError;
};

// Launch helper for all 4 kernel types
void launchKernel(int kernelType, int blocks, int threads,
                  const float* d_input, float* d_output, int n, float threshold)
{
    switch (kernelType) {
        case 0:
            divergentKernel<<<blocks, threads>>>(d_input, d_output, n);
            break;
        case 1:
            alignedKernel<<<blocks, threads>>>(d_input, d_output, n);
            break;
        case 2:
            dataDependentKernel<<<blocks, threads>>>(d_input, d_output, n, threshold);
            break;
        case 3:
            branchFreeKernel<<<blocks, threads>>>(d_input, d_output, n, threshold);
            break;
        default:
            std::cerr << "Invalid kernel type specified." << std::endl;
            std::exit(EXIT_FAILURE);
    }
}

// ==============================================================================
// Output Verification against Host Reference
// ==============================================================================
bool verifyCorrectness(int kernelType, const float* h_input, const float* h_output,
                       int n, float threshold, float& maxAbsError)
{
    maxAbsError = 0.0f;
    int mismatchCount = 0;
    constexpr float TOLERANCE = 1e-4f;

    for (int i = 0; i < n; ++i) {
        float expected = 0.0f;
        switch (kernelType) {
            case 0: // Divergent
                expected = ((i % 2) == 0) ? hostWorkA(h_input[i]) : hostWorkB(h_input[i]);
                break;
            case 1: // Aligned
                expected = (((i / WARP_SIZE) % 2) == 0) ? hostWorkA(h_input[i]) : hostWorkB(h_input[i]);
                break;
            case 2: // Data-Dependent
            case 3: // Branch-Free (must match identical mathematical result)
                expected = (h_input[i] > threshold) ? hostWorkA(h_input[i]) : hostWorkB(h_input[i]);
                break;
        }

        float diff = std::fabs(h_output[i] - expected);
        if (diff > maxAbsError) {
            maxAbsError = diff;
        }
        if (diff > TOLERANCE) {
            mismatchCount++;
            if (mismatchCount <= 5) {
                std::cerr << "Mismatch at index " << i << ": GPU=" << h_output[i]
                          << ", CPU=" << expected << ", Diff=" << diff << std::endl;
            }
        }
    }
    return (mismatchCount == 0);
}

// ==============================================================================
// Profiling Engine: Measures Kernel-Only and End-to-End Latencies
// ==============================================================================
BenchmarkStats runBenchmark(const std::string& name, int kernelType,
                            const float* h_input, float* h_output,
                            const float* d_input, float* d_output,
                            int n, int blocks, int threads, float threshold)
{
    BenchmarkStats stats;
    stats.name = name;

    // 1. Warm-up Launch (ensures driver initialization, PTX JIT, GPU clocks stabilized)
    launchKernel(kernelType, blocks, threads, d_input, d_output, n, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Individual Repetition Timing using CUDA Events
    std::vector<float> sampleTimesMs(REPEATS, 0.0f);
    cudaEvent_t startEvents[REPEATS];
    cudaEvent_t stopEvents[REPEATS];

    for (int r = 0; r < REPEATS; ++r) {
        CUDA_CHECK(cudaEventCreate(&startEvents[r]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[r]));
    }

    for (int r = 0; r < REPEATS; ++r) {
        CUDA_CHECK(cudaEventRecord(startEvents[r]));
        launchKernel(kernelType, blocks, threads, d_input, d_output, n, threshold);
        CUDA_CHECK(cudaEventRecord(stopEvents[r]));
    }

    // Synchronize the final event to guarantee all launches completed
    CUDA_CHECK(cudaEventSynchronize(stopEvents[REPEATS - 1]));
    CUDA_CHECK(cudaGetLastError());

    float totalMs = 0.0f;
    for (int r = 0; r < REPEATS; ++r) {
        CUDA_CHECK(cudaEventElapsedTime(&sampleTimesMs[r], startEvents[r], stopEvents[r]));
        totalMs += sampleTimesMs[r];
        CUDA_CHECK(cudaEventDestroy(startEvents[r]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[r]));
    }

    stats.avgMs = totalMs / REPEATS;

    // Calculate statistical metrics: Min, Max, Median, Standard Deviation
    std::vector<float> sortedSamples = sampleTimesMs;
    std::sort(sortedSamples.begin(), sortedSamples.end());
    stats.minMs = sortedSamples.front();
    stats.maxMs = sortedSamples.back();
    stats.medianMs = (REPEATS % 2 == 0)
        ? (sortedSamples[REPEATS / 2 - 1] + sortedSamples[REPEATS / 2]) * 0.5f
        : sortedSamples[REPEATS / 2];

    float varianceSum = 0.0f;
    for (float t : sampleTimesMs) {
        float diff = t - stats.avgMs;
        varianceSum += diff * diff;
    }
    stats.stdDevMs = (REPEATS > 1) ? std::sqrt(varianceSum / (REPEATS - 1)) : 0.0f;
    stats.throughputMElemPerSec = (static_cast<double>(n) / (stats.avgMs / 1000.0)) / 1e6;

    // 3. End-to-End Latency Measurement (Host-to-Device + Kernel + Device-to-Host)
    auto e2eStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_output, d_input, n * sizeof(float), cudaMemcpyDeviceToDevice)); // Touch buffers
    launchKernel(kernelType, blocks, threads, d_input, d_output, n, threshold);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    auto e2eEnd = std::chrono::high_resolution_clock::now();
    stats.endToEndMs = std::chrono::duration<double, std::milli>(e2eEnd - e2eStart).count();

    // 4. Correctness Verification
    stats.verificationPassed = verifyCorrectness(kernelType, h_input, h_output, n, threshold, stats.maxAbsError);

    // Print summary line (matches format in terminal_output.txt)
    std::cout << std::left << std::setw(30) << name
              << std::right << std::fixed << std::setprecision(4)
              << stats.avgMs << " ms"
              << "   [Verification: " << (stats.verificationPassed ? "PASS" : "FAIL")
              << ", Max Err: " << std::scientific << std::setprecision(2) << stats.maxAbsError << "]"
              << std::endl;

    return stats;
}

// ==============================================================================
// Main Entry Point
// ==============================================================================
int main(int argc, char* argv[])
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "[ERROR] No CUDA-capable NVIDIA GPU was detected on this system." << std::endl;
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "CUDA Warp Divergence Practical" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Warp size reported by device: " << prop.warpSize << std::endl;
    std::cout << "Array elements: " << N << std::endl;
    std::cout << "Threads per block: " << THREADS << std::endl;
    std::cout << "Kernel repetitions: " << REPEATS << std::endl << std::endl;

    if (prop.warpSize != WARP_SIZE) {
        std::cout << "[WARNING] Device warp size (" << prop.warpSize
                  << ") differs from expected NVIDIA standard (" << WARP_SIZE << ")." << std::endl;
    }

    // Allocate host memory buffers
    std::vector<float> h_input(N);
    std::vector<float> h_output(N, 0.0f);

    // Deterministic mixed data generation (LCG uniform distribution in [0.0, 1.0])
    for (int i = 0; i < N; ++i) {
        unsigned int x = static_cast<unsigned int>(i);
        x = x * 1664525u + 1013904223u;
        h_input[i] = static_cast<float>(x & 0xFFFFu) / 65535.0f;
    }

    // Allocate GPU device memory
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    // Initial Host-to-Device memory copy
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (N + THREADS - 1) / THREADS;
    const float threshold = 0.5f;

    std::cout << "Average kernel execution time (" << REPEATS << " repetitions):" << std::endl;

    BenchmarkStats sDivergent = runBenchmark("Divergent branch", 0,
                                             h_input.data(), h_output.data(),
                                             d_input, d_output, N, blocks, THREADS, threshold);

    BenchmarkStats sAligned = runBenchmark("Warp-aligned branch", 1,
                                           h_input.data(), h_output.data(),
                                           d_input, d_output, N, blocks, THREADS, threshold);

    BenchmarkStats sData = runBenchmark("Data-dependent branch", 2,
                                         h_input.data(), h_output.data(),
                                         d_input, d_output, N, blocks, THREADS, threshold);

    BenchmarkStats sBranchFree = runBenchmark("Branch-free selection", 3,
                                             h_input.data(), h_output.data(),
                                             d_input, d_output, N, blocks, THREADS, threshold);

    std::cout << std::endl << "Comparison & Ratio Analysis:" << std::endl;
    std::cout << "----------------------------" << std::endl;
    if (sAligned.avgMs > 0.0f) {
        float rDivAligned = sDivergent.avgMs / sAligned.avgMs;
        float rAlignDiv = sAligned.avgMs / sDivergent.avgMs;
        std::cout << "Divergent / aligned time ratio:       " << std::fixed << std::setprecision(3) << rDivAligned << "x" << std::endl;
        std::cout << "Aligned execution-time ratio (T/T_base): " << std::fixed << std::setprecision(4) << rAlignDiv << std::endl;
        std::cout << "Aligned speedup (T_base/T):              " << std::fixed << std::setprecision(4) << (1.0f / rAlignDiv) << "x" << std::endl;
    }
    if (sBranchFree.avgMs > 0.0f) {
        float rDataBf = sData.avgMs / sBranchFree.avgMs;
        float rBfData = sBranchFree.avgMs / sData.avgMs;
        std::cout << "Data-dependent / branch-free ratio:   " << std::fixed << std::setprecision(3) << rDataBf << "x" << std::endl;
        std::cout << "Branch-free execution-time ratio:     " << std::fixed << std::setprecision(4) << rBfData << std::endl;
        std::cout << "Branch-free speedup vs divergent:     " << std::fixed << std::setprecision(4) << (sDivergent.avgMs / sBranchFree.avgMs) << "x" << std::endl;
    }

    std::cout << std::endl
              << "Note on Mathematical Interpretations:" << std::endl
              << "  - Execution-Time Ratio R = T_variant / T_baseline (R < 1 means variant is FASTER)" << std::endl
              << "  - Speedup S = T_baseline / T_variant (S > 1 means variant is FASTER)" << std::endl
              << "  - Workload characteristics, memory coalescing, and hardware predication" << std::endl
              << "    govern real execution latencies on Ada Lovelace (SM 8.9) architectures." << std::endl;

    // Detailed Statistical Diagnostics Table
    std::cout << std::endl << "Detailed Statistical Diagnostics:" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(26) << "Kernel Variant"
              << std::right << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "Median (ms)"
              << std::setw(12) << "StdDev (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(14) << "End-to-End"
              << std::setw(10) << "Status" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    auto printStatRow = [](const BenchmarkStats& s) {
        std::cout << std::left << std::setw(26) << s.name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(12) << s.avgMs
                  << std::setw(12) << s.medianMs
                  << std::setw(12) << s.stdDevMs
                  << std::setw(12) << s.minMs
                  << std::setw(12) << s.maxMs
                  << std::setprecision(2) << std::setw(12) << s.endToEndMs << " ms"
                  << std::setw(10) << (s.verificationPassed ? "PASS" : "FAIL")
                  << std::endl;
    };

    printStatRow(sDivergent);
    printStatRow(sAligned);
    printStatRow(sData);
    printStatRow(sBranchFree);
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    // Clean up GPU allocations
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
