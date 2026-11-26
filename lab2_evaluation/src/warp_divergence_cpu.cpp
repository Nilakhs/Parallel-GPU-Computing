// ==============================================================================
// CPU Baseline Implementation for Warp Divergence Analysis
// Simulates the equivalent algorithmic operations of warp_divergence.cu on host CPU
//
// Compilation:
//   Windows MSVC:  cl /O2 /EHsc /std:c++14 src\warp_divergence_cpu.cpp /Fe:bin\warp_divergence_cpu.exe
//   Linux/macOS:   g++ -O2 -std=c++14 src/warp_divergence_cpu.cpp -o bin/warp_divergence_cpu
// ==============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

constexpr int N = 1 << 24;          // 16,777,216 elements
constexpr int REPEATS = 20;         // Iterations for stable measurement
constexpr int WARP_SIZE = 32;       // Simulated warp width for aligned indexing

inline float workA(float x) {
    for (int k = 0; k < 8; ++k)
        x = x * 1.0001f + 0.25f;
    return x;
}

inline float workB(float x) {
    for (int k = 0; k < 8; ++k)
        x = x * 0.9999f - 0.25f;
    return x;
}

void cpuDivergent(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        if ((i % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

void cpuAligned(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        if (((i / WARP_SIZE) % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

void cpuDataDependent(const float* input, float* output, int n, float threshold) {
    for (int i = 0; i < n; ++i) {
        if (input[i] > threshold)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}

void cpuBranchFree(const float* input, float* output, int n, float threshold) {
    for (int i = 0; i < n; ++i) {
        float a = workA(input[i]);
        float b = workB(input[i]);
        output[i] = (input[i] > threshold) ? a : b;
    }
}

struct CpuBenchmarkResult {
    std::string name;
    double avgMs;
    double medianMs;
    double minMs;
    double maxMs;
    double stdDevMs;
    double throughputMElemPerSec;
};

template <typename Func>
CpuBenchmarkResult benchmarkFunction(const std::string& name, Func func) {
    // Warm-up run
    func();

    std::vector<double> samples(REPEATS, 0.0);
    for (int r = 0; r < REPEATS; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        samples[r] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double totalMs = std::accumulate(samples.begin(), samples.end(), 0.0);
    double avgMs = totalMs / REPEATS;

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double minMs = sorted.front();
    double maxMs = sorted.back();
    double medianMs = (REPEATS % 2 == 0)
        ? (sorted[REPEATS / 2 - 1] + sorted[REPEATS / 2]) * 0.5
        : sorted[REPEATS / 2];

    double varianceSum = 0.0;
    for (double s : samples) {
        varianceSum += (s - avgMs) * (s - avgMs);
    }
    double stdDevMs = (REPEATS > 1) ? std::sqrt(varianceSum / (REPEATS - 1)) : 0.0;
    double throughput = (static_cast<double>(N) / (avgMs / 1000.0)) / 1e6;

    std::cout << std::left << std::setw(32) << name
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << avgMs << " ms"
              << std::setw(12) << medianMs << " ms"
              << std::setw(10) << stdDevMs << " ms"
              << std::setw(12) << throughput << " M/s"
              << std::endl;

    return {name, avgMs, medianMs, minMs, maxMs, stdDevMs, throughput};
}

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "CPU Baseline Benchmark: Warp Divergence Equivalence" << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << "Array Elements:      " << N << " (" << (N * sizeof(float)) / (1024 * 1024) << " MB FP32)" << std::endl;
    std::cout << "Repeated Iterations: " << REPEATS << std::endl << std::endl;

    std::vector<float> input(N);
    std::vector<float> output(N);

    // Identical deterministic input initialization
    for (int i = 0; i < N; ++i) {
        unsigned int x = static_cast<unsigned int>(i);
        x = x * 1664525u + 1013904223u;
        input[i] = static_cast<float>(x & 0xFFFFu) / 65535.0f;
    }

    const float threshold = 0.5f;

    std::cout << std::left << std::setw(32) << "Algorithm Variant"
              << std::right << std::setw(13) << "Mean (ms)"
              << std::setw(15) << "Median (ms)"
              << std::setw(13) << "StdDev (ms)"
              << std::setw(17) << "Throughput (M/s)" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;

    auto rDiv = benchmarkFunction("CPU Alternating Branch", [&]() {
        cpuDivergent(input.data(), output.data(), N);
    });

    auto rAlign = benchmarkFunction("CPU Chunk-aligned Branch", [&]() {
        cpuAligned(input.data(), output.data(), N);
    });

    auto rData = benchmarkFunction("CPU Data-dependent Branch", [&]() {
        cpuDataDependent(input.data(), output.data(), N, threshold);
    });

    auto rBf = benchmarkFunction("CPU Branch-free (both paths)", [&]() {
        cpuBranchFree(input.data(), output.data(), N, threshold);
    });
    std::cout << "----------------------------------------------------------------------------------" << std::endl;

    std::cout << std::endl << "CPU Performance Ratios vs Baseline (Alternating):" << std::endl;
    std::cout << "  - Chunk-Aligned Speedup:     " << std::fixed << std::setprecision(3) << (rDiv.avgMs / rAlign.avgMs) << "x" << std::endl;
    std::cout << "  - Data-Dependent Speedup:    " << std::fixed << std::setprecision(3) << (rDiv.avgMs / rData.avgMs) << "x" << std::endl;
    std::cout << "  - Branch-Free Speedup:       " << std::fixed << std::setprecision(3) << (rDiv.avgMs / rBf.avgMs) << "x" << std::endl;

    return 0;
}
