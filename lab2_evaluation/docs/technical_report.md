# Microarchitectural Analysis of CUDA Warp Divergence on Modern NVIDIA GPU Architectures
### Technical Research & Laboratory Evaluation Report
**Course:** Parallel GPU Computing (PGC) · **Evaluation:** Mini-Project Lab Evaluation 2 · **Theme 7:** Warp Divergence  
**Authors / Research Team:**  
- Adarsh Walad (`01fe24bci`) — Experimental Design & Slide Deck
- Nilaksha Nath (`01fe24bci087`) — Benchmarking Instrumentation, Data Collection & Statistics
- Bachu venkata siva jaideep (`01fe24bci084`) — GPU Architecture Profiling, Toolchain Configuration & Repository Architect
- Prathamesh Eedigi (`01fe24bci006`) — SIMT Active Mask Diagrams, Visualization & Mathematical Analysis  

---

## 1. Abstract
In modern massively parallel GPU architectures, high computational throughput relies on the Single Instruction, Multiple Threads (SIMT) execution paradigm, wherein groups of 32 threads (termed *warps*) share a common instruction issue unit. When conditional branching causes threads within the same warp to follow divergent control-flow paths, the hardware must serialize the execution of each distinct branch path while disabling non-participating threads via active masking registers. 

This technical report presents an exhaustive empirical and theoretical evaluation of warp divergence on an enterprise-grade **NVIDIA RTX 4500 Ada Generation** GPU (Compute Capability SM 8.9). We evaluate four fundamental branching strategies across $N = 16,777,216$ ($2^{24}$) single-precision floating-point (FP32) elements:
1. **Divergent Branching** (alternating intra-warp lanes: `i % 2 == 0`),
2. **Warp-Aligned Branching** (uniform intra-warp lanes: `(i / 32) % 2 == 0`),
3. **Data-Dependent Branching** (stochastic runtime data condition: `input[i] > 0.5f`), and
4. **Branch-Free Predication** (unconditional candidate computation combined with hardware conditional selection).

Contrary to naive theoretical expectations where warp-alignment trivially yields a $2\times$ speedup, empirical measurements reveal that under memory-bound streaming conditions ($\approx 79.3\%$ of peak GDDR6 bandwidth saturated at $\sim 342.4\text{ GB/s}$), memory access coalescing dominates instruction latency. Furthermore, branch-free predication incurs severe *work inflation* by doubling the arithmetic payload across all 32 lanes, resulting in a measurable performance degradation (0.3952 ms vs. 0.3648 ms for data-dependent branching). We provide formal mathematical formulations, architectural diagrams, statistical timing methodologies, and correctness verification protocols to guide GPU developers in optimizing branching workloads.

---

## 2. Introduction
Modern Graphics Processing Units (GPUs) achieve immense raw computing throughput by dedicating silicon area to massive arrays of arithmetic logic units (ALUs) rather than complex branch predictors and out-of-order execution logic. To amortize instruction fetch, decode, and dispatch overheads, NVIDIA GPUs group 32 consecutive scalar threads into an atomic scheduling entity called a **warp**. 

Under the SIMT execution model, all 32 lanes of a warp execute the exact same instruction at any given cycle. However, modern scientific simulations, computer vision algorithms, and ray tracing pipelines frequently feature conditional branches (`if-else`, loops, and early-exits) that depend on dynamic input data or thread indices. When branch divergence occurs within a warp, the hardware serializes the taken paths, cutting instantaneous execution efficiency. Understanding the exact trade-offs between branch serialization, warp alignment, and branch-free predication on modern architectures is critical for writing high-performance CUDA applications.

---

## 3. Problem Statement
Given an array of $N = 16,777,216$ elements, determine the empirical and microarchitectural performance impact of intra-warp branch divergence on the NVIDIA Ada Lovelace architecture. Specifically, we investigate:
1. Does warp-aligned branching provide a guaranteed $2\times$ speedup over alternating divergent branching on memory-bound workloads?
2. Does branch-free predication (`selp`) always outperform divergent branching, or does the redundant computation (work inflation) offset the benefits of eliminating control-flow divergency?
3. How do modern hardware features, such as Independent Thread Scheduling (ITS) and unified L1/shared memory caches, influence warp divergence penalties?

---

## 4. Objectives
The core objectives of this laboratory investigation are:
1. **Implementation:** Implement four distinct CUDA kernel strategies operating on an identical FP32 dataset ($64\text{ MB}$ input, $64\text{ MB}$ output).
2. **Correctness Verification:** Establish an automated host-side numerical validation suite comparing GPU outputs against a golden CPU reference model to ensure mathematical equivalence.
3. **Benchmarking Rigor:** Profile kernel latencies using hardware CUDA Events across 20 repetitions, capturing statistical mean, median, min, max, and standard deviation while avoiding host launch overhead.
4. **Mathematical Precision:** Strictly formulate and calculate Baseline Latency ($T_{\text{base}}$), Execution-Time Ratio ($R$), Speedup ($S$), and Computational Throughput ($TP$) without metric conflation.
5. **Microarchitectural Analysis:** Correlate measured execution latencies with memory bandwidth saturation and arithmetic work inflation.

---

## 5. GPU and SIMT Architecture
The NVIDIA GPU hardware hierarchy is structured as follows:

```
+--------------------------------------------------------------------------------------------------+
|                                    NVIDIA GPU DEVICE (SM 8.9)                                    |
|                                                                                                  |
|  Global Memory: 24 GB GDDR6 · 192-bit Memory Bus · 432 GB/s Theoretical Peak Bandwidth           |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                     High-Speed Interconnect
                                                 |
+--------------------------------------------------------------------------------------------------+
|  Streaming Multiprocessors (SMs): 60 SMs on RTX 4500 Ada (7,680 FP32 CUDA Cores Total)           |
|                                                                                                  |
|  +--------------------------------------------------------------------------------------------+  |
|  | Sub-Core Partitions (4 Warp Schedulers per SM, 1 Dispatch Unit per Warp Scheduler)        |  |
|  | L1 Data Cache / Shared Memory: 128 KB Configurable per SM                                  |  |
|  | Register File: 64K x 32-bit Registers per SM                                               |  |
|  | 32 Physical Execution Cores (Lanes 0 to 31) executing 1 Warp per Cycle                     |  |
|  +--------------------------------------------------------------------------------------------+  |
+--------------------------------------------------------------------------------------------------+
```

### The SIMT Execution Model
- **Warp Organization:** 32 consecutive threads constitute one warp.
- **Warp Schedulers:** Every clock cycle, the warp scheduler selects an active, instruction-ready warp and dispatches a common instruction to 32 execution units.
- **Active Mask Register:** A dedicated 32-bit register where bit $k$ controls lane $k$. A bit value of `1` indicates that lane $k$ executes the instruction; a bit value of `0` masks the lane, forcing it to remain idle.
- **Independent Thread Scheduling (ITS):** Introduced in Volta and retained in Ada Lovelace (SM 8.9), each thread maintains its own Program Counter (PC) and call stack. This permits fine-grained interleaving of divergent execution paths, resolving deadlocks and enhancing pipeline utilization.

---

## 6. Warp Divergence Theory

### The Mechanism of Serialization
Consider an `if-else` branch statement evaluated by a warp of 32 threads:

```cuda
if (condition) {
    path_A(); // 8 FLOPs
} else {
    path_B(); // 8 FLOPs
}
```

If some lanes evaluate `condition` as `true` and others as `false`, the warp diverges. The execution sequence unfolds as follows:
1. **Path A Evaluation:** The active mask is set to enable only threads where `condition == true`. The execution units compute `path_A()`. Non-participating threads are disabled.
2. **Path B Evaluation:** The active mask is inverted to enable threads where `condition == false`. The execution units compute `path_B()`.
3. **Reconvergence:** Once both paths terminate, the active mask is restored to full participation (`0xFFFFFFFF`), and unified execution resumes.

### Theoretical Multi-Path Serialization Penalty
If a warp diverges across $K$ mutually exclusive paths with execution latencies $t_1, t_2, \dots, t_K$, the total time required to execute the branch construct is:
$$T_{\text{total}} = \sum_{k=1}^K t_k$$

In the worst-case scenario where all 32 lanes take distinct branches ($K = 32$), the active hardware efficiency drops to:
$$\text{Efficiency}_{\text{min}} = \frac{1}{32} \approx 3.125\%$$

---

## 7. Four Evaluated Kernel Strategies

To evaluate branching under identical conditions, four CUDA kernels were developed, each executing on $N = 16,777,216$ elements with 256 threads per block.

### Kernel 1: Divergent Branching (Alternating Lanes)
```cuda
__global__ void divergentKernel(const float* input, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((i % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}
```
- **Condition:** `(i % 2) == 0`.
- **Warp Behavior:** Because `blockDim.x = 256` is an exact multiple of 2, `(i % 2) == (threadIdx.x % 2)`. Lanes 0, 2, 4... 30 take Path A (`workA`), while lanes 1, 3, 5... 31 take Path B (`workB`).
- **Active Mask:** Phase 1 uses `0x55555555` (16 active lanes); Phase 2 uses `0xAAAAAAAA` (16 active lanes).
- **Theoretical Active Lane Efficiency:** $50\%$.

### Kernel 2: Warp-Aligned Branching (Chunk-Aligned)
```cuda
__global__ void alignedKernel(const float* input, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (((i / WARP_SIZE) % 2) == 0)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}
```
- **Condition:** `((i / 32) % 2) == 0`.
- **Warp Behavior:** All 32 threads in a given warp share the exact same global warp index (`i / 32`). Even warps execute Path A in unison; odd warps execute Path B in unison.
- **Active Mask:** Uniform `0xFFFFFFFF` throughout execution.
- **Theoretical Active Lane Efficiency:** $100\%$.

### Kernel 3: Data-Dependent Branching (Runtime Stochastic)
```cuda
__global__ void dataDependentKernel(const float* input, float* output, int n, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (input[i] > threshold)
            output[i] = workA(input[i]);
        else
            output[i] = workB(input[i]);
    }
}
```
- **Condition:** `input[i] > threshold` (with `threshold = 0.5f`).
- **Warp Behavior:** Branches depend on pseudo-random values generated via a Linear Congruential Generator. Lanes within each warp split dynamically based on memory values.
- **Active Mask:** Non-uniform, stochastic mask per warp.
- **Theoretical Active Lane Efficiency:** $\approx 50\%$ on average across the grid.

### Kernel 4: Branch-Free Selection (Predicated Execution)
```cuda
__global__ void branchFreeKernel(const float* input, float* output, int n, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = workA(input[i]);
        float b = workB(input[i]);
        output[i] = (input[i] > threshold) ? a : b;
    }
}
```
- **Mechanism:** Both candidate functions (`workA` and `workB`) are executed unconditionally by every thread. The ternary operator compiles to a PTX `selp` instruction.
- **Warp Behavior:** Zero control-flow divergence. All 32 lanes execute identical instructions every cycle.
- **Active Mask:** Uniform `0xFFFFFFFF` (100% active lanes).
- **Incurred Trade-off (Work Inflation):** Each thread executes 16 arithmetic operations (8 for `workA` + 8 for `workB`) instead of 8, doubling the arithmetic workload.

---

## 8. Verified System & Hardware Configuration

The empirical evaluation was conducted on an enterprise workstation whose toolchain and hardware properties were verified from execution logs and diagnostic screenshots:

| System Parameter | Verified Specification |
| :--- | :--- |
| **GPU Model** | NVIDIA RTX 4500 Ada Generation |
| **Compute Capability** | SM 8.9 (Ada Lovelace Architecture) |
| **Streaming Multiprocessors (SMs)** | 60 SMs |
| **Total FP32 CUDA Cores** | 7,680 Cores |
| **Device Global Memory** | 24,576 MB (24 GB GDDR6) |
| **Memory Bus Width & Peak Bandwidth** | 192-bit bus · $\approx 432\text{ GB/s}$ |
| **L2 Cache Capacity** | 48 MB High-Speed L2 Cache |
| **CUDA Toolkit Version** | CUDA Release 13.4 (V13.4.92) |
| **Host Compiler** | Microsoft Visual C/C++ Optimizing Compiler (MSVC) 19.51.36260 |
| **Host Operating System** | Microsoft Windows 11 Enterprise (x64) |
| **Host Development Environment** | Visual Studio Developer Command Prompt v18.10.2 |

---

## 9. Experimental Methodology & Benchmarking Rigor

To eliminate measurement artifacts, the benchmarking harness implemented the following rigorous protocols:
1. **Isolated Warm-up:** An untimed warm-up launch was performed prior to measurement to ensure GPU clocks were brought to active P-states, instruction caches were primed, and the runtime driver was fully initialized.
2. **GPU Hardware Events:** Timings were captured exclusively on the GPU device using `cudaEventRecord` and `cudaEventElapsedTime`, completely eliminating host CPU launch overhead and OS scheduling jitter.
3. **Hardware Synchronization:** Explicit barriers (`cudaEventSynchronize` and `cudaDeviceSynchronize`) were enforced before querying timestamps.
4. **Repeated Sampling:** 20 iterations were captured per kernel to compute mean execution time, median, standard deviation, minimum, and maximum.
5. **Memory Coalescing Assurance:** All memory accesses (`input[i]` and `output[i]`) were maintained in strict consecutive order across thread lanes, ensuring 128-byte transaction coalescing.
6. **Compiler Optimization Safeguards:** Math loops within `workA` and `workB` were marked with `#pragma unroll 1` to prevent the compiler from unrolling loops or optimizing away identical math terms.

---

## 10. Benchmark Parameters

| Parameter Name | Configuration Value | Architectural Justification |
| :--- | :---: | :--- |
| **Array Element Count ($N$)** | $16,777,216$ ($2^{24}$) | Sufficiently large ($64\text{ MB}$ input / $64\text{ MB}$ output) to exceed L2 cache ($48\text{ MB}$) and test DRAM streaming |
| **Threads per Block** | 256 | Yields 8 warps per block, ensuring high SM occupancy |
| **Grid Dimension (Blocks)** | 65,536 | Fully saturates all 60 SMs ($\approx 1,092$ blocks per SM) |
| **Repeated Iterations** | 20 | Provides statistically reliable latency averages |
| **Branch Threshold** | 0.50f | Divides pseudo-random distribution into balanced 50/50 branches |

---

## 11. Correctness Verification Protocol

To verify that all four kernels produce mathematically accurate results, an automated verification harness was executed on the host:
1. Device memory `d_output` was transferred to host memory `h_output` via `cudaMemcpyDeviceToHost`.
2. A single-threaded host verification loop evaluated the ground-truth calculation for each element $i$.
3. Absolute error was evaluated for every element:
   $$\epsilon_i = |h\_output[i] - \text{expected}_i|$$
4. Output status was reported as `[PASS]` if $\max(\epsilon_i) \le 10^{-4}$, otherwise `[FAIL]`.

**Verification Outcome:**
All four kernels achieved `[PASS]` status with a maximum absolute error of $\epsilon_{\max} = 0.00 \times 10^0$ (exact single-precision floating-point agreement).

---

## 12. Measured Results & Performance Calculations

### Formal Mathematical Definitions
- **Baseline Execution Time ($T_{\text{base}}$):** Time of the divergent baseline kernel ($T_{\text{divergent}} = 0.3738\text{ ms}$).
- **Execution-Time Ratio ($R$):**
  $$R = \frac{T_{\text{variant}}}{T_{\text{base}}}$$
  *(Values $R < 1.0$ indicate that the variant executed faster than baseline).*
- **Speedup ($S$):**
  $$S = \frac{T_{\text{base}}}{T_{\text{variant}}} = \frac{1}{R}$$
  *(Values $S > 1.0$ indicate performance acceleration relative to baseline).*
- **Percentage Performance Change ($\Delta P$):**
  $$\Delta P = \left( \frac{T_{\text{base}} - T_{\text{variant}}}{T_{\text{base}}} \right) \times 100\%$$
- **Effective Throughput ($TP$):**
  $$TP = \frac{N}{T_{\text{variant}} \times 10^{-3} \times 10^6} \quad \text{(Million Elements / second)}$$

### Empirical Performance Table

| Kernel Strategy | Execution Time (ms) | Throughput ($10^6$ elem/s) | Execution-Time Ratio $R$ | Speedup $S$ vs Baseline | Performance Change $\Delta P$ (%) | Active Lane Efficiency | Correctness Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Divergent Branch** *(Baseline)* | **0.3738 ms** | **44,882.87** | **1.0000** | **1.0000x** | **0.00%** | ~50.0% | **PASS** |
| **Warp-Aligned Branch** | **0.3955 ms** | **42,420.27** | **1.0581** | **0.9451x** | **-5.81%** | 100.0% | **PASS** |
| **Data-Dependent Branch** | **0.3648 ms** | **45,990.18** | **0.9759** | **1.0247x** | **+2.41%** | ~50.0% | **PASS** |
| **Branch-Free Selection** | **0.3952 ms** | **42,452.47** | **1.0572** | **0.9458x** | **-5.72%** | 100.0% (predicated) | **PASS** |

---

## 13. Performance Analysis & Architectural Discussion

### 13.1 Why Did Warp-Aligned Branching Not Deliver a 2x Speedup?
In classroom theory, an aligned warp with 100% active lanes is expected to finish in half the time of a divergent warp with 50% active lanes. However, in our benchmark, the warp-aligned kernel took $0.3955\text{ ms}$, slightly slower than the divergent kernel ($0.3738\text{ ms}$, speedup $0.945\times$). 

This counter-intuitive result is explained by three microarchitectural mechanisms:
1. **Memory Bandwidth Dominance (Algorithmic Traffic Estimate):**  
   *Note: This is an estimated algorithmic traffic calculation based on data size, not an empirical hardware performance counter measurement (such as Nsight Compute `dram__bytes_read.sum`).*  
   Processing $16.78\text{M}$ FP32 elements entails reading $64\text{ MiB} \approx 67.11\text{ MB}$ of input and writing $64\text{ MiB} \approx 67.11\text{ MB}$ of output ($128\text{ MiB} \approx 134.22\text{ MB}$ total minimum global memory traffic). At $0.3738\text{ ms}$, the estimated effective memory bandwidth is:
   $$\text{BW}_{\text{eff}} = \frac{134.22 \times 10^6\text{ bytes}}{0.3738 \times 10^{-3}\text{ s}} \approx 359.1\text{ GB/s} \quad \left(\text{or } \frac{128 \times 10^6}{0.3738 \times 10^{-3}} \approx 342.4\text{ GB/s}\right)$$
   Comparing this against the RTX 4500 Ada's theoretical peak memory bandwidth ($432\text{ GB/s}$):
   $$\text{Estimated Bandwidth Saturation} \approx 79.3\% \text{ to } 83.1\%$$
   The kernel operates deep within the **memory bandwidth bound** regime. The instruction pipeline frequently stalls waiting for DRAM requests; hence, eliminating 8 arithmetic cycles does not reduce the wall-clock execution time.
2. **Coalesced Memory Access:** Crucially, both the divergent and aligned kernels maintain perfect 128-byte memory coalescing because lane $k$ reads address $i = \text{base} + k$. Branch divergence did not introduce memory divergence.
3. **Cache Line Spatial Locality:** In the divergent kernel, alternating lanes execute `workA` and `workB`. Because both branches are executed within the same warp during consecutive instruction cycles, cache lines remain warm in L1/L2 caches.

### 13.2 The Work Inflation Penalty of Branch-Free Code
The branch-free kernel achieved a latency of $0.3952\text{ ms}$ (throughput $42,452\text{ M elem/s}$), running slower than the data-dependent divergent kernel ($0.3648\text{ ms}$).
- **Theoretical Instruction Work Inflation:**  
  *Note: 8 vs 16 operations per thread is a theoretical source-level arithmetic count based on loop iterations in `workA` and `workB`, not a disassembled SASS hardware instruction counter measurement.*  
  In the branch-free version, every thread computes both `workA` (8 FMAs) and `workB` (8 FMAs), totaling 16 arithmetic operations per thread, followed by a conditional move (`selp`).
- In contrast, the branching kernels only execute 8 arithmetic operations per active thread lane.
- On modern Ada Lovelace GPUs with deep execution pipelines and Independent Thread Scheduling, computing double the arithmetic workload creates more pipeline execution pressure than allowing hardware predication or warp branch serialization.
- **Engineering Principle:** Branch-free selection is only beneficial when the cost of branch serialization exceeds the cost of redundant computation. For branches with significant computation in both branches, computing both candidate results creates "work inflation".

---

## 14. Real-World Applications & Industry Case Studies

Warp divergence is a primary performance bottleneck in several major industry domains:

### 1. Real-Time Ray Tracing (NVIDIA OptiX & DirectX Raytracing)
When primary rays reflect off glossy or diffuse surfaces, secondary rays scatter in disparate directions. Adjacent threads within a warp traverse completely different bounding volume hierarchies (BVH) in the 3D scene, leading to severe branch divergence.
- **Industry Solution:** NVIDIA RT Cores implement hardware-accelerated BVH traversal. Additionally, production ray tracers employ **ray sorting / binning**, grouping rays by spatial direction before launching traversal kernels to maximize warp coherence.

### 2. Particle & Computational Fluid Dynamics (CFD)
In spatial particle simulations, particles colliding with complex solid boundaries require elaborate friction, collision, and deformation calculations, whereas open-space particles require trivial ballistic updates.
- **Industry Solution:** Engineers employ **spatial grid binning**, sorting particle buffers so that particles within the same spatial cell are processed by consecutive warp lanes.

### 3. Sparse Matrix Computations (SpMV)
In sparse linear algebra, rows containing varying numbers of non-zero entries cause threads in a warp to terminate at different iterations of inner accumulation loops.
- **Industry Solution:** Hybrid matrix representations (e.g., ELLPACK / CSR-Adaptive) allocate balanced work blocks across warps.

---

## 15. Limitations of This Study
1. **Arithmetic-to-Memory Ratio:** The benchmark kernel executes 8 arithmetic operations per 8 bytes of memory transfer ($1\text{ FLOP/byte}$), placing it firmly in the memory-bound regime. Higher arithmetic intensity workloads (e.g., 200+ FLOPs per element) would exhibit more pronounced divergence penalties.
2. **Symmetric Branch Payloads:** Both `workA` and `workB` contain identical operation counts (8 multiply-adds). Highly asymmetric branches (e.g., 1000 instructions in Path A vs. 2 instructions in Path B) would introduce different divergence dynamics.
3. **Single Microarchitecture:** Empirical measurements were conducted exclusively on NVIDIA Ada Lovelace (SM 8.9). Architectures lacking Independent Thread Scheduling (pre-Volta) or with different warp sizes (e.g., AMD Wave64/Wave32) may exhibit different behaviors.

---

## 16. Conclusion
This study provides an empirical and theoretical examination of CUDA warp divergence on the NVIDIA RTX 4500 Ada Generation GPU:
1. **Memory Bound Masking:** On streaming workloads where global memory bandwidth saturation reaches $\approx 80\%$, intra-warp branch divergence does not significantly degrade performance as long as memory coalescing is strictly maintained.
2. **The Branch-Free Trap:** Eliminating branch divergence via branch-free predication is not a universal optimization. When branch bodies contain substantial arithmetic, unconditional computation creates work inflation that slows down kernel execution.
3. **Warp Alignment Principle:** Structuring data and algorithms so that warps follow uniform execution paths remains an essential best practice, but must be paired with arithmetic intensity optimization and memory coalescing to achieve optimal GPU throughput.

---

## 17. References
1. NVIDIA Corporation. *CUDA C++ Programming Guide (Release 13.4)*. NVIDIA Developer Documentation, 2026.
2. NVIDIA Corporation. *NVIDIA Ada Lovelace Architecture In-Depth*. NVIDIA Technical Whitepaper, 2023.
3. Kirk, D. B., & Hwu, W. W. *Programming Massively Parallel Processors: A Hands-on Approach*. 4th Edition, Morgan Kaufmann, 2022.
4. Sanders, J., & Kandrot, E. *CUDA by Example: An Introduction to General-Purpose GPU Programming*. Addison-Wesley, 2010.
5. Volkov, V. *Understanding Latency Hiding on GPUs*. PhD Dissertation, University of California, Berkeley, 2016.
