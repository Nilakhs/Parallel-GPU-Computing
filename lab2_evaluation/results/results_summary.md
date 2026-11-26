# Experimental Results Summary: CUDA Warp Divergence Benchmarking

## 1. Experimental Hardware & Execution Environment
- **Device Under Test (DUT)**: NVIDIA RTX 4500 Ada Generation
- **Microarchitecture**: Ada Lovelace (Compute Capability SM 8.9)
- **Streaming Multiprocessors (SMs)**: 60 SMs (7,680 FP32 CUDA Cores)
- **Device Memory**: 24 GB GDDR6 (192-bit bus, ~432 GB/s peak bandwidth)
- **Host Compiler**: Microsoft Visual C/C++ Optimizing Compiler (MSVC) v19.51.36260 for x64
- **CUDA Toolkit**: NVIDIA CUDA Compiler (NVCC) Release 13.4 (V13.4.92)
- **Host Operating System**: Microsoft Windows 11 Enterprise / x64 Developer Environment
- **Benchmark Problem Size**: $N = 2^{24} = 16,777,216$ Single-Precision Floating-Point (FP32) Elements
- **Thread Block Configuration**: 256 threads per block ($\frac{256}{32} = 8$ warps per block)
- **Grid Dimension**: 65,536 thread blocks ($\lceil 16,777,216 / 256 \rceil$)
- **Benchmark Repetitions**: 20 iterations timed via GPU hardware CUDA Events (`cudaEventElapsedTime`)

---

## 2. Mathematical Metric Definitions

To prevent confusion between execution latency ratios and throughput speedups, the following formal definitions are strictly applied:

1. **Baseline Execution Time ($T_{\text{baseline}}$)**:
   Execution time of the divergent kernel:
   $$T_{\text{baseline}} = T_{\text{divergent}} = 0.3738\text{ ms}$$

2. **Execution-Time Ratio ($R$)**:
   Ratio of variant execution time to baseline execution time:
   $$R = \frac{T_{\text{variant}}}{T_{\text{baseline}}}$$
   - $R < 1.0$: Variant executes in less time than baseline (**faster**).
   - $R > 1.0$: Variant executes in more time than baseline (**slower**).

3. **Speedup ($S$)**:
   Theoretical performance speedup relative to baseline:
   $$S = \frac{T_{\text{baseline}}}{T_{\text{variant}}} = \frac{1}{R}$$
   - $S > 1.0$: Variant achieves higher speed than baseline (**speedup**).
   - $S < 1.0$: Variant achieves lower speed than baseline (**slowdown**).

4. **Percentage Performance Change ($\Delta P$)**:
   $$\Delta P = \left( \frac{T_{\text{baseline}} - T_{\text{variant}}}{T_{\text{baseline}}} \right) \times 100\%$$
   - $\Delta P > 0\%$: Performance improvement relative to divergent baseline.
   - $\Delta P < 0\%$: Performance degradation relative to divergent baseline.

5. **Effective Computational Throughput ($TP$)**:
   $$TP = \frac{N}{T_{\text{variant}} \times 10^{-3}} \times 10^{-6}\quad \text{(Million Elements / second)}$$

---

## 3. Measured Performance Data Table

| Kernel Strategy | Branching Condition | Execution Time $T$ (ms) | Throughput ($10^6$ elem/s) | Execution-Time Ratio $R$ ($T/T_{\text{base}}$) | Speedup $S$ ($T_{\text{base}}/T$) | Performance Change $\Delta P$ (%) | Estimated Active Lane Efficiency | Active Mask Signature |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Divergent Branch** *(Baseline)* | `(i % 2) == 0` | **0.3738 ms** | 44,882.87 | **1.0000** | **1.0000x** | **0.00%** | ~50.0% | Alternating `0x55555555` / `0xAAAAAAAA` |
| **Warp-Aligned Branch** | `((i / 32) % 2) == 0` | **0.3955 ms** | 42,420.27 | **1.0581** | **0.9451x** | **-5.81%** | 100.0% | Uniform Warp `0xFFFFFFFF` |
| **Data-Dependent Branch** | `input[i] > 0.5f` | **0.3648 ms** | 45,990.18 | **0.9759** | **1.0247x** | **+2.41%** | ~50.0% (stochastic) | Dynamic non-uniform bitmask |
| **Branch-Free Selection** | `(input[i] > 0.5f) ? a : b` | **0.3952 ms** | 42,452.47 | **1.0572** | **0.9458x** | **-5.72%** | 100.0% (predicated) | Unconditional execution + `selp` |

---

## 4. Architectural Analysis & Key Findings

### Finding 1: Why Was the Warp-Aligned Kernel Not Faster Than Divergent?
In synthetic microbenchmarks, one might expect a warp-aligned kernel to run up to $2\times$ faster than a divergent kernel due to $100\%$ active lane utilization versus $50\%$ lane serialization. However, on the NVIDIA RTX 4500 Ada Generation:
- **Memory Coalescing Dominance**: Both the divergent and warp-aligned kernels access consecutive 32-bit floats in memory ($i$ to $i+31$). Global memory transactions are $100\%$ coalesced into 128-byte sectors.
- **Estimated Bandwidth Saturation (Theoretical Algorithmic Estimate)**:
  > *Note: This is an estimated algorithmic traffic calculation based on data size, not an empirical hardware performance counter measurement (such as Nsight Compute `dram__bytes_read.sum`).*  
  Transferring $16.78\text{M}$ inputs ($64\text{ MiB} \approx 67.11\text{ MB}$) and $16.78\text{M}$ outputs ($64\text{ MiB} \approx 67.11\text{ MB}$) yields a minimum algorithmic traffic of $128\text{ MiB} \approx 134.22\text{ MB}$. Across $0.3738\text{ ms}$, this yields an estimated effective memory bandwidth of $\sim 342.4\text{ to } 359.1\text{ GB/s}$, which is **$\approx 79.3\%\text{ to } 83.1\%$ of the GPU's theoretical maximum bandwidth ($432\text{ GB/s}$)**.
- **Arithmetic Weight vs. Memory Stalls**: Each branch executes only 8 loop iterations of multiply-add operations. Because memory latency and bus saturation dominate, the instruction pipeline stalls waiting on DRAM rather than branch resolution.

### Finding 2: Why Did Branch-Free Code Exhibit a Slowdown?
The branch-free implementation eliminates control-flow divergence by computing both candidate branches unconditionally (`workA` and `workB`), followed by a conditional selection instruction (`selp`).
- **Theoretical Instruction Work Inflation**:
  > *Note: 8 vs 16 operations is a theoretical source-level arithmetic count based on loop iterations in `workA` and `workB`, not a disassembled SASS hardware instruction counter measurement.*  
  Instead of computing 8 arithmetic operations per thread, every thread computes 16 arithmetic operations ($2\times$ theoretical source compute workload).
- **Latency vs. Instruction Cost**: On modern Ada Lovelace GPUs with deep execution pipelines and Independent Thread Scheduling, computing double the arithmetic workload creates more pipeline execution pressure than allowing hardware predication or warp branch serialization.
- **Rule of Thumb**: Branch-free selection is only profitable when the branch bodies are extremely small (1-2 cheap instructions) or when divergence causes severe memory divergence or divergent barrier stalls.

---

## 5. Summary Ratios
- **Divergent / Aligned Time Ratio ($T_{\text{div}} / T_{\text{align}}$)**: `0.9451x` ($0.3738 / 0.3955$)
- **Data-Dependent / Branch-Free Time Ratio ($T_{\text{data}} / T_{\text{bf}}$)**: `0.9231x` ($0.3648 / 0.3952$)
- **Aligned Execution-Time Ratio ($T_{\text{align}} / T_{\text{div}}$)**: `1.0581`
- **Branch-Free Execution-Time Ratio ($T_{\text{bf}} / T_{\text{data}}$)**: `1.0833`
