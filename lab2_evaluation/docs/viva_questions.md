# Comprehensive Viva Voce Preparation Guide
## CUDA Warp Divergence Analysis | Parallel GPU Computing (Lab Evaluation 2)

This guide contains 28 rigorous, project-specific viva questions and concise, technically authoritative answers designed for faculty evaluations, technical vivas, and architectural interviews.

---

### Q1: What is a Warp in NVIDIA GPU architecture?
**Answer:**  
A warp is the fundamental atomic scheduling and execution unit in NVIDIA GPUs, consisting of **32 consecutive parallel threads**. All 32 threads in a warp are managed by a single warp scheduler and execute the same instruction simultaneously in lockstep.

---

### Q2: What is the SIMT execution model and how does it differ from SIMD?
**Answer:**  
**SIMT (Single Instruction, Multiple Threads)** allows multiple independent scalar threads to execute the same instruction concurrently on different data lanes. Unlike SIMD (Single Instruction, Multiple Data), where vector registers are exposed directly to the programmer and executed as single wide vectors, SIMT allows each thread to maintain its own instruction pointer, registers, and execution state, with hardware automatically handling masking and divergence.

---

### Q3: What is Warp Divergence?
**Answer:**  
Warp divergence occurs when threads within the same 32-thread warp evaluate a conditional branch (`if-else`, `switch`, or loop condition) differently, forcing some lanes to take one execution path while other lanes take another path.

---

### Q4: Why does Warp Divergence degrade GPU execution efficiency?
**Answer:**  
Because all 32 cores in a warp share a single instruction issue unit, the hardware cannot execute two distinct instruction streams simultaneously on the same warp. Instead, it must **serialize** the execution of each branch path sequentially, disabling non-participating threads via an active mask. This cuts the instantaneous computational efficiency by 50% or more.

---

### Q5: What is an Active Mask register and how does it work?
**Answer:**  
The Active Mask is a 32-bit hardware register within each warp execution unit where each bit corresponds to one thread lane (0 to 31). A bit value of `1` signifies that the corresponding lane actively executes the dispatched instruction, while a bit value of `0` masks off the lane, holding it idle and preventing register or memory updates.

---

### Q6: What is Branch Reconvergence?
**Answer:**  
Reconvergence is the point in the control-flow graph where all divergent branch paths rejoin. The GPU hardware detects this point, restores the Active Mask to full participation (`0xFFFFFFFF`), and resumes unified, lockstep execution across all 32 lanes.

---

### Q7: Explain the Divergent Kernel implementation tested in this project.
**Answer:**  
The divergent kernel uses the condition `if ((i % 2) == 0)`. Because block size is 256 (even), `(i % 2)` equals `(threadIdx.x % 2)`. Within every single warp, even lanes (0, 2, 4...) execute Path A with mask `0x55555555`, while odd lanes (1, 3, 5...) execute Path B with mask `0xAAAAAAAA`. The two paths are serialized, achieving 50% theoretical lane efficiency.

---

### Q8: Explain the Warp-Aligned Kernel implementation and why it avoids divergence.
**Answer:**  
The warp-aligned kernel uses the condition `if (((i / 32) % 2) == 0)`. Because `i / 32` represents the global warp index, all 32 threads in any given warp evaluate the exact same condition. An entire warp follows Path A together (mask `0xFFFFFFFF`), and the next warp follows Path B together (mask `0xFFFFFFFF`). Zero intra-warp divergence occurs.

---

### Q9: Why is the global index formula `i = blockIdx.x * blockDim.x + threadIdx.x` used?
**Answer:**  
- `blockIdx.x * blockDim.x` computes the base array offset of the current thread block within the global grid.
- `threadIdx.x` computes the local offset of the thread within that specific block.
- Together, they compute a unique, linear global thread index `i` that maps one-to-one to each element of the $16.78\text{M}$ array.

---

### Q10: What is Data-Dependent Branching?
**Answer:**  
Data-dependent branching occurs when the branch condition depends on dynamic runtime data in memory (`if (input[i] > threshold)`) rather than fixed thread coordinates. In real-world applications (e.g., ray tracing, fluid simulations), this causes stochastic, unpredictable divergence patterns across warps.

---

### Q11: What is Branch-Free Predication and how does it work in CUDA?
**Answer:**  
Branch-free code eliminates control-flow jumps by computing both candidate expressions unconditionally (`float a = workA(x); float b = workB(x);`) and selecting the desired result using a ternary conditional operator (`output[i] = (input[i] > th) ? a : b;`). The compiler translates this into a single hardware conditional selection instruction (`selp` in PTX).

---

### Q12: Why did the Branch-Free kernel run SLOWER than the Data-Dependent kernel on the RTX 4500 Ada?
**Answer:**  
This is caused by **Work Inflation**. While branch-free predication keeps all 32 lanes active (100% active mask), every thread is forced to calculate both `workA` (8 FMAs) and `workB` (8 FMAs) unconditionally—totaling **16 arithmetic operations per thread** instead of 8. On the RTX 4500 Ada, doubling the arithmetic work across all lanes created more pipeline pressure than simply allowing the warp to serialize short branch paths.

---

### Q13: Why did the Warp-Aligned kernel not achieve a 2x speedup over the Divergent kernel?
**Answer:**  
Because the benchmark is **memory bandwidth bound**. Streaming $16.78\text{M}$ FP32 elements in and out ($128\text{ MB}$ total) in $0.3738\text{ ms}$ consumes $\approx 342.4\text{ GB/s}$, which is **$79.3\%$ of the RTX 4500 Ada's theoretical peak memory bandwidth ($432\text{ GB/s}$)**. Furthermore, both kernels preserve perfect 128-byte memory coalescing. Because the hardware execution time was dominated by DRAM transfer latency rather than arithmetic execution, eliminating 8 arithmetic cycles yielded no measurable wall-clock speedup.

---

### Q14: What is Memory Coalescing and why is it critical in divergence studies?
**Answer:**  
Memory coalescing is the hardware mechanism where memory accesses by all 32 threads in a warp are combined into as few 32-, 64-, or 128-byte cache line transactions as possible. In our divergent kernel, consecutive thread lanes access consecutive memory addresses ($i$ to $i+31$), so memory coalescing is 100% preserved despite control-flow divergence.

---

### Q15: What is the formal definition of Execution-Time Ratio ($R$)?
**Answer:**  
The Execution-Time Ratio is defined as:
$$R = \frac{T_{\text{variant}}}{T_{\text{baseline}}}$$
- An execution-time ratio **$R < 1.0$** indicates that the variant executed in less time (faster).
- An execution-time ratio **$R > 1.0$** indicates that the variant took more time (slower).

---

### Q16: What is the formal definition of Speedup ($S$)?
**Answer:**  
Speedup is defined as the baseline latency divided by the variant latency:
$$S = \frac{T_{\text{baseline}}}{T_{\text{variant}}} = \frac{1}{R}$$
- A speedup **$S > 1.0$** indicates performance acceleration relative to baseline.
- A speedup **$S < 1.0$** indicates performance degradation (slowdown).

---

### Q17: Why is a Warm-Up kernel launch essential before taking benchmark timings?
**Answer:**  
The first CUDA kernel launch incurs significant initialization latency due to driver context setup, GPU clock ramping from low-power idle states to boost frequencies, and JIT compilation. An un-timed warm-up launch ensures that subsequent measurements record steady-state GPU execution.

---

### Q18: Why should CUDA Events be used for kernel timing instead of host CPU timers (`std::chrono`)?
**Answer:**  
CUDA kernel launches are asynchronous; the CPU returns immediately after queuing the launch command in the GPU stream. A CPU timer (`std::chrono`) would measure only launch queue overhead unless an explicit `cudaDeviceSynchronize()` is invoked. CUDA Events (`cudaEventRecord`) place hardware timestamps directly into the GPU execution stream, measuring kernel latency with sub-microsecond precision without CPU jitter.

---

### Q19: What is the difference between Kernel-Only timing and End-to-End timing?
**Answer:**  
- **Kernel-Only Timing:** Measures solely the execution duration of the GPU kernel cores on the device via CUDA Events.
- **End-to-End Timing:** Measures the total application workflow time, including Host-to-Device memory copy (`cudaMemcpyHostToDevice`), kernel launch and execution, and Device-to-Host memory copy (`cudaMemcpyDeviceToHost`).

---

### Q20: How many warps are executed in a block of 256 threads?
**Answer:**  
Since a warp contains exactly 32 threads:
$$\text{Warps per block} = \frac{256}{32} = 8 \text{ warps}$$

---

### Q21: How many total thread blocks and threads were launched in this experiment?
**Answer:**  
For $N = 16,777,216$ elements and 256 threads per block:
$$\text{Grid Dimension} = \frac{16,777,216}{256} = 65,536 \text{ blocks}$$
$$\text{Total Threads} = 65,536 \times 256 = 16,777,216 \text{ threads}$$
$$\text{Total Warps} = \frac{16,777,216}{32} = 524,288 \text{ warps}$$

---

### Q22: What is a Streaming Multiprocessor (SM)?
**Answer:**  
An SM is the fundamental hardware processing cluster on an NVIDIA GPU. It contains dedicated warp schedulers, instruction dispatch units, register files, L1 data cache/shared memory, and execution pipelines (FP32, FP64, INT32, Tensor Cores, and RT Cores). The RTX 4500 Ada features 60 SMs.

---

### Q23: What is Independent Thread Scheduling (ITS)?
**Answer:**  
Introduced in the NVIDIA Volta architecture (and present on Ada Lovelace SM 8.9), ITS equips each thread with its own independent Program Counter and call stack. Instead of requiring rigid warp reconvergence, the warp scheduler can interleave instructions from divergent branches dynamically, improving execution efficiency and eliminating lock-free synchronization deadlocks.

---

### Q24: What is the Multi-Path Divergence penalty if all 32 lanes take different branches?
**Answer:**  
If all 32 threads take distinct branch paths, the warp scheduler must execute 32 sequential instruction passes, disabling 31 threads during each pass. Active lane efficiency collapses to:
$$\text{Efficiency} = \frac{1}{32} \approx 3.125\%$$

---

### Q25: How does real-world industry software (e.g., ray tracing) mitigate warp divergence?
**Answer:**  
In real-time ray tracing (e.g., NVIDIA OptiX / DirectX Raytracing), secondary rays scatter randomly across 3D scenes. Production engines solve this via:
1. **Ray Sorting / Binning:** Sorting rays by spatial origin and direction before traversal so adjacent threads traverse similar BVH nodes.
2. **Dedicated Hardware (RT Cores):** Offloading BVH traversal and ray-triangle intersection testing to fixed-function silicon.

---

### Q26: How does data sorting mitigate data-dependent branch divergence?
**Answer:**  
By presorting or binning the input dataset prior to kernel execution (e.g., placing all values $> 0.5$ in contiguous blocks), consecutive threads are assigned similar data values. This converts what would have been random, intra-warp divergence into coarse-grained, warp-aligned branches where whole warps evaluate uniformly.

---

### Q27: How did we verify numerical correctness in our implementation?
**Answer:**  
For each of the four kernels, output buffers were transferred back to the host via `cudaMemcpyDeviceToHost` and compared against a single-threaded CPU reference implementation evaluating the exact same mathematical functions (`hostWorkA` and `hostWorkB`). Maximum absolute error was computed across all $16.78\text{M}$ elements, achieving a verified `[PASS]` status with $0.00$ error.

---

### Q28: What is the primary takeaway regarding branch-free optimization on modern GPUs?
**Answer:**  
Branch-free predication is **not a universal silver bullet**. It is advantageous only when the branch bodies are very short (1-2 cheap instructions) or when divergence causes severe memory divergence. When branch bodies contain non-trivial arithmetic, calculating both candidate paths unconditionally doubles the compute workload (work inflation), making the branch-free code slower than the divergent code.
