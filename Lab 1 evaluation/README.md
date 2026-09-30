# GPU Vector Operations using CUDA

Welcome to the Parallel Computing Lab Evaluation project for **Theme 7**. This project explores parallel computing techniques by accelerating standard vector operations on modern hardware.

- **Parallel Model:** CUDA
- **Main Task:** Vector Addition and Vector Multiplication
- **Comparison:** CPU vs GPU
- **Performance metrics:** Execution Time, Speedup and Efficiency

*Note: This repository structure is currently being prepared. Actual performance results and benchmarks will be added after the experimental data is collected.*

---

## Team

| Role | Team Member |
|------|-------------|
| Member 1 | TBD |
| Member 2 | TBD |
| Member 3 | TBD |
| Member 4 | TBD |

---

## 1. Problem Statement

The goal of this project is to perform vector addition and vector multiplication using CUDA threads and compare the execution performance between the CPU and the GPU. Large numerical vectors will be processed element-by-element, providing a clear illustration of how workloads can be distributed across many parallel processing units.

---

## 2. Objectives

- Implement sequential CPU vector operations.
- Implement CUDA vector operations.
- Verify CPU and GPU produce identical results.
- Run experiments using multiple vector sizes.
- Record execution time.
- Calculate speedup.
- Analyze efficiency.
- Visualize the results using graphs.
- Understand the effect of data size on CPU/GPU performance.

---

## 3. Background

### CPU execution
One sequential execution flow processes vector elements one after the other. It leverages the CPU's high clock speed but is limited by the number of CPU cores when handling massively parallel tasks.

### CUDA execution
CUDA launches many GPU threads so different vector elements can be processed concurrently. The GPU's massive parallelism allows thousands of threads to execute simultaneously.

**Key Concepts:**
- **Host:** The CPU and its system memory.
- **Device:** The GPU and its dedicated memory.
- **Kernel:** A function written in C/C++ that is executed on the device by an array of threads.
- **Thread:** A single, lightweight execution unit on the GPU.
- **Block:** A grouped collection of threads that execute concurrently and can cooperate via shared memory.
- **Grid:** A collection of thread blocks that execute the same kernel.
- **Device memory:** The memory physically located on the GPU (e.g., global memory).
- **Host-to-device transfer:** The process of copying input data from the CPU memory to the GPU memory prior to execution.
- **Device-to-host transfer:** The process of copying the computed output data from the GPU memory back to the CPU memory.

---

## 4. System Architecture

```text
CPU / Host
   |
   | Allocate and initialize vectors
   v
Input Vectors A, B
   |
   +-----------------------------+
   |                             |
   v                             v
CPU Sequential               CUDA GPU
Execution                    Kernel
   |                             |
   |                             |
   v                             v
CPU Result C               GPU Result C
   |                             |
   +-------------+---------------+
                 |
                 v
        Result Verification
                 |
                 v
       Performance Analysis
                 |
       +---------+---------+
       |         |         |
       v         v         v
    Time      Speedup   Efficiency
                 |
                 v
              Graphs
```

---

## 5. Parallel Design

The design maps the data perfectly to the CUDA architecture using a **one-thread-per-element** strategy. 

For **vector addition**, each thread is responsible for computing one element of the output vector:
`C[i] = A[i] + B[i]`

For **vector multiplication**, each thread computes:
`C[i] = A[i] * B[i]`

The specific array index `i` mapped to a thread is computed using the **CUDA index concept**:
`i = blockIdx.x * blockDim.x + threadIdx.x`

**Bounds Checking:**
Because the total number of threads launched (Grid size × Block size) must be a multiple of the block size, it often exceeds the actual vector size `N`. Therefore, a bounds check is required inside the kernel to prevent out-of-bounds memory access:
```c
if (i < N) {
    // Perform computation
}
```

---

## 6. Algorithms

### Sequential Vector Addition
```text
for i = 0 to N-1
    C[i] = A[i] + B[i]
```

### CUDA Vector Addition
```text
calculate global thread index
if index is within vector bounds
    compute C[index] = A[index] + B[index]
```

### Sequential Vector Multiplication
```text
for i = 0 to N-1
    C[i] = A[i] * B[i]
```

### CUDA Vector Multiplication
```text
calculate global thread index
if index is within vector bounds
    compute C[index] = A[index] * B[index]
```

---

## 7. Experimental Methodology

This project will compare the CPU and CUDA implementations across multiple vector sizes to evaluate scalability. All final values will be populated from actual lab measurements once the experiments are executed.

| Parameter | Status |
|----------|--------|
| CPU implementation | Planned |
| CUDA implementation | Planned |
| Vector addition | Planned |
| Vector multiplication | Planned |
| Multiple vector sizes | Planned |
| Timing collection | Pending |
| Speedup calculation | Pending |
| Efficiency analysis | Pending |
| Graph generation | Pending |

---

## 8. Performance Metrics

- **Execution Time:** The total time taken to perform the vector operation. We will independently measure the kernel execution time and the total time including memory transfers.
- **Speedup:** The ratio indicating how much faster the GPU executes the workload compared to the CPU. 
  `Speedup = CPU Execution Time / GPU Execution Time`
- **Efficiency:** The efficiency metric will be defined consistently with the experimental CUDA configuration and documented with the final results.

---

## 9. Planned Results Table

*(Note: This is a placeholder until actual benchmark data is collected.)*

| Vector Size | CPU Addition | GPU Addition | CPU Multiplication | GPU Multiplication |
|------------:|-------------:|-------------:|-------------------:|-------------------:|
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |

---

## 10. Result Analysis

Once the data is collected, we intend to analyze:
- How execution time changes as vector size increases.
- CPU vs GPU execution time.
- Speedup obtained from GPU execution.
- Effect of CUDA thread configuration.
- Possible overhead from memory transfers.
- Behavior for smaller vs larger workloads.

---

## 11. Graphs

*(Real graphs will be added here after benchmark collection.)*

- **Execution Time vs Vector Size:** To visualize the raw time taken by CPU and GPU.
- **Speedup vs Vector Size:** To identify at which vector size the GPU provides the maximum relative advantage.
- **Efficiency vs Vector Size:** To evaluate resource utilization.

---

## 12. Screenshots / Evidence

*(Space reserved for lab execution evidence)*

- NVIDIA GPU verification (`nvidia-smi`)
- CUDA compiler verification (`nvcc --version`)
- Successful compilation
- Vector addition execution
- Vector multiplication execution
- Benchmark outputs
- Final GitHub/project evidence

---

## 13. Project Structure

The repository is structured cleanly for the evaluation:

```text
Lab 1 evaluation/
├── README.md
├── src/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── graphs/
│   └── .gitkeep
├── screenshots/
│   └── .gitkeep
├── presentation/
│   └── .gitkeep
└── report/
    └── .gitkeep
```

**Directory breakdown:**
- `src/`: Contains all C/C++ and CUDA source code files.
- `data/`: Contains any raw datasets or input matrices/vectors.
- `results/`: Contains raw benchmark logs or CSVs outputted by the program.
- `graphs/`: Contains generated visualizations and charts.
- `screenshots/`: Contains execution logs and environment verification images.
- `presentation/`: Reserved for PPT or slide decks.
- `report/`: Reserved for the final technical report.

---

## 14. Build and Run

**To be completed.** *(Actual build/run commands will be added after the source files are finalized.)*

*Example / Placeholder Command:*
```bash
nvcc -O2 <cuda_source>.cu -o <output>
./<output>
```

---

## 15. Technologies

| Technology | Purpose |
|-----------|---------|
| CUDA | GPU parallel programming |
| C/C++ | CPU implementation / host-side code |
| NVIDIA GPU | GPU execution |
| Visual Studio / MSVC | Windows C/C++ host compiler |
| CUDA Toolkit / nvcc | CUDA compilation |
| GitHub | Version control and submission |

---

## 16. Viva Preparation Topics

This experiment covers the following core topics:
- CUDA
- CPU vs GPU
- Thread
- Block
- Grid
- Kernel
- Host and Device
- Global thread index
- Memory transfer
- Vector addition
- Vector multiplication
- Execution time
- Speedup
- Efficiency
- Parallelism

---

## 17. References

- Course-provided Parallel Computing lab material.
- NVIDIA CUDA documentation, when relevant during implementation.
