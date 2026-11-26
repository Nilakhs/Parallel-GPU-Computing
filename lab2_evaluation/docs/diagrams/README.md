# Technical Diagrams Catalog: CUDA Warp Divergence Analysis

This directory contains formal architectural, SIMT execution, and microarchitectural analysis diagrams for **Lab Evaluation 2: CUDA Warp Divergence Analysis** on NVIDIA Ada Lovelace architecture (RTX 4500 Ada Generation, SM 8.9).

---

## Diagram 1 — Project Architecture
```mermaid
flowchart TD
    subgraph Host["CPU Host (System Memory)"]
        H1["1. Deterministic Input Generation (LCG 16.78M FP32)"]
        H2["2. Host Memory Allocation (h_input, h_output)"]
        H7["7. Host-Side Verification (CPU Reference Match)"]
        H8["8. Statistical Timing Analysis (CUDA Events)"]
        H9["9. Data Export (CSV) & Graph Generation (Matplotlib)"]
    end

    subgraph PCIe["PCIe Bus Interface"]
        T1["Host-to-Device Memory Transfer (cudaMemcpyHostToDevice)"]
        T2["Device-to-Host Memory Transfer (cudaMemcpyDeviceToHost)"]
    end

    subgraph Device["NVIDIA GPU Device (Ada Lovelace SM 8.9)"]
        D1["3. Device Memory Allocation (d_input, d_output)"]
        subgraph Grid["CUDA Grid (65,536 Blocks)"]
            subgraph Block["Thread Block (256 Threads)"]
                subgraph Warps["8 Warps (32 Threads / Warp)"]
                    K1["4. Kernel Strategy Execution (Divergent, Aligned, Data-Dep, Branch-Free)"]
                end
            end
        end
        D2["5. CUDA Event Timing (cudaEventRecord & Synchronize)"]
        D3["6. Result Buffer (d_output)"]
    end

    H1 --> H2
    H2 --> T1
    T1 --> D1
    D1 --> Grid
    Grid --> D2
    D2 --> D3
    D3 --> T2
    T2 --> H7
    H7 --> H8
    H8 --> H9
```

---

## Diagram 2 — CUDA Execution Hierarchy
```mermaid
flowchart TD
    Grid["<b>CUDA Grid</b><br/>65,536 Thread Blocks<br/>GridDim.x = 65536"]
    
    Grid --> Block0["Block 0"]
    Grid --> Block1["Block 1"]
    Grid --> BlockDots["..."]
    Grid --> BlockN["Block 65535"]

    subgraph BlockDetail["<b>Thread Block 0</b> (256 Threads, BlockDim.x = 256)"]
        Warp0["Warp 0 (Lanes 0-31)"]
        Warp1["Warp 1 (Lanes 32-63)"]
        WarpDots["..."]
        Warp7["Warp 7 (Lanes 224-255)"]
    end

    Block0 --> BlockDetail

    subgraph IndexFormula["<b>Global Thread Index Calculation</b>"]
        Formula["<b>idx = blockIdx.x * blockDim.x + threadIdx.x</b>"]
        Term1["<b>blockIdx.x * blockDim.x</b>:<br/>Base offset of the current block in the global array"]
        Term2["<b>threadIdx.x</b>:<br/>Local thread offset within the current block (0 to 255)"]
        Term3["<b>Warp Membership</b>:<br/>Warp Index in Block = threadIdx.x / 32 (0 to 7)<br/>Global Warp Index = idx / 32 (0 to 524,287)"]
        Formula --- Term1
        Formula --- Term2
        Formula --- Term3
    end

    BlockDetail --> IndexFormula
```

---

## Diagram 3 — Warp Divergence (SIMT Execution Trace)
```mermaid
sequenceDiagram
    autonumber
    participant WS as Warp Scheduler
    participant L0_15 as Lanes 0, 2, 4... 30 (Path A: Even)
    participant L1_15 as Lanes 1, 3, 5... 31 (Path B: Odd)

    Note over WS,L1_15: Condition: (i % 2 == 0) Evaluated at Cycle 0
    rect rgb(230, 245, 230)
    Note over WS,L0_15: Step 1: Active Mask = 0x55555555 (16 Active, 16 Inactive)
    WS->>L0_15: Issue Instruction: workA(x) [8 FMAs]
    Note over L1_15: Masked Off / Idle (Hardware Inactive)
    end

    rect rgb(255, 235, 235)
    Note over WS,L1_15: Step 2: Active Mask = 0xAAAAAAAA (16 Active, 16 Inactive)
    WS->>L1_15: Issue Instruction: workB(x) [8 FMAs]
    Note over L0_15: Masked Off / Idle (Hardware Inactive)
    end

    Note over WS,L1_15: Step 3: Reconvergence at Branch Target (Total Latency: Phase A + Phase B)
    Note over WS,L1_15: Theoretical Active Lane Efficiency = 16 / 32 = 50%
```

---

## Diagram 4 — Warp-Aligned Branching
```mermaid
flowchart TD
    subgraph WarpEven["Warp 0 (Even Global Warp: (idx / 32) % 2 == 0)"]
        E_Cond["Condition Evaluates TRUE for all 32 Lanes"]
        E_Mask["Active Mask = 0xFFFFFFFF (32/32 Active)"]
        E_Exec["Execute workA(x) in Unison (1 Single Phase)"]
        E_Cond --> E_Mask --> E_Exec
    end

    subgraph WarpOdd["Warp 1 (Odd Global Warp: (idx / 32) % 2 != 0)"]
        O_Cond["Condition Evaluates FALSE for all 32 Lanes"]
        O_Mask["Active Mask = 0xFFFFFFFF (32/32 Active)"]
        O_Exec["Execute workB(x) in Unison (1 Single Phase)"]
        O_Cond --> O_Mask --> O_Exec
    end

    subgraph Summary["Warp-Aligned Execution Characteristics"]
        S1["<b>No Intra-Warp Divergence</b>: Every thread in a warp takes the identical branch"]
        S2["<b>Branch Uniformity</b>: Hardware never executes both paths for the same warp"]
        S3["<b>Active Efficiency</b>: 100% active lanes during instruction issue"]
    end

    WarpEven --- Summary
    WarpOdd --- Summary
```

---

## Diagram 5 — Four Kernel Strategies Comparison
```mermaid
classDiagram
    class DivergentKernel {
        Condition: (i % 2) == 0
        Branch Granularity: 1-thread alternation
        Warp Lane Behavior: 16 Lanes Path A, 16 Lanes Path B
        Active Mask: 0x55555555 then 0xAAAAAAAA
        Execution Phases: 2 Sequential Phases
        Estimated Lane Efficiency: ~50%
        Trade-off: Intra-warp serialization penalty
    }

    class WarpAlignedKernel {
        Condition: ((i / 32) % 2) == 0
        Branch Granularity: 32-thread chunk alignment
        Warp Lane Behavior: All 32 lanes take identical path
        Active Mask: 0xFFFFFFFF
        Execution Phases: 1 Unified Phase
        Estimated Lane Efficiency: 100%
        Trade-off: Zero intra-warp divergence; grid load balance
    }

    class DataDependentKernel {
        Condition: input[i] > threshold (0.5f)
        Branch Granularity: Stochastic / data-value dependent
        Warp Lane Behavior: Arbitrary split based on runtime input
        Active Mask: Dynamic non-uniform per warp
        Execution Phases: Up to 2 Phases per divergent warp
        Estimated Lane Efficiency: ~50% (stochastic average)
        Trade-off: Real-world workload behavior; requires data binning
    }

    class BranchFreeKernel {
        Condition: (input[i] > threshold) ? a : b
        Branch Granularity: No branching (predicated selp)
        Warp Lane Behavior: Both workA and workB computed unconditionally
        Active Mask: 0xFFFFFFFF (100% active)
        Execution Phases: 1 Phase computing 16 FMAs (both paths)
        Estimated Lane Efficiency: 100% (predicated)
        Trade-off: 2x arithmetic instructions (Work Inflation)
    }
```

---

## Diagram 6 — Benchmark Workflow
```mermaid
flowchart TD
    A["<b>1. Environment Verification</b><br/>Verify GPU (RTX 4500 Ada), CUDA 13.4, MSVC 19.51"] --> B["<b>2. Compilation</b><br/>nvcc -O2 -std=c++14 -arch=sm_89"]
    B --> C["<b>3. Memory Setup & Input Generation</b><br/>Allocate 16.78M FP32; generate deterministic LCG data"]
    C --> D["<b>4. Warm-up Launch</b><br/>Un-timed execution to initialize caches and device clocks"]
    D --> E["<b>5. Benchmark Repetitions (20x)</b><br/>Launch kernels with CUDA Event record before & after"]
    E --> F["<b>6. CUDA Event Synchronization</b><br/>cudaEventSynchronize & cudaEventElapsedTime"]
    F --> G["<b>7. Device-to-Host Memory Transfer</b><br/>Transfer results for verification (d_output -> h_output)"]
    G --> H["<b>8. Correctness Check</b><br/>Compare GPU output vs golden CPU reference (PASS/FAIL)"]
    H --> I["<b>9. CSV & Metrics Export</b><br/>Export execution times, speedups, and ratios to timing_data.csv"]
    I --> J["<b>10. Visualization & Graphs</b><br/>Generate 7 high-resolution figures via Matplotlib"]
    J --> K["<b>11. Performance Analysis</b><br/>Microarchitectural evaluation of bandwidth and instruction inflation"]
```

---

## Diagram 7 — Performance Analysis & Metric Correlation
```mermaid
flowchart LR
    subgraph LatencyMetrics["<b>Latency & Ratio Metrics</b>"]
        Tbase["<b>Baseline Latency (T_base)</b><br/>Divergent Kernel: 0.3738 ms"]
        Tvar["<b>Variant Latency (T_variant)</b><br/>Aligned: 0.3955 ms<br/>Data-Dep: 0.3648 ms<br/>Branch-Free: 0.3952 ms"]
        
        Ratio["<b>Execution-Time Ratio (R)</b><br/>R = T_variant / T_base<br/>R < 1: Faster | R > 1: Slower"]
        Speedup["<b>Speedup (S)</b><br/>S = T_base / T_variant = 1 / R<br/>S > 1: Faster | S < 1: Slower"]
        
        Tbase --> Ratio
        Tvar --> Ratio
        Tbase --> Speedup
        Tvar --> Speedup
    end

    subgraph ArchitectureFactors["<b>Underlying Microarchitectural Factors</b>"]
        BW["<b>Global Memory Bandwidth</b><br/>342.4 GB/s effective throughput<br/>(~79.3% of 432 GB/s peak)<br/>Memory coalescing fully preserved"]
        WorkInf["<b>Work Inflation Factor</b><br/>Branch-Free calculates 16 FMAs/thread<br/>Branching calculates 8 FMAs/thread"]
        Pred["<b>Hardware Predication & ITS</b><br/>Independent Thread Scheduling (SM 8.9)<br/>Efficient short branch handling"]
    end

    subgraph Correlation["<b>Performance Outcome Correlation</b>"]
        Outcome1["Memory-bound regime masks intra-warp branch stalls"]
        Outcome2["Work inflation causes branch-free to be slower than divergent"]
    end

    ArchitectureFactors --> Correlation
    LatencyMetrics --- Correlation
```
