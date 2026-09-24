# Experiment 1 — Matrix Multiplication

## Overview

This experiment implements matrix multiplication using two different CPU-based approaches:

- **Part A:** Sequential Matrix Multiplication
- **Part B:** OpenMP Parallel Matrix Multiplication

The same `4000 × 4000` matrix multiplication problem is used for both implementations. Each element of matrices `A` and `B` is initialized to `1.0`, so every element of the resulting matrix `C` is expected to be `4000.00`.

---

# Part A — Sequential Matrix Multiplication

## Objective

To implement matrix multiplication using a single sequential CPU execution flow and establish a baseline execution time.

## Implementation

The program was written in C using three nested loops to perform:

```text
C = A × B

The program was compiled using GCC with -O2 optimization:
gcc -O2 matrix.c -o matrix

The executable was then run using:
./matrix

Configuration
- Matrix Size: 4000 × 4000
- Execution Model: Sequential
- Compiler: GCC
- Optimization: -O2
Result
Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = 256.171620 seconds
Verification C[0][0] = 4000.00

Execution Time: 256.171620 seconds
Verification: C[0][0] = 4000.00


Part B — OpenMP Matrix Multiplication
Objective
To parallelize matrix multiplication using OpenMP and multiple CPU threads.
Implementation
The program was written in C using OpenMP to parallelize the outer loop of the matrix multiplication.
The main parallel section uses:
#pragma omp parallel for private(j, k)

This distributes the outer-loop iterations among multiple CPU threads.
OpenMP Configuration
The experiment was configured to use 8 OpenMP threads:
export OMP_NUM_THREADS=8

The setting was verified using:
echo $OMP_NUM_THREADS

The system reported 32 logical CPUs using:
nproc

Compilation
The OpenMP program was compiled using GCC with OpenMP support:
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp

The program was executed using:
./matrix_openmp

Configuration
- Matrix Size: 4000 × 4000
- Execution Model: OpenMP
- Threads Used: 8
- Logical CPUs Available: 32
- Compiler: GCC
- Optimization: -O2
Result
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 8
Execution Time = 40.364523 seconds
Verification C[0][0] = 4000.00

Execution Time: 40.364523 seconds
Verification: C[0][0] = 4000.00
 
Performance Comparison
Implementation	Execution Time	Speedup
Sequential	256.171620 s	1.00×
OpenMP	40.364523 s	6.34×


Speedup Calculation
Speedup = Sequential Time / OpenMP Time

Speedup = 256.171620 / 40.364523

Speedup ≈ 6.34×

Conclusion
Both implementations successfully performed 4000 × 4000 matrix multiplication and produced the expected verification value:
C[0][0] = 4000.00

The OpenMP implementation reduced the execution time from 256.171620 seconds to 40.364523 seconds by using 8 CPU threads, achieving an approximately 6.34× speedup over the sequential implementation.

This will render cleanly in GitHub's README preview.
