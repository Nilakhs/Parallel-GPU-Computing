# Experiment 1 — Matrix Multiplication

## Overview

This experiment implements matrix multiplication using two CPU-based approaches:

- **Part A:** Sequential Matrix Multiplication
- **Part B:** OpenMP Parallel Matrix Multiplication

Both implementations use `4000 × 4000` matrices. Matrices `A` and `B` are initialized with `1.0`, therefore the expected value of every element in the result matrix `C` is `4000.00`.

---

## Part A — Sequential Matrix Multiplication

### Objective

To perform matrix multiplication using a single CPU execution flow and establish a baseline execution time.

### Configuration

| Parameter | Value |
|---|---|
| Matrix Size | 4000 × 4000 |
| Execution Model | Sequential |
| Compiler | GCC |
| Optimization | `-O2` |

### Compilation

```bash
gcc -O2 matrix.c -o matrix


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
