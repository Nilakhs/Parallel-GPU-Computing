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
