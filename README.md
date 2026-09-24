# Experiment 1 — Matrix Multiplication

## Overview

This experiment implements matrix multiplication using two different CPU-based approaches:

- **Part A:** Sequential Matrix Multiplication
- **Part B:** OpenMP Parallel Matrix Multiplication

The same `4000 × 4000` matrix multiplication problem is used for both implementations. Each element of matrices `A` and `B` is initialized to `1.0`, so every element of the resulting matrix `C` is expected to be `4000.00`. :chatgpt-content-reference{index="0"}

---

# Part A — Sequential Matrix Multiplication

## Objective

To implement matrix multiplication using a single sequential CPU execution flow and establish a baseline execution time.

## Implementation

The program was written in C using three nested loops to perform:

```text
C = A × B
