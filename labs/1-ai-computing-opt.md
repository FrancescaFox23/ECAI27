---
title: "1. AI Computing Optimization"
---

## Learning objectives

By the end of this lab, you should be able to:

- Learn and understand the main optimizations for AI computational workloads, including SIMD, parallelism, data reuse, and tiling.
- Describe the memory hierarchy and use the roofline model's **operational intensity** (operations per byte transferred) to reason about memory-bound and compute-bound workloads.
- Explain how **SIMD**, multiple levels of parallelism, **data reuse**, and **tiling** affect neural-network computations.
- Implement and compare scalar and ARM NEON versions of basic kernels, then use GEMM and a convolution workload to observe the effects of data layout and data movement.
- Report correctness and performance using FLOPs/MACs, latency, throughput, and stated measurement assumptions.

## Introduction

In PyTorch, an operation such as `y = layer(x)` is presented as a high-level call. At runtime, operations are carried out by optimized low-level routines, often called **micro-kernels**. Those routines use techniques such as SIMD, tiling, data reuse, and parallel execution, i.e., the same ideas discussed in lecture, along with more advanced techniques outside this lab.

This lab makes the effects of those optimizations visible on the computational workload. The best organization depends on the processor architecture: instructions, vector width, caches, and memory bandwidth differ across hardware. We focus on the ARM CPU in the Raspberry Pi 4, whose CPU has **four cores**, not on a universal implementation. Real production micro-kernels are substantially more complex than these educational examples; this lab is not a production-library implementation.

The Pi 4's AArch64 CPU supports ARM NEON SIMD. **Single instruction, multiple data (SIMD)** applies one operation to several values at once. For example, a scalar add handles one pair:

```text
a0 + b0 -> c0
```

A four-lane FP32 NEON add handles four pairs in parallel:

```text
[a0 a1 a2 a3] + [b0 b1 b2 b3] -> [c0 c1 c2 c3]
```

This is useful for AI workloads because operations such as vector arithmetic, dot products, and matrix multiplication contain many similar operations over tensor elements. SIMD alone does not guarantee a speedup: data access, reductions, tails, and available memory bandwidth also matter.

We will write **C with ARM NEON intrinsics**, which explicitly expresses vector operations and is translated by the compiler to ARM instructions. Inline assembly is another way to express instructions, but is intentionally **not used** in this lab: intrinsics keep the focus on kernel ideas without requiring assembly programming.

Some later exercises also use **OpenMP**. OpenMP is a simple way to make a C program use multiple threads: you add a directive such as `#pragma omp parallel for` to a loop whose iterations can run independently, and the compiler/runtime distributes that work across threads. Here, think of it as a way to use multiple CPU cores; you do not need prior threading experience. It is separate from NEON: NEON processes several data values within a core, while OpenMP lets multiple threads run work across cores. OpenMP is enabled for the relevant build with `-fopenmp`.

## Exercise sequence

Work through the exercises in order. Starter functions contain TODOs; keep their declared names and argument layouts. **Implement each kernel's scalar version first.** It is the simple correctness reference used by the tests to check the optimized version (such as NEON, tiled, indirect, or fused code). Run `make test` after the scalar implementation, then use the same tests as you add optimized variants.

| Stage | Kernel / workload | Variants and optimization studied |
|---|---|---|
| 1 | Vector addition | Scalar baseline; explicit NEON vector load, add, store, and tail handling. Observe a simple memory-bound workload. |
| 2 | Sum reduction | Scalar reference; NEON SIMD reduction. Study horizontal reduction and partial sums. |
| 3 | GEMV (`y = Ax`) | Scalar dot product; NEON vector accumulation and reduction. Study reuse of the input vector and weights. |
| 4 | GEMM (`C = AB`) | Scalar GEMM; educational loop-blocked GEMM with tile sizes 4, 8, 16, and 32. Study cache working sets and data reuse. |
| 5 | Batched GEMM | Scalar and tiled forms; optional OpenMP multi-threading over batch items or output tiles. Study shared-weight reuse, temporal/spatial tiling, and throughput versus per-item latency. |
| 6 | Transformer connection | Identify GEMM-like, elementwise, and reduction operations in scaled dot-product attention; discussion only. |

## 1. Get set up and find your way around

Project structure:

```text
lab1
├── README.md                 Setup, concepts, and non-Conv1D exercises
├── Makefile                  Build/test shortcuts
├── include/kernels.h         Public function declarations and layouts
├── src/
│   ├── vector_add/            Vector-add kernels and benchmark
│   ├── reduce_sum/             Sum-reduction kernels, test, and benchmark
│   ├── gemv/                  GEMV kernels and benchmark
│   ├── gemm/                  GEMM kernels, benchmark, and tests
│   └── conv1d/                Conv1D kernels, benchmark, and tests
```

You will mainly edit files in `src/`. Keep function names and argument order as declared in `include/kernels.h`. Arrays are flat in memory; for example, row-major `A[M,K]` stores each row's `K` values consecutively. `size_t` is used for dimensions and indices; `const float *` means that the function reads, but does not modify, those values.

### Build, edit, test

Build only the exercise you are working on, then run its benchmark. For example:

```bash
make vector_add
./vector-add/benchmark
```

After implementing a TODO, run the correctness tests. `make test` builds and
runs each exercise's test from its corresponding source folder:

```bash
make test
```

You can also run one exercise's test directly after building it, for example:

```bash
make src/vector_add/test_kernel
./src/vector_add/test_kernel
```

### Run the C benchmarks at different input sizes

The C benchmark programs use dimensions defined as constants near the start
of each `main` function. To measure a different size, edit the corresponding 
constants, save the file, rebuild that target, and run it again. 
The benchmark reports average latency for the selected size.

For vector addition, change `n` in `src/vector_add/benchmark.c` (for example,
try `1024`, `16384`, and `1048576`):

```bash
# Edit n in src/vector_add/benchmark.c, then:
make vector_add
./vector-add/benchmark
```

For the reduction, change `n` in `src/reduce_sum/benchmark.c`, rebuild, and run:

```bash
# Edit n in src/reduce_sum/benchmark.c, then:
make reduce_sum
./reduce-sum/benchmark
```

The default benchmark compares scalar and SIMD. To include the SIMD+OpenMP
variant, build its optional OpenMP executable and choose its thread count:

```bash
make reduce_sum_omp
OMP_NUM_THREADS=1 ./reduce-sum/benchmark_omp
OMP_NUM_THREADS=2 ./reduce-sum/benchmark_omp
OMP_NUM_THREADS=4 ./reduce-sum/benchmark_omp
```

For GEMV, change `m` and `k` in `src/gemv/benchmark.c`. For example, compare
`m = 32, k = 64` with `m = 128, k = 256`:

```bash
# Edit m and k in src/gemv/benchmark.c, then:
make gemv
./gemv/benchmark
```

For GEMM, change `m`, `n`, and `k` in `src/gemm/benchmark.c`. The benchmark
compares scalar GEMM with tiled GEMM using the `tile` value in that file; you
can change the dimensions and tile size between runs:

```bash
# Edit m, n, k (and optionally tile) in src/gemm/benchmark.c, then:
make gemm
./gemm/benchmark
```

The Makefile tracks each benchmark source, so rebuilding the target after
saving your edit recompiles it. Record the dimensions, tile size, and compiler
flags alongside each measurement. Compare variants at the same dimensions;
do not compare raw latency across sizes as if they performed the same work.

Tests may fail while relevant TODOs remain unfinished. Read the first failure, fix the implementation, and rerun; do not remove failing tests. A successful build establishes only that the code compiles, not that a kernel is correct or fast. Start with the first compiler error because later errors may be consequences of it.

Useful flags in this lab:

- `gcc`: compile C code;
- `-O0`: the Makefile's default compilation effort; useful for debugging and keeping generated code closer to the source;
- `-O3`: enable more speed optimizations;
- `-mcpu=cortex-a72`: tune for the Pi 4 CPU;
- `-fopenmp`: enable an optional OpenMP build;
- `-lm`: link the math library when needed;
- `-fPIC`: only needed if building a shared library.

A one-file compile example using the Makefile's default optimization level:

```bash
gcc -O0 -Wall -Wextra -std=c11 -mcpu=cortex-a72 -Iinclude -c src/vector_add/vector_add.c -o /tmp/vector_add.o
```

### Optional: compare compiler optimization

The Makefile keeps compilation effort separate from the other compiler flags in `OPT`, which defaults to `-O0`. Override `OPT` to compare the same workload with `-O3`; record the flags used for every measurement:

```bash
make clean
make vector_add OPT=-O0
./vector-add/benchmark
make clean
make vector_add OPT=-O3
./vector-add/benchmark
```

You can compare compiler output with `gcc -O0 ... -S src/vector_add/vector_add.c -o ./vector_add_O0.s` and the corresponding `-O3` command, or inspect a built executable with `objdump -d ./vector-add/benchmark`.

## 2. ARMv8-A and NEON quick reference

The **instruction set architecture (ISA)** is the software/hardware contract. A CPU executes instructions using its datapath, registers, ALU/FPU, and control logic; caches and main memory form a hierarchy. AArch64 NEON is ARM's SIMD technology. Its vector register file has 32 registers of 128 bits, so one vector holds four FP32 lanes.

```c
#include <arm_neon.h>

float32x4_t zero = vdupq_n_f32(0.0f); // broadcast one scalar to all 4 lanes
float32x4_t a = vld1q_f32(ptr);       // load 4 adjacent FP32 values
vst1q_f32(ptr, a);                    // store 4 FP32 values
float32x4_t s = vaddq_f32(a, b);      // add corresponding lanes
float32x4_t d = vsubq_f32(a, b);      // subtract corresponding lanes
float32x4_t p = vmulq_f32(a, b);      // multiply corresponding lanes
float32x4_t acc = vfmaq_f32(acc, a, b); // acc += a*b, lane by lane
float sum = vaddvq_f32(acc);          // add the four lanes into one scalar
```

Use these intrinsics in the SIMD exercises:

| Intrinsic | Purpose in this lab |
|---|---|
| `vdupq_n_f32(value)` | Initialize a vector accumulator by copying `value` into all four lanes. |
| `vld1q_f32(address)` | Load four contiguous FP32 values from memory. Only use it when four valid elements remain. |
| `vst1q_f32(address, value)` | Store four FP32 lanes to contiguous memory. |
| `vaddq_f32(a, b)` | Add four pairs of values lane by lane; used by vector addition and reduction. |
| `vsubq_f32(a, b)` | Subtract four pairs lane by lane; available for elementwise SIMD arithmetic. |
| `vmulq_f32(a, b)` | Multiply four pairs lane by lane. |
| `vfmaq_f32(acc, a, b)` | Fused multiply-add in each lane: `acc[lane] += a[lane] * b[lane]`; useful for GEMV/GEMM/convolution accumulation. |
| `vaddvq_f32(value)` | Horizontally add the four FP32 lanes to produce one scalar; used at the end of a reduction or dot product. |

The `q` in these AArch64 intrinsics denotes a 128-bit vector; `float32x4_t` therefore holds four FP32 values. 

## 3. Vector addition: introduce SIMD

Implement scalar vector addition first, then the NEON version. The tests compare the NEON result with the scalar result, so the scalar implementation must be correct before that comparison is meaningful:

```c
void vector_add(const float *a, const float *b, float *c, size_t n);
```

Use the scalar version as the correctness baseline. The NEON version uses vector loads, addition, stores, and a tail if needed. This is a simple, often memory-bound workload.

Before measuring, calculate additions, FP32 values loaded/stored, and approximate operational intensity:

```text
OI = operations / bytes transferred
```

Does NEON achieve a 4× speedup? Explain using work per instruction and data movement.

| N | Scalar time | NEON time | Speedup |
|---:|---:|---:|---:|
| 1K | | | |
| 16K | | | |
| 1M | | | |

## 4. Sum reduction: from SIMD lanes to CPU cores

After vector addition, implement a function that sums all elements of a 1D FP32
array:

```c
float reduce_sum_scalar(const float *x, size_t n);
```

Implement the scalar version first; tests use it as the reference for both
parallel variants. For `n == 0`, return zero. Then implement these versions in
order:

1. **Scalar:** add each element to one scalar accumulator.
2. **SIMD:** accumulate groups of four values in a `float32x4_t`, horizontally
   add its four lanes to a scalar, and handle any remaining elements with a
   scalar tail. The vector lanes are partial sums, not four final answers.
3. **SIMD + OpenMP:** divide the input into independent chunks. Each thread
   accumulates its chunk with NEON, reduces its lanes to a partial scalar, and
   combines partial scalars using an OpenMP reduction. Do not update one shared
   sum without a reduction clause: that would create a data race.

Build and run the non-OpenMP benchmark (scalar and SIMD):

```bash
make reduce_sum
./reduce-sum/benchmark
```

Build the optional multi-core benchmark separately. Compare the same input size
at 1, 2, and 4 threads; small inputs may not benefit because parallel setup and
combining partial sums have a cost.

```bash
make reduce_sum_omp
OMP_NUM_THREADS=1 ./reduce-sum/benchmark_omp
OMP_NUM_THREADS=2 ./reduce-sum/benchmark_omp
OMP_NUM_THREADS=4 ./reduce-sum/benchmark_omp
```

The test includes lengths that are shorter than a NEON vector, divisible by
four, and have a SIMD tail. Compare each result to the scalar reference within
a floating-point tolerance: changing the order of additions can change the
last few bits. Count `n-1` additions for `n > 0`; the input occupies `4*n`
bytes. Estimate operational intensity and explain why this reduction moves
through the input only once, while SIMD and OpenMP expose different kinds of
parallelism.

## 5. GEMV: dot products and reductions

Implement the scalar `y = A x` first, where `A` is `M × K`, `x` is `K`, and `y` is `M`. This scalar result is the reference used to test the NEON version. Then implement a NEON version that processes four input terms at a time and reduces the partial vector sum to one scalar output.

Calculate the MAC count, sizes of `x` and `A`, and identify which values are reused. Compare latency and GFLOP/s. State whether you count one MAC as one MAC or as two FLOPs (multiply plus add).

| Variant | Latency (µs) | GFLOP/s | Speedup vs scalar |
|---|---:|---:|---:|
| Scalar | | | 1.0× |
| NEON | | | |

## 6. GEMM and loop tiling: increase data reuse

Implement the scalar `C = AB` function first; `make test` uses its output as the reference for the tiled version. For row-major matrices `A[M,K]`, `B[K,N]`, and `C[M,N]`, compute

```text
C[i,j] = sum(k=0..K-1) A[i,k] * B[k,j]
```

Use the flat-array indices `A[i*K + k]`, `B[k*N + j]`, and `C[i*N + j]`. Initialize an accumulator to zero for each output element, sum over all `k`, then store the result. Do not rely on the incoming contents of `C`.

Once the scalar version passes its test, implement the tiled function and compare it against the scalar result. A tile groups nearby rows, columns, and reduction indices so portions of `A` and `B` can be reused while computing a smaller output region. The outer-loop structure is:

```c
for (ii = 0; ii < M; ii += T) {
    for (jj = 0; jj < N; jj += T) {
        for (kk = 0; kk < K; kk += T) {
            /* compute this portion of the output tile */
        }
    }
}
```

Inside these loops, clamp each tile's end to the matrix dimension, for example `i_end = min(ii + T, M)`. The same is needed for `j` and `k`: dimensions may not be multiples of `T`, and the test deliberately checks edge tiles. For each output element, accumulate contributions across all `kk` tiles; initialize its output before the first `kk` contribution (or use a local accumulator over the complete `k` range).

Try `T = 4, 8, 16, 32` on the same matrix dimensions. Before timing, predict which tile sizes may help and why. Then measure: a larger tile is not automatically faster, because its working set may use cache less effectively, and loop overhead and compiler behavior also matter. Record the dimensions, tile size, latency, and speedup relative to scalar. This is an educational blocked GEMM, not a production microkernel.

| Variant / tile | Latency (µs) | GFLOP/s | Speedup vs scalar | Observation |
|---|---:|---:|---:|---|
| Scalar | | | 1.0× | |
| Tiled, T=4 | | | | |
| Tiled, T=8 | | | | |
| Tiled, T=16 | | | | |
| Tiled, T=32 | | | | |

For this GEMM, the work is `M*N*K` MACs (or `2*M*N*K` FLOPs when counting a multiply and an add separately). Explain what is reused in your loop order: an `A[i,k]` value contributes to multiple output columns, and a `B[k,j]` value contributes to multiple output rows. Connect your measurements to data reuse, the tile's cache working set, and operational intensity.

| Variant / tile | Latency (µs) | GFLOP/s | Speedup vs scalar |
|---|---:|---:|---:|
| Scalar | | | 1.0× |
| Tiled, T=4 | | | |
| Tiled, T=8 | | | |
| Tiled, T=16 | | | |
| Tiled, T=32 | | | |

### Extension: Batched GEMM, reuse, and parallel schedules

Apply one shared weight matrix to multiple input matrices:

```text
A: [B,M,K]   W: [K,N]   Y: [B,M,N]
Y[b,i,j] = sum_k A[b,i,k] * W[k,j]
```

Implement the declared functions in `src/gemm/gemm_batched.c`; do not copy `W` for each batch item. Build and run:

```bash
make batched_gemm_bench
./batched_gemm_bench
make batched_gemm_bench_omp
OMP_NUM_THREADS=1 ./batched_gemm_bench_omp
OMP_NUM_THREADS=2 ./batched_gemm_bench_omp
OMP_NUM_THREADS=4 ./batched_gemm_bench_omp
```

The benchmark uses `M=N=K=64`, batch sizes `B=1, 4, 16, 64`, and tile size 16. It checks each result before timing, allocates its arrays outside the timed region, performs warm-up iterations, and reports latency, GFLOP/s, and speedup relative to scalar GEMM. Compare schedules at the same batch size and thread count; change one factor at a time so you can interpret the result.

The schedules are serial scalar GEMM, serial tiled GEMM, OpenMP over batch items, and OpenMP over output tiles. OpenMP over batch items gives each parallel task a different `b`; with `B=1`, that schedule has only one batch task. OpenMP over output tiles instead distributes independent output tiles, so it can expose parallel work even when `B=1`. Neither schedule is guaranteed to be faster: small tasks, memory traffic, and thread overhead can limit scaling. NEON is SIMD within a core; OpenMP runs work across cores. Tiling for cache reuse and distributing tiles across cores are related but distinct: a tile can improve local data reuse, while parallel scheduling decides which core computes it.

Before running, calculate the work for each batch size:

```text
MACs = B * M * N * K
FLOPs = 2 * B * M * N * K   (if one multiply and one add count as two FLOPs)
```

For FP32, estimate the tensor storage footprint as:

```text
bytes = 4 * (B*M*K + K*N + B*M*N)
         input A      shared W    output Y
```

This is the size of the arrays, **not** the number of bytes transferred from main memory during execution: cache reuse can reduce those transfers. To reason about weight reuse across batches, compare the idealized cases of reading `B*K*N` weight elements (no reuse between batches) and `K*N` elements (perfect reuse). Actual traffic depends on cache behavior and the implementation.

Record both total latency and throughput (`GFLOP/s`). Since increasing `B` increases the total amount of work, compare throughput as well as latency. In your discussion, explain how batch size changes available tasks, how weights may be reused, and how the observed speedup compares with the number of threads.

| B | Schedule | Threads | Tile | Latency (µs) | GFLOP/s | Speedup vs scalar | Observation |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | scalar / tiled / batch-OMP / tile-OMP | | 16 | | | | |
| 4 | scalar / tiled / batch-OMP / tile-OMP | | 16 | | | | |
| 16 | scalar / tiled / batch-OMP / tile-OMP | | 16 | | | | |
| 64 | scalar / tiled / batch-OMP / tile-OMP | | 16 | | | | |

### Transformer connection

Scaled dot-product attention combines the same kernel families:

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(P)) V
QK^T and Attention·V  -> GEMM-like kernels
scaling                 -> elementwise operation
softmax                 -> reduction kernel
```

SIMD, parallelism, data reuse, memory traffic, tiling, and fusion are relevant to Transformer inference too. Implementing attention is outside this lab.

## Final discussion and submission

### Complexity-estimation worksheet (complete before benchmarking)

Complete one row for **every listed kernel variant**, using the symbols and
dimensions shown in each exercise (and the dimensions selected for your run).
Do the calculation before timing, then compare the prediction with your
measurements. For every row state assumptions, especially what you count as
memory traffic. Use FP32 = 4 bytes/element. In this lab report **1 MAC = 2
FLOPs** (one multiply and one add); also show MACs separately. For add-only
operations, report additions as FLOPs and MACs as zero.

Use `OI = FLOPs / bytes transferred`. First calculate the tensor storage
footprint, then estimate ideal traffic (inputs read once + outputs written
once). These are not always equal: a kernel can load values repeatedly at
source level, while cache reuse can avoid repeated main-memory transfers.
Write down which model you use. Include any temporary buffers, pointer tables,
and initialization or conversion traffic when estimating end-to-end variants.

#### Per-variant calculation table

Fill a separate row for each variant, not just one row per kernel family. For
variants with different layouts or additional passes, make separate rows for
the compute-only and end-to-end cases. Use equations first, then substitute
your benchmark dimensions.

| Kernel / variant | Dimensions | MACs | FLOPs | Storage (bytes) | Estimated bytes transferred | OI (FLOP/B) |
|---|---|---:|---:|---:|---|---:|
| Vector add — scalar | `n = ____` | | | | | |
| Vector add — NEON | `n = ____` | | | | | |
| Reduce sum — scalar | `n = ____` | | | | | |
| Reduce sum — NEON | `n = ____` | | | | | |
| Reduce sum — NEON + OpenMP | `n = ____`, threads = ____ | | | | | |
| GEMV — scalar | `M = ____`, `K = ____` | | | | | |
| GEMV — NEON | `M = ____`, `K = ____` | | | | | |
| GEMM — scalar | `M = ____`, `N = ____`, `K = ____` | | | | | |
| GEMM — tiled, `T = 4` | `M = ____`, `N = ____`, `K = ____` | | | | | |
| GEMM — tiled, `T = 8` | `M = ____`, `N = ____`, `K = ____` | | | | | |
| GEMM — tiled, `T = 16` | `M = ____`, `N = ____`, `K = ____` | | | | | |
| GEMM — tiled, `T = 32` | `M = ____`, `N = ____`, `K = ____` | | | | | |
| Batched GEMM — scalar | `B = ____`, `M = ____`, `N = ____`, `K = ____` | | | | | |
| Batched GEMM — tiled | `B = ____`, `M = ____`, `N = ____`, `K = ____`, `T = ____` | | | | | |
| Batched GEMM — OpenMP over batch | `B = ____`, threads = ____ | | | | | |
| Batched GEMM — OpenMP over tiles | `B = ____`, threads = ____ | | | | | |

Use these operation-count checks when filling in the table:

```text
Vector add:       n additions
Reduce sum:       n-1 additions 
GEMV:             M*K MACs
GEMM:             M*N*K MACs
Batched GEMM:     B*M*N*K MACs
```

#### Results and prediction comparison

Complete this after running the benchmarks. Keep dimensions, compiler flags,
and thread count with each result. Do not expect OI alone to predict exact
latency; use the last column to explain any difference between your
memory-bound/compute-bound prediction and observation.

| Kernel / variant | Dimensions / tile / threads | Estimated OI (FLOP/B) | Latency (µs) | Throughput (GFLOP/s, if applicable) | Predicted vs observed; explanation |
|---|---|---:|---:|---:|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

In your discussion, identify at least one example each of input reuse, weight
reuse, and output-accumulator reuse. Explain how tiling changes reuse without
changing the MAC count, and distinguish NEON data-level parallelism within a
core from OpenMP parallelism across cores. State whether each workload appears
memory-bound or compute-bound based on both its estimated OI and measured
behavior.

The final lesson is: **efficient AI software is not only about reducing FLOPs; it is also about arranging computation and data so that the hardware can do useful work efficiently.**
