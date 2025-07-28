Neon
====

This chapter focuses on implementing the first kernels using ARM64 `Neon <https://developer.arm.com/Architectures/Neon>`_ instructions.
The goal is to develop highly optimized kernels for matrix-matrix multiplication and batch-reduced matrix multiplication.

All files related to the tasks of this chapter can be found under ``submissions/neon/``.

Execution Throughput and Latency
--------------------------------

First, we will microbenchmark the execution throughput and latency of selected FP32 NEON instructions. This will provide a better
understanding of their performance characteristics and serve as a reference point for performance expectations.

1. Execution Throughput
^^^^^^^^^^^^^^^^^^^^^^^

**Task**: Microbenchmark the execution throughput of the following instructions:

Each subtask is structured into four parts: the file containing the implementation of the subtask, the driver file that runs the assembly code,
a compilation command to create an executable, and a short description of the results. The results of the microbenchmarks are documented in the
image below:

.. image:: ../_static/images/report_25_05_01/neon_1_1.png
    :align: center

**Subtask**: ``FMLA`` (vector) with arrangement specifier ``4S``.

- File: ``neon_1_1.s``
- Driver: ``neon_1_1_driver.cpp``
- Compilation: ``g++ -o neon_1_1.exe neon_1_1_driver.cpp neon_1_1.s``
- We have :math:`13.2304 \cdot 10^{10}` instructions per second.
  That are :math:`132.304` GFLOPs in total.

**Subtask**: ``FMLA`` (vector) with arrangement specifier ``2S``.

- File: ``neon_1_1.s``
- Driver: ``neon_1_1_driver.cpp``
- Compilation: ``g++ -o neon_1_1.exe neon_1_1_driver.cpp neon_1_1.s``
- We have :math:`6.65221 \cdot 10^{10}` instructions per second.
  That are :math:`66.5221` GFLOPs in total.

**Subtask**: ``FMADD`` (scalar), single-precision variant.

- File: ``neon_1_1.s``
- Driver: ``neon_1_1_driver.cpp``
- Compilation: ``g++ -o neon_1_1.exe neon_1_1_driver.cpp neon_1_1.s``
- We have :math:`1.12728 \cdot 10^{10}` instructions per second.
  That are :math:`11.2728` GFLOPs in total.

**Summary**

It can be seen that the usage of SIMD lanes can increase the throughput significantly. From the scala ``FMADD`` instruction to the vector
``FMLA`` instruction with arrangement specifier ``2S`` the throughput increases by a factor of about 6. The throughput of the vector
``FMLA`` instruction with arrangement specifier ``4S`` is about twice as high as the one with ``2S``, resulting in a factor of about 12 compared to
the scalar ``FMADD`` instruction. This shows the power of SIMD instructions and how they can be used to increase the throughput.

2. Execution Latency
^^^^^^^^^^^^^^^^^^^^

**Task**: Microbenchmark the execution latency of ``FMLA`` (vector) with arrangement specifier ``4s``. Consider the following two cases:

Same structure as above, with the file containing the implementation of the subtask, the driver file that runs the assembly code,
a compilation command to create an executable, and a short description of the results. The results of the microbenchmarks are documented
in the image below:

.. image:: ../_static/images/report_25_05_01/neon_1_2.png
    :align: center

**Subtask**: Dependency on one of the source registers.

- File: ``neon_1_2.s``
- Driver: ``neon_1_2_driver.cpp``
- Compilation: ``g++ -o neon_1_2.exe neon_1_2_driver.cpp neon_1_2.s``
- We have :math:`11.4961 \cdot 10^9` instruction per seconds in a single ALU.
  Resulting in a **latency of** :math:`\approx 1/3` **cycle** for the known clock speed of 4.4 GHz.

**Subtask**: Dependency on the destination register only.

- File: ``neon_1_2.s``
- Driver: ``neon_1_2_driver.cpp``
- Compilation: ``g++ -o neon_1_2.exe neon_1_2_driver.cpp neon_1_2.s``
- We have :math:`11.7019 \cdot 10^9` instruction per seconds in a single ALU.
  Resulting in a **latency of** :math:`\approx 1/3` **cycle** for the known clock speed of 4.4 GHz.

**Summary**

We see that the latency of the ``FMLA`` instruction is about 1/3 of a cycle, regardless of whether there is a dependency on one of the
source registers or only on the destination register.

Microkernel
-----------

Next, we implement the first microkernel for the matrix-matrix multiplication of :math:`16 \times 1` matrices with a :math:`1 \times 6` matrix
which uses a :math:`16 \times 6` accumulator matrix C and computes C+=AB.

1. matmul_16_6_1
^^^^^^^^^^^^^^^^

**Task**: Implement a Neon microkernel that computes C+=AB for M=16, N=6, and K=1. Wrap your microkernel in the ``matmul_16_6_1`` function.

- File: ``neon_2_simple.s``
- Driver: ``neon_2_driver.cpp``

Implementation of the microkernel looping over each of the six columns of the matrix C:

.. code-block:: asm
    :linenos:
    
        ...
        // Offset the used leading dimension by the size of floats (4byte == 2 lshifts)
        lsl x3, x3, #2 // x3 * 4 = x3 * sizeof(float)
        lsl x4, x4, #2 // x4 * 4 = x4 * sizeof(float)
        lsl x5, x5, #2 // x5 * 4 = x5 * sizeof(float)

        // Load all data from the 16x1 matrix a
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]

        // Init the loop counter
        mov x6, #6
    process_next_column:
        // Iteration -= 1
        subs x6, x6, #1

        // Load next element from the 1x6 matrix 
        // ldr s4, [x1], #4 // one-liner but not using the argument offset
        ldr s4, [x1]
        add x1, x1, x4

        // Load next column from the 16x6 matrix c
        ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2]
        
        // Calculate the next row of c
        fmla v17.4s, v0.4s, v4.s[0]
        fmla v18.4s, v1.4s, v4.s[0]
        fmla v19.4s, v2.4s, v4.s[0]
        fmla v20.4s, v3.4s, v4.s[0]

        // Store the result back to memory
        st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5

        // Compare and branch on not-zero
        cbnz x6, process_next_column
        ...

.. _neon_2_optimization:

2. Performance
^^^^^^^^^^^^^^

**Task**: Test and optimize your microkernel. Report its performance in GFLOPS.

- Files: 
    - ``neon_2.h`` using a loop over the columns
    - ``neon_2_unrolled.s`` using an unrolled version of the loop
- Tests: ``neon_2.test.cpp``
- Benchmarks: ``neon_2.bench.cpp``

**Subtask**: Optimization

To optimize the kernel we unrolled the loop into 3 different register ranges (v15-v28, v17-v20, v21-v24),
to allow for less dependency between the calculation of columns.
These 3 different ``fmla`` blocks gets repeated with ``.rept 2`` to achieve the total of 6 column of calculation.

.. code-block:: asm
    :linenos:

    ...
    .rept 2
    // Load first element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4
    // Load first column from the 16x6 matrix c
    ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2]

    // Calculate first column of c
    fmla v25.4s, v0.4s, v4.s[0]
    fmla v26.4s, v1.4s, v4.s[0]
    fmla v27.4s, v2.4s, v4.s[0]
    fmla v28.4s, v3.4s, v4.s[0]

    // Store first column back to memory
    st1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5 

    // Load second element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4
    // Load second column from the 16x6 matrix c
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2]

    // Calculate second column of c
    fmla v17.4s, v0.4s, v4.s[0]
    fmla v18.4s, v1.4s, v4.s[0]
    fmla v19.4s, v2.4s, v4.s[0]
    fmla v20.4s, v3.4s, v4.s[0]

    // Store second column back to memory
    st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5
    
    // Load third element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4
    // Load third column from the 16x6 matrix c
    ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2]

    // Calculated third column of c
    fmla v21.4s, v0.4s, v4.s[0]
    fmla v22.4s, v1.4s, v4.s[0]
    fmla v23.4s, v2.4s, v4.s[0]
    fmla v24.4s, v3.4s, v4.s[0]

    // Store third column back to memory
    st1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
    .endr
    ...

**Subtask**: Benchmarks

We run the benchmark with the following command:

.. code-block::
 
  ./benchmarks --benchmark_counters_tabular=true --benchmark_repetitions=10 --benchmark_report_aggregates_only=true

Therefore we do 10 repetitions of the benchmark which do about ``120 000 000`` iterations each on our matmul kernels.

.. code-block::
  :emphasize-lines: 4, 8
     
  ----------------------------------------------------------------------------------------------------------------------------------
  Benchmark                                                                             Time             CPU   Iterations      FLOPS
  ----------------------------------------------------------------------------------------------------------------------------------
  Gemm16x6x1Fixture/BM_matmul_16_6_1_simple/min_warmup_time:1.000_mean               5.84 ns         5.82 ns           10 33.0036G/s
  Gemm16x6x1Fixture/BM_matmul_16_6_1_simple/min_warmup_time:1.000_median             5.83 ns         5.81 ns           10 33.0317G/s
  Gemm16x6x1Fixture/BM_matmul_16_6_1_simple/min_warmup_time:1.000_stddev            0.025 ns        0.025 ns           10 143.339M/s
  Gemm16x6x1Fixture/BM_matmul_16_6_1_simple/min_warmup_time:1.000_cv                 0.43 %          0.44 %            10      0.43%
  Gemm16x6x1Fixture/BM_matmul_16_6_1_unrolled/min_warmup_time:1.000_mean             5.71 ns         5.69 ns           10 33.7234G/s
  Gemm16x6x1Fixture/BM_matmul_16_6_1_unrolled/min_warmup_time:1.000_median           5.70 ns         5.68 ns           10 33.7732G/s
  Gemm16x6x1Fixture/BM_matmul_16_6_1_unrolled/min_warmup_time:1.000_stddev          0.038 ns        0.038 ns           10 224.892M/s
  Gemm16x6x1Fixture/BM_matmul_16_6_1_unrolled/min_warmup_time:1.000_cv               0.67 %          0.67 %            10      0.67

We see that the simple first implementation of our matmul kernel gets about **33.0 GFLOPS**.
The optimized unrolled version gets about 0.7 GFLOPS more resulting in **33.7 GFLOPS**.


Loops
-----

To scale the microkernel to larger matrices, we will introduce loops over the *K*, *M*, and *N* dimensions.

1. Loop over K
^^^^^^^^^^^^^^

**Task**: Loop over K: Implement a kernel that computes C+=AB for M=16, N=6 and K=64. Wrap your kernel in the ``matmul_16_6_64`` function.

The first loop implemented is over the *K* dimension, which is the most inner loop in the matrix multiplication. The result is a microkernel
that computes C+=AB for M=16, N=6 and K=64.

- File ``neon_3_1.s``

.. code-block:: asm
  :linenos:

    ...
    // Offset the used leading dimension by the size of floats
    lsl x3, x3, #2 // x3 * 4 = x3 * sizeof(float)
    lsl x4, x4, #2 // x4 * 4 = x4 * sizeof(float)
    lsl x5, x5, #2 // x5 * 4 = x5 * sizeof(float)

    mov x6, x1 // Store the initial value of x1, to be restored in the next loop iteration
    mov x7, x2 // Store the initial value of x2, to be restored after the loop

    // Load first column from the 16x6 matrix c
    ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5
    // Load second column from the 16x6 matrix c
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5
    // Load third column from the 16x6 matrix c
    ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
    // Load fourth column from the 16x6 matrix c
    ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x2], x5
    // Load fifth column from the 16x6 matrix c
    ld1 {v9.4s, v10.4s, v11.4s, v12.4s}, [x2], x5
    // Load sixth column from the 16x6 matrix c
    ld1 {v13.4s, v14.4s, v15.4s, v16.4s}, [x2], x5

    mov x9, #64 // x9 iterator for K loop
  matmul_loop_over_K:
    sub x9, x9, #1

    // Load first column data from the 16x1 matrix a
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x3

    // run the known matmul_16_6_1_unrolled kernel
    // Load first element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4

    // Calculate first column of c
    fmla v25.4s, v0.4s, v4.s[0]
    fmla v26.4s, v1.4s, v4.s[0]
    fmla v27.4s, v2.4s, v4.s[0]
    fmla v28.4s, v3.4s, v4.s[0]


    // Load second element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4

    // Calculate second column of c
    fmla v17.4s, v0.4s, v4.s[0]
    fmla v18.4s, v1.4s, v4.s[0]
    fmla v19.4s, v2.4s, v4.s[0]
    fmla v20.4s, v3.4s, v4.s[0]

    
    // Load third element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4

    // Calculated third column of c
    fmla v21.4s, v0.4s, v4.s[0]
    fmla v22.4s, v1.4s, v4.s[0]
    fmla v23.4s, v2.4s, v4.s[0]
    fmla v24.4s, v3.4s, v4.s[0]


    // Load fourth element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4

    // Calculate fourth column of c
    fmla v5.4s, v0.4s, v4.s[0]
    fmla v6.4s, v1.4s, v4.s[0]
    fmla v7.4s, v2.4s, v4.s[0]
    fmla v8.4s, v3.4s, v4.s[0]


    // Load fifth element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4

    // Calculate fifth column of c
    fmla v9.4s, v0.4s, v4.s[0]
    fmla v10.4s, v1.4s, v4.s[0]
    fmla v11.4s, v2.4s, v4.s[0]
    fmla v12.4s, v3.4s, v4.s[0]

    
    // Load sixth element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4

    // Calculated sixth column of c
    fmla v13.4s, v0.4s, v4.s[0]
    fmla v14.4s, v1.4s, v4.s[0]
    fmla v15.4s, v2.4s, v4.s[0]
    fmla v16.4s, v3.4s, v4.s[0]


    // offset x6 to the next element in the column
    add x6, x6, #4 // #4 = sizeof(float)

    // Restore x1 to be incremented again
    mov x1, x6

    // Loop back
    cbnz x9, matmul_loop_over_K

    // Restore initial value of x2 that was changed by the loads
    mov x2, x7

    // Store first column back to memory
    st1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5 
    // Store second column back to memory
    st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5
    // Store third column back to memory
    st1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
    // Store fourth column back to memory
    st1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x2], x5 
    // Store fifth column back to memory
    st1 {v9.4s, v10.4s, v11.4s, v12.4s}, [x2], x5
    // Store sixth column back to memory
    st1 {v13.4s, v14.4s, v15.4s, v16.4s}, [x2], x5


2. Loop over M
^^^^^^^^^^^^^^

**Task**: Loop over M: Implement a kernel that computes C+=AB for M=64, N=6 and K=64. Wrap your kernel in the ``matmul_64_6_64`` function.

The next extension is to loop over the *M* dimension to allow computation of C+=AB for M=64, N=6 and K=64.

- File ``neon_3_2.s``

.. code-block:: asm
  :linenos:

      // Offset the used leading dimension by the size of floats
      lsl x3, x3, #2 // x3 * 4 = x3 * sizeof(float)
      lsl x4, x4, #2 // x4 * 4 = x4 * sizeof(float)
      lsl x5, x5, #2 // x5 * 4 = x5 * sizeof(float)

      mov x6, x1 // Store the initial value of x1, to be restored in the K loop iteration
      mov x7, x2 // Store the initial value of x2, to be restored in the K loop iteration

      mov x8, x0 // Store the initial value of x0, to be restored in the M loop iteration
      mov x9, x1 // Store the initial value of x1, to be restored in the M loop iteration

      mov x16, #4 // x16 iterator for M loop
  matmul_loop_over_M:
      sub x16, x16, #1

      // ... <logic of loop over K - neon_3_1>

      // next M iteration on the matrix c and matrix a, both need offset about 16 values
      // also matrix b needs to start at the initial location again
      // Updates for the matrix c
      add x7, x7, #16*4 // column height * sizeof(float)
      mov x2, x7 // also apply offset to x2

      // Updates for the matrix a
      add x8, x8, #16*4 // column height * sizeof(float)
      mov x0, x8 // also apply offset to x0

      // Updates for the matrix b
      mov x6, x9 // Update the restore register for x1 for the K loop
      mov x1, x9 // Update the x1 register itself

      // Loop back to M
      cbnz x16, matmul_loop_over_M

.. _neon_3_loop_over_N:

3. Loop over N
^^^^^^^^^^^^^^

**Task**: Loop over N: Implement a kernel that computes C+=AB for M=64, N=48 and K=64. Wrap your kernel in the ``matmul_64_48_64`` function.

The final extension is to loop over the *N* dimension to allow computation of C+=AB for M=64, N=48 and K=64.

- File ``neon_3_3.s``

.. code-block:: asm
  :linenos:
  
      // Offset the used leading dimension by the size of floats
      lsl x3, x3, #2 // x3 * 4 = x3 * sizeof(float)
      lsl x4, x4, #2 // x4 * 4 = x4 * sizeof(float)
      lsl x5, x5, #2 // x5 * 4 = x5 * sizeof(float)

      mov x6, x1 // Store the initial value of x1, to be restored in the K loop iteration
      mov x7, x2 // Store the initial value of x2, to be restored in the K loop iteration

      mov x8, x0 // Store the initial value of x0, to be restored in the M loop iteration
      mov x9, x1 // Store the initial value of x1, to be restored in the M loop iteration

      mov x10, x0 // Store the initial value of x0, to be restored in the N loop iteration
      mov x11, x2 // Store the initial value of x2, to bes restored in the N loop iteration
      mov x12, #6 // hold the size of N that are processed in one loop, needed for offset calculation 

      mov x17, #8 // x17 iterator for N loop
  matmul_loop_over_N:
      sub x17, x17, #1

    // ... <logic of loop over M - neon_3_2>

      // next M iteration on the matrix b and matrix c, both need offset about 6*ldb/ldc values
      // also matrix a needs to start at the initial location again
      // Update for the matrix a
      mov x8, x10 // Update the restore register for x0 for the M loop
      mov x0, x10 // Update the x0 register itself

      // Updates for the matrix b
      madd x9, x4, x12, x9 // ldb * 6 + initial position
      mov x6, x9 // Update the restore register of x1 for the K loop
      mov x1, x9 // Update the x1 register itself

      // Updates for the matrix c
      madd x11, x5, x12, x11 // ldc * 6 + initial position
      mov x7, x11 // Update the restore register of x2 for the K loop
      mov x2, x11 // Update the x2 register itself

      // Loop back to N
      cbnz x17, matmul_loop_over_N

4. Performance
^^^^^^^^^^^^^^

**Task**: Test and optimize the kernels. Report your performance in GFLOPS.

- File ``neon_3.h``
- Tests ``neon_3.test.cpp``
- Benchmarks ``neon_3.bench.cpp``

**Subtask**: Optimization

Usage of already optimized `matmul_16_6_1` from task :ref:`neon_2_optimization` to as inner microkernel for the
loop over K, M, and N.

**Subtask**: Benchmarks

We run the benchmark with the following command: 

.. code-block:: 
  
  ./benchmarks --benchmark_counters_tabular=true --benchmark_repetitions=10 --benchmark_report_aggregates_only=true


.. code-block::
  :emphasize-lines: 4, 8, 12
     
  ----------------------------------------------------------------------------------------------------------------------------------
  Benchmark                                                                             Time             CPU   Iterations      FLOPS
  ----------------------------------------------------------------------------------------------------------------------------------
  GemmMxNxKFixture<16, 6, 64>/BM_matmul_16_6_64/min_warmup_time:1.000_mean           97.8 ns         97.4 ns           10  126.12G/s
  GemmMxNxKFixture<16, 6, 64>/BM_matmul_16_6_64/min_warmup_time:1.000_median         97.7 ns         97.3 ns           10 126.245G/s
  GemmMxNxKFixture<16, 6, 64>/BM_matmul_16_6_64/min_warmup_time:1.000_stddev        0.581 ns        0.563 ns           10 720.109M/s
  GemmMxNxKFixture<16, 6, 64>/BM_matmul_16_6_64/min_warmup_time:1.000_cv             0.59 %          0.58 %            10      0.57%
  GemmMxNxKFixture<64, 6, 64>/BM_matmul_64_6_64/min_warmup_time:1.000_mean            386 ns          385 ns           10 127.812G/s
  GemmMxNxKFixture<64, 6, 64>/BM_matmul_64_6_64/min_warmup_time:1.000_median          385 ns          384 ns           10  127.95G/s
  GemmMxNxKFixture<64, 6, 64>/BM_matmul_64_6_64/min_warmup_time:1.000_stddev         2.16 ns         2.11 ns           10 693.069M/s
  GemmMxNxKFixture<64, 6, 64>/BM_matmul_64_6_64/min_warmup_time:1.000_cv             0.56 %          0.55 %            10      0.54%
  GemmMxNxKFixture<64, 48, 64>/BM_matmul_64_48_64/min_warmup_time:1.000_mean         3103 ns         3092 ns           10 95.3736G/s
  GemmMxNxKFixture<64, 48, 64>/BM_matmul_64_48_64/min_warmup_time:1.000_median       3097 ns         3087 ns           10 95.5363G/s
  GemmMxNxKFixture<64, 48, 64>/BM_matmul_64_48_64/min_warmup_time:1.000_stddev       16.0 ns         15.6 ns           10 475.851M/s
  GemmMxNxKFixture<64, 48, 64>/BM_matmul_64_48_64/min_warmup_time:1.000_cv           0.52 %          0.50 %            10      0.50%


- Mean FLOPS for loop over K: **126.1 GFLOPS**.
- Mean FLOPS for loop over M: **127.8 GFLOPS**.
- Mean FLOPS for loop over N: **95.4 GFLOPS**.

SIMD Lanes
----------

Up to this point, our *M* and *K* dimensions have always been multiples of 4. This allowed us to fully utilize all SIMD lanes when loading
and storing data from memory. That means we could load or store 4 floats at once using a single instruction, which reduces complexity and
improves the performance of our kernels.

However, this assumption doesn't always exist in real-world applications. To make our implementation more robust, we need to adapt our
kernels to handle cases where the *M* and *K* dimensions are not multiples of 4. Therefore Neon supports loading 4, 2, or 1 float(s) at a
time, which enables us to manage these edge cases.

1. matmul_14_6_64
^^^^^^^^^^^^^^^^^

**Task**: Implement a kernel that computes C+=AB for M=14, N=6 and K=64. Wrap your kernel in the ``matmul_14_6_64`` function.

We first have a look at the case where we have a *M* dimension of 14. Data management can be done by loading/storing three columns of 4
floats and one column of 2 floats.

File: ``neon_4_1.s``

For this kernel ``matmul_14_6_64`` we adapt the already implemented kernel ``matmul_16_6_64``. The only change is that we now use 3
``ld1/st1`` instructions that loads/stores on 4 scalars, and one ``ldr/st1`` instruction that load/store the last 2 scalars: :math:`4 \cdot 3 + 1 \cdot 2 = 14`.
The ``fmla`` remain unchanged as we "mask" the operation by the correct load and store operations.

We load the first 14 floats and additional the last 2 floats:

.. code-block:: asm
    :linenos:

    ...
    // Load first column from the 14x6 matrix c - load 12 + 2 entries
    ldr d28, [x2, #12*4]
    ld1 {v25.4s, v26.4s, v27.4s}, [x2], x5
    // Load second column from the 14x6 matrix c
    ldr d20, [x2, #12*4]
    ld1 {v17.4s, v18.4s, v19.4s}, [x2], x5
    // Load third column from the 14x6 matrix c
    ldr d24, [x2, #12*4]
    ld1 {v21.4s, v22.4s, v23.4s}, [x2], x5
    // Load fourth column from the 14x6 matrix c
    ldr d8, [x2, #12*4]
    ld1 {v5.4s, v6.4s, v7.4s}, [x2], x5
    // Load fifth column from the 14x6 matrix c
    ldr d12, [x2, #12*4]
    ld1 {v9.4s, v10.4s, v11.4s}, [x2], x5
    // Load sixth column from the 14x6 matrix c
    ldr d16, [x2, #12*4]
    ld1 {v13.4s, v14.4s, v15.4s}, [x2], x5
    ...

Next the loop over K:

.. code-block:: asm
    :linenos:

    ...
        mov x9, #64 // x9 iterator for K loop
    matmul_loop_over_K:
        sub x9, x9, #1

        // Load first column data from the 14x1 matrix a  (loading 2 + 12 entries)
        ldr d3, [x0, #12*4]
        ld1 {v0.4s, v1.4s, v2.4s}, [x0], x3

        // run the known matmul_16_6_1_unrolled kernel with modification to matmult_14_6_1
        // Load first element from the 1x6 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculate first column of c
        fmla v25.4s, v0.4s, v4.s[0] // 4 floats
        fmla v26.4s, v1.4s, v4.s[0] // 4 floats
        fmla v27.4s, v2.4s, v4.s[0] // 4 floats
        fmla v28.4s, v3.4s, v4.s[0] // 4 floats

        // Load second element from the 1x6 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculate second column of c
        fmla v17.4s, v0.4s, v4.s[0]
        fmla v18.4s, v1.4s, v4.s[0]
        fmla v19.4s, v2.4s, v4.s[0]
        fmla v20.4s, v3.4s, v4.s[0]
    ...

To store the matrix c back to memory, we use the exact same code we used to load the matrix c, but replace the load with store instructions.

.. code-block:: asm
    :linenos:

    ...
    // Store first column from the 14x6 matrix c - store 12 + 2 entries
    str d28, [x2, #12*4]
    st1 {v25.4s, v26.4s, v27.4s}, [x2], x5
    // Store second column from the 14x6 matrix c
    str d20, [x2, #12*4]
    st1 {v17.4s, v18.4s, v19.4s}, [x2], x5
    // Store third column from the 14x6 matrix c
    str d24, [x2, #12*4]
    st1 {v21.4s, v22.4s, v23.4s}, [x2], x5
    // Store fourth column from the 14x6 matrix c
    str d8, [x2, #12*4]
    st1 {v5.4s, v6.4s, v7.4s}, [x2], x5
    // Store fifth column from the 14x6 matrix c
    str d12, [x2, #12*4]
    st1 {v9.4s, v10.4s, v11.4s}, [x2], x5
    // Store sixth column from the 14x6 matrix c
    str d16, [x2, #12*4]
    st1 {v13.4s, v14.4s, v15.4s}, [x2], x5
    ...

2. matmul_15_6_64
^^^^^^^^^^^^^^^^^

**Task**: Implement a kernel that computes C+=AB for M=15, N=6 and K=64. Wrap your kernel in the ``matmul_15_6_64`` function.

The second edge case we manage is the case where we have a *M* dimension of 15. Data management can be done by loading/storing three columns
of 4 floats, one column of 2 floats, and one time 1 float.

File: ``neon_4_2.s``

For this kernel ``matmul_15_6_64`` we adapt the already implemented kernel ``matmul_16_6_64``. Similar to ``matmul_14_6_64`` we load/store 
the first 12 float and handel the last 3 elements separately. For the last 3 float we divide into a load/store of 2 elements + load/store of
1 element. Again we "mask" the computation by the load and store operation. 

We load the loads 12 + 2 + 1 floats:

.. code-block:: asm
    :linenos:

    ...
    // Load first column from the 15x6 matrix c - load 12 + 2 + 1 entries
    ldr d28, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    ld1 {v28.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    ld1 {v25.4s, v26.4s, v27.4s}, [x2], x5
    // Load second column from the 14x6 matrix c
    ldr d20, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    ld1 {v20.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    ld1 {v17.4s, v18.4s, v19.4s}, [x2], x5
    // Load third column from the 14x6 matrix c
    ldr d24, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    ld1 {v24.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    ld1 {v21.4s, v22.4s, v23.4s}, [x2], x5
    // Load fourth column from the 14x6 matrix c
    ldr d8, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    ld1 {v8.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    ld1 {v5.4s, v6.4s, v7.4s}, [x2], x5
    // Load fifth column from the 14x6 matrix c
    ldr d12, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    ld1 {v12.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    ld1 {v9.4s, v10.4s, v11.4s}, [x2], x5
    // Load sixth column from the 14x6 matrix c
    ldr d16, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    ld1 {v16.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    ld1 {v13.4s, v14.4s, v15.4s}, [x2], x5
    ...

Next the loop over K:

.. code-block:: asm
    :linenos:

    ...
        mov x9, #64 // x9 iterator for K loop
    matmul_loop_over_K:
        sub x9, x9, #1

        // Load first column data from the 15x1 matrix a
        ldr d3, [x0, #12*4]!
        add x0, x0, #2*4 // offset 2 elements
        ld1 {v3.s}[2],[x0]
        sub x0, x0, #14*4 // revert offset 2+12 elements
        ld1 {v0.4s, v1.4s, v2.4s}, [x0], x3

        // run the known matmul_16_6_1_unrolled kernel with modification to matmul_15_6_1
        // Load first element from the 1x6 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculate first column of c
        fmla v25.4s, v0.4s, v4.s[0]
        fmla v26.4s, v1.4s, v4.s[0]
        fmla v27.4s, v2.4s, v4.s[0]
        fmla v28.4s, v3.4s, v4.s[0]

        // Load second element from the 1x6 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculate second column of c
        fmla v17.4s, v0.4s, v4.s[0]
        fmla v18.4s, v1.4s, v4.s[0]
        fmla v19.4s, v2.4s, v4.s[0]
        fmla v20.4s, v3.4s, v4.s[0]
    ...

To store the matrix c back to memory, we use the exact same code we used to load the matrix c, but replace the load with store instructions.

.. code-block:: asm
    :linenos:

    ...
    // Load first column from the 15x6 matrix c - load 12 + 2 + 1 entries
    str d28, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    st1 {v28.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    st1 {v25.4s, v26.4s, v27.4s}, [x2], x5
    // Load second column from the 14x6 matrix c
    str d20, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    st1 {v20.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    st1 {v17.4s, v18.4s, v19.4s}, [x2], x5
    // Load third column from the 14x6 matrix c
    str d24, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    st1 {v24.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    st1 {v21.4s, v22.4s, v23.4s}, [x2], x5
    // Load fourth column from the 14x6 matrix c
    str d8, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    st1 {v8.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    st1 {v5.4s, v6.4s, v7.4s}, [x2], x5
    // Load fifth column from the 14x6 matrix c
    str d12, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    st1 {v12.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    st1 {v9.4s, v10.4s, v11.4s}, [x2], x5
    // Load sixth column from the 14x6 matrix c
    str d16, [x2, #12*4]!
    add x2, x2, #2*4 // offset 2 elements
    st1 {v16.s}[2],[x2]
    sub x2, x2, #14*4 // revert offset 2+12 elements
    st1 {v13.4s, v14.4s, v15.4s}, [x2], x5
    ...

3. Performance
^^^^^^^^^^^^^^

**Task**: Test and optimize the kernels. Report your performance in GFLOPS.

Since we already optimized the base kernel ``matmul_16_6_1`` in task :ref:`neon_2_optimization`, we do not found any further
optimizations for the kernels ``matmul_14_6_64`` and ``matmul_15_6_64``.

Optimized benchmark results:

#TODO rerun benchmark

.. code-block:: 
    :emphasize-lines: 4, 8

    --------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                       Time             CPU   Iterations      FLOPS
    --------------------------------------------------------------------------------------------------------------------------------------------
    GemmMxNxKFixture<14, 6, 64>/BM_matmul_14_6_64/min_warmup_time:1.000_mean                     94.8 ns         94.5 ns           10 113.789G/s
    GemmMxNxKFixture<14, 6, 64>/BM_matmul_14_6_64/min_warmup_time:1.000_median                   94.8 ns         94.5 ns           10 113.775G/s
    GemmMxNxKFixture<14, 6, 64>/BM_matmul_14_6_64/min_warmup_time:1.000_stddev                  0.671 ns        0.659 ns           10 790.609M/s
    GemmMxNxKFixture<14, 6, 64>/BM_matmul_14_6_64/min_warmup_time:1.000_cv                       0.71 %          0.70 %            10      0.69%
    GemmMxNxKFixture<15, 6, 64>/BM_matmul_15_6_64/min_warmup_time:1.000_mean                     95.5 ns         95.1 ns           10 121.074G/s
    GemmMxNxKFixture<15, 6, 64>/BM_matmul_15_6_64/min_warmup_time:1.000_median                   95.4 ns         95.1 ns           10  121.09G/s
    GemmMxNxKFixture<15, 6, 64>/BM_matmul_15_6_64/min_warmup_time:1.000_stddev                  0.295 ns        0.293 ns           10 373.529M/s
    GemmMxNxKFixture<15, 6, 64>/BM_matmul_15_6_64/min_warmup_time:1.000_cv                       0.31 %          0.31 %            10      0.31%


- **matmul_14_6_64** kernel: :math:`113.8` GFLOPS
- **matmul_15_6_64** kernel: :math:`121.1` GFLOPS

Accumulator Shapes
------------------

This section considers a matrix-matrix multiplication where a high-performance implementation may require accumulator blocks with different shapes.

1. matmul_64_64_64
^^^^^^^^^^^^^^^^^^

**Task**: Implement a kernel that computes C+=AB for M=64, N=64 and K=64. Wrap your kernel in the ``matmul_64_64_64`` function.

File: ``neon_5_1.s``

For this kernel ``matmul_64_64_64`` we adapt the already implemented kernel ``matmul_64_48_64``. The only changes is that we removed
two ``fmla`` blocks from the inner loop i.e. our microkernel becomes a ``matmul_16_4_1`` matrix multiplication.

.. code-block:: asm
    :linenos:
    
    ...
        mov x15, #64 // x15 iterator for K loop
    matmul_loop_over_K:
        sub x15, x15, #1

        // Load first column data from the 16x1 matrix a
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x3

        // run the matmul_16_4_1_unrolled kernel
        // Load first element from the 1x4 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculate first column of c
        fmla v25.4s, v0.4s, v4.s[0]
        fmla v26.4s, v1.4s, v4.s[0]
        fmla v27.4s, v2.4s, v4.s[0]
        fmla v28.4s, v3.4s, v4.s[0]


        // Load second element from the 1x4 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculate second column of c
        fmla v17.4s, v0.4s, v4.s[0]
        fmla v18.4s, v1.4s, v4.s[0]
        fmla v19.4s, v2.4s, v4.s[0]
        fmla v20.4s, v3.4s, v4.s[0]

        
        // Load third element from the 1x4 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculated third column of c
        fmla v21.4s, v0.4s, v4.s[0]
        fmla v22.4s, v1.4s, v4.s[0]
        fmla v23.4s, v2.4s, v4.s[0]
        fmla v24.4s, v3.4s, v4.s[0]


        // Load fourth element from the 1x4 matrix b
        ldr s4, [x1]
        add x1, x1, x4

        // Calculate fourth column of c
        fmla v5.4s, v0.4s, v4.s[0]
        fmla v6.4s, v1.4s, v4.s[0]
        fmla v7.4s, v2.4s, v4.s[0]
        fmla v8.4s, v3.4s, v4.s[0]


        // offset x6 to the next element in the column
        add x6, x6, #4 // #4 = sizeof(float)

        // Restore x1 to be incremented again
        mov x1, x6

        // Loop back to K
        cbnz x15, matmul_loop_over_K
    ...

Then changed the number of loops over M to four to achieve :math:`4 \cdot 16 = 64`:

.. code-block:: asm
    :linenos:
    
    ...
        mov x16, #4 // x16 iterator for M loop
    matmul_loop_over_M:
        sub x16, x16, #1

        // Load first column from the 16x4 matrix c
        ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5
        // Load second column from the 16x4 matrix c
        ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5
        // Load third column from the 16x4 matrix c
        ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
        // Load fourth column from the 16x4 matrix c
        ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x2], x5

        mov x15, #64 // x15 iterator for K loop
    matmul_loop_over_K:
        sub x15, x15, #1
    ...

And finally changed the number of loops over N to 16 :math:`16 \cdot 4 = 64`:

.. code-block:: asm
    :linenos:
    
    ...
        mov x17, #16 // x17 iterator for N loop
    matmul_loop_over_N:
        sub x17, x17, #1

        mov x16, #4 // x16 iterator for M loop
    matmul_loop_over_M:
        sub x16, x16, #1
    ...

2. Performance
^^^^^^^^^^^^^^

**Task**: Test and optimize the kernel. Report your performance in GFLOPS.

After experimenting with different loop orders, we stay with the current order of loops over N, M, and K. The benchmark results are listed below.

.. code-block:: 

    --------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                       Time             CPU   Iterations      FLOPS
    --------------------------------------------------------------------------------------------------------------------------------------------
    GemmMxNxKFixture<64, 64, 64>/BM_matmul_64_64_64/min_warmup_time:1.000_mean                   4111 ns         4097 ns           10 127.964G/s
    GemmMxNxKFixture<64, 64, 64>/BM_matmul_64_64_64/min_warmup_time:1.000_median                 4110 ns         4096 ns           10 127.988G/s
    GemmMxNxKFixture<64, 64, 64>/BM_matmul_64_64_64/min_warmup_time:1.000_stddev                 13.7 ns         13.8 ns           10 431.794M/s
    GemmMxNxKFixture<64, 64, 64>/BM_matmul_64_64_64/min_warmup_time:1.000_cv                     0.33 %          0.34 %            10      0.34%


- **matmul_64_64_64** kernel: :math:`128.0` GFLOPS

Batch-Reduce GEMM
-----------------

This section examines a batch-reduced matrix-matrix multiplication that introduces a fourth dimension *C* alongside the known
*M*, *N*, and *K* dimensions. A batch-reduced matrix-matrix multiplication (BRGEMM or BRMM) is an operation where multiple pairs
of matrices are multiplied, and their results are accumulated into a single output matrix. This operation is commonly used in
machine learning to efficiently perform repeated matrix multiplications with summation across a batch dimension.

1. matmul_64_48_64_16
^^^^^^^^^^^^^^^^^^^^^

**Task**: Implement a kernel that computes C+=∑AᵢBᵢ for M=64, N=48 and K=64 and a batch-reduce dimension size of 16. Wrap your kernel
in the ``matmul_64_48_64_16`` function.

- File: ``neon_6_1.s``

We started by using our ``matmul_64_48_64`` from :ref:`neon_3_loop_over_N` kernel and replaced the microkernel with the ``matmul_16_4_1`` as
it achieves higher performance, resulting in the file ``neon_6_1_no_batch.s``.

.. code-block:: asm
    :linenos:
    :emphasize-lines: 18

    ...
        mov x17, #12 // x17 iterator for N loop
    matmul_loop_over_N:
        sub x17, x17, #1

        ...

        mov x16, #4 // x16 iterator for M loop
    matmul_loop_over_M:
        sub x16, x16, #1

        ...

        mov x15, #64 // x15 iterator for K loop
    matmul_loop_over_K:
        sub x15, x15, #1

        ... matmul_16_4_1 kernel ...

        // Loop back to K
        cbnz x15, matmul_loop_over_K

        ...

        // Loop back to M
        cbnz x16, matmul_loop_over_M
        
        ...

        // Loop back to N
        cbnz x17, matmul_loop_over_N

Then we wrapped the ``matmul_64_48_64`` kernel inside another loop of size 16, representing the batch dimension:

.. code-block:: asm
    :linenos:
    :emphasize-lines: 3, 41
  
    ...
        mov x19, #16 // x19 iterator for the batch dimension
    matmul_loop_batch_dimension:
        sub x19, x19, #1

        ...

        mov x17, #12 // x17 iterator for N loop
    matmul_loop_over_N:
        sub x17, x17, #1

        ...

        mov x16, #4 // x16 iterator for M loop
    matmul_loop_over_M:
        sub x16, x16, #1

        ...

        mov x15, #64 // x15 iterator for K loop
    matmul_loop_over_K:
        sub x15, x15, #1

        ...

        // Loop back to K
        cbnz x15, matmul_loop_over_K

        ... matmul_16_4_1 kernel ...

        // Loop back to M
        cbnz x16, matmul_loop_over_M
        
        ...

        // Loop back to N
        cbnz x17, matmul_loop_over_N

        ...

        // Loop back to batch dimension
        cbnz x19, matmul_loop_batch_dimension


2. Performance
^^^^^^^^^^^^^^

**Task**: Test and optimize the kernel. Report your performance in GFLOPS.

We tested a variation in which the batch loop was positioned between the M and K loops. This approach achieved around :math:`73` GFLOPS. 
We suspect that the reason for this was that the matrices did not fit into the cache. Therefore, we do not follow this approach due to
the poor performance.

However, this leads us to assume that our result of putting the batch loop outside is a good choice. The benchmark results are listed below.

.. code-block::
    :emphasize-lines: 4, 8

    -----------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                          Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------
    GemmMxNxKxBatchFixture<64, 48, 64, 1>/BM_matmul_64_48_64/min_warmup_time:1.000_mean             3104 ns         3093 ns           10 127.138G/s
    GemmMxNxKxBatchFixture<64, 48, 64, 1>/BM_matmul_64_48_64/min_warmup_time:1.000_median           3102 ns         3092 ns           10  127.19G/s
    GemmMxNxKxBatchFixture<64, 48, 64, 1>/BM_matmul_64_48_64/min_warmup_time:1.000_stddev           10.1 ns         8.08 ns           10 331.319M/s
    GemmMxNxKxBatchFixture<64, 48, 64, 1>/BM_matmul_64_48_64/min_warmup_time:1.000_cv               0.33 %          0.26 %            10      0.26%
    GemmMxNxKxBatchFixture<64, 48, 64, 16>/BM_matmul_64_48_64_16/min_warmup_time:1.000_mean        51072 ns        50890 ns           10 123.628G/s
    GemmMxNxKxBatchFixture<64, 48, 64, 16>/BM_matmul_64_48_64_16/min_warmup_time:1.000_median      51027 ns        50840 ns           10 123.749G/s
    GemmMxNxKxBatchFixture<64, 48, 64, 16>/BM_matmul_64_48_64_16/min_warmup_time:1.000_stddev        120 ns          119 ns           10 287.993M/s
    GemmMxNxKxBatchFixture<64, 48, 64, 16>/BM_matmul_64_48_64_16/min_warmup_time:1.000_cv           0.24 %          0.23 %            10      0.23%


- **matmul_64_48_64** kernel: :math:`127.1` GFLOPS
- **matmul_64_48_64_16** kernel: :math:`123.6` GFLOPS

Transposition
-------------

The final topic of this chapter covers matrix transposition. Transposing a matrix means swapping its rows and columns which is a common
operation in many matrix computations. We developed a kernel that performs the identity operation on the elements of an :math:`8 \times 8`
matrix stored in column-major format matrix A and writes the result in row-major format to matrix B.

1. Transpose
^^^^^^^^^^^^

**Task**: Implement a Neon kernel that transposes an 8x8 matrix: B:=Aᵀ.

File: ``neon_7_1.s``

From the lecture, we already know the :math:`4 \times 4` transpose kernel. Therefore, we have the following idea:

1. Divide the 8x8 matrix A into four 4x4 sub-matrices
2. Transpose each 4x4 sub-matrix
3. Save T(A) and T(D) sub-matrix to matrix B
4. Swap sub-matrix B and C: Save T(B) to bottom-left sub-matrix of B and T(C) to top-right sub-matrix of B

.. image:: ../_static/images/report_25_05_22/trans_8_8.png
    :align: left

Code:

.. code-block:: asm
    :linenos:

    ...
    /*
    * Part 1:
    * Load 4x4 sub-matrix A.
    * Transpose 4x4 block.
    * Store 4x4 block of A into B.
    */
    // Load
    ldr q0, [x4]
    add x4, x4, x2
    ldr q1, [x4]
    add x4, x4, x2
    ldr q2, [x4]
    add x4, x4, x2
    ldr q3, [x4]

    // Transpose
    trn1 v4.4s, v0.4s, v1.4s
    trn2 v5.4s, v0.4s, v1.4s
    trn1 v6.4s, v2.4s, v3.4s
    trn2 v7.4s, v2.4s, v3.4s

    zip1  v8.2d, v4.2d, v6.2d
    zip1  v9.2d, v5.2d, v7.2d
    zip2 v10.2d, v4.2d, v6.2d
    zip2 v11.2d, v5.2d, v7.2d

    // Store
    str q8, [x5]
    add x5, x5, x3
    str q9, [x5]
    add x5, x5, x3
    str q10, [x5]
    add x5, x5, x3
    str q11, [x5]

    /*
    * Part 2:
    * Load 4x4 sub-matrix B and C.
    * Transpose both 4x4 blocks.
    * Store both 4x4 blocks of C and B into B.
    */
    // Load right-top
    mov x4, x0       // A
    add x4, x4, #128 // Offset to top-left corner of right half of A (32th element)
    ...

    // Transpose right-top
    ...

    // Load left-bottom
    mov x4, x0      // A
    add x4, x4, #16 // Offset to next 4 elements of column in A (4th element)
    ...

    // Transpose left-bottom
    ...

    // Store after transpose to avoid conflicts when input matrix A = B
    // Store B to C (right-top of A to left-bottom of B)
    mov x5, x1
    add x5, x5, #16
    ...

    // Store C to B (left-bottom of A to right-top of B)
    mov x5, x1
    add x5, x5, #128
    ...

    /*
    * Part 3:
    * Load 4x4 sub-matrix D.
    * Transpose 4x4 block.
    * Store 4x4 block of A into B.
    */
    // Load
    mov x4, x0       // A
    add x4, x4, #144 // 128 + 16 -> left-top corner of right-bottom 4x4 sub-matrix of A
    ...

    // Transpose
    ...

    // Store
    mov x5, x1       // A
    add x5, x5, #144 // 128 + 16 -> left-top corner of right-bottom 4x4 sub-matrix of B
    ...

2. Performance
^^^^^^^^^^^^^^

**Task**: Test and optimize your kernel. Report its performance in GiB/s.

We benchmarked the performance of our transpose kernel and achieved the following results:

.. code-block::
    :emphasize-lines: 4

    --------------------------------------------------------------------------------------------------------------
    Benchmark                                                         Time             CPU   Iterations       Byte
    --------------------------------------------------------------------------------------------------------------
    Trans8x8Fixture/BT_tran_8_8/min_warmup_time:1.000_mean         5.08 ns         5.06 ns           10 101.188G/s
    Trans8x8Fixture/BT_tran_8_8/min_warmup_time:1.000_median       5.07 ns         5.06 ns           10 101.277G/s
    Trans8x8Fixture/BT_tran_8_8/min_warmup_time:1.000_stddev      0.030 ns        0.030 ns           10 590.962M/s
    Trans8x8Fixture/BT_tran_8_8/min_warmup_time:1.000_cv           0.59 %          0.59 %            10      0.58%


- **tran_8_8** kernel: :math:`101.2` GiB/s