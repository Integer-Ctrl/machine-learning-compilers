// using the neon_2_1_unrolled as base kernel as it is the fast based on benchmarks

/**
    * @param x0 = a pointer to column-major 16x64 matrix A.
    * @param x1 = b pointer to column-major 64x6 matrix B.
    * @param x2 = c pointer to column-major 16x6 matrix C.
    * @param x3 = lda leading dimension of A.
    * @param x4 = ldb leading dimension of B.
    * @param x5 = ldc leading dimension of C.
    **/
.text
.type matmul_64_6_64, %function
.global matmul_64_6_64
matmul_64_6_64:

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

    // Load first column data from the 16x1 matrix a (+ offset based on M iteration)
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x3

    mov x15, #64 // x15 iterator for K loop
matmul_loop_over_K:
    sub x15, x15, #1

    // run the known matmul_16_6_1_unrolled kernel
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

    // offset x6 to the next element in the column
    add x6, x6, #4 // #4 = sizeof(float)

    // Restore x1 and x2 to be incremented again
    mov x1, x6
    mov x2, x7

    // Load next column data from the 16x1 matrix a
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x3

    // Loop back to K
    cbnz x15, matmul_loop_over_K

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

    ret
    .size matmul_64_6_64, (. - matmul_64_6_64)
