// using the neon_2_unrolled as base kernel as it is the fast based on benchmarks

/**
    * @param x0 = a pointer to column-major 64x64 matrix A.
    * @param x1 = b pointer to column-major 64x48 matrix B.
    * @param x2 = c pointer to column-major 64x48 matrix C.
    * @param x3 = lda leading dimension of A.
    * @param x4 = ldb leading dimension of B.
    * @param x5 = ldc leading dimension of C.
    **/
.type matmul_64_48_64, %function
.global matmul_64_48_64
matmul_64_48_64:
    
    // Procedural Call Standard
    // save frame pointer and link register
    stp fp, lr, [sp, #-16]!
    // update frame pointer to current stack pointer
    mov fp, sp

    // save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

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

    mov x16, #4 // x16 iterator for M loop
matmul_loop_over_M:
    sub x16, x16, #1

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

    mov x15, #64 // x15 iterator for K loop
matmul_loop_over_K:
    sub x15, x15, #1

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

    // Loop back to K
    cbnz x15, matmul_loop_over_K

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

    // Procedural Call Standard
    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // restore frame pointer and link register
    ldp fp, lr, [sp], #16

    ret
    .size matmul_64_48_64, (. - matmul_64_48_64)
