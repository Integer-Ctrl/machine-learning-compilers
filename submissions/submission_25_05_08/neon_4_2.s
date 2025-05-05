/**
    * @param x0 = a pointer to column-major 15x64 matrix A.
    * @param x1 = b pointer to column-major 64x6 matrix B.
    * @param x2 = c pointer to column-major 15x6 matrix C.
    * @param x3 = lda leading dimension of A.
    * @param x4 = ldb leading dimension of B.
    * @param x5 = ldc leading dimension of C.
**/
.text
.type matmul_15_6_64, %function
.global matmul_15_6_64
matmul_15_6_64:

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

    mov x6, x1 // Store the initial value of x1, to be restored in the next loop iteration
    mov x7, x2 // Store the initial value of x2, to be restored after the loop

    // Load first column from the 15x6 matrix c - load full 16 entries - ignore last
    ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5
    // Load second column from the 15x6 matrix c
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5
    // Load third column from the 15x6 matrix c
    ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
    // Load fourth column from the 15x6 matrix c
    ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x2], x5
    // Load fifth column from the 15x6 matrix c
    ld1 {v9.4s, v10.4s, v11.4s, v12.4s}, [x2], x5
    // Load sixth column from the 15x6 matrix c
    ld1 {v13.4s, v14.4s, v15.4s, v16.4s}, [x2], x5

    mov x9, #64 // x9 iterator for K loop
matmul_loop_over_K:
    sub x9, x9, #1

    // Load first column data from the 15x1 matrix a
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x3
    // ldp q0, q1, [x0] // 4 + 4 values
    // ldr q2, [x0, #32] // 4 values
    // ldr d3, [x0, #48] // 2 values

    // run the known matmul_16_6_1_unrolled kernel with modification to matmult_15_6_1
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
    st1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5 // offset of 15 floats
    // Store second column back to memory
    st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5 // offset of 15 floats
    // Store third column back to memory
    st1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5 // offset of 15 floats
    // Store fourth column back to memory
    st1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x2], x5 // offset of 15 floats
    // Store fifth column back to memory
    st1 {v9.4s, v10.4s, v11.4s, v12.4s}, [x2], x5 // offset of 15 floats
    // Store sixth column back to memory (exactly last 15 elements)
    stp q13, q14, [x2] // 8 floats
    str q15, [x2, #32] // 4 floats
    str d16, [x2, #48] // 2 floats
    mov w9, v16.s[2]
    str w9, [x2, #56] // 1 floats

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
    .size matmul_15_6_64, (. - matmul_15_6_64)
