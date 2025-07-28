/**
    * Identity primitive that transposes an 4x4 matrix.
    *
    * @param x0 = a pointer to column-major matrix A.
    * @param x1 = b pointer to row-major matrix B.
    * @param x2 = lda leading dimension of A.
    * @param x3 = ldb leading dimension of B.
**/
.type trans_neon_4_4, %function
.global trans_neon_4_4
trans_neon_4_4:
    
    // Procedural Call Standard
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

    // hold addresses to A and B in work registers
    mov x4, x0 // A
    mov x5, x1 // B

    // convert strides to bytes
    lsl x2, x2, #2 // stride of A
    lsl x3, x3, #2 // stride of B

    /*
    * Part 1:
    * Load 4x4 block of A.
    */
    ldr q0, [x4]
    add x4, x4, x2
    ldr q1, [x4]
    add x4, x4, x2
    ldr q2, [x4]
    add x4, x4, x2
    ldr q3, [x4]

    /*
    * Part 2:
    * Transpose 4x4 block.
    */
    trn1 v4.4s, v0.4s, v1.4s
    trn2 v5.4s, v0.4s, v1.4s
    trn1 v6.4s, v2.4s, v3.4s
    trn2 v7.4s, v2.4s, v3.4s

    zip1  v8.2d, v4.2d, v6.2d
    zip1  v9.2d, v5.2d, v7.2d
    zip2 v10.2d, v4.2d, v6.2d
    zip2 v11.2d, v5.2d, v7.2d

    /*
    * Part 3:
    * Store 4x4 block of A into B.
    */
    str q8, [x5]
    add x5, x5, x3
    str q9, [x5]
    add x5, x5, x3
    str q10, [x5]
    add x5, x5, x3
    str q11, [x5]

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

    ret