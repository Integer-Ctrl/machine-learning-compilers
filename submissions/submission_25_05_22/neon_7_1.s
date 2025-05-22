/**
    * Identity primitive that transposes an 8x8 matrix.
    *
    * @param x0 = a pointer to column-major matrix A.
    * @param x1 = b pointer to row-major matrix B.
    * @param x2 = lda leading dimension of A.
    * @param x3 = ldb leading dimension of B.
**/
.type trans_neon_8_8, %function
.global trans_neon_8_8
trans_neon_8_8:

    /*
    * Idea: Divide matrix A into 4 4x4 sub-matrices -> transpose each 4x4 -> swap B top-right and bottom-right
    * A | B      T(A) | T(B)      T(A) | T(C)
    * -----  ->  -----------  ->  -----------
    * C | D      T(C) | T(D)      T(B) | T(D)
    */
    
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
    add x4, x4, #128 // Offset to top-right corner of right half of A (32th element)
    
    ldr q12, [x4]
    add x4, x4, x2
    ldr q13, [x4]
    add x4, x4, x2
    ldr q14, [x4]
    add x4, x4, x2
    ldr q15, [x4]

    // Transpose right-top
    trn1 v16.4s, v12.4s, v14.4s
    trn1 v17.4s, v13.4s, v15.4s
    trn2 v18.4s, v12.4s, v14.4s
    trn2 v19.4s, v13.4s, v15.4s

    zip1 v20.4s, v16.4s, v17.4s 
    zip1 v21.4s, v18.4s, v19.4s 
    zip2 v22.4s, v16.4s, v17.4s 
    zip2 v23.4s, v18.4s, v19.4s

    // Load left-bottom
    mov x4, x0      // A
    add x4, x4, #16 // Offset to next 4 elements of column in A (4th element)

    ldr q0, [x4]
    add x4, x4, x2
    ldr q1, [x4]
    add x4, x4, x2
    ldr q2, [x4]
    add x4, x4, x2
    ldr q3, [x4]

    // Transpose left-bottom
    trn1 v4.4s, v0.4s, v2.4s
    trn1 v5.4s, v1.4s, v3.4s
    trn2 v6.4s, v0.4s, v2.4s
    trn2 v7.4s, v1.4s, v3.4s

    zip1 v8.4s, v4.4s, v5.4s    
    zip1 v9.4s, v6.4s, v7.4s    
    zip2 v10.4s, v4.4s, v5.4s   
    zip2 v11.4s, v6.4s, v7.4s

    // Store after transpose to avoid conflicts when input matrix A = B
    // Store B to C (right-top of A to left-bottom of B)
    mov x5, x1
    add x5, x5, #16

    str q20, [x5]
    add x5, x5, x3
    str q21, [x5]
    add x5, x5, x3
    str q22, [x5]
    add x5, x5, x3
    str q23, [x5]

    // Store C to B (left-bottom of A to right-top of B)
    mov x5, x1
    add x5, x5, #128

    str q8, [x5]
    add x5, x5, x3
    str q9, [x5]
    add x5, x5, x3
    str q10, [x5]
    add x5, x5, x3
    str q11, [x5]

    /*
    * Part 3:
    * Load 4x4 sub-matrix D.
    * Transpose 4x4 block.
    * Store 4x4 block of A into B.
    */
    // Load
    mov x4, x0       // A
    add x4, x4, #144 // 128 + 16 -> left-top corner of right-bottom 4x4 sub-matrix of A

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
    mov x5, x1       // A
    add x5, x5, #144 // 128 + 16 -> left-top corner of right-bottom 4x4 sub-matrix of B

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