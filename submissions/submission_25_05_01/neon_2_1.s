/**
    * @param x0 = a pointer to column-major 16x1 matrix A.
    * @param x1 = b pointer to column-major 1x6 matrix B.
    * @param x2 = c pointer to column-major 16x6 matrix C.
    * @param x3 = lda leading dimension of A.
    * @param x4 = ldb leading dimension of B.
    * @param x5 = ldc leading dimension of C.
    **/
.text
.type matmul_16_6_1, %function
.global matmul_16_6_1
matmul_16_6_1:
    // Optimization: Only use caller-saved registers to avoid stack usage

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

    ret
    .size matmul_16_6_1, (. - matmul_16_6_1)
