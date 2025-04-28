/**
    * @param x0 = a pointer to column-major 16x1 matrix A.
    * @param x1 = b pointer to column-major 1x6 matrix B.
    * @param x2 = c pointer to column-major 16x6 matrix C.
    * @param x3 = lda leading dimension of A.
    * @param x4 = ldb leading dimension of B.
    * @param x5 = ldc leading dimension of C.
    **/
.text
.type matmul_16_6_1_optimized, %function
.global matmul_16_6_1_optimized
matmul_16_6_1_optimized:

    // Offset the used leading dimension by the size of floats
    lsl x3, x3, #2 // x3 * 4 = x3 * sizeof(float)
    lsl x4, x4, #2 // x4 * 4 = x4 * sizeof(float)
    lsl x5, x5, #2 // x5 * 4 = x5 * sizeof(float)

    // Load all data from the 16x1 matrix a
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]

    // We will use x6 and x2 sequential.
    mov x6, x2
    add x6, x6, x5
    lsl x5, x5, #1

    // We will use x1 and x7 sequential.
    mov x7, x1
    add x7, x7, x4
    lsl x4, x4, #1

    // Load first element from the 1x6 matrix b
    ldr s4, [x1]
    add x1, x1, x4
    // Load first column from the 16x6 matrix c
    ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2]

    // Calculate first column of c
    fmla v25.4s, v0.4s, v4.s[0]
    fmla v26.4s, v1.4s, v4.s[0]
    // Load second column from the 16x6 matrix c
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x6]
    fmla v27.4s, v2.4s, v4.s[0]
    // Load second element from the 1x6 matrix b
    ldr s5, [x7]
    fmla v28.4s, v3.4s, v4.s[0]

    // Store first column back to memory
    st1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5 


    // Calculate second column of c
    fmla v17.4s, v0.4s, v5.s[0]
    add x7, x7, x4
    fmla v18.4s, v1.4s, v5.s[0]
    // Load third column from the 16x6 matrix c
    ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2]
    fmla v19.4s, v2.4s, v5.s[0]
    // Load third element from the 1x6 matrix b
    ldr s4, [x1]
    fmla v20.4s, v3.4s, v5.s[0]

    // Store second column back to memory
    st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x6], x5
    

    // Calculated third column of c
    fmla v21.4s, v0.4s, v4.s[0]
    add x1, x1, x4
    fmla v22.4s, v1.4s, v4.s[0]
    // Load fourth column from the 16x6 matrix c
    ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x6]
    fmla v23.4s, v2.4s, v4.s[0]
    // Load fourth element from the 1x6 matrix b
    ldr s5, [x7]
    fmla v24.4s, v3.4s, v4.s[0]

    // Store third column back to memory
    st1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5


    // Calculate fourth column of c
    fmla v25.4s, v0.4s, v5.s[0]
    add x7, x7, x4
    fmla v26.4s, v1.4s, v5.s[0]
    // Load fifth column from the 16x6 matrix c
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2]
    fmla v27.4s, v2.4s, v5.s[0]
    // Load fifth element from the 1x6 matrix b
    ldr s4, [x1]
    fmla v28.4s, v3.4s, v5.s[0]

    // Store fourth column back to memory
    st1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x6], x5 

    // Calculate fifth column of c
    fmla v17.4s, v0.4s, v4.s[0]
    fmla v18.4s, v1.4s, v4.s[0]
    // Load sixth column from the 16x6 matrix c
    ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x6]
    fmla v19.4s, v2.4s, v4.s[0]
    // Load sixth element from the 1x6 matrix b
    ldr s5, [x7]
    fmla v20.4s, v3.4s, v4.s[0]

    // Store fifth column back to memory
    st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2]

    // Calculated sixth column of c
    fmla v21.4s, v0.4s, v5.s[0]
    fmla v22.4s, v1.4s, v5.s[0]
    fmla v23.4s, v2.4s, v5.s[0]
    fmla v24.4s, v3.4s, v5.s[0]

    // Store sixth column back to memory
    st1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x6]

    ret
    .size matmul_16_6_1_optimized, (. - matmul_16_6_1_optimized)
