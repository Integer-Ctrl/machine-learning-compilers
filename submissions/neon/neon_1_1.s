/// @brief Execute the fmla 4s instruction for throughput benchmark.
/// @param iterations (X0) The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.text
.type throughput_fmla_4s, %function
.global throughput_fmla_4s
throughput_fmla_4s:
    // Save frame pointer and link register
    stp fp, lr, [sp, #-16]!
    // Update frame pointer to current stack pointer
    mov fp, sp

    // Save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // Init SIMD registers
    eor v0.16b, v0.16b, v0.16b
    eor v1.16b, v1.16b, v1.16b
    eor v2.16b, v2.16b, v2.16b
    eor v3.16b, v3.16b, v3.16b
    eor v4.16b, v4.16b, v4.16b
    eor v5.16b, v5.16b, v5.16b
    eor v6.16b, v6.16b, v6.16b
    eor v7.16b, v7.16b, v7.16b
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b
    eor v12.16b, v12.16b, v12.16b
    eor v13.16b, v13.16b, v13.16b
    eor v14.16b, v14.16b, v14.16b
    eor v15.16b, v15.16b, v15.16b
    eor v16.16b, v16.16b, v16.16b
    eor v17.16b, v17.16b, v17.16b
    eor v18.16b, v18.16b, v18.16b
    eor v19.16b, v19.16b, v19.16b
    eor v20.16b, v20.16b, v20.16b
    eor v21.16b, v21.16b, v21.16b
    eor v22.16b, v22.16b, v22.16b
    eor v23.16b, v23.16b, v23.16b
    eor v24.16b, v24.16b, v24.16b
    eor v25.16b, v25.16b, v25.16b
    eor v26.16b, v26.16b, v26.16b
    eor v27.16b, v27.16b, v27.16b
    eor v28.16b, v28.16b, v28.16b
    eor v29.16b, v29.16b, v29.16b
    eor v30.16b, v30.16b, v30.16b
    eor v31.16b, v31.16b, v31.16b

loop_throughput_fmla_4s:
    // Iteration -= 1
    sub X0, X0, #1

    // Execute (32 * 8 * rept) instruction
    .rept 100
    fmla  v0.4s,  v8.4s, v16.4s
    fmla  v1.4s,  v9.4s, v17.4s
    fmla  v2.4s, v10.4s, v18.4s
    fmla  v3.4s, v11.4s, v19.4s

    fmla  v4.4s, v12.4s, v20.4s
    fmla  v5.4s, v13.4s, v21.4s
    fmla  v6.4s, v14.4s, v22.4s
    fmla  v7.4s, v15.4s, v23.4s

    fmla  v8.4s, v16.4s, v24.4s
    fmla  v9.4s, v17.4s, v25.4s
    fmla v10.4s, v18.4s, v26.4s
    fmla v11.4s, v19.4s, v27.4s

    fmla v12.4s, v20.4s, v28.4s
    fmla v13.4s, v21.4s, v29.4s
    fmla v14.4s, v22.4s, v30.4s
    fmla v15.4s, v23.4s, v31.4s

    fmla v16.4s, v24.4s,  v0.4s
    fmla v17.4s, v25.4s,  v1.4s
    fmla v18.4s, v26.4s,  v2.4s
    fmla v19.4s, v27.4s,  v3.4s

    fmla v20.4s, v28.4s,  v4.4s
    fmla v21.4s, v29.4s,  v5.4s
    fmla v22.4s, v30.4s,  v6.4s
    fmla v23.4s, v31.4s,  v7.4s

    fmla v24.4s,  v0.4s,  v8.4s
    fmla v25.4s,  v1.4s,  v9.4s
    fmla v26.4s,  v2.4s, v10.4s
    fmla v27.4s,  v3.4s, v11.4s

    fmla v28.4s,  v4.4s, v12.4s
    fmla v29.4s,  v5.4s, v13.4s
    fmla v30.4s,  v6.4s, v14.4s
    fmla v31.4s,  v7.4s, v15.4s
    .endr

    // Loop back if iteration != 0
    cbnz X0, loop_throughput_fmla_4s

    // Restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp fp, lr, [sp], #16

    // Return value to instructions * rept
    mov X0, #32*8*100
    ret
    .size throughput_fmla_4s, (. - throughput_fmla_4s)



/// @brief Execute the fmla 2s instruction for throughput benchmark.
/// @param iterations (X0) The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.text
.type throughput_fmla_2s, %function
.global throughput_fmla_2s
throughput_fmla_2s:
    // Save frame pointer and link register
    stp fp, lr, [sp, #-16]!
    // Update frame pointer to current stack pointer
    mov fp, sp

    // Save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // Init SIMD registers
    eor v0.16b, v0.16b, v0.16b
    eor v1.16b, v1.16b, v1.16b
    eor v2.16b, v2.16b, v2.16b
    eor v3.16b, v3.16b, v3.16b
    eor v4.16b, v4.16b, v4.16b
    eor v5.16b, v5.16b, v5.16b
    eor v6.16b, v6.16b, v6.16b
    eor v7.16b, v7.16b, v7.16b
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b
    eor v12.16b, v12.16b, v12.16b
    eor v13.16b, v13.16b, v13.16b
    eor v14.16b, v14.16b, v14.16b
    eor v15.16b, v15.16b, v15.16b
    eor v16.16b, v16.16b, v16.16b
    eor v17.16b, v17.16b, v17.16b
    eor v18.16b, v18.16b, v18.16b
    eor v19.16b, v19.16b, v19.16b
    eor v20.16b, v20.16b, v20.16b
    eor v21.16b, v21.16b, v21.16b
    eor v22.16b, v22.16b, v22.16b
    eor v23.16b, v23.16b, v23.16b
    eor v24.16b, v24.16b, v24.16b
    eor v25.16b, v25.16b, v25.16b
    eor v26.16b, v26.16b, v26.16b
    eor v27.16b, v27.16b, v27.16b
    eor v28.16b, v28.16b, v28.16b
    eor v29.16b, v29.16b, v29.16b
    eor v30.16b, v30.16b, v30.16b
    eor v31.16b, v31.16b, v31.16b

loop_throughput_fmla_2s:
    // Iteration -= 1
    sub X0, X0, #1

    // Execute (32 * 4 * rept) instruction
    .rept 100
    fmla  v0.2s,  v8.2s, v16.2s
    fmla  v1.2s,  v9.2s, v17.2s
    fmla  v2.2s, v10.2s, v18.2s
    fmla  v3.2s, v11.2s, v19.2s

    fmla  v4.2s, v12.2s, v20.2s
    fmla  v5.2s, v13.2s, v21.2s
    fmla  v6.2s, v14.2s, v22.2s
    fmla  v7.2s, v15.2s, v23.2s

    fmla  v8.2s, v16.2s, v24.2s
    fmla  v9.2s, v17.2s, v25.2s
    fmla v10.2s, v18.2s, v26.2s
    fmla v11.2s, v19.2s, v27.2s

    fmla v12.2s, v20.2s, v28.2s
    fmla v13.2s, v21.2s, v29.2s
    fmla v14.2s, v22.2s, v30.2s
    fmla v15.2s, v23.2s, v31.2s

    fmla v16.2s, v24.2s,  v0.2s
    fmla v17.2s, v25.2s,  v1.2s
    fmla v18.2s, v26.2s,  v2.2s
    fmla v19.2s, v27.2s,  v3.2s

    fmla v20.2s, v28.2s,  v4.2s
    fmla v21.2s, v29.2s,  v5.2s
    fmla v22.2s, v30.2s,  v6.2s
    fmla v23.2s, v31.2s,  v7.2s

    fmla v24.2s,  v0.2s,  v8.2s
    fmla v25.2s,  v1.2s,  v9.2s
    fmla v26.2s,  v2.2s, v10.2s
    fmla v27.2s,  v3.2s, v11.2s

    fmla v28.2s,  v4.2s, v12.2s
    fmla v29.2s,  v5.2s, v13.2s
    fmla v30.2s,  v6.2s, v14.2s
    fmla v31.2s,  v7.2s, v15.2s
    .endr

    // Loop back if iteration != 0
    cbnz X0, loop_throughput_fmla_2s

    // Restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp fp, lr, [sp], #16

    // Return value to instructions * rept
    mov X0, #32*4*100
    ret
    .size throughput_fmla_2s, (. - throughput_fmla_2s)



/// @brief Execute the fmadd instruction for throughput benchmark.
/// @param iterations (X0) The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.text
.type throughput_fmadd, %function
.global throughput_fmadd
throughput_fmadd:
    // Save frame pointer and link register
    stp fp, lr, [sp, #-16]!
    // Update frame pointer to current stack pointer
    mov fp, sp

    // Save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // Init SIMD registers
    eor v0.16b, v0.16b, v0.16b
    eor v1.16b, v1.16b, v1.16b
    eor v2.16b, v2.16b, v2.16b
    eor v3.16b, v3.16b, v3.16b
    eor v4.16b, v4.16b, v4.16b
    eor v5.16b, v5.16b, v5.16b
    eor v6.16b, v6.16b, v6.16b
    eor v7.16b, v7.16b, v7.16b
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b
    eor v12.16b, v12.16b, v12.16b
    eor v13.16b, v13.16b, v13.16b
    eor v14.16b, v14.16b, v14.16b
    eor v15.16b, v15.16b, v15.16b
    eor v16.16b, v16.16b, v16.16b
    eor v17.16b, v17.16b, v17.16b
    eor v18.16b, v18.16b, v18.16b
    eor v19.16b, v19.16b, v19.16b
    eor v20.16b, v20.16b, v20.16b
    eor v21.16b, v21.16b, v21.16b
    eor v22.16b, v22.16b, v22.16b
    eor v23.16b, v23.16b, v23.16b
    eor v24.16b, v24.16b, v24.16b
    eor v25.16b, v25.16b, v25.16b
    eor v26.16b, v26.16b, v26.16b
    eor v27.16b, v27.16b, v27.16b
    eor v28.16b, v28.16b, v28.16b
    eor v29.16b, v29.16b, v29.16b
    eor v30.16b, v30.16b, v30.16b
    eor v31.16b, v31.16b, v31.16b

loop_throughput_fmadd:
    // Iteration -= 1
    sub X0, X0, #1

    // Execute (32 * 2 * rept) instruction
    .rept 100
    fmadd s0, s8, s9, s10
    fmadd s1, s11, s12, s13
    fmadd s2, s14, s15, s16
    fmadd s3, s17, s18, s19

    fmadd s4, s20, s21, s22
    fmadd s5, s23, s24, s25
    fmadd s6, s26, s27, s28
    fmadd s7, s29, s30, s31

    fmadd s8, s0, s1, s2
    fmadd s9, s3, s4, s5
    fmadd s10, s6, s7, s8
    fmadd s11, s9, s10, s11

    fmadd s12, s12, s13, s14
    fmadd s13, s14, s15, s16
    fmadd s14, s17, s18, s19
    fmadd s15, s20, s21, s22

    fmadd s16, s23, s24, s25
    fmadd s17, s26, s27, s28
    fmadd s18, s29, s30, s31
    fmadd s19, s0, s1, s2

    fmadd s20, s3, s4, s5
    fmadd s21, s6, s7, s8
    fmadd s22, s9, s10, s11
    fmadd s23, s12, s13, s14

    fmadd s24, s15, s16, s17
    fmadd s25, s18, s19, s20
    fmadd s26, s21, s22, s23
    fmadd s27, s24, s25, s26

    fmadd s28, s27, s28, s29
    fmadd s29, s30, s31, s0
    fmadd s30, s1, s2, s3
    fmadd s31, s4, s5, s6
    .endr

    // loop back if iteration != 0
    cbnz X0, loop_throughput_fmadd

    // Restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp fp, lr, [sp], #16

    // Return value to instructions * rept
    mov X0, #32*2*100
    ret
    .size throughput_fmadd, (. - throughput_fmadd)
