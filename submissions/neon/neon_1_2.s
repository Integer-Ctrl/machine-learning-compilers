/// @brief Execute the fmla 4s instruction for latency benchmarks with dependency on one of the source registers.
/// @param iterations The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.text
.type latency_fmla_4s_source, %function
.global latency_fmla_4s_source
latency_fmla_4s_source:
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

loop_latency_fmla_4s_source:
    // Iteration -= 1
    sub X0, X0, #1

    // Execute (32 * 8 * rept) instruction
    .rept 100
    fmla v0.4s, v0.4s,  v1.4s
    fmla v0.4s, v0.4s,  v2.4s
    fmla v0.4s, v0.4s,  v3.4s
    fmla v0.4s, v0.4s,  v4.4s

    fmla v0.4s, v0.4s,  v5.4s
    fmla v0.4s, v0.4s,  v6.4s
    fmla v0.4s, v0.4s,  v7.4s
    fmla v0.4s, v0.4s,  v8.4s

    fmla v0.4s, v0.4s,  v9.4s
    fmla v0.4s, v0.4s, v10.4s
    fmla v0.4s, v0.4s, v11.4s
    fmla v0.4s, v0.4s, v12.4s

    fmla v0.4s, v0.4s, v13.4s
    fmla v0.4s, v0.4s, v14.4s
    fmla v0.4s, v0.4s, v15.4s
    fmla v0.4s, v0.4s, v16.4s

    fmla v0.4s, v0.4s, v17.4s
    fmla v0.4s, v0.4s, v18.4s
    fmla v0.4s, v0.4s, v19.4s
    fmla v0.4s, v0.4s, v20.4s

    fmla v0.4s, v0.4s, v21.4s
    fmla v0.4s, v0.4s, v22.4s
    fmla v0.4s, v0.4s, v23.4s
    fmla v0.4s, v0.4s, v24.4s

    fmla v0.4s, v0.4s, v25.4s
    fmla v0.4s, v0.4s, v26.4s
    fmla v0.4s, v0.4s, v27.4s
    fmla v0.4s, v0.4s, v28.4s

    fmla v0.4s, v0.4s, v29.4s
    fmla v0.4s, v0.4s, v30.4s
    fmla v0.4s, v0.4s, v31.4s
    .endr

    // Loop back if iteration != 0
    cbnz X0, loop_latency_fmla_4s_source

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
    .size latency_fmla_4s_source, (. - latency_fmla_4s_source)



/// @brief Execute the fmla 4s instruction for latency benchmarks with dependency on the destination register.
/// @param iterations The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.text
.type latency_fmla_4s_destination, %function
.global latency_fmla_4s_destination
latency_fmla_4s_destination:
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

loop_latency_fmla_4s_destination:
    // Iteration -= 1
    sub X0, X0, #1

    // Execute (32 * 8 * rept) instruction
    .rept 100
    fmla v0.4s,  v1.4s,  v9.4s
    fmla v0.4s,  v2.4s, v10.4s
    fmla v0.4s,  v3.4s, v11.4s
    fmla v0.4s,  v4.4s, v12.4s

    fmla v0.4s,  v5.4s, v13.4s
    fmla v0.4s,  v6.4s, v14.4s
    fmla v0.4s,  v7.4s, v15.4s
    fmla v0.4s,  v8.4s, v16.4s

    fmla v0.4s,  v9.4s, v17.4s
    fmla v0.4s, v10.4s, v18.4s
    fmla v0.4s, v11.4s, v19.4s
    fmla v0.4s, v12.4s, v20.4s

    fmla v0.4s, v13.4s, v21.4s
    fmla v0.4s, v14.4s, v22.4s
    fmla v0.4s, v15.4s, v23.4s
    fmla v0.4s, v16.4s, v24.4s

    fmla v0.4s, v17.4s, v25.4s
    fmla v0.4s, v18.4s, v26.4s
    fmla v0.4s, v19.4s, v27.4s
    fmla v0.4s, v20.4s, v28.4s

    fmla v0.4s, v21.4s, v29.4s
    fmla v0.4s, v22.4s, v30.4s
    fmla v0.4s, v23.4s, v31.4s
    fmla v0.4s, v24.4s,  v1.4s

    fmla v0.4s, v25.4s,  v2.4s
    fmla v0.4s, v26.4s,  v3.4s
    fmla v0.4s, v27.4s,  v4.4s
    fmla v0.4s, v28.4s,  v5.4s

    fmla v0.4s, v29.4s,  v6.4s
    fmla v0.4s, v30.4s,  v7.4s
    fmla v0.4s, v31.4s,  v8.4s
    .endr

    // Loop back if iteration != 0
    cbnz X0, loop_latency_fmla_4s_destination

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
    .size latency_fmla_4s_destination, (. - latency_fmla_4s_destination)
