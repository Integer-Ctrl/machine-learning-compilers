
/// @brief Execute the add instruction for throughput benchmark.
/// @param iterations (X0) The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.text
.type throughput_add, %function
.global throughput_add
throughput_add:
    // save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    mov X27, #12
    mov X28, #25

loop_throughput_add:
    sub X0, X0, #1 // iteration -= 1

    // execute (25 * rept) add instruction for throughput test
    .rept 100
    add X1, X27, X28
    add X2, X27, X28
    add X3, X27, X28
    add X4, X27, X28
    add X5, X27, X28
    
    add X6, X27, X28
    add X7, X27, X28
    add X8, X27, X28
    add X9, X27, X28
    add X10, X27, X28

    add X11, X27, X28
    add X12, X27, X28
    add X13, X27, X28
    add X14, X27, X28
    add X15, X27, X28

    add X16, X27, X28
    add X17, X27, X28 // Ignore X18 because its platform register
    add X19, X27, X28
    add X20, X27, X28
    add X21, X27, X28

    add X22, X27, X28
    add X23, X27, X28
    add X24, X27, X28
    add X25, X27, X28
    add X26, X27, X28
    .endr

    // loop back if iteration != 0
    cbnz X0, loop_throughput_add

    // restore callee-saved registers
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    mov X0, #25*100 // set return value to instructions * rept
    ret
    .size throughput_add, (. - throughput_add)

/// @brief Execute the mul instruction for throughput benchmark.
/// @param iterations The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.type throughput_mul, %function
.global throughput_mul
throughput_mul:
    // save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    mov X27, #12
    mov X28, #25

loop_throughput_mul:
    sub X0, X0, #1 // iteration -= 1

    // execute (25 * rept) mul instruction for throughput test
    .rept 100
    mul X1, X27, X28
    mul X2, X27, X28
    mul X3, X27, X28
    mul X4, X27, X28
    mul X5, X27, X28
    
    mul X6, X27, X28
    mul X7, X27, X28
    mul X8, X27, X28
    mul X9, X27, X28
    mul X10, X27, X28

    mul X11, X27, X28
    mul X12, X27, X28
    mul X13, X27, X28
    mul X14, X27, X28
    mul X15, X27, X28

    mul X16, X27, X28
    mul X17, X27, X28 // Ignore X18 because its platform register
    mul X19, X27, X28
    mul X20, X27, X28
    mul X21, X27, X28

    mul X22, X27, X28
    mul X23, X27, X28
    mul X24, X27, X28
    mul X25, X27, X28
    mul X26, X27, X28
    .endr

    // loop back if iteration != 0
    cbnz X0, loop_throughput_mul

    // restore callee-saved registers
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    mov X0, #25*100 // set return value to instructions * rept 
    ret
    .size throughput_mul, (. - throughput_mul)

/// @brief Execute the add instruction for latency benchmarks.
/// @param iterations The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.type latency_add, %function
.global latency_add
latency_add:
    mov X1, #25
    mov X2, #45

loop_latency_add:
    sub X0, X0, #1 // iterations -= 1

    // Benchmark the latency using read-after-write using (5 * rept) Instructions
    .rept 5*100
    add X1, X1, X2
    add X1, X1, X2
    add X1, X1, X2
    add X1, X1, X2
    add X1, X1, X2
    .endr

    cbnz X0, loop_latency_add
    
    mov X0, #5*5*100 // set return value to instructions * rept
    ret
    .size latency_add, (. - latency_add)

/// @brief Execute the mul instruction for latency benchmarks.
/// @param iterations The number of iterations the instructions are run.
/// @return The number of processed instructions in a single loop.
.type latency_mul, %function
.global latency_mul
latency_mul:
    mov X1, #25
    mov X2, #45

loop_latency_mul:
    sub X0, X0, #1 // iterations -= 1

    // Benchmark the latency using read-after-write using (5 * rept) Instructions
    .rept 5*100
    mul X1, X1, X2
    mul X1, X1, X2
    mul X1, X1, X2
    mul X1, X1, X2
    mul X1, X1, X2
    .endr

    cbnz X0, loop_latency_mul
    
    mov X0, #5*5*100 // set return value to instructions * rept
    ret
    .size latency_mul, (. - latency_mul)
