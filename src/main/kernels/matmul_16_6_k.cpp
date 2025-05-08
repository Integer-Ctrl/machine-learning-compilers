#include "matmul_16_6_k.h"
#include "../arm_instructions/arm_all.h"
#include "../Brgemm.h"
#include "../Kernel.h"


void mini_jit::kernels::matmul_16_6_k(mini_jit::Kernel &kernel, const uint32_t k_loop)
{
  using namespace mini_jit::arm_instructions;

  // Procedural Call Standard
  // save frame pointer and link register
  kernel.add({

    stpPre(fp, lr, sp, -16),  // stp fp, lr, [sp, #-16]!
    // update frame pointer to current stack pointer
    movSp(fp, sp),  // mov fp, sp
        
    // save callee-saved registers
    stpPre(x19, x20, sp, -16),  // stp x19, x20, [sp, #-16]!
    stpPre(x21, x22, sp, -16),  // stp x21, x22, [sp, #-16]!
    stpPre(x23, x24, sp, -16),  // stp x23, x24, [sp, #-16]!
    stpPre(x25, x26, sp, -16),  // stp x25, x26, [sp, #-16]!
    stpPre(x27, x28, sp, -16),  // stp x27, x28, [sp, #-16]!

    stpPre(d8, d9, sp, -16),  // stp  d8,  d9, [sp, #-16]!
    stpPre(d10, d11, sp, -16),  // stp d10, d11, [sp, #-16]!
    stpPre(d12, d13, sp, -16),  // stp d12, d13, [sp, #-16]!
    stpPre(d14, d15, sp, -16),  // stp d14, d15, [sp, #-16]!

    // Offset the used leading dimension by the size of floats
    lsl(x3, x3, 2),  // lsl x3, x3, #2
    lsl(x4, x4, 2),  // lsl x4, x4, #2
    lsl(x5, x5, 2),  // lsl x5, x5, #2

    mov(x6, x1),  // mov x6, x1
    mov(x7, x2),  // mov x7, x2

    // Load first column from the 16x6 matrix c
    ld1Post(v25, t4s, v26, t4s, v27, t4s, v28, t4s, x2, x5),  // ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5
    // Load second column from the 16x6 matrix c
    ld1Post(v17, t4s, v18, t4s, v19, t4s, v20, t4s, x2, x5),  // ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5
    // Load third column from the 16x6 matrix c
    ld1Post(v21, t4s, v22, t4s, v23, t4s, v24, t4s, x2, x5),  // ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
    // Load fourth column from the 16x6 matrix c
    ld1Post(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x2, x5),  // ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x2], x5
    // Load fifth column from the 16x6 matrix c
    ld1Post(v9, t4s, v10, t4s, v11, t4s, v12, t4s, x2, x5),  // ld1 {v9.4s, v10.4s, v11.4s, v12.4s}, [x2], x5
    // Load sixth column from the 16x6 matrix c
    ld1Post(v13, t4s, v14, t4s, v15, t4s, v16, t4s, x2, x5),  // ld1 {v13.4s, v14.4s, v15.4s, v16.4s}, [x2], x5

    movz(x9, k_loop),  // mov x9, "iterator for K loop"
    
    // #############################
    // #### matmul_loop_over_K: ####
    // #############################
    sub(x9, x9, 1),  // sub x9, x9, #1

    // Load first column data from the 16x1 matrix a
    ld1Post(v0, t4s, v1, t4s, v2, t4s, v3, t4s, x0, x3),  // ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x3

    // run the known matmul_16_6_1_unrolled kernel
    // Load first element from the 1x6 matrix b
    ldr(s4, x1),  // ldr s4, [x1]
    add(x1, x1, x4),  // add x1, x1, x4

    // Calculate first column of c
    fmla(v25, t4s, v0, t4s, v4, 0),  // fmla v25.4s, v0.4s, v4.s[0]
    fmla(v26, t4s, v1, t4s, v4, 0),  // fmla v26.4s, v1.4s, v4.s[0]
    fmla(v27, t4s, v2, t4s, v4, 0),  // fmla v27.4s, v2.4s, v4.s[0]
    fmla(v28, t4s, v3, t4s, v4, 0),  // fmla v28.4s, v3.4s, v4.s[0]


    // Load second element from the 1x6 matrix b
    ldr(s4, x1),  // ldr s4, [x1]
    add(x1, x1, x4),  // add x1, x1, x4

    // Calculate second column of c
    fmla(v17, t4s, v0, t4s, v4, 0),  // fmla v17.4s, v0.4s, v4.s[0]
    fmla(v18, t4s, v1, t4s, v4, 0),  // fmla v18.4s, v1.4s, v4.s[0]
    fmla(v19, t4s, v2, t4s, v4, 0),  // fmla v19.4s, v2.4s, v4.s[0]
    fmla(v20, t4s, v3, t4s, v4, 0),  // fmla v20.4s, v3.4s, v4.s[0]

        
    // Load third element from the 1x6 matrix b
    ldr(s4, x1),  // ldr s4, [x1]
    add(x1, x1, x4),  // add x1, x1, x4

    // Calculated third column of c
    fmla(v21, t4s, v0, t4s, v4, 0),  // fmla v21.4s, v0.4s, v4.s[0]
    fmla(v22, t4s, v1, t4s, v4, 0),  // fmla v22.4s, v1.4s, v4.s[0]
    fmla(v23, t4s, v2, t4s, v4, 0),  // fmla v23.4s, v2.4s, v4.s[0]
    fmla(v24, t4s, v3, t4s, v4, 0),  // fmla v24.4s, v3.4s, v4.s[0]


    // Load fourth element from the 1x6 matrix b
    ldr(s4, x1),  // ldr s4, [x1]
    add(x1, x1, x4),  // add x1, x1, x4

    // Calculate fourth column of c
    fmla(v5, t4s, v0, t4s, v4, 0),  // fmla v5.4s, v0.4s, v4.s[0]
    fmla(v6, t4s, v1, t4s, v4, 0),  // fmla v6.4s, v1.4s, v4.s[0]
    fmla(v7, t4s, v2, t4s, v4, 0),  // fmla v7.4s, v2.4s, v4.s[0]
    fmla(v8, t4s, v3, t4s, v4, 0),  // fmla v8.4s, v3.4s, v4.s[0]


    // Load fifth element from the 1x6 matrix b
    ldr(s4, x1),  // ldr s4, [x1]
    add(x1, x1, x4),  // add x1, x1, x4

    // Calculate fifth column of c
    fmla(v9, t4s, v0, t4s, v4, 0),  // fmla v9.4s, v0.4s, v4.s[0]
    fmla(v10, t4s, v1, t4s, v4, 0),  // fmla v10.4s, v1.4s, v4.s[0]
    fmla(v11, t4s, v2, t4s, v4, 0),  // fmla v11.4s, v2.4s, v4.s[0]
    fmla(v12, t4s, v3, t4s, v4, 0),  // fmla v12.4s, v3.4s, v4.s[0]

        
    // Load sixth element from the 1x6 matrix b
    ldr(s4, x1),  // ldr s4, [x1]
    add(x1, x1, x4),  // add x1, x1, x4

    // Calculated sixth column of c
    fmla(v13, t4s, v0, t4s, v4, 0),  // fmla v13.4s, v0.4s, v4.s[0]
    fmla(v14, t4s, v1, t4s, v4, 0),  // fmla v14.4s, v1.4s, v4.s[0]
    fmla(v15, t4s, v2, t4s, v4, 0),  // fmla v15.4s, v2.4s, v4.s[0]
    fmla(v16, t4s, v3, t4s, v4, 0),  // fmla v16.4s, v3.4s, v4.s[0]


    // offset x6 to the next element in the column
    add(x6, x6, 4),  // add x6, x6, #4 // #4 = sizeof(float)

    // Restore x1 to be incremented again
    mov(x1, x6),  // mov x1, x6

    // Loop back
    cbnz(x9, -40*4),  // cbnz x9, matmul_loop_over_K

    // Restore initial value of x2 that was changed by the loads
    mov(x2, x7),  // mov x2, x7

    // Store first column back to memory
    st1Post(v25, t4s, v26, t4s, v27, t4s, v28, t4s, x2, x5),  // st1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5 
    // Store second column back to memory
    st1Post(v17, t4s, v18, t4s, v19, t4s, v20, t4s, x2, x5),  // st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5
    // Store third column back to memory
    st1Post(v21, t4s, v22, t4s, v23, t4s, v24, t4s, x2, x5),  // st1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
    // Store fourth column back to memory
    st1Post(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x2, x5),  // st1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x2], x5 
    // Store fifth column back to memory
    st1Post(v9, t4s, v10, t4s, v11, t4s, v12, t4s, x2, x5),  // st1 {v9.4s, v10.4s, v11.4s, v12.4s}, [x2], x5
    // Store sixth column back to memory
    st1Post(v13, t4s, v14, t4s, v15, t4s, v16, t4s, x2, x5),  // st1 {v13.4s, v14.4s, v15.4s, v16.4s}, [x2], x5

    // Procedural Call Standard
    // restore callee-saved registers
    ldpPost(d14, d15, sp, 16),  // ldp d14, d15, [sp], #16
    ldpPost(d12, d13, sp, 16),  // ldp d12, d13, [sp], #16
    ldpPost(d10, d11, sp, 16),  // ldp d10, d11, [sp], #16
    ldpPost(d8, d9, sp, 16),  // ldp  d8,  d9, [sp], #16

    ldpPost(x27, x28, sp, 16),  // ldp x27, x28, [sp], #16
    ldpPost(x25, x26, sp, 16),  // ldp x25, x26, [sp], #16
    ldpPost(x23, x24, sp, 16),  // ldp x23, x24, [sp], #16
    ldpPost(x21, x22, sp, 16),  // ldp x21, x22, [sp], #16
    ldpPost(x19, x20, sp, 16),  // ldp x19, x20, [sp], #16

    // restore frame pointer and link register
    ldpPost(fp, lr, sp, 16),  // ldp fp, lr, [sp], #16

    ret()  // ret
  });

  kernel.write("matmul_16_6_k.bin");
}