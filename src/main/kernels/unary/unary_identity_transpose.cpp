#include "unary_identity_transpose.h"
#include "../../arm_instructions/arm_all.h"
#include <stdio.h>

void identity(mini_jit::Kernel &, const mini_jit::arm_instructions::VGeneral)
{
  return;
}

void mini_jit::kernels::unary_identity_transpose(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop)
{
  internal::unary_ops_transpose(kernel, m_loop, n_loop, identity, "unary_identity_transpose.bin");
}

void mini_jit::kernels::internal::unary_ops_transpose(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop, ops_t ops,
                                                      char const *path)
{

  using namespace mini_jit::arm_instructions;
  using namespace mini_jit::kernels::internal;

  release_assert(m_loop != 0, "Cannot use a matrix with a m loop of size zero.");
  release_assert(n_loop != 0, "Cannot use a matrix with a n loop of size zero.");

  const uint32_t m_transpose_block = 4;
  const uint32_t n_transpose_block = 4;
  const uint32_t m_transpose_rest = m_loop % m_transpose_block;
  const uint32_t n_transpose_rest = n_loop % n_transpose_block;

  kernel.add({
    // /**
    //     * @param x0 = a pointer to column-major matrix A (Input). Unused for zero unary kerne.
    //     * @param x1 = b pointer to column-major matrix B (Output).
    //     * @param x2 = lda leading dimension of A. Unused for for zero unary kernel.
    //     * @param x3 = ldb leading dimension of B.

    //     // Procedural Call Standard
    //     // save frame pointer and link register
    //     // stp fp, lr, [sp, #-16]!
    //     // update frame pointer to current stack pointer
    //     // mov fp, sp

    //     // save callee-saved registers
    //     // stp x19, x20, [sp, #-16]!
    //     // stp x21, x22, [sp, #-16]!
    //     // stp x23, x24, [sp, #-16]!
    //     // stp x25, x26, [sp, #-16]!
    //     // stp x27, x28, [sp, #-16]!

    stpPre(d8, d9, sp, -16),    //     // stp  d8,  d9, [sp, #-16]!
    stpPre(d10, d11, sp, -16),  //     // stp d10, d11, [sp, #-16]!
    stpPre(d12, d13, sp, -16),  //     // stp d12, d13, [sp, #-16]!
    stpPre(d14, d15, sp, -16),  //     // stp d14, d15, [sp, #-16]!

    // Offset the used leading dimension by the size of floats
    lsl(x2, x2, 2),  // x2 * 4 = x2 * sizeof(float)
    lsl(x3, x3, 2),  // x3 * 4 = x3 * sizeof(float)

    // hold addresses to A and B in work registers for the transpose
    mov(x4, x0),  // mov x4, x0 // A for next 4 lda element in transpose (row)
    mov(x5, x1),  // mov x5, x1 // B for next 4 ldb element in transpose (row)

    mov(x6, x0),  // Holds the current block of transpose that is calculated in the matrix a
    mov(x7, x1),  // Holds the current block of transpose that is calculated in the matrix b

    mov(x8, 4),  // sizeof(float)

    mov(x10, x0),  // Holds the initial state of A matrix to offset to the next inner transpose, also need to set x4, x6
    mov(x11, x1),  // Holds the initial state of B matrix to offset to the next inner transpose, also need to set x5, x7

    mov(x12, m_transpose_block),  // Holds the m_transpose block value
    mov(x13, n_transpose_block),  // Holds the n_transpose block value
  });

  uint32_t transpose_m_loops = static_cast<int32_t>(m_loop / m_transpose_block);
  uint32_t transpose_n_loops = static_cast<int32_t>(n_loop / n_transpose_block);

  // n loop
  if (n_loop / n_transpose_block > 0)
  {
    kernel.add({
      // x16 iterator for the n_loop
      mov(x16, transpose_n_loops),
      // loop over n
      sub(x16, x16, 1),
    });

    int32_t n_jump_start = kernel.get_instruction_count() - 1;

    if (transpose_m_loops > 0)
    {
      kernel.add({

        // x17 iterator for the m_loop
        mov(x17, transpose_m_loops),
        // loop over m
        sub(x17, x17, 1),
      });

      int32_t m_jump_start = kernel.get_instruction_count() - 1;

      transpose(kernel, m_transpose_block, n_transpose_block, ops);

      kernel.add({
        // Next transpose position
        madd(x6, x8, x12, x6),  // matrix_a: x10 += m_transpose_block * sizeof(float)
        madd(x7, x3, x12, x7),  // matrix_b: x11 += ldb * m_transpose_block

        mov(x4, x6),  // Update matrix a
        mov(x5, x7),  // Update matrix b
      });

      int32_t m_jump_end = kernel.get_instruction_count();

      kernel.add(

        // loop back to m
        cbnz(x17, -(m_jump_end - m_jump_start) * 4));
    }

    // Handel the rest of the m loop
    if (m_transpose_rest > 0)
    {
      transpose(kernel, m_transpose_rest, n_transpose_block, ops);
    }

    kernel.add({
      madd(x10, x2, x13, x10),  // matrix_a: x10 += lda * n_transpose_block
      madd(x11, x8, x13, x11),  // matrix_b: x11 += n_transpose_block * sizeof(float)

      // Restore the transpose block pointers
      // matrix_a
      mov(x4, x10),
      mov(x6, x10),

      // matrix_b
      mov(x5, x11),
      mov(x7, x11),
    });

    if (transpose_m_loops > 0)
    {
      kernel.add(sub(x14, x14, 1));
    }

    int32_t n_jump_end = kernel.get_instruction_count();
    // loop back to n
    kernel.add(cbnz(x16, -(n_jump_end - n_jump_start) * 4));
  }

  if (n_transpose_rest > 0)
  {
    if (transpose_m_loops > 0)
    {
      kernel.add({

        // x17 iterator for the m_loop
        mov(x17, transpose_m_loops),
        // loop over m
        sub(x17, x17, 1),
      });

      int32_t m_jump_start = kernel.get_instruction_count() - 1;

      transpose(kernel, m_transpose_block, n_transpose_rest, ops);

      kernel.add({
        // Next transpose position
        madd(x6, x8, x12, x6),  // matrix_a: x10 += m_transpose_block * sizeof(float)
        madd(x7, x3, x12, x7),  // matrix_b: x11 += ldb * m_transpose_block

        mov(x4, x6),  // Update matrix a
        mov(x5, x7),  // Update matrix b
      });

      int32_t m_jump_end = kernel.get_instruction_count();

      kernel.add(

        // loop back to m
        cbnz(x17, -(m_jump_end - m_jump_start) * 4));
    }

    if (m_transpose_rest > 0)
    {
      transpose(kernel, m_transpose_rest, n_transpose_rest, ops);
    }
  }

  kernel.add({
    //   //     // Procedural Call Standard
    //   //     // restore callee-saved registers
    ldpPost(d14, d15, sp, 16),  //   //     // ldp d14, d15, [sp], #16
    ldpPost(d12, d13, sp, 16),  //   //     // ldp d12, d13, [sp], #16
    ldpPost(d10, d11, sp, 16),  //   //     // ldp d10, d11, [sp], #16
    ldpPost(d8, d9, sp, 16),    //   //     ldp  d8,  d9, [sp], #16

    //   //     // ldp x27, x28, [sp], #16
    //   //     // ldp x25, x26, [sp], #16
    //   //     // ldp x23, x24, [sp], #16
    //   //     // ldp x21, x22, [sp], #16
    //   //     // ldp x19, x20, [sp], #16

    //   //     // restore frame pointer and link register
    //   //     // ldp fp, lr, [sp], #16

    ret(),
  });

#ifdef SAVE_JITS_TO_FILE
  kernel.write(path);
#else
  (void)path;
#endif  // SAVE_JITS_TO_FILE
}

void mini_jit::kernels::internal::transpose(mini_jit::Kernel &kernel, const uint32_t m, const uint32_t n, ops_t ops)
{
  using namespace mini_jit::arm_instructions;

  release_assert(m > 0, "m should be larger than 0.");
  release_assert(m <= 4, "m should be less equal than 4.");
  release_assert(n > 0, "m should be larger than 0.");
  release_assert(n <= 4, "m should be less equal than 4.");

  switch (m)
  {
  case 1:
    switch (n)
    {
    case 1:  // m=1 n=1
      kernel.add(
        //    // Load
        ldr(s0, x4)  //    ldr q0, [x4]
      );
      ops(kernel, v0);
      kernel.add(
        //    // Store
        str(s0, x5)  //    str q8, [x5]
      );
      break;

    case 2:  // m=1 n=2
      kernel.add({
        //    // Load
        ldr(s0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s1, x4),      //    ldr q1, [x4]

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
      });
      ops(kernel, v8);
      kernel.add(
        //    // Store
        str(d8, x5)  //    str q8, [x5]
      );
      break;

    case 3:  // m=1 n=3
      kernel.add({
        //    // Load
        ldr(s0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s2, x4),      //    ldr q2, [x4]

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
      });
      ops(kernel, v8);
      kernel.add({
        //    // Store
        strPost(d8, x5, 2 * 4),              //    str q8, [x5]
        st1(s8, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s8
      });
      break;

    case 4:  // m=1 n=4
      kernel.add({
        //    // Load
        ldr(s0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s2, x4),      //    ldr q2, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s3, x4),      //    ldr q3, [x4]

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
      });
      ops(kernel, v8);
      kernel.add(
        //    // Store
        str(q8, x5)  //    str q8, [x5]
      );
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose axis");
      break;
    }
    break;

  case 2:
    switch (n)
    {
    case 1:  // m=2 n=1
      kernel.add(
        //    // Load
        ldr(d0, x4)  //    ldr q0, [x4]
      );
      ops(kernel, v0);
      kernel.add({

        //    // Store
        str(s0, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        st1(s0, 1, x5),   // Store the second element back
      });
      break;

    case 2:  // m=2 n=2
      kernel.add({
        //    // Load
        ldr(d0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(d1, x4),      //    ldr q1, [x4]
      });
      ops(kernel, v0);
      ops(kernel, v1);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),  //    trn2 v5.4s, v0.4s, v1.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),  //    zip1  v9.2d, v5.2d, v7.2d

        //    // Store
        str(d8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(d9, x5),      //    str q9, [x5]
      });
      break;

    case 3:  // m=2 n=3
      kernel.add({
        //    // Load
        ldr(d0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(d1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(d2, x4),      //    ldr q2, [x4]

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),  //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),  //    trn2 v7.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),  //    zip1  v9.2d, v5.2d, v7.2d

      });
      ops(kernel, v8);
      ops(kernel, v9);
      kernel.add({
        //    // Store
        strPost(d8, x5, 2 * 4),              //    str q8, [x5]
        st1(s8, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s8
        add(x5, x5, x3),                     //    add x5, x5, x3
        strPost(d9, x5, 2 * 4),              //    str q9, [x5]
        st1(s9, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s9
      });
      break;

    case 4:  // m=2 n=4
      kernel.add({
        //    // Load
        ldr(d0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(d1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(d2, x4),      //    ldr q2, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(d3, x4),      //    ldr q3, [x4]

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),  //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),  //    trn2 v7.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),  //    zip1  v9.2d, v5.2d, v7.2d
      });
      ops(kernel, v8);
      ops(kernel, v9);
      kernel.add({

        //    // Store
        str(q8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q9, x5),      //    str q9, [x5]
      });
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose axis");
      break;
    }
    break;

  case 3:
    switch (n)
    {
    case 1:  // m=3 n=1
      kernel.add({
        //    // Load
        ldrPost(d0, x4, 2 * 4),              //    ldr q0, [x4]
        ld1(s0, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s0
      });
      ops(kernel, v0);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d
        zip2(v11, t2d, v5, t2d, v7, t2d),  //    zip2 v11.2d, v5.2d, v7.2d

        //    // Store
        str(s8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(s9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(s10, x5),     //    str q10, [x5]
      });
      break;

    case 2:  // m=3 n=2
      kernel.add({
        // We transpose the same as a 4x4 but we only load and store relevent data
        //    // Load
        ldrPost(d0, x4, 2 * 4),              //    ldr q0, [x4]
        ld1(s0, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s0
        add(x4, x4, x2),                     //    add x4, x4, x2
        ldrPost(d1, x4, 2 * 4),              //    ldr q1, [x4]
        ld1(s1, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s1
      });
      ops(kernel, v0);
      ops(kernel, v1);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d

        //    // Store
        str(d8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(d9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(d10, x5),     //    str q10, [x5]
      });
      break;

    case 3:  // m=3 n=3
      kernel.add({
        // We transpose the same as a 4x4 but we only load and store relevent data
        //    // Load
        ldrPost(d0, x4, 2 * 4),              //    ldr q0, [x4]
        ld1(s0, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s0
        add(x4, x4, x2),                     //    add x4, x4, x2
        ldrPost(d1, x4, 2 * 4),              //    ldr q1, [x4]
        ld1(s1, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s1
        add(x4, x4, x2),                     //    add x4, x4, x2
        ldrPost(d2, x4, 2 * 4),              //    ldr q2, [x4]
        ld1(s2, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s2
      });
      ops(kernel, v0);
      ops(kernel, v1);
      ops(kernel, v2);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),   //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),   //    trn2 v7.4s, v2.4s, v3.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d

        //    // Store
        strPost(d8, x5, 2 * 4),               //    str q8, [x5]
        st1(s8, 2, x5), sub(x5, x5, 2 * 4),   // revert offset from load of s8
        add(x5, x5, x3),                      //    add x5, x5, x3
        strPost(d9, x5, 2 * 4),               //    str q9, [x5]
        st1(s9, 2, x5), sub(x5, x5, 2 * 4),   // revert offset from load of s9
        add(x5, x5, x3),                      //    add x5, x5, x3
        strPost(d10, x5, 2 * 4),              //    str q10, [x5]
        st1(s10, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s10
      });
      break;

    case 4:  // m=3 n=4
      kernel.add({
        //    // Load
        ldrPost(d0, x4, 2 * 4),              //    ldr q0, [x4]
        ld1(s0, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s0
        add(x4, x4, x2),                     //    add x4, x4, x2
        ldrPost(d1, x4, 2 * 4),              //    ldr q1, [x4]
        ld1(s1, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s1
        add(x4, x4, x2),                     //    add x4, x4, x2
        ldrPost(d2, x4, 2 * 4),              //    ldr q2, [x4]
        ld1(s2, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s2
        add(x4, x4, x2),                     //    add x4, x4, x2
        ldrPost(d3, x4, 2 * 4),              //    ldr q3, [x4]
        ld1(s3, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s3

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),   //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),   //    trn2 v7.4s, v2.4s, v3.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d
      });
      ops(kernel, v8);
      ops(kernel, v9);
      ops(kernel, v10);
      kernel.add({

        //    // Store
        str(q8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q10, x5),     //    str q10, [x5]
      });
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose axis");
      break;
    }
    break;

  case 4:
    switch (n)
    {
    case 1:  // m=4 n=1
      kernel.add(
        //    // Load
        ldr(q0, x4)  //    ldr q0, [x4]
      );
      ops(kernel, v0);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d
        zip2(v11, t2d, v5, t2d, v7, t2d),  //    zip2 v11.2d, v5.2d, v7.2d

        // We only need to store back 1 element
        //    // Store
        str(s8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(s9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(s10, x5),     //    str q10, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(s11, x5),     //    str q11, [x5]
      });
      break;

    case 2:  // m=4 n=2
      kernel.add({
        //    // Load
        ldr(q0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q1, x4),      //    ldr q1, [x4]
      });
      ops(kernel, v0);
      ops(kernel, v1);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d
        zip2(v11, t2d, v5, t2d, v7, t2d),  //    zip2 v11.2d, v5.2d, v7.2d

        // We only need to store back 2 elements
        //    // Store
        str(d8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(d9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(d10, x5),     //    str q10, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(d11, x5),     //    str q11, [x5]
      });
      break;

    case 3:  // m=4 n=3
      kernel.add({
        //    // Load
        ldr(q0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q2, x4),      //    ldr q2, [x4]
      });
      ops(kernel, v0);
      ops(kernel, v1);
      ops(kernel, v2);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),   //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),   //    trn2 v7.4s, v2.4s, v3.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d
        zip2(v11, t2d, v5, t2d, v7, t2d),  //    zip2 v11.2d, v5.2d, v7.2d

        // We only need to store back 3 elements
        //    // Store
        strPost(d8, x5, 2 * 4),               //    str q8, [x5]
        st1(s8, 2, x5), sub(x5, x5, 2 * 4),   // revert offset from load of s8
        add(x5, x5, x3),                      //    add x5, x5, x3
        strPost(d9, x5, 2 * 4),               //    str q9, [x5]
        st1(s9, 2, x5), sub(x5, x5, 2 * 4),   // revert offset from load of s9
        add(x5, x5, x3),                      //    add x5, x5, x3
        strPost(d10, x5, 2 * 4),              //    str q10, [x5]
        st1(s10, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s10
        add(x5, x5, x3),                      //    add x5, x5, x3
        strPost(d11, x5, 2 * 4),              //    str q11, [x5]
        st1(s11, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s11c
      });
      break;

    case 4:  // m=4 n=4
      kernel.add({
        //    // Load
        ldr(q0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q2, x4),      //    ldr q2, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q3, x4),      //    ldr q3, [x4]
      });
      ops(kernel, v0);
      ops(kernel, v1);
      ops(kernel, v2);
      ops(kernel, v3);
      kernel.add({

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),   //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),   //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),   //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),   //    trn2 v7.4s, v2.4s, v3.4s
                                           //
        zip1(v8, t2d, v4, t2d, v6, t2d),   //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),   //    zip1  v9.2d, v5.2d, v7.2d
        zip2(v10, t2d, v4, t2d, v6, t2d),  //    zip2 v10.2d, v4.2d, v6.2d
        zip2(v11, t2d, v5, t2d, v7, t2d),  //    zip2 v11.2d, v5.2d, v7.2d

        //    // Store
        str(q8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q10, x5),     //    str q10, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q11, x5),     //    str q11, [x5]
      });
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose axis");
      break;
    }
    break;

  default:
    release_assert(false, "Out of range m dimension on transpose axis");
    break;
  }
}
