#include "unary_identity_transpose.h"
#include "../../arm_instructions/arm_all.h"

void mini_jit::kernels::unary_identity_transpose(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop)
{
  using namespace mini_jit::arm_instructions;

  release_assert(m_loop != 0, "Cannot use a matrix with a m loop of size zero.");
  release_assert(n_loop != 0, "Cannot use a matrix with a n loop of size zero.");

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

    //     // stp  d8,  d9, [sp, #-16]!
    //     // stp d10, d11, [sp, #-16]!
    //     // stp d12, d13, [sp, #-16]!
    //     // stp d14, d15, [sp, #-16]!

    // Offset the used leading dimension by the size of floats
    lsl(x2, x2, 2),  // x2 * 4 = x2 * sizeof(float)
    lsl(x3, x3, 2),  // x3 * 4 = x3 * sizeof(float)

    // hold addresses to A and B in work registers for the transpose
    mov(x4, x0),  // mov x4, x0 // A for next 4 lda element in transpose (row)
    mov(x5, x1),  // mov x5, x1 // B for next 4 ldb element in transpose (row)
    mov(x6, x0),  // A for next 4 consecutive element in transpose (column)
    mov(x7, x1),  // B for next 4 consecutive element in transposes (column)

    // LOCKED mov(x9, 0), // Used as a temporary register, always set the specific size of this register before using it in any context

    mov(x7, x0),  // Store the inital value of x0, to be restored in the N loop
    mov(x8, x1),  // Store the inital value of x1, to be restored in the N loop

    // x16 iterator for the n_loop
    mov(x16, n_loop),
    // loop over n
    sub(x16, x16, 1),

    mov(x0, x7),  // Restore x0 for the m loop
    mov(x1, x8),  // Restore x1 for the m loop

  });

  int32_t n_jump_start = kernel.get_instruction_count() - 3;

  if (m_loop < 4)
  {
    kernel.add({
      // x17 iterator for the m_loop
      mov(x17, m_loop / 4),
      // loop over m
      sub(x17, x17, 1),

      // loop back to m
      cbnz(x17, -3 * 4),
    });
  }

  uint32_t m_loop_rest = m_loop % 4;
  // Handel the rest of m
  if (m_loop_rest != 0)
  {
    uint32_t m_loop_rest_multiple_4 = m_loop_rest / 4;
    switch (m_loop_rest_multiple_4)
    {
    case 1:
      break;

    case 2:
      break;

    case 3:
      break;

    default:
      release_assert(false, "Out of range loop rest detected for multiple of 4 instructions.");
      break;
    }
  }

  int32_t n_jump_end = kernel.get_instruction_count() + 2;

  kernel.add({
    add(x7, x2, x7),  // lda + initial position
    add(x8, x3, x8),  // ldb + initial position

    // loop back to n
    cbnz(x16, -(n_jump_end - n_jump_start) * 4),

    //   //     // Procedural Call Standard
    //   //     // restore callee-saved registers
    //   //     // ldp d14, d15, [sp], #16
    //   //     // ldp d12, d13, [sp], #16
    //   //     // ldp d10, d11, [sp], #16
    //   //     ldp  d8,  d9, [sp], #16

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
  kernel.write("unary_identity.bin");
#endif  // SAVE_JITS_TO_FILE
}

void mini_jit::kernels::internal::transpose_axis(mini_jit::Kernel &kernel, const uint32_t m, const uint32_t n)
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
      kernel.add({
        //    // Load
        ldr(s0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Store
        str(s0, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
      });
      break;

    case 2:  // m=1 n=2
      kernel.add({
        //    // Load
        ldr(s0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d

        //    // Store
        str(d8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
      });
      break;

    case 3:  // m=1 n=3
      kernel.add({
        //    // Load
        ldr(s0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(s2, x4),      //    ldr q2, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d

        //    // Store
        strPost(d8, x5, 2 * 4),              //    str q8, [x5]
        st1(s8, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s8
        add(x5, x5, x3),                     //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d

        //    // Store
        str(q8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
      });
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
      kernel.add({
        //    // Load
        ldr(d0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Store
        str(s0, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        st1(s0, 1, x5),   // Store the second element back
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
      });
      break;

    case 2:  // m=2 n=2
      kernel.add({
        //    // Load
        ldr(d0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(d1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

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
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),  //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),  //    trn2 v7.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),  //    zip1  v9.2d, v5.2d, v7.2d

        //    // Store
        strPost(d8, x5, 2 * 4),              //    str q8, [x5]
        st1(s8, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s8
        add(x5, x5, x3),                     //    add x5, x5, x3
        strPost(d9, x5, 2 * 4),              //    str q9, [x5]
        st1(s9, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s9
        add(x5, x5, x3),                     //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Transpose
        trn1(v4, t4s, v0, t4s, v1, t4s),  //    trn1 v4.4s, v0.4s, v1.4s
        trn2(v5, t4s, v0, t4s, v1, t4s),  //    trn2 v5.4s, v0.4s, v1.4s
        trn1(v6, t4s, v2, t4s, v3, t4s),  //    trn1 v6.4s, v2.4s, v3.4s
        trn2(v7, t4s, v2, t4s, v3, t4s),  //    trn2 v7.4s, v2.4s, v3.4s
                                          //
        zip1(v8, t2d, v4, t2d, v6, t2d),  //    zip1  v8.2d, v4.2d, v6.2d
        zip1(v9, t2d, v5, t2d, v7, t2d),  //    zip1  v9.2d, v5.2d, v7.2d

        //    // Store
        str(q8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),                     //    add x4, x4, x2

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
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),                     //    add x4, x4, x2

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
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),                     //    add x4, x4, x2

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
        add(x5, x5, x3),                      //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        ldrPost(q3, x4, 2 * 4),              //    ldr q3, [x4]
        ld1(s3, 2, x4), sub(x4, x4, 2 * 4),  // revert offset from load of s3
        add(x4, x4, x2),                     //    add x4, x4, x2

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
        str(q8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q10, x5),     //    str q10, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
      kernel.add({
        //    // Load
        ldr(q0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

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
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
      });
      break;

    case 2:  // m=4 n=2
      kernel.add({
        //    // Load
        ldr(q0, x4),      //    ldr q0, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q1, x4),      //    ldr q1, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

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
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),  //    add x4, x4, x2

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
        strPost(q10, x5, 2 * 4),              //    str q10, [x5]
        st1(s10, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s10
        add(x5, x5, x3),                      //    add x5, x5, x3
        strPost(q11, x5, 2 * 4),              //    str q11, [x5]
        st1(s11, 2, x5), sub(x5, x5, 2 * 4),  // revert offset from load of s11c
        add(x5, x5, x3),                      //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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
        add(x4, x4, x2),  //    add x4, x4, x2

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
        add(x5, x5, x3),  //    add x5, x5, x3

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
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

void mini_jit::kernels::internal::transpose_else(mini_jit::Kernel &kernel, const uint32_t m, const uint32_t n)
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
      break;

    case 2:  // m=1 n=2
      break;

    case 3:  // m=1 n=3
      break;

    case 4:  // m=1 n=4
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose else");
      break;
    }
    break;

  case 2:
    switch (n)
    {
    case 1:  // m=2 n=1
      break;

    case 2:  // m=2 n=2
      break;

    case 3:  // m=2 n=3
      break;

    case 4:  // m=2 n=4
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose else");
      break;
    }
    break;

  case 3:
    switch (n)
    {
    case 1:  // m=3 n=1
      break;

    case 2:  // m=3 n=2
      break;

    case 3:  // m=3 n=3
      break;

    case 4:  // m=3 n=4
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose else");
      break;
    }
    break;

  case 4:
    switch (n)
    {
    case 1:  // m=4 n=1
      break;

    case 2:  // m=4 n=2
      break;

    case 3:  // m=4 n=3
      break;

    case 4:  // m=4 n=4
      kernel.add({
        //    // Load right-top
        ldr(q12, x4),     //    ldr q12, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q13, x4),     //    ldr q13, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q14, x4),     //    ldr q14, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2
        ldr(q15, x4),     //    ldr q15, [x4]
        add(x4, x4, x2),  //    add x4, x4, x2

        //    // Transpose right-top
        trn1(v16, t4s, v12, t4s, v14, t4s),  //    trn1 v16.4s, v12.4s, v14.4s
        trn1(v17, t4s, v13, t4s, v15, t4s),  //    trn1 v17.4s, v13.4s, v15.4s
        trn2(v18, t4s, v12, t4s, v14, t4s),  //    trn2 v18.4s, v12.4s, v14.4s
        trn2(v19, t4s, v13, t4s, v15, t4s),  //    trn2 v19.4s, v13.4s, v15.4s
                                             //
        zip1(v20, t4s, v16, t4s, v17, t4s),  //    zip1 v20.4s, v16.4s, v17.4s
        zip1(v21, t4s, v18, t4s, v19, t4s),  //    zip1 v21.4s, v18.4s, v19.4s
        zip2(v22, t4s, v16, t4s, v17, t4s),  //    zip2 v22.4s, v16.4s, v17.4s
        zip2(v23, t4s, v18, t4s, v19, t4s),  //    zip2 v23.4s, v18.4s, v19.4s

        //    // Load left-bottom
        ldr(q0, x6),                        //    ldr q0, [x4]
        add(x6, x6, x2),                    //    add x4, x4, x2
        ldr(q1, x6),                        //    ldr q1, [x4]
        add(x6, x6, x2),                    //    add x4, x4, x2
        ldr(q2, x6),                        //    ldr q2, [x4]
        add(x6, x6, x2),                    //    add x4, x4, x2
        ldr(q3, x6),                        //    ldr q3, [x4]
        mov(x9, -3), madd(x6, x2, x9, x6),  // Revert store offset

        //    // Transpose left-bottom
        trn1(v4, t4s, v0, t4s, v2, t4s),   //    trn1 v4.4s, v0.4s, v2.4s
        trn1(v5, t4s, v1, t4s, v3, t4s),   //    trn1 v5.4s, v1.4s, v3.4s
        trn2(v6, t4s, v0, t4s, v2, t4s),   //    trn2 v6.4s, v0.4s, v2.4s
        trn2(v7, t4s, v1, t4s, v3, t4s),   //    trn2 v7.4s, v1.4s, v3.4s
                                           //
        zip1(v8, t4s, v4, t4s, v5, t4s),   //    zip1 v8.4s, v4.4s, v5.4s
        zip1(v9, t4s, v6, t4s, v7, t4s),   //    zip1 v9.4s, v6.4s, v7.4s
        zip2(v10, t4s, v4, t4s, v5, t4s),  //    zip2 v10.4s, v4.4s, v5.4s
        zip2(v11, t4s, v6, t4s, v7, t4s),  //    zip2 v11.4s, v6.4s, v7.4s

        //    // Store after transpose to avoid conflicts when input matrix A = B
        //    // Store B to C (right-top of A to left-bottom of B)
        str(q20, x7),                       //    str q20, [x7]
        add(x7, x7, x3),                    //    add x7, x7, x3
        str(q21, x7),                       //    str q21, [x7]
        add(x7, x7, x3),                    //    add x7, x7, x3
        str(q22, x7),                       //    str q22, [x7]
        add(x7, x7, x3),                    //    add x7, x7, x3
        str(q23, x7),                       //    str q23, [x7]
        mov(x9, -3), madd(x7, x3, x9, x7),  //  Revert store offset

        //    // Store C to B (left-bottom of A to right-top of B)
        str(q8, x5),      //    str q8, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q9, x5),      //    str q9, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q10, x5),     //    str q10, [x5]
        add(x5, x5, x3),  //    add x5, x5, x3
        str(q11, x5),     //    str q11, [x5]

        // Offset the consecutive elements
        add(x6, x6, m * 4),  // offset 4 * sizeof(float)
        add(x7, x7, n * 4),  // offset 4 * sizeof(float)
      });
      break;

    default:
      release_assert(false, "Out of range n dimension on transpose else");
      break;
    }
    break;

  default:
    release_assert(false, "Out of range m dimension on transpose axis");
    break;
  }
}
