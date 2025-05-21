#include "unary_zero_4m_n.h"
#include "../../arm_instructions/arm_all.h"

void mini_jit::kernels::unary_zero_4m_n(mini_jit::Kernel &kernel, const uint32_t m_loop_4, const uint32_t n_loop,
                                        const bool use_init_and_end)
{
  using namespace mini_jit::arm_instructions;

  release_assert(m_loop_4 != 0, "Cannot use a matrix with a m loop of size zero.");
  release_assert(n_loop != 0, "Cannot use a matrix with a n loop of size zero.");

  if (use_init_and_end)
  {

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
      lsl(x3, x3, 2),  // x3 * 4 = x3 * sizeof(float)

      mov(x8, x1),      // Store the inital value of x1, to be restored in the N loop
      mov(x9, 4 * 16),  // 4 * 16Byte Hold the number of bytes that are stored in the loop

      // Zero four register so we can fill the matrix with zeros
      eor(v0, t16b, v0, t16b, v0, t16b),  // Zero the v0 register
      eor(v1, t16b, v1, t16b, v1, t16b),  // Zero the v1 register
      eor(v2, t16b, v2, t16b, v2, t16b),  // Zero the v2 register
      eor(v3, t16b, v3, t16b, v3, t16b),  // Zero the v3 register
    });
  }

  kernel.add({
    // x16 iterator for the n_loop
    mov(x16, n_loop),
    // loop over n
    sub(x16, x16, 1),

    mov(x1, x9),  // Restore x1 for the m loop

    // x17 iterator for the m_loop
    mov(x17, m_loop_4),
    // loop over m
    sub(x17, x17, 1),

    st1Post(v0, t4s, v1, t4s, v2, t4s, v3, t4s, x1, x9),  // increase x1 after store with value of x2 i.e. x1 += 4 * 16 Byte

    // loop back to m
    cbnz(x17, -2 * 4),

    // Updates for the matrix B
    add(x9, x3, x9),  // lda + initial position

    // loop back to n
    cbnz(x16, -7 * 4),
  });

  if (use_init_and_end)
  {
    // kernel.add({
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

    // });
    kernel.add(ret());
  }

#ifdef SAVE_JITS_TO_FILE
  kernel.write("unary_zero_m_4n.bin");
#endif  // SAVE_JITS_TO_FILE
}
