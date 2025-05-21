#include "unary_relu.h"
#include "../../arm_instructions/arm_all.h"

void mini_jit::kernels::unary_relu(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop)
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

    mov(x7, x0),         // Store the inital value of x0, to be restored in the N loop
    mov(x8, x1),         // Store the inital value of x1, to be restored in the N loop
    mov(x9, 4 * 4 * 4),  // 4 * 4 * sizeof(float) Hold the number of bytes that are stored in the loop

    // x16 iterator for the n_loop
    mov(x16, n_loop),
    // loop over n
    sub(x16, x16, 1),

    mov(x0, x7),  // Restore x0 for the m loop
    mov(x1, x8),  // Restore x1 for the m loop

    eor(v5, t16b, v5, t16b, v5, t16b),  // Zero the v5 register to use fmax vector
  });

  int32_t n_jump_start = kernel.get_instruction_count() - 4;

  if (m_loop < 16)
  {
    kernel.add({

      // x17 iterator for the m_loop
      mov(x17, m_loop / 16),
      // loop over m
      sub(x17, x17, 1),

      ld1Post(v0, t4s, v1, t4s, v2, t4s, v3, t4s, x0, x9),  // increase x0 after load with value of x9 i.e. x0 += 4 * 4 * sizeof(float)
      fmax(v0, t4s, v0, t4s, v5, t4s),
      fmax(v1, t4s, v1, t4s, v5, t4s),
      fmax(v2, t4s, v2, t4s, v5, t4s),
      fmax(v3, t4s, v3, t4s, v5, t4s),
      st1Post(v0, t4s, v1, t4s, v2, t4s, v3, t4s, x1, x9),  // increase x1 after store with value of x9 i.e. x1 += 4 * 4 * sizeof(float)

      // loop back to m
      cbnz(x17, -3 * 4),

    });
  }

  uint32_t m_loop_rest = m_loop % 16;
  // Handel the rest of m
  if (m_loop_rest != 0)
  {

    uint32_t m_loop_rest_multiple_4 = m_loop_rest / 4;
    switch (m_loop_rest_multiple_4)
    {
    case 0:
      // nothing to do
      break;

    case 1:
      kernel.add({
        mov(x9, 1 * 4 * 4),        // 1 * 4 * sizeof(float)
        ld1Post(v0, t4s, x0, x9),  // increase x0 after load with value of x9 i.e. x0 += 1 * 4 * sizeof(float)
        fmax(v0, t4s, v0, t4s, v5, t4s),
        st1Post(v0, t4s, x1, x9),  // increase x1 after store with value of x9 i.e. x1 += 1 * 4 * sizeof(float)
      });
      break;

    case 2:
      kernel.add({
        mov(x9, 2 * 4 * 4),                 // 2 * 4 * sizeof(float)
        ld1Post(v0, t4s, v1, t4s, x0, x9),  // increase x0 after load with value of x9 i.e. x0 += 4 * 4 * sizeof(float)
        fmax(v0, t4s, v0, t4s, v5, t4s), fmax(v1, t4s, v1, t4s, v5, t4s),
        st1Post(v0, t4s, v1, t4s, x1, x9),  // increase x1 after store with value of x9 i.e. x1 += 4 * 4 * sizeof(float)
      });
      break;

    case 3:
      kernel.add({
        mov(x9, 3 * 4 * 4),                          // 3 * 4 * sizeof(float)
        ld1Post(v0, t4s, v1, t4s, v2, t4s, x0, x9),  // increase x0 after load with value of x9 i.e. x0 += 4 * 4 * sizeof(float)
        fmax(v0, t4s, v0, t4s, v5, t4s), fmax(v1, t4s, v1, t4s, v5, t4s), fmax(v2, t4s, v2, t4s, v5, t4s),
        st1Post(v0, t4s, v1, t4s, v2, t4s, x1, x9),  // increase x1 after store with value of x9 i.e. x1 += 4 * 4 * sizeof(float)
      });
      break;

    default:
      release_assert(false, "Out of range loop rest detected for multiple of 4 instructions.");
      break;
    }

    uint32_t m_loop_rest_less_than_4 = m_loop_rest % 4;
    switch (m_loop_rest_less_than_4)
    {
    case 0:
      // noting to do
      break;

    case 1:
      kernel.add({
        // load single element
        ldrPost(s0, x0, 4),
        fmax(s0, s0, s5),
        strPost(s0, x1, 4),
      });
      break;

    case 2:
      kernel.add({
        // load two elements
        ldpPost(s0, s1, x0, 4 * 2),
        fmax(s0, s0, s5),
        fmax(s1, s1, s5),
        stpPost(s0, s1, x1, 4 * 2),
      });
      break;

    case 3:
      kernel.add({
        // load three elements
        ldpPost(s0, s1, x0, 4 * 2),
        fmax(s0, s0, s5),
        fmax(s1, s1, s5),
        stpPost(s0, s1, x1, 4 * 2),
        ldrPost(s0, x0, 4),
        fmax(s0, s0, s5),
        strPost(s0, x1, 4),
      });
      break;

    default:
      release_assert(false, "Out of range loop rest detected for less than 4 instructions.");
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
  kernel.write("unary_relu.bin");
#endif  // SAVE_JITS_TO_FILE
}