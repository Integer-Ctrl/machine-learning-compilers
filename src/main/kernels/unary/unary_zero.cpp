#include "unary_zero.h"
#include "../../arm_instructions/arm_all.h"

void mini_jit::kernels::unary_zero(mini_jit::Kernel &kernel, const uint32_t m_loop_16, const uint32_t n_loop, const uint32_t m_rest)
{
  using namespace mini_jit::arm_instructions;

  release_assert(m_rest < 16, "M dimension cannot be larger than 15.");

  // Hold the number of instruction to jump for each loop
  int32_t kernel_size_stamp;
  int32_t n_loop_jump_inst = 0;

  kernel.add({

    /**
     * @param x0 = a pointer to column-major matrix A (Input). Unused for zero unary kerne.
     * @param x1 = b pointer to column-major matrix B (Output).
     * @param x2 = lda leading dimension of A. Unused for for zero unary kernel.
     * @param x3 = ldb leading dimension of B.
     */

    // Offset the used leading dimension by the size of floats
    lsl(x3, x3, 2),  // x3 * 4 = x3 * sizeof(float)

    mov(x8, x1),         // Store the inital value of x1, to be restored in the N loop
    mov(x9, 4 * 4 * 4),  // 4 * 4 * sizeof(float) Hold the number of bytes that are stored in the loop

    // Zero four register so we can fill the 16 values with zeros at a time
    eor(v0, t16b, v0, t16b, v0, t16b),  // Zero the v0 register
    eor(v1, t16b, v1, t16b, v1, t16b),  // Zero the v1 register
    eor(v2, t16b, v2, t16b, v2, t16b),  // Zero the v2 register
    eor(v3, t16b, v3, t16b, v3, t16b),  // Zero the v3 register
  });

  kernel.add({
    // x16 iterator for the n_loop
    mov(x16, n_loop),
    // loop over n
    sub(x16, x16, 1),

    mov(x1, x8),  // Restore x1 for the m loop
  });

  n_loop_jump_inst += 2;
  kernel_size_stamp = kernel.get_instruction_count();

  if (m_loop_16 > 0)
  {
    kernel.add({
      // x17 iterator for the m_loop
      mov(x17, m_loop_16),
      // loop over m
      sub(x17, x17, 1),

      st1Post(v0, t4s, v1, t4s, v2, t4s, v3, t4s, x1, x9),  // increase x1 after store with value of x2 i.e. x1 += 4 * 16 Byte

      // loop back to m
      cbnz(x17, -2 * 4),
    });
  }

  switch (m_rest)
  {
  case 0:
    break;
  case 1:
    kernel.add(str(s0, x1));
    break;
  case 2:
    kernel.add(str(d0, x1));
    break;
  case 3:
    kernel.add(strPost(d0, x1, 8));
    kernel.add(str(s0, x1));
    break;

  case 4:
    kernel.add(str(q0, x1));
    break;
  case 5:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(str(s0, x1));
    break;
  case 6:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(str(d0, x1));
    break;
  case 7:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(d0, x1, 8));
    kernel.add(str(s0, x1));
    break;

  case 8:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    break;
  case 9:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(str(s0, x1));
    break;
  case 10:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(str(d0, x1));
    break;
  case 11:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(d0, x1, 8));
    kernel.add(str(s0, x1));
    break;

  case 12:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    break;
  case 13:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(str(s0, x1));
    break;
  case 14:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(str(d0, x1));
    break;
  case 15:
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(q0, x1, 16));
    kernel.add(strPost(d0, x1, 8));
    kernel.add(str(s0, x1));
    break;

  default:
    break;
  }

  // Updates for the matrix B
  kernel.add(add(x8, x3, x8));  // lda + initial position

  n_loop_jump_inst += kernel.get_instruction_count() - kernel_size_stamp;

  kernel.add({
    // loop back to n
    cbnz(x16, -n_loop_jump_inst * 4),
    ret(),
  });

#ifdef SAVE_JITS_TO_FILE
  kernel.write("unary_zero.bin");
#endif  // SAVE_JITS_TO_FILE
}