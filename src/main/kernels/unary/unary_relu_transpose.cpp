#include "unary_relu_transpose.h"
#include "../../arm_instructions/arm_all.h"

void mini_jit::kernels::unary_relu_transpose(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop)
{
  using namespace mini_jit::arm_instructions;

  release_assert(m_loop != 0, "Cannot use a matrix with a m loop of size zero.");
  release_assert(n_loop != 0, "Cannot use a matrix with a n loop of size zero.");

  kernel.get_kernel();

#ifdef SAVE_JITS_TO_FILE
  kernel.write("unary_relu_transpose.bin");
#endif  // SAVE_JITS_TO_FILE
}