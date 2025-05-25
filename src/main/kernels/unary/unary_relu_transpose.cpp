#include "unary_relu_transpose.h"
#include "../../arm_instructions/arm_all.h"
#include "unary_identity_transpose.h"

void relu(mini_jit::Kernel &kernel, mini_jit::arm_instructions::VGeneral vRegister)
{
  using namespace mini_jit::arm_instructions;
  kernel.add(fmax(vRegister, t4s, vRegister, t4s, v31, t4s));
}

void mini_jit::kernels::unary_relu_transpose(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop)
{
  using namespace mini_jit::arm_instructions;

  release_assert(m_loop != 0, "Cannot use a matrix with a m loop of size zero.");
  release_assert(n_loop != 0, "Cannot use a matrix with a n loop of size zero.");

  kernel.add(eor(v31, t16b, v31, t16b, v31, t16b));  // LOCKED as hard zero
  internal::unary_ops_transpose(kernel, m_loop, n_loop, relu, "unary_relu_transpose.bin");
}