#ifndef MINI_JIT_KERNELS_UNARY_RELU_H
#define MINI_JIT_KERNELS_UNARY_RELU_H

#include "../../Kernel.h"
#include <cstdint>

namespace mini_jit
{
  namespace kernels
  {
    /**
     * @brief Generates a M x N unary identity kernel.
     *
     * @param kernel The kernel to add instructions too.
     * @param m_loop The repetitions of the m dimensions.
     * @param n_loop The repetitions of the n dimensions.
     */
    void unary_relu(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop);

  }  // namespace kernels
}  // namespace mini_jit

#endif  // MINI_JIT_KERNELS_UNARY_RELU_H