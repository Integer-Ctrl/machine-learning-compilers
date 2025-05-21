#ifndef MINI_JIT_KERNELS_UNARY_ZERO_H
#define MINI_JIT_KERNELS_UNARY_ZERO_H

#include "../../Kernel.h"
#include <cstdint>

namespace mini_jit
{
  namespace kernels
  {
    /**
     * @brief Generates a M x 4*N unary zero kernel.
     *
     * @param kernel The kernel to add instructions too.
     * @param m_loop_4 The repetitions of the m block of size 4.
     * @param n_loop The repetitions of the n block of size 4.
     * @param m_rest The reminder of the m repetitions.
     * @param use_init_and_end Indicates if the procedural call standard, initializing setup and the ret instruction are used. Defaults to
     */
    void unary_zero(mini_jit::Kernel &kernel, const uint32_t m_loop_4, const uint32_t n_loop, const uint32_t m_rest);

  }  // namespace kernels
}  // namespace mini_jit

#endif  // MINI_JIT_KERNELS_UNARY_ZERO_H