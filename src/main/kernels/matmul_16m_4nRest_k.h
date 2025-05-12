#ifndef MINI_JIT_KERNELS_MATMUL_16M_4NRest_K_H
#define MINI_JIT_KERNELS_MATMUL_16M_4NRest_K_H

#include "../Kernel.h"
#include <cstdint>

namespace mini_jit
{
  namespace kernels
  {

    /**
     * @brief Generates a 16*M x 4*N + Rest x k matmul kernel.
     *
     * @param kernel The kernel to add instructions to.
     * @param m_loop_16 The repetitions of the m block of size 16.
     * @param n_loop_4 The repetitions of the n block of size 4.
     * @param k_loop The loops in the k dimensions.
     * @param n_loop_rest The rest/remainder of the n loop that is not dividable by 4
     */
    void matmul_16m_4nRest_k(mini_jit::Kernel &kernel, const uint32_t m_loop_16, const uint32_t n_loop_4, const uint32_t k_loop,
                             const uint32_t n_loop_rest);

  }  // namespace kernels
}  // namespace mini_jit
#endif  // MINI_JIT_KERNELS_MATMUL_16M_4NRest_K_H