#ifndef MINI_JIT_KERNELS_BR_MATMUL_LT16_4N_K_H
#define MINI_JIT_KERNELS_BR_MATMUL_LT16_4N_K_H

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
     * @param n_loop_4 The repetitions of the n block of size 4.
     * @param k_loop The loops in the k dimensions.
     * @param br_size number of batch dimensions.
     * @param m_loop_rest The rest/remainder of the m loop that is not dividable by 16.
     * @param use_init_and_end Indicates if the procedural call standard, initializing setup and the ret instruction are used. Defaults to
     * true.
     */
    void br_matmul_lt16_4n_k(mini_jit::Kernel &kernel, const uint32_t n_loop_4, const uint32_t k_loop, const uint32_t br_size,
                             const uint32_t m_loop_rest, const bool use_init_and_end = true);

  }  // namespace kernels
}  // namespace mini_jit
#endif  // MINI_JIT_KERNELS_BR_MATMUL_LT16_4N_K_H