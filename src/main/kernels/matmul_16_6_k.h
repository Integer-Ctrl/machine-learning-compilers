#ifndef MINI_JIT_KERNELS_MATMUL_16_6_K_H
#define MINI_JIT_KERNELS_MATMUL_16_6_K_H

#include "../Kernel.h"
#include <cstdint>

namespace mini_jit
{
  namespace kernels
  {

    /**
     * @brief Generates a 16 x 6 x k matmul kernel.
     *
     * @param kernel The kernel to add instructions to.
     * @param k_loop The loop over the k dimension.
     */
    void matmul_16_6_k(mini_jit::Kernel &kernel, const uint32_t k_loop);

  }  // namespace kernels
}  // namespace mini_jit
#endif  // MINI_JIT_KERNELS_MATMUL_16_6_K_H