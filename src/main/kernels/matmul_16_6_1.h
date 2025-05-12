#ifndef MINI_JIT_KERNELS_MATMUL_16_6_1_H
#define MINI_JIT_KERNELS_MATMUL_16_6_1_H

#include "../Kernel.h"
#include <cstdint>

namespace mini_jit
{
  namespace kernels
  {

    /**
     * @brief Generates a 16 x 6 x 1 matmul kernel.
     *
     * @param kernel The kernel to add instructions to.
     */
    void matmul_16_6_1(mini_jit::Kernel &kernel);

  }  // namespace kernels
}  // namespace mini_jit
#endif  // MINI_JIT_KERNELS_MATMUL_16_6_1_H