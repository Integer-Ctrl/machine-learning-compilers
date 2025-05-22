#ifndef MINI_JIT_KERNELS_UNARY_IDENTITY_TRANSPOSE_H
#define MINI_JIT_KERNELS_UNARY_IDENTITY_TRANSPOSE_H

#include "../../Kernel.h"
#include <cstdint>

namespace mini_jit
{
  namespace kernels
  {
    namespace internal
    {
      /**
       * @brief Adds a transpose instructions to the kernel that is located on the matrix axis.
       *
       * @param kernel The kernel to add instructions too.
       * @param m The m dimension size in range of 0 < m <= 4.
       * @param n The n dimension size in range of 0 < n <= 4.
       */
      void transpose_axis(mini_jit::Kernel &kernel, const uint32_t m, const uint32_t n);

      /**
       * @brief Adds a transpose instructions to the kernel that is located outside of the matrix axis.
       *
       * @param kernel The kernel to add instructions too.
       * @param m The m dimension size in range of 0 < m <= 4.
       * @param n The n dimension size in range of 0 < n <= 4.
       */
      void transpose_else(mini_jit::Kernel &kernel, const uint32_t m, const uint32_t n);
    }  // namespace internal

    /**
     * @brief Generates a M x N unary identity transpose kernel.
     *
     * @param kernel The kernel to add instructions too.
     * @param m_loop The repetitions of the m dimensions.
     * @param n_loop The repetitions of the n dimensions.
     */
    void unary_identity_transpose(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop);

  }  // namespace kernels
}  // namespace mini_jit

#endif  // MINI_JIT_KERNELS_UNARY_identity_H