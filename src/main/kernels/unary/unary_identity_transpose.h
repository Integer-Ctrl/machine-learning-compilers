#ifndef MINI_JIT_KERNELS_UNARY_identity_H
#define MINI_JIT_KERNELS_UNARY_identity_H

#include "../../Kernel.h"
#include <cstdint>

namespace mini_jit
{
  namespace kernels
  {
    namespace internal
    {
      /**
       * @brief Transpose a matrix, this function is only called as subfunction from unary_identity_transpose
       *
       * M=1 N=0 and M=0 and N=1 are pairs and only called ones, thus we only need to implement a triangular matrix.
       *
       *  M=N=0  | M=0 N=1 |
       *---------------------
       * M=1 N=0 |
       *
       * @param kernel The kernel to add instruction too.
       * @param m_position The current position on the m dimension.
       * @param n_position The current position on the n dimension.
       * @param M The size of the M dimension
       * @param N The size of the N dimension
       */
      void transpose_symmetric(mini_jit::Kernel &kernel, const uint32_t m_position, const uint32_t n_position, const uint32_t M,
                               const uint32_t N);
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