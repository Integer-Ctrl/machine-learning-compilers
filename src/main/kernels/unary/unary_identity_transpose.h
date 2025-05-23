#ifndef MINI_JIT_KERNELS_UNARY_IDENTITY_TRANSPOSE_H
#define MINI_JIT_KERNELS_UNARY_IDENTITY_TRANSPOSE_H

#include "../../Kernel.h"
#include "../../arm_instructions/register.h"
#include <cstdint>
#include <string>

namespace mini_jit
{
  namespace kernels
  {

    namespace internal
    {
      /**
       * @brief The operation that should be done on 4x fp32 element.
       *
       * @param kernel The kernel to add the instructions to.
       * @param vRegister The vX.t4s register to be used.
       */
      using ops_t = void (*)(mini_jit::Kernel &kernel, const mini_jit::arm_instructions::VGeneral vRegister);

      /**
       * @brief Adds a transpose instructions to the kernel that is located on the matrix axis.
       *
       * @param kernel The kernel to add instructions too.
       * @param m The m dimension size in range of 0 < m <= 4.
       * @param n The n dimension size in range of 0 < n <= 4.
       * @param ops The operation to do on a 4x fp32 element
       */
      void transpose_axis(mini_jit::Kernel &kernel, const uint32_t m, const uint32_t n, ops_t ops);

      /**
       * @brief Adds a transpose instructions to the kernel that is located outside of the matrix axis.
       *
       * @param kernel The kernel to add instructions too.
       * @param m The m dimension size in range of 0 < m <= 4.
       * @param n The n dimension size in range of 0 < n <= 4.
       * @param ops The operation to do on a 4x fp32 element
       */
      void transpose_else(mini_jit::Kernel &kernel, const uint32_t m, const uint32_t n, ops_t ops);

      /**
       * @brief Generate a transpose kernel.
       *
       * @param kernel  The kernel to add instructions too.
       * @param m_loop  The repetitions of the m dimension.
       * @param n_loop  The repetitions of the n dimension.
       * @param ops  The operation that should extract add.
       * @param path The path to write the bin dumps to.
       */
      void unary_ops_transpose(mini_jit::Kernel &kernel, const uint32_t m_loop, const uint32_t n_loop, ops_t ops, char const *path);
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