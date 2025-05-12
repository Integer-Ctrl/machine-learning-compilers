#ifndef NEON_5_H
#define NEON_5_H

#include <cstdint>
#include <ctime>

extern "C"
{
  /**
   * @brief Matmul that loops over the NMK dimension of an original matmul of (M=64, N=64, K=64) now with loop over K=64.
   * @param a pointer to column-major matrix A.
   * @param b pointer to column-major matrix B.
   * @param c pointer to column-major matrix C.
   * @param lda leading dimension of A.
   * @param ldb leading dimension of B.
   * @param ldc leading dimension of C.
   **/
  void matmul_64_64_64(float const *a, float const *b, float *c, int64_t lda, int64_t ldb, int64_t ldc);

  /**
   * @brief Matmul that loops over the NMK dimension of an original matmul of (M=64, N=64, K=64) now with loop over K=64.
   * @param a pointer to column-major 64x64 matrix A.
   * @param b pointer to column-major 64x64 matrix B.
   * @param c pointer to column-major 64x64 matrix C.
   **/
  void matmul_64_64_64_base_line(float const *a, float const *b, float *c);
}

/// @brief Fill the given matrix with random values.
/// @tparam TSize The total size of the matrix.
/// @param matrix The matrix to write to.
template <uint32_t TSize> void fill_random_matrix(float (&matrix)[TSize])
{
  std::srand(std::time(0));
  for (size_t i = 0; i < TSize; i++)
  {
    matrix[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
  }
}

/// @brief Fill the given matrix with counting up values, starting from 0.
/// @tparam TSize The total size of the matrix.
/// @param matrix The matrix to write to.
template <uint32_t TSize> void fill_counting_matrix(float (&matrix)[TSize])
{
  for (size_t i = 0; i < TSize; i++)
  {
    matrix[i] = i;
  }
}

/// @brief Copy the values of matrix to another matrix.
/// @tparam TSize The equal size of the matrices.
/// @param input The matrix to copy from.
/// @param output The matrix to copy to.
template <uint32_t TSize> void copy_matrix(float (&input)[TSize], float (&output)[TSize])
{
  std::copy(std::begin(input), std::end(input), std::begin(output));
}

/// @brief Naive matmul of column-major C [MxN] = A [MxK] mul B [KxN].
/// @tparam TMDim The size of the M dimension.
/// @tparam TNDim The size of the N dimension.
/// @tparam TKDim The size of the K dimension.
/// @param a The pointer of matrix A.
/// @param b The pointer of matrix B.
/// @param c The pointer of matrix C.
/// @param lda The leading dimension of A.
/// @param ldb The leading dimension of B.
/// @param ldc The leading dimension of C.
template <uint32_t TMDim, uint32_t TNDim, uint32_t TKDim>
void naive_matmul_M_N_K(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int64_t lda, int64_t ldb,
                        int64_t ldc)
{
  for (size_t iM = 0; iM < TMDim; iM++)
  {
    for (size_t iN = 0; iN < TNDim; iN++)
    {
      for (size_t iK = 0; iK < TKDim; ++iK)
      {
        c[iM + iN * ldc] += a[iM + iK * lda] * b[iK + iN * ldb];
      }
    }
  }
}
#endif  // NEON_5_H