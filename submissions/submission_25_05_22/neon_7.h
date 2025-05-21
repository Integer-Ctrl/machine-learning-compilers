#ifndef NEON_7_H
#define NEON_7_H

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iterator>

extern "C"
{
  /**
   * @brief Identity primitive that transposes an 8x8 matrix.
   * @param a    Pointer to column-major matrix A.
   * @param b    Pointer to row-major matrix B.
   * @param ld_a Leading dimension of A.
   * @param ld_b Leading dimension of B.
   **/
  void trans_neon_8_8(float const *a, float *b, int64_t lda, int64_t ldb);
}

/**
 * @brief Fill the given matrix with random values.
 *
 * @tparam TSize The total size of the matrix.
 * @param matrix The matrix to write to.
 */
template <uint32_t TSize> void fill_random_matrix(float (&matrix)[TSize])
{
  std::srand(std::time(0));
  for (size_t i = 0; i < TSize; i++)
  {
    matrix[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
  }
}

/**
 * @brief Fill the given matrix with counting up values, starting from 0.
 *
 * @tparam TSize The total size of the matrix.
 * @param matrix The matrix to write to.
 */
template <uint32_t TSize> void fill_counting_matrix(float (&matrix)[TSize])
{
  for (size_t i = 0; i < TSize; i++)
  {
    matrix[i] = i;
  }
}

/**
 * @brief Copy the values of matrix to another matrix.
 *
 * @tparam TSize The equal size of the matrices.
 * @param input The matrix to copy from.
 * @param output The matrix to copy to.
 */
template <uint32_t TSize> void copy_matrix(float (&input)[TSize], float (&output)[TSize])
{
  std::copy(std::begin(input), std::end(input), std::begin(output));
}

/**
 * @brief Naive transposition of column-major B [8x8] = A [8x8] transposed.
 *
 * @param a The pointer of matrix A.
 * @param b The pointer of matrix B.
 * @param lda The leading dimension of A.
 * @param ldb The leading dimension of B.
 */
void naive_trans_8_8(const float *__restrict__ a, float *__restrict__ b, int64_t lda, int64_t ldb)
{
  for (int64_t i = 0; i < 8; ++i)
  {
    for (int64_t j = 0; j < 8; ++j)
    {
      b[j * ldb + i] = a[i * lda + j];
    }
  }
}

// TODO: remove when debugging is not needed
void print_matrix(const float *mat, int64_t ld, const char *label)
{
  printf("%s:\n", label);
  for (int64_t i = 0; i < 8; ++i)
  {
    for (int64_t j = 0; j < 8; ++j)
    {
      printf("%8.4f ", mat[i * ld + j]);
    }
    printf("\n");
  }
  printf("\n");
}
#endif  // NEON_7_H