#ifndef MINIJIT_KERNELS_MATMUL_TEST_H
#define MINIJIT_KERNELS_MATMUL_TEST_H

#include "../../main/Brgemm.h"
#include "../../main/Kernel.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <limits>

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
    float denominator = 1;
    do
    {
      denominator = static_cast<float>(std::rand());
    } while (denominator == 0);

    float numerator = 1;
    do
    {
      numerator = static_cast<float>(std::rand());
    } while (numerator == 0);

    matrix[i] = numerator / denominator;
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
 * @brief Naive matmul of column-major C [MxN] = A [MxK] mul B [KxN].
 *
 * @tparam TMDim The size of the M dimension.
 * @tparam TNDim The size of the N dimension.
 * @tparam TKDim The size of the K dimension.
 * @param a The pointer of matrix A.
 * @param b The pointer of matrix B.
 * @param c The pointer of matrix C.
 * @param lda The leading dimension of A.
 * @param ldb The leading dimension of B.
 * @param ldc The leading dimension of C.
 * @param batch_stride_a The elements to jump to the next matrix a in the batch.
 * @param batch_stride_b The elements to jump to the next matrix b in the batch.
 */
template <uint32_t TMDim, uint32_t TNDim, uint32_t TKDim, uint32_t TBatchDim>
void naive_matmul_M_N_K_Batch(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int64_t lda, int64_t ldb,
                              int64_t ldc, int64_t batch_stride_a, int64_t batch_stride_b)
{
  for (size_t iB = 0; iB < TBatchDim; iB++)
  {
    for (size_t iM = 0; iM < TMDim; iM++)
    {
      for (size_t iN = 0; iN < TNDim; iN++)
      {
        for (size_t iK = 0; iK < TKDim; ++iK)
        {
          c[iM + iN * ldc] += a[iM + iK * lda + iB * batch_stride_a] * b[iK + iN * ldb + iB * batch_stride_b];
        }
      }
    }
  }
}

/**
 * @brief Verify that all values of a matmul are equal.
 *
 * @tparam The size of the matmul
 */
template <uint TSize> void verify_matmul(const float (&expected)[TSize], const float (&result)[TSize])
{
  for (size_t i = 0; i < TSize; i++)
  {
    CAPTURE(i, result[i], expected[i]);
    REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
  }
}

/**
 * @brief List possible fill types for matrix used in this test.
 */
enum class TestInfill
{
  /// @brief Fill with random data.
  Random,

  /// @brief Fill with couting data starting from 0.
  Counting,
};

class GemmMxNxKxBatchTestFixture
{
protected:
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t BatchSize;
  uint32_t lda = 0;
  uint32_t ldb = 0;
  uint32_t ldc = 0;
  uint32_t batch_stride_a = 0;
  uint32_t batch_stride_b = 0;
  float *matrix_a;
  float *matrix_b;
  float *matrix_c;
  float *matrix_c_verify;

  /**
   * @brief Fills the given matrix with random values.
   *
   * @param matrix The matrix to fill.
   * @param size The total size of the matrix.
   */
  void fill_random_matrix(float *matrix, uint32_t size);

  /**
   * @brief Fills the given matrix with counting values starting from 0.
   *
   * @param matrix The matrix to fill.
   * @param size The total size of the matrix.
   */
  void fill_counting_matrix(float *matrix, uint32_t size);

  /**
   * @brief Does a naive matmul for verification usage.
   *
   * @param a The a matrix.
   * @param b The b matrix.
   * @param c The c matrix.
   * @param lda The leading dimension of matrix a.
   * @param ldb The leading dimension of matrix b.
   * @param ldc The leading dimension of matrix c.
   * @param batch_stride_a The batch stride of matrix a.
   * @param batch_stride_b The batch stride of matrix b.
   */
  void naive_matmul_M_N_K_Batch(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int64_t lda, int64_t ldb,
                                int64_t ldc, int64_t batch_stride_a, int64_t batch_stride_b);

  /**
   * @brief Compares the two matrices by comparing each values.
   *
   * @param expected The matrix results that are expected.
   * @param result The actual matrix values.
   * @param size The total size of the matrix.
   */
  void verify_matmul(const float *__restrict__ expected, const float *__restrict__ result, uint32_t size);

public:
  mini_jit::Kernel native_kernel;

  GemmMxNxKxBatchTestFixture() = delete;
  GemmMxNxKxBatchTestFixture(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize);
  GemmMxNxKxBatchTestFixture(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize, uint32_t lda, uint32_t ldb, uint32_t ldc,
                             uint32_t batch_stride_a, uint32_t bach_stride_b);
  ~GemmMxNxKxBatchTestFixture();

  /**
   * @brief Set up the test fixture object.
   *
   * @param fillType Fills the matrices with the given infill type.
   */
  void SetUp(TestInfill fillType);

  /**
   * @brief Executes the Test von an BRGemm with the given input.
   *
   * @param lda: leading dimension of A.
   * @param ldb: leading dimension of B.
   * @param ldc: leading dimension of C.
   * @param br_stride_a: stride between two A matrices (in elements, not bytes).
   * @param br_stride_b: stride between two B matrices (in elements, not bytes).
   */
  void RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a, const uint32_t batch_stride_b)
  {
    _RunTest(lda, ldb, ldc, batch_stride_a, batch_stride_b);
  }

protected:
  /**
   * @brief Executes the Test von an BRGemm with the given input.
   *
   * @param lda: leading dimension of A.
   * @param ldb: leading dimension of B.
   * @param ldc: leading dimension of C.
   * @param br_stride_a: stride between two A matrices (in elements, not bytes).
   * @param br_stride_b: stride between two B matrices (in elements, not bytes).
   */
  void _RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a, const uint32_t batch_stride_b);
};

class GemmMxNxKTestFixture : public GemmMxNxKxBatchTestFixture
{

  void RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a,
               const uint32_t batch_stride_b) = delete;  // delete so not visible in a GemmMxNxKTestFixture object.

public:
  GemmMxNxKTestFixture() = delete;
  GemmMxNxKTestFixture(uint32_t M, uint32_t N, uint32_t K);
  GemmMxNxKTestFixture(uint32_t M, uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc);
  ~GemmMxNxKTestFixture();

  /**
   * @brief Executes the Test von an BRGemm with the given input.
   *
   * @param lda: leading dimension of A.
   * @param ldb: leading dimension of B.
   * @param ldc: leading dimension of C.
   */
  void RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc)
  {
    GemmMxNxKxBatchTestFixture::_RunTest(lda, ldb, ldc, lda * K, ldb * N);
  }
};

#endif  // MINIJIT_KERNELS_MATMUL_TEST_H