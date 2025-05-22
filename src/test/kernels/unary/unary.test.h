#ifndef MINI_JIT_KERNELS_UNARY_TEST_H
#define MINI_JIT_KERNELS_UNARY_TEST_H

#include "../../../main/Kernel.h"
#include "../../../main/Unary.h"
#include <cstdint>

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

/**
 * @brief List available unary operations.
 */
enum class UnaryType
{
  /// @brief Fills the matrix b with zeros.
  Zero,

  /// @brief Copies the matrix a to matrix b.
  Identity,

  /// @brief Applies the relu function (i.e. max(x, 0)) from matrix a to matrix b.
  ReLu,

  /// @brief Copies the matrix a with a transpose to matrix b.
  Identity_Transpose,

  /// @brief Applies the relu function (i.e. max(x, 0)) from matrix a with transpose to matrix b.
  ReLu_Transpose,
};

class UnaryTestFixture
{
private:
  uint32_t M;
  uint32_t N;
  uint32_t lda;
  uint32_t ldb;
  float *matrix_a;
  float *matrix_b;
  float *matrix_b_verify;
  int32_t counting_state = 1;

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
  void naive_unary_M_N(const float *__restrict__ a, float *__restrict__ b, int64_t lda, int64_t ldb, UnaryType type);

  /**
   * @brief Compares the two matrices by comparing each values.
   *
   * @param expected The matrix results that are expected.
   * @param result The actual matrix values.
   * @param size The total size of the matrix.
   */
  void verify_matrix(const float *__restrict__ expected, const float *__restrict__ result, uint32_t size);

public:
  mini_jit::Kernel native_kernel;

  UnaryTestFixture(uint32_t M, uint32_t N);
  UnaryTestFixture(uint32_t M, uint32_t N, uint32_t lda, uint32_t ldb);
  ~UnaryTestFixture();

  /**
   * @brief Prints a matrix
   *
   * @param matrix  The matrix to print.
   * @param M The size in the m dimension.
   * @param N The size in the n dimension.
   * @param ld The leading dimension on the m dimension.
   * @param label The name of the matrix.
   */
  static void print_matrix(const float *matrix, int64_t M, int64_t N, int64_t ld, const char *label);

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
   */
  void RunTest(const uint32_t lda, const uint32_t ldb, UnaryType type);
};

#endif  // MINI_JIT_KERNELS_UNARY_TEST_H