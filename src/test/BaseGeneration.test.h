#ifndef MINI_JIT_BASEGENERATION_TEST_H
#define MINI_JIT_BASEGENERATION_TEST_H

#include "kernels/matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

/**
 * @brief List available unary operations.
 */
enum class UnaryType
{
  /// @brief None type for init
   None,

  /// @brief Fills the matrix b with zeros.
  Zero,

  /// @brief Copies the matrix a to matrix b.
  Identity,

  /// @brief Applies the relu function (i.e. max(x, 0)) from matrix a to matrix b.
  ReLu,
};

class GenerationTest
{
public:
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t BatchSize;
  uint32_t lda = 0;
  uint32_t ldb = 0;
  uint32_t ldc = 0;
  uint32_t batch_stride_a = 0;
  uint32_t batch_stride_b = 0;
  std::vector<float> matrix_a;
  std::vector<float> matrix_b;
  std::vector<float> matrix_c;
  std::vector<float> matrix_c_verify;
  mini_jit::Brgemm::kernel_t kernel = nullptr;

  /**
   * @brief Fills the given matrix with random values.
   *
   * @param matrix The matrix to fill.
   * @param size The total size of the matrix.
   */
  static void fill_random_matrix(float *matrix, uint32_t size);

  /**
   * @brief Fills the given matrix with counting values starting from 0.
   *
   * @param matrix The matrix to fill.
   * @param size The total size of the matrix.
   */
  static void fill_counting_matrix(float *matrix, uint32_t size);

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
   * @param trans_b True if b is transposed.
   * @param type The type of unary operation to do.
   */
  void naive_unary_M_N(const float *a, float *b, int64_t lda, int64_t ldb, bool trans_b, UnaryType type);

  /**
   * @brief Compares the two matrices by comparing each values.
   *
   * @param expected The matrix results that are expected.
   * @param result The actual matrix values.
   * @param size The total size of the matrix.
   */
  static void verify_matmul(const float *__restrict__ expected, const float *__restrict__ result, uint32_t size);

  GenerationTest() = delete;
  GenerationTest(uint32_t M, uint32_t N, uint32_t K);
  GenerationTest(uint32_t M, uint32_t N, uint32_t K, std::vector<uint32_t> matrix_a_sizes, std::vector<uint32_t> matrix_b_sizes,
                 std::vector<uint32_t> matrix_c_sizes);
  GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc);
  GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize);
  GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize, std::vector<uint32_t> matrix_a_sizes,
                 std::vector<uint32_t> matrix_b_sizes, std::vector<uint32_t> matrix_c_sizes);
  GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t batch_stride_a,
                 uint32_t batch_stride_b);

  /**
   * @brief Set up the test fixture object.
   *
   * @param fillType Fills the matrices with the given infill type.
   */
  void SetUp(TestInfill fillType);

  /**
   * @brief Set the kernel to be used in the test.
   *
   * @param kernel The kernel to run in the test.
   */
  void SetKernel(mini_jit::Brgemm::kernel_t kernel);

  /**
   * @brief Executes the Test von an BRGemm with the given input.
   *
   * @param lda: leading dimension of A.
   * @param ldb: leading dimension of B.
   * @param ldc: leading dimension of C.
   * @param br_stride_a: stride between two A matrices (in elements, not bytes).
   * @param br_stride_b: stride between two B matrices (in elements, not bytes).
   */
  void RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a, const uint32_t batch_stride_b);
};

#endif  // MINI_JIT_BASEGENERATION_TEST_H