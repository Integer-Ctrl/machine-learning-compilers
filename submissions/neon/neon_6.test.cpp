#include "neon_6.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>

template <uint TSize> void verify_matmul(const float (&expected)[TSize], const float (&result)[TSize])
{
  for (size_t i = 0; i < TSize; i++)
  {
    CAPTURE(i, result[i], expected[i]);
    REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
  }
}

TEST_CASE("Test 64x48x64 batch=1 gemm correctness random data", "[neon_6][correctness][gemm]")
{
  const uint32_t M = 64;
  const uint32_t N = 48;
  const uint32_t K = 64;

  float matrix_a[M * K];
  float matrix_b[K * N];
  float matrix_c[M * N];
  float matrix_c_verify[M * N];

  fill_random_matrix(matrix_a);
  fill_random_matrix(matrix_b);
  fill_random_matrix(matrix_c);
  copy_matrix(matrix_c, matrix_c_verify);

  // Run matmuls
  matmul_64_48_64(matrix_a, matrix_b, matrix_c, 64, 64, 64);
  naive_matmul_M_N_K_Batch<M, N, K, 1>(matrix_a, matrix_b, matrix_c_verify, 64, 64, 64, 0, 0);

  verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 64x48x64 batch=1 gemm correctness counting data", "[neon_6][correctness][gemm]")
{
  const uint32_t M = 64;
  const uint32_t N = 48;
  const uint32_t K = 64;

  float matrix_a[M * K];
  float matrix_b[K * N];
  float matrix_c[M * N];
  float matrix_c_verify[M * N];

  fill_counting_matrix(matrix_a);
  fill_counting_matrix(matrix_b);
  fill_counting_matrix(matrix_c);
  copy_matrix(matrix_c, matrix_c_verify);

  // Run matmuls
  matmul_64_48_64(matrix_a, matrix_b, matrix_c, 64, 64, 64);
  naive_matmul_M_N_K_Batch<M, N, K, 1>(matrix_a, matrix_b, matrix_c_verify, 64, 64, 64, 0, 0);

  verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 64x48x64 batch=16 gemm correctness random data", "[neon_6][correctness][gemm]")
{
  const uint32_t M = 64;
  const uint32_t N = 48;
  const uint32_t K = 64;
  const uint32_t Batch = 16;

  float matrix_a[M * K * Batch];
  float matrix_b[K * N * Batch];
  float matrix_c[M * N];
  float matrix_c_verify[M * N];

  fill_random_matrix(matrix_a);
  fill_random_matrix(matrix_b);
  fill_random_matrix(matrix_c);
  copy_matrix(matrix_c, matrix_c_verify);

  // Run matmuls
  matmul_64_48_64_16(matrix_a, matrix_b, matrix_c, 64, 64, 64, M * K, K * N);
  naive_matmul_M_N_K_Batch<M, N, K, 16>(matrix_a, matrix_b, matrix_c_verify, 64, 64, 64, M * K, K * N);

  verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 64x48x64 batch=16 gemm correctness counting data", "[neon_6][correctness][gemm]")
{
  const uint32_t M = 64;
  const uint32_t N = 48;
  const uint32_t K = 64;
  const uint32_t Batch = 16;

  float matrix_a[M * K * Batch];
  float matrix_b[K * N * Batch];
  float matrix_c[M * N];
  float matrix_c_verify[M * N];

  fill_counting_matrix(matrix_a);
  fill_counting_matrix(matrix_b);
  fill_counting_matrix(matrix_c);
  copy_matrix(matrix_c, matrix_c_verify);

  // Run matmuls
  matmul_64_48_64_16(matrix_a, matrix_b, matrix_c, 64, 64, 64, M * K, K * N);
  naive_matmul_M_N_K_Batch<M, N, K, 16>(matrix_a, matrix_b, matrix_c_verify, 64, 64, 64, M * K, K * N);

  verify_matmul(matrix_c_verify, matrix_c);
}