#include "neon_7.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>

template <uint TSize> void verify_trans(const float (&expected)[TSize], const float (&result)[TSize])
{
  for (size_t i = 0; i < TSize; i++)
  {
    CAPTURE(i, result[i], expected[i]);
    REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
  }
}

TEST_CASE("Test 8x8 transposition correctness random data", "[neon_7][correctness][gemm]")
{
  const uint32_t M = 8;
  const uint32_t N = 8;
  const uint32_t K = 8;

  float matrix_a[M * K];
  float matrix_b[K * N];
  float matrix_b_verify[M * N];

  fill_random_matrix(matrix_a);
  fill_random_matrix(matrix_b);
  copy_matrix(matrix_b, matrix_b_verify);

  // Run transposition
  trans_neon_8_8(matrix_a, matrix_b, 8, 8);
  naive_trans_8_8(matrix_a, matrix_b_verify, 8, 8);

  // Debug output
  print_matrix(matrix_a, 8, "Input A");
  print_matrix(matrix_b_verify, 8, "Expected Transposed (Naive)");
  print_matrix(matrix_b, 8, "Actual Transposed (Neon)");

  verify_trans(matrix_b_verify, matrix_b);
}

TEST_CASE("Test 8x8 transposition correctness counting data", "[neon_7][correctness][gemm]")
{
  const uint32_t M = 8;
  const uint32_t N = 8;
  const uint32_t K = 8;

  float matrix_a[M * K];
  float matrix_b[K * N];
  float matrix_b_verify[M * N];

  fill_counting_matrix(matrix_a);
  fill_counting_matrix(matrix_b);
  copy_matrix(matrix_b, matrix_b_verify);

  // Run transposition
  trans_neon_8_8(matrix_a, matrix_b, 8, 8);
  naive_trans_8_8(matrix_a, matrix_b_verify, 8, 8);

  // Debug output
  print_matrix(matrix_a, 8, "Input A");
  print_matrix(matrix_b_verify, 8, "Expected Transposed (Naive)");
  print_matrix(matrix_b, 8, "Actual Transposed (Neon)");

  verify_trans(matrix_b_verify, matrix_b);
}
