#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include "../../main/kernels/matmul_16_6_1.h"
#include "matmul.test.h"
#include "../../main/Brgemm.h"

template <uint TSize>
void verify_matmul(const float (&expected)[TSize], const float (&result)[TSize])
{
  for (size_t i = 0; i < TSize; i++)
  {
    CAPTURE(i, result[i], expected[i]);
    REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
  }
}

TEST_CASE("Test 16x6x1 jited gemm correctness random data", "[jit][correctness][gemm]")
{
  float matrix_a[16 * 1];
  float matrix_b[1 * 6];
  float matrix_c[16 * 6];
  float matrix_c_verify[16 * 6];

  fill_random_matrix(matrix_a);
  fill_random_matrix(matrix_b);
  fill_random_matrix(matrix_c);
  copy_matrix(matrix_c, matrix_c_verify);

  // Generate kernel
  mini_jit::Brgemm brgemm;
  brgemm.generate(16, 6, 1, 1, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
  mini_jit::Brgemm::kernel_t kernel = brgemm.get_kernel();

  

  // Run matmuls
  kernel(matrix_a, matrix_b, matrix_c, 16, 1, 16, 1, 1);
  naive_matmul_M_N_K<16, 6, 1>(matrix_a, matrix_b, matrix_c_verify, 16, 1, 16);

  verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 16x6x1 jited gemm correctness counting data", "[jit][neon_3][correctness][gemm]")
{
  float matrix_a[16 * 1];
  float matrix_b[1 * 6];
  float matrix_c[16 * 6];
  float matrix_c_verify[16 * 6];

  fill_counting_matrix(matrix_a);
  fill_counting_matrix(matrix_b);
  fill_counting_matrix(matrix_c);
  copy_matrix(matrix_c, matrix_c_verify);

  // Generate kernel
  mini_jit::Brgemm brgemm;
  brgemm.generate(16, 6, 1, 1, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
  mini_jit::Brgemm::kernel_t kernel = brgemm.get_kernel();

  // Run matmuls
  kernel(matrix_a, matrix_b, matrix_c, 16, 1, 16, 1, 1);
  naive_matmul_M_N_K<16, 6, 1>(matrix_a, matrix_b, matrix_c_verify, 16, 1, 16);

  verify_matmul(matrix_c_verify, matrix_c);
}
