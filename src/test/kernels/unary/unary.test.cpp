#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

UnaryTestFixture::UnaryTestFixture(uint32_t M, uint32_t N) : UnaryTestFixture(M, N, M, M, false)
{
}

UnaryTestFixture::UnaryTestFixture(uint32_t M, uint32_t N, uint32_t lda, uint32_t ldb) : UnaryTestFixture(M, N, lda, ldb, false)
{
}

UnaryTestFixture::UnaryTestFixture(uint32_t M, uint32_t N, uint32_t lda, uint32_t ldb, bool trans_b)
    : M(M), N(N), lda(lda), ldb(ldb), trans_b(trans_b)
{
  if (trans_b == true)  // Transpose
  {
    REQUIRE(lda >= M);
    REQUIRE(ldb >= N);

    matrix_a = new float[lda * N];
    matrix_b = new float[ldb * M];
    matrix_b_verify = new float[ldb * M];
  }
  else
  {
    REQUIRE(lda >= M);
    REQUIRE(ldb >= M);

    matrix_a = new float[lda * N];
    matrix_b = new float[ldb * N];
    matrix_b_verify = new float[ldb * N];
  }
}

UnaryTestFixture::~UnaryTestFixture()
{
  delete[] matrix_a;
  delete[] matrix_b;
  delete[] matrix_b_verify;
}

void UnaryTestFixture::print_matrix(const float *matrix, int64_t M, int64_t N, int64_t ld, const char *label)
{
  printf("%s:\n", label);
  for (int64_t i = 0; i < N; ++i)
  {
    for (int64_t j = 0; j < M; ++j)
    {
      printf("%8.4f ", matrix[i * ld + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void UnaryTestFixture::SetUp(TestInfill fillType)
{
  switch (fillType)
  {
  case TestInfill::Random:
    if (trans_b == true)
    {
      fill_random_matrix(matrix_a, lda * N);
      fill_random_matrix(matrix_b, ldb * M);
    }
    else
    {
      fill_random_matrix(matrix_a, lda * N);
      fill_random_matrix(matrix_b, ldb * N);
    }
    break;
  case TestInfill::Counting:
    if (trans_b == true)
    {
      fill_counting_matrix(matrix_a, lda * N);
      fill_counting_matrix(matrix_b, ldb * M);
    }
    else
    {
      fill_counting_matrix(matrix_a, lda * N);
      fill_counting_matrix(matrix_b, ldb * N);
    }
    break;
  default:
    FAIL("Undefined infill type found.");
    break;
  }

  if (trans_b)
  {
    std::copy(matrix_b, matrix_b + ldb * M, matrix_b_verify);
  }
  else
  {
    std::copy(matrix_b, matrix_b + ldb * N, matrix_b_verify);
  }
}

void UnaryTestFixture::RunTest(const uint32_t lda, const uint32_t ldb, UnaryType type)
{
  if (native_kernel.get_size() <= 0)
  {
    INFO("The kernel should contain instructions before the test is executed.");
    REQUIRE(native_kernel.get_size() > 0);
  }

  // Generate executable kernel
  native_kernel.set_kernel();
  mini_jit::Unary::kernel_t kernel = reinterpret_cast<mini_jit::Unary::kernel_t>(
    const_cast<void *>(native_kernel.get_kernel()));  // Properly cast from const void* to kernel_t

  // Verification of same lda, ldb
  REQUIRE(UnaryTestFixture::lda == lda);
  REQUIRE(UnaryTestFixture::ldb == ldb);

  UnaryTestFixture::print_matrix(matrix_a, M, N, lda, "Initial");

  naive_unary_M_N(matrix_a, matrix_b_verify, lda, ldb, trans_b, type);
  UnaryTestFixture::print_matrix(matrix_b_verify, N, M, ldb, "Expected");

  kernel(matrix_a, matrix_b, lda, ldb);
  UnaryTestFixture::print_matrix(matrix_b, N, M, ldb, "Result");

  if (trans_b)
  {
    verify_matrix(matrix_b_verify, matrix_b, ldb * M);
  }
  else
  {
    verify_matrix(matrix_b_verify, matrix_b, ldb * N);
  }
}

void UnaryTestFixture::naive_unary_M_N(const float *__restrict__ a, float *__restrict__ b, int64_t lda, int64_t ldb, bool trans_b,
                                       UnaryType type)
{
  for (size_t iN = 0; iN < N; iN++)
  {
    for (size_t iM = 0; iM < M; iM++)
    {
      switch (type)
      {
      case UnaryType::Zero:
        if (trans_b == true)
        {
          b[ldb * iM + iN] = 0;
        }
        else
        {
          b[ldb * iN + iM] = 0;
        }

        break;

      case UnaryType::Identity:
        if (trans_b == true)
        {
          b[ldb * iM + iN] = a[lda * iN + iM];
        }
        else
        {
          b[ldb * iN + iM] = a[lda * iN + iM];
        }
        break;

      case UnaryType::ReLu:
        if (trans_b == true)
        {
          b[ldb * iM + iN] = std::max(a[lda * iN + iM], 0.f);
        }
        else
        {
          b[ldb * iN + iM] = std::max(a[lda * iN + iM], 0.f);
        }
        break;

      default:
        FAIL("Found unary invalid type for testing");
        break;
      }
    }
  }
}

void UnaryTestFixture::verify_matrix(const float *__restrict__ expected, const float *__restrict__ result, uint32_t size)
{
  for (size_t i = 0; i < size; i++)
  {
    CAPTURE(i, result[i], expected[i]);

    if (std::isnan(expected[i]))
    {
      REQUIRE_THAT(result[i], Catch::Matchers::IsNaN());
    }
    else
    {
      REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
    }
  }
}

void UnaryTestFixture::fill_random_matrix(float *matrix, uint32_t size)
{
  for (int64_t i = 0; i < size; i++)
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

    matrix[i] = numerator / denominator * (1 - (2 * (i % 2)));
  }
}

void UnaryTestFixture::fill_counting_matrix(float *matrix, uint32_t size)
{
  for (int64_t i = 0; i < size; i++)
  {
    matrix[i] = i * counting_state;
  }
  counting_state *= -1;
}
