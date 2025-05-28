#include "BaseGeneration.test.h"
#include "kernels/matmul.test.h"
#include <cmath>
#include <numeric>

void GenerationTest::fill_random_matrix(float *matrix, uint32_t size)
{
  std::srand(std::time(0));
  for (size_t i = 0; i < size; i++)
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

void GenerationTest::fill_counting_matrix(float *matrix, uint32_t size)
{
  for (size_t i = 0; i < size; i++)
  {
    matrix[i] = i;
  }
}

void GenerationTest::naive_matmul_M_N_K_Batch(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int64_t lda,
                                              int64_t ldb, int64_t ldc, int64_t batch_stride_a, int64_t batch_stride_b)
{
  for (size_t iB = 0; iB < BatchSize; iB++)
  {
    for (size_t iM = 0; iM < M; iM++)
    {
      for (size_t iN = 0; iN < N; iN++)
      {
        for (size_t iK = 0; iK < K; ++iK)
        {
          c[iM + iN * ldc] += a[iM + iK * lda + iB * batch_stride_a] * b[iK + iN * ldb + iB * batch_stride_b];
        }
      }
    }
  }
}

void GenerationTest::verify_matmul(const float *__restrict__ expected, const float *__restrict__ result, uint32_t size)
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

GenerationTest::GenerationTest(uint32_t M, uint32_t N, uint32_t K) : GenerationTest(M, N, K, 1)
{
}


GenerationTest::GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc)
    : GenerationTest(M, N, K, 1, lda, ldb, ldc, lda * K, ldb * N)
{
}

GenerationTest::GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize)
    : GenerationTest(M, N, K, BatchSize, M, K, M, M * K, K * N)
{
}

GenerationTest::GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize, uint32_t matrix_a_sizes, uint32_t matrix_b_sizes,
                               uint32_t matrix_c_sizes)
    : M(M), N(N), K(K), BatchSize(BatchSize)
{
  matrix_a = std::vector<float>(matrix_a_sizes);
  matrix_b = std::vector<float>(matrix_b_sizes);
  matrix_c = std::vector<float>(matrix_c_sizes);
  matrix_c_verify = std::vector<float>(matrix_c_sizes);
}

GenerationTest::GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize, uint32_t lda, uint32_t ldb, uint32_t ldc,
                               uint32_t batch_stride_a, uint32_t batch_stride_b)
    : M(M), N(N), K(K), BatchSize(BatchSize), lda(lda), ldb(ldb), ldc(ldc), batch_stride_a(batch_stride_a), batch_stride_b(batch_stride_b)
{
  matrix_a = std::vector<float>(batch_stride_a * BatchSize);
  matrix_b = std::vector<float>(batch_stride_b * BatchSize);
  matrix_c = std::vector<float>(ldc * N);
  matrix_c_verify = std::vector<float>(ldc * N);
}

void GenerationTest::SetUp(TestInfill fillType)
{
  switch (fillType)
  {
  case TestInfill::Random:
    fill_random_matrix(matrix_a.data(), matrix_a.size());
    fill_random_matrix(matrix_b.data(), matrix_b.size());
    fill_random_matrix(matrix_c.data(), matrix_c.size());
    break;
  case TestInfill::Counting:
    fill_counting_matrix(matrix_a.data(), matrix_a.size());
    fill_counting_matrix(matrix_b.data(), matrix_b.size());
    fill_counting_matrix(matrix_c.data(), matrix_c.size());
    break;
  default:
    FAIL("Undefined infill type found.");
    break;
  }

  std::copy(matrix_c.begin(), matrix_c.end(), matrix_c_verify.begin());
}

void GenerationTest::naive_unary_M_N(const float *a, float *b, int64_t lda, int64_t ldb, bool trans_b, UnaryType type)
{
  for (size_t iK = 0; iK < K; iK++)
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
            b[ldb * iM + iN] = a[lda * iK + iM];
          }
          else
          {
            b[ldb * iN + iM] = a[lda * iK + iM];
          }
          break;

        case UnaryType::ReLu:
          if (trans_b == true)
          {
            b[ldb * iM + iN] = std::max(a[lda * iK + iM], 0.f);
          }
          else
          {
            b[ldb * iN + iM] = std::max(a[lda * iK + iM], 0.f);
          }
          break;

        default:
          FAIL("Found unary invalid type for testing");
          break;
        }
      }
    }
  }
}

void GenerationTest::SetKernel(mini_jit::Brgemm::kernel_t kernel)
{
  GenerationTest::kernel = kernel;
}

void GenerationTest::RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a,
                             const uint32_t batch_stride_b)
{
  if (kernel == nullptr)
  {
    // Do this because Catch2 does not support fail messages with REQUIRE :(
    FAIL("The kernel should be set before the test is executed.");
  }

  // Verification of same lda, ldb, batch_stride_a, batch_stride_b
  REQUIRE(GenerationTest::lda == lda);
  REQUIRE(GenerationTest::ldb == ldb);
  REQUIRE(GenerationTest::ldc == ldc);
  REQUIRE(GenerationTest::batch_stride_a == batch_stride_a);
  REQUIRE(GenerationTest::batch_stride_b == batch_stride_b);

  // Run matmuls
  kernel(matrix_a.data(), matrix_b.data(), matrix_c.data(), lda, ldb, ldc, batch_stride_a, batch_stride_b);

  naive_matmul_M_N_K_Batch(matrix_a.data(), matrix_b.data(), matrix_c_verify.data(), lda, ldb, ldc, batch_stride_a, batch_stride_b);

  verify_matmul(matrix_c_verify.data(), matrix_c.data(), ldc * N);
}