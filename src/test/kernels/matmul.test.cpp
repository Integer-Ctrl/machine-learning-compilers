#include "matmul.test.h"
#include <cmath>

void GemmMxNxKxBatchTestFixture::_RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a,
                                          const uint32_t batch_stride_b)
{
  if (native_kernel.get_size() <= 0)
  {
    INFO("The kernel should contain instructions before the test is executed.");
    REQUIRE(native_kernel.get_size() > 0);
  }

  // Generate executable kernel
  native_kernel.set_kernel();
  mini_jit::Brgemm::kernel_t kernel = reinterpret_cast<mini_jit::Brgemm::kernel_t>(
    const_cast<void *>(native_kernel.get_kernel()));  // Properly cast from const void* to kernel_t

  if (GemmMxNxKxBatchTestFixture::lda != 0)
  {
    // Verification of same lda, ldb, batch_stride_a, batch_stride_b
    REQUIRE(GemmMxNxKxBatchTestFixture::lda == lda);
    REQUIRE(GemmMxNxKxBatchTestFixture::ldb == ldb);
    REQUIRE(GemmMxNxKxBatchTestFixture::ldc == ldc);
    REQUIRE(GemmMxNxKxBatchTestFixture::batch_stride_a == batch_stride_a);
    REQUIRE(GemmMxNxKxBatchTestFixture::batch_stride_b == batch_stride_b);
  }

  // Run matmuls
  kernel(matrix_a, matrix_b, matrix_c, lda, ldb, ldc, batch_stride_a, batch_stride_b);
  naive_matmul_M_N_K_Batch(matrix_a, matrix_b, matrix_c_verify, lda, ldb, ldc, batch_stride_a, batch_stride_b);

  if (lda != 0)
  {
    verify_matmul(matrix_c_verify, matrix_c, ldc * N);
  }
  else
  {
    verify_matmul(matrix_c_verify, matrix_c, M * N);
  }
};

void GemmMxNxKxBatchTestFixture::fill_random_matrix(float *matrix, uint32_t size)
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

void GemmMxNxKxBatchTestFixture::fill_counting_matrix(float *matrix, uint32_t size)
{
  for (size_t i = 0; i < size; i++)
  {
    matrix[i] = i;
  }
}

void GemmMxNxKxBatchTestFixture::naive_matmul_M_N_K_Batch(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c,
                                                          int64_t lda, int64_t ldb, int64_t ldc, int64_t batch_stride_a,
                                                          int64_t batch_stride_b)
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

void GemmMxNxKxBatchTestFixture::verify_matmul(const float *__restrict__ expected, const float *__restrict__ result, uint32_t size)
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

GemmMxNxKxBatchTestFixture::GemmMxNxKxBatchTestFixture(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize)
    : M(M), N(N), K(K), BatchSize(BatchSize)
{

  matrix_a = new float[M * K * BatchSize];
  matrix_b = new float[K * N * BatchSize];
  matrix_c = new float[M * N];
  matrix_c_verify = new float[M * N];
}

GemmMxNxKxBatchTestFixture::GemmMxNxKxBatchTestFixture(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize, uint32_t lda, uint32_t ldb,
                                                       uint32_t ldc, uint32_t batch_stride_a, uint32_t batch_stride_b)
    : M(M), N(N), K(K), BatchSize(BatchSize), lda(lda), ldb(ldb), ldc(ldc), batch_stride_a(batch_stride_a), batch_stride_b(batch_stride_b)
{
  REQUIRE(lda >= M);
  REQUIRE(ldb >= K);
  REQUIRE(ldc >= M);
  REQUIRE(batch_stride_a >= lda * K);
  REQUIRE(batch_stride_b >= ldb * N);

  matrix_a = new float[batch_stride_a * BatchSize];
  matrix_b = new float[batch_stride_b * BatchSize];
  matrix_c = new float[ldc * N];
  matrix_c_verify = new float[ldc * N];
}

GemmMxNxKxBatchTestFixture::~GemmMxNxKxBatchTestFixture()
{
  delete[] matrix_a;
  delete[] matrix_b;
  delete[] matrix_c;
  delete[] matrix_c_verify;
}

void GemmMxNxKxBatchTestFixture::SetUp(TestInfill fillType)
{
  if (lda != 0)
  {
    switch (fillType)
    {
    case TestInfill::Random:
      fill_random_matrix(matrix_a, batch_stride_a * BatchSize);
      fill_random_matrix(matrix_b, batch_stride_b * BatchSize);
      fill_random_matrix(matrix_c, ldc * N);
      break;
    case TestInfill::Counting:
      fill_counting_matrix(matrix_a, batch_stride_a * BatchSize);
      fill_counting_matrix(matrix_b, batch_stride_b * BatchSize);
      fill_counting_matrix(matrix_c, ldc * N);
      break;
    default:
      FAIL("Undefined infill type found.");
      break;
    }

    std::copy(matrix_c, matrix_c + ldc * N, matrix_c_verify);
  }
  else
  {
    switch (fillType)
    {
    case TestInfill::Random:
      fill_random_matrix(matrix_a, M * K * BatchSize);
      fill_random_matrix(matrix_b, K * N * BatchSize);
      fill_random_matrix(matrix_c, M * N);
      break;
    case TestInfill::Counting:
      fill_counting_matrix(matrix_a, M * K * BatchSize);
      fill_counting_matrix(matrix_b, K * N * BatchSize);
      fill_counting_matrix(matrix_c, M * N);
      break;
    default:
      FAIL("Undefined infill type found.");
      break;
    }

    std::copy(matrix_c, matrix_c + M * N, matrix_c_verify);
  }
}

GemmMxNxKTestFixture::GemmMxNxKTestFixture(uint32_t M, uint32_t N, uint32_t K) : GemmMxNxKxBatchTestFixture(M, N, K, 1)
{
}

GemmMxNxKTestFixture::GemmMxNxKTestFixture(uint32_t M, uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc)
    : GemmMxNxKxBatchTestFixture(M, N, K, 1, lda, ldb, ldc, lda * K, ldb * N)
{
}

GemmMxNxKTestFixture::~GemmMxNxKTestFixture()
{
}
