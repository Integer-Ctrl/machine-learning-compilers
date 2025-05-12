#include "BaseGeneration.test.h"
#include "kernels/matmul.test.h"

void GenerationTest::fill_random_matrix(float *matrix, uint32_t size)
{
  std::srand(std::time(0));
  for (size_t i = 0; i < size; i++)
  {
    matrix[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
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
    REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
  }
}

GenerationTest::GenerationTest(uint32_t M, uint32_t N, uint32_t K) : GenerationTest(M, N, K, 1)
{
}

GenerationTest::GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize) : M(M), N(N), K(K), BatchSize(BatchSize)
{

  matrix_a = new float[M * K * BatchSize];
  matrix_b = new float[K * N * BatchSize];
  matrix_c = new float[M * N];
  matrix_c_verify = new float[M * N];
}

GenerationTest::~GenerationTest()
{
  delete[] matrix_a;
  delete[] matrix_b;
  delete[] matrix_c;
  delete[] matrix_c_verify;
}

void GenerationTest::SetUp(TestInfill fillType)
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

  // Run matmuls
  kernel(matrix_a, matrix_b, matrix_c, lda, ldb, ldc, batch_stride_a, batch_stride_b);

  naive_matmul_M_N_K_Batch(matrix_a, matrix_b, matrix_c_verify, lda, ldb, ldc, batch_stride_a, batch_stride_b);

  verify_matmul(matrix_c_verify, matrix_c, M * N);
}