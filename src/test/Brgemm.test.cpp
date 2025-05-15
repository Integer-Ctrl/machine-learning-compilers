#include "BaseGeneration.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test gemm generation (1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128],lda=M, ldb=K, and ldc=M) on random data",
          "[generation][correctness][gemm]")
{
  auto M = GENERATE(range(1u, 64u + 1u, 1u));
  auto N = GENERATE(range(1u, 64u + 1u, 1u));
  auto K = GENERATE(1u, 16u, 32u, 64u, 128u);

  CAPTURE(M, N, K);

  GenerationTest generatorTest(M, N, K);
  generatorTest.SetUp(TestInfill::Random);

  mini_jit::Brgemm gemm;
  mini_jit::Brgemm::error_t error = gemm.generate(M, N, K, 1, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);

  switch (error)
  {
  case mini_jit::Brgemm::error_t::success:
    break;
  case mini_jit::Brgemm::error_t::err_batch_reduce_size_not_supported:
    FAIL("Error batch reduce size not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_row_major_order_not_supported:
    FAIL("Error row major order not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dimension:
    FAIL("Error err wrong dimension.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dtype:
    FAIL("Error wrong dtype.");
    break;
  default:
    FAIL("Found unprocessed error type");
    break;
  }

  mini_jit::Brgemm::kernel_t kernel = gemm.get_kernel();
  generatorTest.SetKernel(kernel);
  generatorTest.RunTest(M, K, M, 0, 0);
}

TEST_CASE("Test gemm generation (1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128],lda=M, ldb=K, and ldc=M) on counting data",
          "[generation][correctness][gemm]")
{
  auto M = GENERATE(range(1u, 64u + 1u, 1u));
  auto N = GENERATE(range(1u, 64u + 1u, 1u));
  auto K = GENERATE(1u, 16u, 32u, 64u, 128u);

  CAPTURE(M, N, K);

  GenerationTest generatorTest(M, N, K);
  generatorTest.SetUp(TestInfill::Counting);

  mini_jit::Brgemm gemm;
  mini_jit::Brgemm::error_t error = gemm.generate(M, N, K, 1, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);

  switch (error)
  {
  case mini_jit::Brgemm::error_t::success:
    break;
  case mini_jit::Brgemm::error_t::err_batch_reduce_size_not_supported:
    FAIL("Error batch reduce size not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_row_major_order_not_supported:
    FAIL("Error row major order not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dimension:
    FAIL("Error err wrong dimension.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dtype:
    FAIL("Error wrong dtype.");
    break;
  default:
    FAIL("Found unprocessed error type");
    break;
  }

  mini_jit::Brgemm::kernel_t kernel = gemm.get_kernel();
  generatorTest.SetKernel(kernel);
  generatorTest.RunTest(M, K, M, 0, 0);
}

TEST_CASE("Test gemm generation (1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128],lda>M, ldb>K, and ldc>M) on random data",
          "[generation][correctness][gemm]")
{
  auto M = GENERATE(range(1u, 64u + 1u, 1u));
  auto N = GENERATE(range(1u, 64u + 1u, 1u));
  auto K = GENERATE(1u, 16u, 32u, 64u, 128u);
  const uint32_t lda = M + 5;
  const uint32_t ldb = K + 3;
  const uint32_t ldc = M + 7;

  CAPTURE(M, N, K, lda, ldb, ldc);

  GenerationTest generatorTest(M, N, K, lda, ldb, ldc);
  generatorTest.SetUp(TestInfill::Random);

  mini_jit::Brgemm gemm;
  mini_jit::Brgemm::error_t error = gemm.generate(M, N, K, 1, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);

  switch (error)
  {
  case mini_jit::Brgemm::error_t::success:
    break;
  case mini_jit::Brgemm::error_t::err_batch_reduce_size_not_supported:
    FAIL("Error batch reduce size not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_row_major_order_not_supported:
    FAIL("Error row major order not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dimension:
    FAIL("Error err wrong dimension.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dtype:
    FAIL("Error wrong dtype.");
    break;
  default:
    FAIL("Found unprocessed error type");
    break;
  }

  mini_jit::Brgemm::kernel_t kernel = gemm.get_kernel();
  generatorTest.SetKernel(kernel);
  generatorTest.RunTest(lda, ldb, ldc, lda * K, ldb * N);
}

TEST_CASE("Test gemm generation (1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128],lda>M, ldb>K, and ldc>M) on counting data",
          "[generation][correctness][gemm]")
{
  auto M = GENERATE(range(1u, 64u + 1u, 1u));
  auto N = GENERATE(range(1u, 64u + 1u, 1u));
  auto K = GENERATE(1u, 16u, 32u, 64u, 128u);
  const uint32_t lda = M + 5;
  const uint32_t ldb = K + 3;
  const uint32_t ldc = M + 7;

  CAPTURE(M, N, K, lda, ldb, ldc);

  GenerationTest generatorTest(M, N, K, lda, ldb, ldc);
  generatorTest.SetUp(TestInfill::Counting);

  mini_jit::Brgemm gemm;
  mini_jit::Brgemm::error_t error = gemm.generate(M, N, K, 1, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);

  switch (error)
  {
  case mini_jit::Brgemm::error_t::success:
    break;
  case mini_jit::Brgemm::error_t::err_batch_reduce_size_not_supported:
    FAIL("Error batch reduce size not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_row_major_order_not_supported:
    FAIL("Error row major order not supported.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dimension:
    FAIL("Error err wrong dimension.");
    break;
  case mini_jit::Brgemm::error_t::err_wrong_dtype:
    FAIL("Error wrong dtype.");
    break;
  default:
    FAIL("Found unprocessed error type");
    break;
  }

  mini_jit::Brgemm::kernel_t kernel = gemm.get_kernel();
  generatorTest.SetKernel(kernel);
  generatorTest.RunTest(lda, ldb, ldc, lda * K, ldb * N);
}
