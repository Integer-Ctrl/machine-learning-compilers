#include "../../main/kernels/br_matmul_16m_4n_k.h"
#include "matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

TEST_CASE("Test br_matmul_16m_4n_k (M=16, N=4, K=1, B=1) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16;
  const uint32_t N = 4;
  const uint32_t K = 1;
  const uint32_t B = 1;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16, N=4, K=1, B=5) jited br gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16;
  const uint32_t N = 4;
  const uint32_t K = 1;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16, N=4, K=18, B=5) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16;
  const uint32_t N = 4;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16, N=4, K=18, B=5) jited br gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16;
  const uint32_t N = 4;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16, N=4*50, K=18, B=5) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16;
  const uint32_t N = 4 * 50;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16, N=4*50, K=18, B=5) jited br gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16;
  const uint32_t N = 4 * 50;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16*7, N=4*10, K=18, B=5) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7;
  const uint32_t N = 4 * 10;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16*7, N=4*10, K=18, B=5) jited br gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7;
  const uint32_t N = 4 * 10;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16*4, N=4*16, K=1, B=5) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4;
  const uint32_t N = 4 * 16;
  const uint32_t K = 1;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_4n_k (M=16*4, N=4*16, K=1, B=5) jited br gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4;
  const uint32_t N = 4 * 16;
  const uint32_t K = 1;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K, B);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}