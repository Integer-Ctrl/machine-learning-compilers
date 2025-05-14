#include "../../main/kernels/br_matmul_16m_lt4nRest_k.h"
#include "matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=4+[1-3], K=1, B=1) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 1;
  const uint32_t B = 1;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=4+[1-3], K=1, B=5) jited br gemm correctness counting data", "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 1;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=4+[1-3], K=18, B=5) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=4+[1-3], K=18, B=5) jited br gemm correctness counting data", "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=4*50+[1-3], K=18, B=5) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 4 * 50 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=4*50+[1-3], K=18, B=5) jited br gemm correctness counting data",
          "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 4 * 50 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16*7, N=4*10+[1-3], K=18, B=5) jited br gemm correctness random data",
          "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 7;
  const uint32_t N = 4 * 10 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16*7, N=4*10+[1-3], K=18, B=5) jited br gemm correctness counting data",
          "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 7;
  const uint32_t N = 4 * 10 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16*4, N=4*16+[1-3], K=1, B=5) jited br gemm correctness random data",
          "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 4;
  const uint32_t N = 4 * 16 + NRest;
  const uint32_t K = 1;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16*4, N=4*16+[1-3], K=1, B=5) jited br gemm correctness couting data",
          "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 4;
  const uint32_t N = 4 * 16 + NRest;
  const uint32_t K = 1;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=0+[1-3], K=18, B=5) jited br gemm correctness random data", "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 0 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}

TEST_CASE("Test br_matmul_16m_lt4nRest_k (M=16, N=0+[1-3], K=18, B=5) jited br gemm correctness counting data", "[jit][correctness][gemm]")
{
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16;
  const uint32_t N = 0 + NRest;
  const uint32_t K = 18;
  const uint32_t B = 5;
  GemmMxNxKxBatchTestFixture gemmTest(M, N, K, B);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::br_matmul_16m_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, B, N % 4);
  gemmTest.RunTest(M, K, M, M * K, K * N);
}