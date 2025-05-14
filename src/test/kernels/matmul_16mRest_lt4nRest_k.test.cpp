#include "../../main/kernels/matmul_16mRest_lt4nRest_k.h"
#include "matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=4+[1-3], K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 1;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=4+[1-3], K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 1;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=4+[1-3], K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=4+[1-3], K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 4 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=4*50+[1-3], K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 4 * 50 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=4*50+[1-3], K=18) jited gemm correctness counting data",
          "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 4 * 50 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16*7+[1-15], N=4*10+[1-3], K=18) jited gemm correctness random data",
          "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 7 + MRest;
  const uint32_t N = 4 * 10 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16*7+[1-15], N=4*10+[1-3], K=18) jited gemm correctness counting data",
          "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 7 + MRest;
  const uint32_t N = 4 * 10 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16*4+[1-15], N=4*16+[1-3], K=1) jited gemm correctness random data",
          "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 4 + MRest;
  const uint32_t N = 4 * 16 + NRest;
  const uint32_t K = 1;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16*4+[1-15], N=4*16+[1-3], K=1) jited gemm correctness couting data",
          "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 * 4 + MRest;
  const uint32_t N = 4 * 16 + NRest;
  const uint32_t K = 1;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=0+[1-3], K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 0 + NRest;
  const uint32_t K = 1;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=0+[1-3], K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 0 + NRest;
  const uint32_t K = 1;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=0+[1-3], K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 0 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_lt4nRest_k (M=16+[1-15], N=0+[1-3], K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto NRest = GENERATE(range(1u, 3u + 1u, 1u));
  const uint32_t M = 16 + MRest;
  const uint32_t N = 0 + NRest;
  const uint32_t K = 18;
  CAPTURE(MRest, NRest);
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_lt4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}