#include "../../main/kernels/matmul_16m_4n_k.h"
#include "matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 4, 1);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, 1);
  gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 4, 1);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, 1);
  gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  GemmMxNxKTestFixture gemmTest(16, 4, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, K);
  gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  GemmMxNxKTestFixture gemmTest(16, 4, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, K);
  gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4*50, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 50;
  GemmMxNxKTestFixture gemmTest(16, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, N / 4, K);
  gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4*50, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 50;
  GemmMxNxKTestFixture gemmTest(16, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, N / 4, K);
  gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*7, N=4*10, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 10;
  const uint32_t M = 16 * 7;
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*7, N=4*10, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 10;
  const uint32_t M = 16 * 7;
  GemmMxNxKTestFixture gemmTest(M, N, K);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*4, N=4*16, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t N = 4 * 16;
  const uint32_t M = 16 * 4;
  GemmMxNxKTestFixture gemmTest(M, N, 1);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, 1);
  gemmTest.RunTest(M, 1, M);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*4, N=4*16, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t N = 4 * 16;
  const uint32_t M = 16 * 4;
  GemmMxNxKTestFixture gemmTest(M, N, 1);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, 1);
  gemmTest.RunTest(M, 1, M);
}

// ============================================
// Leading dimension larger than size
// ============================================

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=1) higher leading jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 4, 1, 20, 10, 20);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, 1);
  gemmTest.RunTest(20, 10, 20);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=1) higher leading jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 4, 1, 20, 10, 20);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, 1);
  gemmTest.RunTest(20, 10, 20);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=18) higher leading jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  GemmMxNxKTestFixture gemmTest(16, 4, K, 20, K + 5, 20);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, K);
  gemmTest.RunTest(20, K + 5, 20);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=18) higher leading jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  GemmMxNxKTestFixture gemmTest(16, 4, K, 20, K + 5, 20);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, K);
  gemmTest.RunTest(20, K + 5, 20);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4*50, K=18) higher leading jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 50;
  GemmMxNxKTestFixture gemmTest(16, N, K, 20, K + 5, 20);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, N / 4, K);
  gemmTest.RunTest(20, K + 5, 20);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4*50, K=18) higher leading jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 50;
  GemmMxNxKTestFixture gemmTest(16, N, K, 20, K + 5, 20);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, N / 4, K);
  gemmTest.RunTest(20, K + 5, 20);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*7, N=4*10, K=18) higher leading jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 10;
  const uint32_t M = 16 * 7;
  GemmMxNxKTestFixture gemmTest(M, N, K, M + 5, K + 5, M + 5);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K);
  gemmTest.RunTest(M + 5, K + 5, M + 5);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*7, N=4*10, K=18) higher leading jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t K = 18;
  const uint32_t N = 4 * 10;
  const uint32_t M = 16 * 7;
  GemmMxNxKTestFixture gemmTest(M, N, K, M + 5, K + 5, M + 5);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K);
  gemmTest.RunTest(M + 5, K + 5, M + 5);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*4, N=4*16, K=1) higher leading jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t N = 4 * 16;
  const uint32_t M = 16 * 4;
  GemmMxNxKTestFixture gemmTest(M, N, 1, M + 5, 5, M + 5);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, 1);
  gemmTest.RunTest(M + 5, 5, M + 5);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*4, N=4*16, K=1) higher leading jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t N = 4 * 16;
  const uint32_t M = 16 * 4;
  GemmMxNxKTestFixture gemmTest(M, N, 1, M + 5, 5, M + 5);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, 1);
  gemmTest.RunTest(M + 5, 5, M + 5);
}