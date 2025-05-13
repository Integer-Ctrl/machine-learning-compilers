#include "../../main/kernels/matmul_16mRest_4nRest_k.h"
#include "matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

// =====================================================
// Rest m=1, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+1, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 1;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+1, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 1;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+1, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 1;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+1, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 1;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+1, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 1;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+1, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 1;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+1, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 1;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+1, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 1;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+1, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 1;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+1, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 1;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=2, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+2, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 2;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+2, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 2;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+2, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 2;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+2, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 2;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+2, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 2;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+2, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 2;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+2, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 2;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+2, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 2;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+2, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 2;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+2, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 2;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=3, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+3, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 3;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+3, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 3;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+3, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 3;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+3, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 3;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+3, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 3;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+3, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 3;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+3, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 3;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+3, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 3;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+3, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 3;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+3, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 3;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=4, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+4, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 4;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+4, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 4;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+4, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 4;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+4, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 4;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+4, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 4;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+4, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 4;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+4, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 4;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+4, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 4;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+4, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 4;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+4, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 4;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=5, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+5, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 5;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+5, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 5;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+5, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 5;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+5, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 5;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+5, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 5;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+5, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 5;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+5, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 5;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+5, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 5;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+5, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 5;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+5, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 5;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=6, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+6, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 6;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+6, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 6;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+6, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 6;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+6, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 6;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+6, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 6;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+6, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 6;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+6, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 6;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+6, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 6;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+6, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 6;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+6, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 6;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=7, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+7, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 7;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+7, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 7;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+7, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 7;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+7, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 7;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+7, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 7;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+7, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 7;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+7, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 7;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+7, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 7;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+7, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 7;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+7, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 7;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=8, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+8, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 8;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+8, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 8;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+8, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 8;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+8, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 8;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+8, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 8;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+8, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 8;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+8, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 8;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+8, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 8;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+8, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 8;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+8, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 8;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=9, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+9, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 9;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+9, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 9;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+9, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 9;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+9, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 9;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+9, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 9;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+9, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 9;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+9, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 9;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+9, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 9;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+9, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 9;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+9, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 9;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=10, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+10, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 10;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+10, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 10;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+10, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 10;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+10, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 10;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+10, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 10;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+10, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 10;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+10, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 10;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+10, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 10;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+10, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 10;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+10, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 10;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=11, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+11, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 11;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+11, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 11;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+11, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 11;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+11, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 11;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+11, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 11;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+11, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 11;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+11, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 11;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+11, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 11;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+11, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 11;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+11, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 11;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=12, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+12, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 12;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+12, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 12;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+12, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 12;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+12, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 12;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+12, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 12;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+12, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 12;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+12, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 12;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+12, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 12;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+12, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 12;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+12, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 12;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=13, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+13, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 13;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+13, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 13;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+13, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 13;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+13, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 13;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+13, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 13;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+13, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 13;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+13, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 13;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+13, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 13;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+13, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 13;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+13, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 13;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=14, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+14, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 14;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+14, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 14;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+14, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 14;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+14, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 14;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+14, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 14;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+14, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 14;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+14, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 14;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+14, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 14;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+14, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 14;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+14, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 14;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest m=15, n=2 Tests
// =====================================================
TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+15, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 15;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+15, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 15;
  const uint32_t N = 4 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+15, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 15;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+15, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 15;
  const uint32_t N = 4 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+15, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 15;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16+15, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 + 15;
  const uint32_t N = 4 * 50 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+15, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 15;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*7+15, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 7 + 15;
  const uint32_t N = 4 * 10 + 2;
  const uint32_t K = 18;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+15, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 15;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16mRest_4nRest_k (M=16*4+15, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
  const uint32_t M = 16 * 4 + 15;
  const uint32_t N = 4 * 16 + 2;
  const uint32_t K = 1;
  GemmMxNxKTestFixture<M, N, K> gemmTest;
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16mRest_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, M % 16, N % 4);
  gemmTest.RunTest(M, K, M);
}