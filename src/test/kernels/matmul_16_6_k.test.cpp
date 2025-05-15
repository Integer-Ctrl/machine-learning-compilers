#include "../../main/kernels/matmul_16_6_k.h"
#include "matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 1);
  gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 1);
  gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=128) jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 128);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 128);
  gemmTest.RunTest(16, 128, 16);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=128) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 128);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 128);
  gemmTest.RunTest(16, 128, 16);
}

// ============================================
// Leading dimension larger than size
// ============================================

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=1) higher leading jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1, 20, 10, 20);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 1);
  gemmTest.RunTest(20, 10, 20);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=1) higher leading jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1, 20, 10, 20);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 1);
  gemmTest.RunTest(20, 10, 20);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=128) higher leading jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 128, 20, 200, 20);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 128);
  gemmTest.RunTest(20, 200, 20);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=128) higher leading jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 128, 20, 200, 20);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 128);
  gemmTest.RunTest(20, 200, 20);
}