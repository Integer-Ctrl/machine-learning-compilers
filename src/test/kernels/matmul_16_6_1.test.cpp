#include "../../main/kernels/matmul_16_6_1.h"
#include "matmul.test.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

TEST_CASE("Test matmul_16_6_1 jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16_6_1(gemmTest.native_kernel);
  gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16_6_1 jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16_6_1(gemmTest.native_kernel);
  gemmTest.RunTest(16, 1, 16);
}

// ============================================
// Leading dimension larger than size
// ============================================

TEST_CASE("Test matmul_16_6_1 higher leading dim jited gemm correctness random data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1, 20, 10, 20);
  gemmTest.SetUp(TestInfill::Random);
  mini_jit::kernels::matmul_16_6_1(gemmTest.native_kernel);
  gemmTest.RunTest(20, 10, 20);
}

TEST_CASE("Test matmul_16_6_1 higher leading dim jited gemm correctness counting data", "[jit][correctness][gemm]")
{
  GemmMxNxKTestFixture gemmTest(16, 6, 1, 20, 10, 20);
  gemmTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::matmul_16_6_1(gemmTest.native_kernel);
  gemmTest.RunTest(20, 10, 20);
}