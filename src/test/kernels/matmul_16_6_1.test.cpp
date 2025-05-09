#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "../../main/kernels/matmul_16_6_1.h"
#include "matmul.test.h"

TEST_CASE("Test matmul_16_6_1 jited gemm correctness random data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 6, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16_6_1(gemmTest.native_kernel);
    gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16_6_1 jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 6, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16_6_1(gemmTest.native_kernel);
    gemmTest.RunTest(16, 1, 16);
}