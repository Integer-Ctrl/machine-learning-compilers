#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "../../main/kernels/matmul_16_6_k.h"
#include "matmul.test.h"


TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 6, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 1);
    gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 6, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 1);
    gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=128) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 6, 128> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 128);
    gemmTest.RunTest(16, 128, 16);
}

TEST_CASE("Test matmul_16_6_k (M=16, N=6, K=128) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 6, 128> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16_6_k(gemmTest.native_kernel, 128);
    gemmTest.RunTest(16, 128, 16);
}
