#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "../../main/kernels/matmul_16m_4n_k.h"
#include "matmul.test.h"

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 4, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, 1);
    gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    GemmMxNxKTestFixture<16, 4, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, 1);
    gemmTest.RunTest(16, 1, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t K = 18;
    GemmMxNxKTestFixture<16, 4, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, K);
    gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t K = 18;
    GemmMxNxKTestFixture<16, 4, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, 1, K);
    gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4*50, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t K = 18;
    const uint32_t N = 4 * 50;
    GemmMxNxKTestFixture<16, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    CAPTURE(N / 4);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, N / 4, K);
    gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16, N=4*50, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t K = 18;
    const uint32_t N = 4 * 50;
    GemmMxNxKTestFixture<16, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    CAPTURE(N / 4);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, 1, N / 4, K);
    gemmTest.RunTest(16, K, 16);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*7, N=4*10, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t K = 18;
    const uint32_t N = 4 * 10;
    const uint32_t M = 16 * 7;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    CAPTURE(M / 16, N / 4);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*7, N=4*10, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t K = 18;
    const uint32_t N = 4 * 10;
    const uint32_t M = 16 * 7;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    CAPTURE(M / 16, N / 4);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, K);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*4, N=4*16, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t N = 4 * 16;
    const uint32_t M = 16 * 4;
    GemmMxNxKTestFixture<M, N, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, 1);
    gemmTest.RunTest(M, 1, M);
}

TEST_CASE("Test matmul_16m_4n_k (M=16*4, N=4*16, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
    const uint32_t N = 4 * 16;
    const uint32_t M = 16 * 4;
    GemmMxNxKTestFixture<M, N, 1> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4n_k(gemmTest.native_kernel, M / 16, N / 4, 1);
    gemmTest.RunTest(M, 1, M);
}