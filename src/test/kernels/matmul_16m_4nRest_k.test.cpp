#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "../../main/kernels/matmul_16m_4nRest_k.h"
#include "matmul.test.h"

// =====================================================
// Rest 1 Tests
// =====================================================
TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+1, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 1;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+1, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 1;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+1, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 1;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+1, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 1;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4*50+1, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 * 50 + 1;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4*50+1, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 * 50 + 1;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*7, N=4*10+1, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 7;
    const uint32_t N = 4 * 10 + 1;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*7, N=4*10+1, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 7;
    const uint32_t N = 4 * 10 + 1;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*4, N=4*16+1, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 4;
    const uint32_t N = 4 * 16 + 1;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*4, N=4*16+1, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 4;
    const uint32_t N = 4 * 16 + 1;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest 2 Tests
// =====================================================
TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 2;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+2, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 2;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 2;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 2;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4*50+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 * 50 + 2;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4*50+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 * 50 + 2;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*7, N=4*10+2, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 7;
    const uint32_t N = 4 * 10 + 2;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*7, N=4*10+2, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 7;
    const uint32_t N = 4 * 10 + 2;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*4, N=4*16+2, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 4;
    const uint32_t N = 4 * 16 + 2;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*4, N=4*16+2, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 4;
    const uint32_t N = 4 * 16 + 2;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

// =====================================================
// Rest 3 Tests
// =====================================================
TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+3, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 3;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+3, K=1) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 3;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+3, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 3;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4+3, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 + 3;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4*50+3, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 * 50 + 3;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16, N=4*50+3, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16;
    const uint32_t N = 4 * 50 + 3;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*7, N=4*10+3, K=18) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 7;
    const uint32_t N = 4 * 10 + 3;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*7, N=4*10+3, K=18) jited gemm correctness counting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 7;
    const uint32_t N = 4 * 10 + 3;
    const uint32_t K = 18;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*4, N=4*16+3, K=1) jited gemm correctness random data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 4;
    const uint32_t N = 4 * 16 + 3;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Random);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}

TEST_CASE("Test matmul_16m_4nRest_k (M=16*4, N=4*16+3, K=1) jited gemm correctness couting data", "[jit][correctness][gemm]")
{
    const uint32_t M = 16 * 4;
    const uint32_t N = 4 * 16 + 3;
    const uint32_t K = 1;
    GemmMxNxKTestFixture<M, N, K> gemmTest;
    gemmTest.SetUp(TestInfill::Counting);
    mini_jit::kernels::matmul_16m_4nRest_k(gemmTest.native_kernel, M / 16, N / 4, K, N % 4);
    gemmTest.RunTest(M, K, M);
}