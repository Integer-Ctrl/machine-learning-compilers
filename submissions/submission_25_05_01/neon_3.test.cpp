#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include "neon_3.h"

template <uint TSize>
void verify_matmul(const float (&expected)[TSize], const float (&result)[TSize])
{
    for (size_t i = 0; i < TSize; i++)
    {
        CAPTURE(i, result[i], expected[i]);
        REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
    }
}

TEST_CASE("Test 16x6x64 gemm correctness random data", "[neon_3][correctness][gemm]")
{
    float matrix_a[16 * 64];
    float matrix_b[64 * 6];
    float matrix_c[16 * 6];
    float matrix_c_verify[16 * 6];

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
    fill_random_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_16_6_64(matrix_a, matrix_b, matrix_c, 16, 64, 16);
    naive_matmul_M_N_K<16, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 16, 64, 16);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 16x6x64 gemm correctness counting data", "[neon_3][correctness][gemm]")
{
    float matrix_a[16 * 64];
    float matrix_b[64 * 6];
    float matrix_c[16 * 6];
    float matrix_c_verify[16 * 6];

    fill_counting_matrix(matrix_a);
    fill_counting_matrix(matrix_b);
    fill_counting_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_16_6_64(matrix_a, matrix_b, matrix_c, 16, 64, 16);
    naive_matmul_M_N_K<16, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 16, 64, 16);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 64x6x64 gemm correctness random data", "[neon_3][correctness][gemm]")
{
    float matrix_a[64 * 64];
    float matrix_b[64 * 6];
    float matrix_c[64 * 6];
    float matrix_c_verify[64 * 6];

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
    fill_random_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_64_6_64(matrix_a, matrix_b, matrix_c, 64, 64, 64);
    naive_matmul_M_N_K<64, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 64, 64, 64);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 64x6x64 gemm correctness counting data", "[neon_3][correctness][gemm]")
{
    float matrix_a[64 * 64];
    float matrix_b[64 * 6];
    float matrix_c[64 * 6];
    float matrix_c_verify[64 * 6];

    fill_counting_matrix(matrix_a);
    fill_counting_matrix(matrix_b);
    fill_counting_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_64_6_64(matrix_a, matrix_b, matrix_c, 64, 64, 64);
    naive_matmul_M_N_K<64, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 64, 64, 64);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 64x48x64 gemm correctness random data", "[neon_3][correctness][gemm]")
{
    float matrix_a[64 * 64];
    float matrix_b[64 * 48];
    float matrix_c[64 * 48];
    float matrix_c_verify[64 * 48];

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
    fill_random_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_64_48_64(matrix_a, matrix_b, matrix_c, 64, 64, 64);
    naive_matmul_M_N_K<64, 48, 64>(matrix_a, matrix_b, matrix_c_verify, 64, 64, 64);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 64x48x64 gemm correctness counting data", "[neon_3][correctness][gemm]")
{
    float matrix_a[64 * 64];
    float matrix_b[64 * 48];
    float matrix_c[64 * 48];
    float matrix_c_verify[64 * 48];

    fill_counting_matrix(matrix_a);
    fill_counting_matrix(matrix_b);
    fill_counting_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_64_48_64(matrix_a, matrix_b, matrix_c, 16, 64, 16);
    naive_matmul_M_N_K<64, 48, 64>(matrix_a, matrix_b, matrix_c_verify, 16, 64, 16);

    verify_matmul(matrix_c_verify, matrix_c);
}