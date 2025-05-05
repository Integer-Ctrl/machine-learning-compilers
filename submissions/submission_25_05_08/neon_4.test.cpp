#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include "neon_4.h"

template <uint TSize>
void verify_matmul(const float (&expected)[TSize], const float (&result)[TSize])
{
    for (size_t i = 0; i < TSize; i++)
    {
        CAPTURE(i, result[i], expected[i]);
        REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
    }
}

TEST_CASE("Test 14x6x64 gemm correctness random data", "[neon_4][correctness][gemm]")
{
    float matrix_a[14 * 64];
    float matrix_b[64 * 6];
    float matrix_c[14 * 6];
    float matrix_c_verify[14 * 6];

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
    fill_random_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_14_6_64(matrix_a, matrix_b, matrix_c, 14, 64, 14);
    naive_matmul_M_N_K<14, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 14, 64, 14);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 14x6x64 gemm correctness counting data", "[neon_4][correctness][gemm]")
{
    float matrix_a[14 * 64];
    float matrix_b[64 * 6];
    float matrix_c[14 * 6];
    float matrix_c_verify[14 * 6];

    fill_counting_matrix(matrix_a);
    fill_counting_matrix(matrix_b);
    fill_counting_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_14_6_64(matrix_a, matrix_b, matrix_c, 14, 64, 14);
    naive_matmul_M_N_K<14, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 14, 64, 14);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 15x6x64 gemm correctness random data", "[neon_4][correctness][gemm]")
{
    float matrix_a[15 * 64];
    float matrix_b[64 * 6];
    float matrix_c[15 * 6];
    float matrix_c_verify[15 * 6];

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
    fill_random_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_15_6_64(matrix_a, matrix_b, matrix_c, 15, 64, 15);
    naive_matmul_M_N_K<15, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 15, 64, 15);

    verify_matmul(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 15x6x64 gemm correctness counting data", "[neon_4][correctness][gemm]")
{
    float matrix_a[15 * 64];
    float matrix_b[64 * 6];
    float matrix_c[15 * 6];
    float matrix_c_verify[15 * 6];

    fill_counting_matrix(matrix_a);
    fill_counting_matrix(matrix_b);
    fill_counting_matrix(matrix_c);
    copy_matrix(matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_15_6_64(matrix_a, matrix_b, matrix_c, 15, 64, 15);
    naive_matmul_M_N_K<15, 6, 64>(matrix_a, matrix_b, matrix_c_verify, 15, 64, 15);

    verify_matmul(matrix_c_verify, matrix_c);
}