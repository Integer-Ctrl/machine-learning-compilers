#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include "neon_2_1.h"


void verify_matmul_16_6_1(const float * __restrict__ expected, const float * __restrict__ result)
{
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < 16; j++)
        {
            REQUIRE_THAT(result[j + i * 16], Catch::Matchers::WithinRel(expected[j + i * 16]));
        }   
    }
}

TEST_CASE("Test 16x6x1 simple gemm correctness random data", "[neon_2_1][correctness][gemm]") {
    float matrix_a[16*1];
    float matrix_b[1*6];
    float matrix_c[16*6];
    float matrix_c_verify[16*6];
    
    fill_matmul_16_6_1(matrix_a, matrix_b, matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_16_6_1_simple(matrix_a, matrix_b, matrix_c, 16, 1, 16);
    naive_matmul_16_6_1(matrix_a, matrix_b, matrix_c_verify, 16, 1, 16);

    verify_matmul_16_6_1(matrix_c_verify, matrix_c);
}

TEST_CASE("Test 16x6x1 unrolled gemm correctness random data", "[neon_2_1][correctness][gemm]") {
    float matrix_a[16*1];
    float matrix_b[1*6];
    float matrix_c[16*6];
    float matrix_c_verify[16*6];
    
    fill_matmul_16_6_1(matrix_a, matrix_b, matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_16_6_1_unrolled(matrix_a, matrix_b, matrix_c, 16, 1, 16);
    naive_matmul_16_6_1(matrix_a, matrix_b, matrix_c_verify, 16, 1, 16);

    verify_matmul_16_6_1(matrix_c_verify, matrix_c);
}
