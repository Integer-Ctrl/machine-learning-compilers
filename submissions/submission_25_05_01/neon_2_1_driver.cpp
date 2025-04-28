#include <iostream>
#include <cstdint>
#include "neon_2_1.h"

void matmul_16_6_1(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c,
                   int64_t lda, int64_t ldb, int64_t ldc)
{
    matmul_16_6_1_simple(a, b, c, lda, ldb, ldc);
    // matmul_16_6_1_unrolled(a, b, c, lda, ldb, ldc);
}

int main()
{
    float matrix_a[16 * 1];
    float matrix_b[1 * 6];
    float matrix_c[16 * 6];
    float matrix_c_verify[16 * 6];

    // Fill with random values
    fill_matmul_16_6_1(matrix_a, matrix_b, matrix_c, matrix_c_verify);

    // Run matmuls
    matmul_16_6_1(matrix_a, matrix_b, matrix_c, 16, 1, 16);
    naive_matmul_16_6_1(matrix_a, matrix_b, matrix_c_verify, 16, 1, 16);

    // Verify results
    uint32_t success_count = 0;
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < 16; j++)
        {
            success_count += (std::abs(matrix_c[j + i * 16] - matrix_c_verify[j + i * 16]) < 0.01f);
        }
    }

    std::cout << success_count / static_cast<float>(16 * 6) * 100 << "% Successful" << std::endl;

    return 0;
}