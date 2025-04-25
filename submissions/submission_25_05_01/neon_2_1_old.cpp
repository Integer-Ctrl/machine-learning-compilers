#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>

extern "C"
{
    /**
     * @param a pointer to column-major matrix A.
     * @param b pointer to column-major matrix B.
     * @param c pointer to column-major matrix C.
     * @param lda leading dimension of A.
     * @param ldb leading dimension of B.
     * @param ldc leading dimension of C.
     **/
    void matmul_16_6_1(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, 
        int64_t lda, int64_t ldb, int64_t ldc);
}

void verify_matmul_16_6_1(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, 
    int64_t lda, int64_t ldb, int64_t ldc)
{
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < 16; j++)
        {
            c[j + i * ldc] += a[j] * b[i * ldb];
        }
    }
}

int main() 
{
    float matrix_a[16*1];
    float matrix_b[1*6];
    float matrix_c[16*6];
    float matrix_c_verify[16*6];

    // Fill with random values
    std::srand(std::time(0));
    for (size_t i = 0; i < 16*1; i++)
    {
        matrix_a[i] = (static_cast<float>(std::rand()))/(static_cast<float>(std::rand()));
    }
    for (size_t i = 0; i < 1*6; i++)
    {
        matrix_b[i] = (static_cast<float>(std::rand()))/(static_cast<float>(std::rand()));
    }
    for (size_t i = 0; i < 16*6; i++)
    {
        matrix_c[i] = (static_cast<float>(std::rand()))/(static_cast<float>(std::rand()));
        matrix_c_verify[i] = matrix_c[i];
    }
    
    matmul_16_6_1(matrix_a, matrix_b, matrix_c, 16, 1, 16);

    verify_matmul_16_6_1(matrix_a, matrix_b, matrix_c_verify, 16, 1, 16);

    uint32_t success_count = 0;
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < 16; j++)
        {
            success_count += std::abs(matrix_c[i * 16 + j] - matrix_c_verify[i * 16 + j]) < std::numeric_limits<float>::epsilon();
            std::cout << "Element " << i << ", " << j << ": asm matrix: " << matrix_c[i * 16 + j] << " cpp matrix: " << matrix_c_verify[i * 16 + j] << std::endl;
        }   
    }
    
    std::cout << success_count/static_cast<float>(16*6) << "% Successful" << std::endl;

    return 0;
}