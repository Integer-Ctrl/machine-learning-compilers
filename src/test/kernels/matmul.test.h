#include <cstdint>

/**
 * @brief Fill the given matrix with random values. 
 * 
 * @tparam TSize The total size of the matrix.
 * @param matrix The matrix to write to.
 */
template <uint32_t TSize>
void fill_random_matrix(float (&matrix)[TSize])
{
    std::srand(std::time(0));
    for (size_t i = 0; i < TSize; i++)
    {
        matrix[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
    }
}

/**
 * @brief Fill the given matrix with counting up values, starting from 0. 
 * 
 * @tparam TSize The total size of the matrix.
 * @param matrix The matrix to write to.
 */
template <uint32_t TSize>
void fill_counting_matrix(float (&matrix)[TSize])
{
    for (size_t i = 0; i < TSize; i++)
    {
        matrix[i] = i;
    }
}

/**
 * @brief Copy the values of matrix to another matrix.
 * 
 * @tparam TSize The equal size of the matrices.
 * @param input The matrix to copy from.
 * @param output The matrix to copy to.
 */
template <uint32_t TSize>
void copy_matrix(float (&input)[TSize], float (&output)[TSize])
{
    std::copy(std::begin(input), std::end(input), std::begin(output));
}

/**
 * @brief Naive matmul of column-major C [MxN] = A [MxK] mul B [KxN].
 * 
 * @tparam TMDim The size of the M dimension.
 * @tparam TNDim The size of the N dimension.
 * @tparam TKDim The size of the K dimension.
 * @param a The pointer of matrix A.
 * @param b The pointer of matrix B.
 * @param c The pointer of matrix C.
 * @param lda The leading dimension of A.
 * @param ldb The leading dimension of B.
 * @param ldc The leading dimension of C.
 */
template <uint32_t TMDim, uint32_t TNDim, uint32_t TKDim>
void naive_matmul_M_N_K(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c,
                        int64_t lda, int64_t ldb, int64_t ldc)
{
    for (size_t iM = 0; iM < TMDim; iM++)
    {
        for (size_t iN = 0; iN < TNDim; iN++)
        {
            for (size_t iK = 0; iK < TKDim; ++iK)
            {
                c[iM + iN * ldc] += a[iM + iK * lda] * b[iK + iN * ldb];
            }
        }
    }
}