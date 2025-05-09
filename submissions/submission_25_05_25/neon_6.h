#ifndef NEON_6_H
#define NEON_6_H

#include <cstdint>
#include <ctime>

extern "C"
{
    /**
     * @brief Matmul that loops over the NMK dimension of an original matmul of (M=64, N=64, K=64) now with loop over K=64.
     * @param a pointer to column-major matrix A.
     * @param b pointer to column-major matrix B.
     * @param c pointer to column-major matrix C.
     * @param lda leading dimension of A.
     * @param ldb leading dimension of B.
     * @param ldc leading dimension of C.
     **/
    void matmul_64_48_64(float const* a, float const* b, float* c, int64_t lda, int64_t ldb, int64_t ldc);

    /**
     * @brief Batch-reduce GEMM that computes: C+=sum(Ai*Bi) over a batch.
     * @param a           Pointer to first of a batch of column-major A matrices.
     * @param b           Pointer to first of a batch of column-major B matrices.
     * @param c           Pointer to column-major C matrix.
     * @param ld_a        Leading dimension of A.
     * @param ld_b        Leading dimension of B.
     * @param ld_c        Leading dimension of C.
     * @param br_stride_a Stride (in elements, not bytes) between A matrices.
     * @param br_stride_b Stride (in elements, not bytes) between B matrices.
     **/
    void matmul_64_48_64_16(float const* a, float const* b, float* c, int64_t ld_a, int64_t ld_b, int64_t ld_c,
        int64_t br_stride_a, int64_t br_stride_b);
}

/**
 * @brief Fill the given matrix with random values.
 *
 * @tparam TSize The total size of the matrix.
 * @param matrix The matrix to write to.
 */
template <uint32_t TSize>
void fill_random_matrix(float(&matrix)[TSize])
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
void fill_counting_matrix(float(&matrix)[TSize])
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
void copy_matrix(float(&input)[TSize], float(&output)[TSize])
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
 * @param batch_stride_a The elements to jump to the next matrix a in the batch.
 * @param batch_stride_b The elements to jump to the next matrix b in the batch.
 */
template <uint32_t TMDim, uint32_t TNDim, uint32_t TKDim, uint32_t TBatchDim>
void naive_matmul_M_N_K_Batch(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t batch_stride_a, int64_t batch_stride_b)
{
    for (size_t iB = 0; iB < TBatchDim; iB++)
    {
        for (size_t iM = 0; iM < TMDim; iM++)
        {
            for (size_t iN = 0; iN < TNDim; iN++)
            {
                for (size_t iK = 0; iK < TKDim; ++iK)
                {
                    c[iM + iN * ldc] += a[iM + iK * lda + iB * batch_stride_a] * b[iK + iN * ldb + iB * batch_stride_b];
                }
            }
        }
    }
}
#endif // NEON_6_H