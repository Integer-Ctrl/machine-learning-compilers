#ifndef MINIJIT_KERNELS_MATMUL_TEST_H
#define MINIJIT_KERNELS_MATMUL_TEST_H

#include <cstdint>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../main/Kernel.h"
#include "../../main/Brgemm.h"

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

/**
 * @brief Verify that all values of a matmul are equal.
 *
 * @tparam The size of the matmul
 */
template <uint TSize>
void verify_matmul(const float(&expected)[TSize], const float(&result)[TSize])
{
    for (size_t i = 0; i < TSize; i++)
    {
        CAPTURE(i, result[i], expected[i]);
        REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
    }
}

/**
 * @brief List possible fill types for matrix used in this test.
 */
enum class TestInfill
{
    /// @brief Fill with random data.
    Random,

    /// @brief Fill with couting data starting from 0.
    Counting,
};

template <uint32_t TMdim, uint32_t TNdim, uint32_t TKdim, uint32_t TBatchDim>
class GemmMxNxKxBatchTestFixture
{
public:
    float matrix_a[TMdim * TKdim * TBatchDim];
    float matrix_b[TKdim * TNdim * TBatchDim];
    float matrix_c[TMdim * TNdim];
    float matrix_c_verify[TMdim * TNdim];
    const uint32_t lda = TMdim;
    const uint32_t ldb = TKdim;
    const uint32_t ldc = TMdim;
    const uint32_t batch_stride_a = TMdim * TKdim;
    const uint32_t batch_stride_b = TKdim * TNdim;
    mini_jit::Kernel native_kernel;

    /**
     * @brief Set up the test fixture object.
     *
     * @param fillType Fills the matrices with the given infill type.
     */
    void SetUp(TestInfill fillType)
    {
        switch (fillType)
        {
        case TestInfill::Random:
            fill_random_matrix(matrix_a);
            fill_random_matrix(matrix_b);
            fill_random_matrix(matrix_c);
            break;
        case TestInfill::Counting:
            fill_counting_matrix(matrix_a);
            fill_counting_matrix(matrix_b);
            fill_counting_matrix(matrix_c);
            break;
        default:
            FAIL("Undefined infill type found.");
            break;
        }

        copy_matrix(matrix_c, matrix_c_verify);
    }

    /**
     * @brief Executes the Test von an BRGemm with the given input.
     *
     * @param lda: leading dimension of A.
     * @param ldb: leading dimension of B.
     * @param ldc: leading dimension of C.
     * @param br_stride_a: stride between two A matrices (in elements, not bytes).
     * @param br_stride_b: stride between two B matrices (in elements, not bytes).
     */
    void RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a,
        const uint32_t batch_stride_b)
    {
        _RunTest(lda, ldb, ldc, batch_stride_a, batch_stride_b);
    }

protected:
    /**
     * @brief Executes the Test von an BRGemm with the given input.
     *
     * @param lda: leading dimension of A.
     * @param ldb: leading dimension of B.
     * @param ldc: leading dimension of C.
     * @param br_stride_a: stride between two A matrices (in elements, not bytes).
     * @param br_stride_b: stride between two B matrices (in elements, not bytes).
     */
    void _RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a,
        const uint32_t batch_stride_b)
    {
        if (native_kernel.get_size() <= 0)
        {
            INFO("The kernel should contain instructions before the test is executed.");
            REQUIRE(native_kernel.get_size() > 0);
        }


        // Generate executable kernel
        native_kernel.set_kernel();
        mini_jit::Brgemm::kernel_t kernel = reinterpret_cast<mini_jit::Brgemm::kernel_t>(
            const_cast<void*>(native_kernel.get_kernel()));  // Properly cast from const void* to kernel_t

        // Run matmuls
        kernel(matrix_a, matrix_b, matrix_c, lda, ldb, ldc, batch_stride_a, batch_stride_b);
        naive_matmul_M_N_K_Batch<TMdim, TNdim, TKdim, TBatchDim>(matrix_a, matrix_b, matrix_c_verify, lda, ldb, ldc,
            batch_stride_a, batch_stride_b);

        verify_matmul(matrix_c_verify, matrix_c);
    }

};

template <uint32_t TMdim, uint32_t TNdim, uint32_t TKdim>
class GemmMxNxKTestFixture : public GemmMxNxKxBatchTestFixture<TMdim, TNdim, TKdim, 1>
{

    void RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc, const uint32_t batch_stride_a,
        const uint32_t batch_stride_b) = delete; // delete so not visible in a GemmMxNxKTestFixture object.

public:
    /**
     * @brief Executes the Test von an BRGemm with the given input.
     *
     * @param lda: leading dimension of A.
     * @param ldb: leading dimension of B.
     * @param ldc: leading dimension of C.
     */
    void RunTest(const uint32_t lda, const uint32_t ldb, const uint32_t ldc)
    {
        GemmMxNxKxBatchTestFixture<TMdim, TNdim, TKdim, 1>::_RunTest(lda, ldb, ldc, 0, 0);
    }
};

#endif // MINIJIT_KERNELS_MATMUL_TEST_H