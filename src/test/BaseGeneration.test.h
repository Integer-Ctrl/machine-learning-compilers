#ifndef MINI_JIT_BASEGENERATION_TEST_H
#define MINI_JIT_BASEGENERATION_TEST_H

#include <cstdint>
#include <catch2/catch_test_macros.hpp>
#include "kernels/matmul.test.h"

class GenerationTest
{
private:
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t BatchSize;
    float* matrix_a;
    float* matrix_b;
    float* matrix_c;
    float* matrix_c_verify;
    mini_jit::Brgemm::kernel_t kernel = nullptr;

    /**
     * @brief Fills the given matrix with random values.
     *
     * @param matrix The matrix to fill.
     * @param size The total size of the matrix.
     */
    void fill_random_matrix(float* matrix, uint32_t size);

    /**
     * @brief Fills the given matrix with counting values starting from 0.
     *
     * @param matrix The matrix to fill.
     * @param size The total size of the matrix.
     */
    void fill_counting_matrix(float* matrix, uint32_t size);

    /**
     * @brief Does a naive matmul for verification usage.
     *
     * @param a The a matrix.
     * @param b The b matrix.
     * @param c The c matrix.
     * @param lda The leading dimension of matrix a.
     * @param ldb The leading dimension of matrix b.
     * @param ldc The leading dimension of matrix c.
     * @param batch_stride_a The batch stride of matrix a.
     * @param batch_stride_b The batch stride of matrix b.
     */
    void naive_matmul_M_N_K_Batch(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
        int64_t lda, int64_t ldb, int64_t ldc, int64_t batch_stride_a, int64_t batch_stride_b);

    /**
     * @brief Compares the two matrices by comparing each values.
     *
     * @param expected The matrix results that are expected.
     * @param result The actual matrix values.
     * @param size The total size of the matrix.
     */
    void verify_matmul(const float* __restrict__ expected, const float* __restrict__ result, uint32_t size);

public:
    GenerationTest(uint32_t M, uint32_t N, uint32_t K);
    GenerationTest(uint32_t M, uint32_t N, uint32_t K, uint32_t BatchSize);
    ~GenerationTest();

    /**
     * @brief Set up the test fixture object.
     *
     * @param fillType Fills the matrices with the given infill type.
     */
    void SetUp(TestInfill fillType);

    /**
     * @brief Set the kernel to be used in the test.
     *
     * @param kernel The kernel to run in the test.
     */
    void SetKernel(mini_jit::Brgemm::kernel_t kernel);

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
        const uint32_t batch_stride_b);
};

#endif // MINI_JIT_BASEGENERATION_TEST_H