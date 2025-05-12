#ifndef MLC_neon_2_H
#define MLC_neon_2_H

#include <cstdint>
#include <cstdlib>
#include <ctime>

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
  void matmul_16_6_1_simple(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int64_t lda, int64_t ldb,
                            int64_t ldc);

  /**
   * @param a pointer to column-major matrix A.
   * @param b pointer to column-major matrix B.
   * @param c pointer to column-major matrix C.
   * @param lda leading dimension of A.
   * @param ldb leading dimension of B.
   * @param ldc leading dimension of C.
   **/
  void matmul_16_6_1_unrolled(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int64_t lda, int64_t ldb,
                              int64_t ldc);
}

void naive_matmul_16_6_1(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int64_t _, int64_t ldb,
                         int64_t ldc)
{
  for (size_t i = 0; i < 6; i++)
  {
    for (size_t j = 0; j < 16; j++)
    {
      c[j + i * ldc] += a[j] * b[i * ldb];
    }
  }
}

void fill_matmul_16_6_1(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, float *__restrict__ verify_c)
{

  // Fill with random values
  std::srand(std::time(0));
  for (size_t i = 0; i < 16 * 1; i++)
  {
    a[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
  }
  for (size_t i = 0; i < 1 * 6; i++)
  {
    b[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
  }
  for (size_t i = 0; i < 16 * 6; i++)
  {
    c[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
    verify_c[i] = c[i];
  }
}
#endif  // MLC_neon_2_H