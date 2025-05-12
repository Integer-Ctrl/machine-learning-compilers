#include "neon_6.h"
#include <benchmark/benchmark.h>

template <uint32_t TMdim, uint32_t TNdim, uint32_t TKdim, uint32_t TBatchDim> class GemmMxNxKxBatchFixture : public benchmark::Fixture
{
public:
  float matrix_a[TMdim * TKdim * TBatchDim];
  float matrix_b[TKdim * TNdim * TBatchDim];
  float matrix_c[TMdim * TNdim];
  const uint32_t lda = TMdim;
  const uint32_t ldb = TKdim;
  const uint32_t ldc = TMdim;
  const uint32_t batch_stride_a = TMdim * TKdim;
  const uint32_t batch_stride_b = TKdim * TNdim;
  double flops;

  void SetUp(::benchmark::State &) override
  {
    flops = 0;

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
    fill_random_matrix(matrix_c);
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["FLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKxBatchFixture, BM_matmul_64_48_64, 64, 48, 64, 1)(benchmark::State &state)
{
  for (auto _ : state)
  {
    matmul_64_48_64(matrix_a, matrix_b, matrix_c, lda, ldb, ldc);
  }

  flops = (64 * 48 * 64) * 2;  // M * N * K * 2 instructions (add & mul)
  flops *= state.iterations();
};

BENCHMARK_REGISTER_F(GemmMxNxKxBatchFixture, BM_matmul_64_48_64)->MinWarmUpTime(1.0);  // WarmUp in seconds

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKxBatchFixture, BM_matmul_64_48_64_16, 64, 48, 64, 16)(benchmark::State &state)
{
  for (auto _ : state)
  {
    matmul_64_48_64_16(matrix_a, matrix_b, matrix_c, lda, ldb, ldc, batch_stride_a, batch_stride_b);
  }

  flops = (64 * 48 * 64 * 16) * 2;  // M * N * K * Batch * 2 instructions (add & mul)
  flops *= state.iterations();
};

BENCHMARK_REGISTER_F(GemmMxNxKxBatchFixture, BM_matmul_64_48_64_16)->MinWarmUpTime(1.0);  // WarmUp in seconds
