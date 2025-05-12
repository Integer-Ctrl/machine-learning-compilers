#include "neon_4.h"
#include <benchmark/benchmark.h>

template <uint32_t TMdim, uint32_t TNdim, uint32_t TKdim> class GemmMxNxKFixture : public benchmark::Fixture
{
public:
  float matrix_a[TMdim * TKdim];
  float matrix_b[TKdim * TNdim];
  float matrix_c[TMdim * TNdim];
  double flops;

  void SetUp(::benchmark::State &_) override
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

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKFixture, BM_matmul_14_6_64, 14, 6, 64)(benchmark::State &state)
{
  for (auto _ : state)
  {
    matmul_14_6_64(matrix_a, matrix_b, matrix_c, 14, 64, 14);
  }

  flops = (14 * 6 * 64) * 2;  // M * N * K * 2 instructions (add & mul)
  flops *= state.iterations();
};

BENCHMARK_REGISTER_F(GemmMxNxKFixture, BM_matmul_14_6_64)->MinWarmUpTime(1.0);  // WarmUp in seconds

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKFixture, BM_matmul_15_6_64, 15, 6, 64)(benchmark::State &state)
{
  for (auto _ : state)
  {
    matmul_15_6_64(matrix_a, matrix_b, matrix_c, 15, 64, 15);
  }

  flops = (15 * 6 * 64) * 2;  // M * N * K * 2 instructions (add & mul)
  flops *= state.iterations();
};

BENCHMARK_REGISTER_F(GemmMxNxKFixture, BM_matmul_15_6_64)->MinWarmUpTime(1.0);  // WarmUp in seconds