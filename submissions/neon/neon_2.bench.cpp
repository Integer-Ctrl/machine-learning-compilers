#include "neon_2.h"
#include <benchmark/benchmark.h>

class Gemm16x6x1Fixture : public benchmark::Fixture
{
public:
  float matrix_a[16 * 1];
  float matrix_b[1 * 6];
  float matrix_c[16 * 6];
  double flops;

  void SetUp(::benchmark::State &_) override
  {
    flops = 0;

    // Fill with random values
    std::srand(std::time(0));
    for (size_t i = 0; i < 16 * 1; i++)
    {
      matrix_a[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
    }
    for (size_t i = 0; i < 1 * 6; i++)
    {
      matrix_b[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
    }
    for (size_t i = 0; i < 16 * 6; i++)
    {
      matrix_c[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
    }
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["FLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_DEFINE_F(Gemm16x6x1Fixture, BM_matmul_16_6_1_simple)(benchmark::State &state)
{
  for (auto _ : state)
  {
    matmul_16_6_1_simple(matrix_a, matrix_b, matrix_c, 16, 1, 16);
  }

  flops = 4 * 6 * 4 * 2;  // 4 fmla * 4 floats each * 2 instructions (add & mul) * 6 columns
  flops *= state.iterations();
}

BENCHMARK_REGISTER_F(Gemm16x6x1Fixture, BM_matmul_16_6_1_simple)->MinWarmUpTime(1.0);  // WarmUp in seconds

BENCHMARK_DEFINE_F(Gemm16x6x1Fixture, BM_matmul_16_6_1_unrolled)(benchmark::State &state)
{
  for (auto _ : state)
  {
    matmul_16_6_1_unrolled(matrix_a, matrix_b, matrix_c, 16, 1, 16);
  }

  flops = 4 * 6 * 4 * 2;  // 4 fmla * 4 floats each * 2 instructions (add & mul) * 6 columns
  flops *= state.iterations();
}

BENCHMARK_REGISTER_F(Gemm16x6x1Fixture, BM_matmul_16_6_1_unrolled)->MinWarmUpTime(1.0);  // WarmUp in seconds
