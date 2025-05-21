#include "neon_7.h"
#include <benchmark/benchmark.h>

class Trans8x8Fixture : public benchmark::Fixture
{
public:
  float matrix_a[8 * 8];
  float matrix_b[8 * 8];
  const uint32_t lda = 8;
  const uint32_t ldb = 8;
  double bytes;

  void SetUp(::benchmark::State &) override
  {
    bytes = 0;

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["Byte"] = benchmark::Counter(bytes, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_DEFINE_F(Trans8x8Fixture, BT_tran_8_8)(benchmark::State &state)
{
  for (auto _ : state)
  {
    trans_neon_8_8(matrix_a, matrix_b, 8, 8);
  }

  bytes = (8 * 8) * 4;  // 8x8 matrix with 4 byte (32 bit) elements
  bytes *= state.iterations();
};

BENCHMARK_REGISTER_F(Trans8x8Fixture, BT_tran_8_8)->MinWarmUpTime(1.0);  // WarmUp in seconds
