#include "../../../main/Brgemm.h"
#include "../../../main/Unary.h"
#include "../../../main/kernels/unary/unary_identity.h"
#include "unary.bench.h"
#include <benchmark/benchmark.h>

class UnaryFixture : public benchmark::Fixture
{
public:
  std::vector<float> matrix_a, matrix_b;
  double bytes;

  void SetUp(::benchmark::State &state) override
  {
    bytes = 0;

    int M = state.range(0);
    int N = state.range(1);

    matrix_a.resize(M * N);
    matrix_b.resize(M * N);

    fill_random_matrix_args(matrix_a.data(), M * N);
    fill_random_matrix_args(matrix_b.data(), M * N);
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["Bytes"] = benchmark::Counter(bytes, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_DEFINE_F(UnaryFixture, BM_unary_identity)(benchmark::State &state)
{
  int M = state.range(0);
  int N = state.range(1);

  mini_jit::Kernel native_kernel;
  mini_jit::kernels::unary_identity(native_kernel, M, N);
  native_kernel.set_kernel();
  mini_jit::Unary::kernel_t kernel = reinterpret_cast<mini_jit::Unary::kernel_t>(
    const_cast<void *>(native_kernel.get_kernel()));  // Properly cast from const void* to kernel_t

  for (auto _ : state)
  {
    kernel(matrix_a.data(), matrix_b.data(), M, N);
  }

  bytes = M * N * 4 * 2 * state.iterations();  // M * N * 4 bytes (fp32) * 2 (load/store)
}

static void CustomArguments(benchmark::internal::Benchmark *b)
{
  for (int S : {50, 64, 512, 2048})
    b->Args({S, S});
}

BENCHMARK_REGISTER_F(UnaryFixture, BM_unary_identity)
  ->ArgNames({"M", "N"})
  ->DisplayAggregatesOnly(true)
  ->Apply(CustomArguments)
  ->MinWarmUpTime(1.0);  // WarmUp in seconds