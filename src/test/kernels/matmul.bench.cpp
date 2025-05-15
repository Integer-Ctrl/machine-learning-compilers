#include "matmul.bench.h"
#include "../../main/Brgemm.h"
#include <benchmark/benchmark.h>

class GemmFixture : public benchmark::Fixture
{
public:
  std::vector<float> matrix_a, matrix_b, matrix_c;
  double flops;

  void SetUp(::benchmark::State &state) override
  {
    flops = 0;

    int M = state.range(0);
    int N = state.range(1);
    int K = state.range(2);

    matrix_a.resize(M * K);
    matrix_b.resize(K * N);
    matrix_c.resize(M * N);

    fill_random_matrix_args(matrix_a.data(), M * K);
    fill_random_matrix_args(matrix_b.data(), K * N);
    fill_random_matrix_args(matrix_c.data(), M * N);
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["FLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_DEFINE_F(GemmFixture, BM_matmul)(benchmark::State &state)
{
  int M = state.range(0);
  int N = state.range(1);
  int K = state.range(2);

  mini_jit::Brgemm brgemm;
  brgemm.generate(M, N, K, 1, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
  auto kernel = brgemm.get_kernel();

  for (auto _ : state)
  {
    kernel(matrix_a.data(), matrix_b.data(), matrix_c.data(), M, K, M, 1, 1);
  }

  flops = M * N * K * 2 * state.iterations();
}

static void CustomArguments(benchmark::internal::Benchmark *b)
{
  for (int M = 1; M <= 64; M += 1)
    for (int N = 1; N <= 64; N += 1)
      for (int K : {1, 16, 32, 64, 128})
        b->Args({M, N, K});
}

BENCHMARK_REGISTER_F(GemmFixture, BM_matmul)
  ->ArgNames({"M", "N", "K"})
  ->DisplayAggregatesOnly(true)
  ->Apply(CustomArguments)
  ->MinWarmUpTime(1.0);  // WarmUp in seconds

class BrGemmFixture : public benchmark::Fixture
{
public:
  std::vector<float> matrix_a, matrix_b, matrix_c;
  double flops;

  void SetUp(::benchmark::State &state) override
  {
    flops = 0;

    int M = state.range(0);
    int N = state.range(1);
    int K = state.range(2);
    int Batch = state.range(3);

    matrix_a.resize(M * K * Batch);
    matrix_b.resize(K * N * Batch);
    matrix_c.resize(M * N * Batch);

    fill_random_matrix_args(matrix_a.data(), M * K * Batch);
    fill_random_matrix_args(matrix_b.data(), K * N * Batch);
    fill_random_matrix_args(matrix_c.data(), M * N * Batch);
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["FLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_DEFINE_F(BrGemmFixture, BM_brMatmul)(benchmark::State &state)
{
  int M = state.range(0);
  int N = state.range(1);
  int K = state.range(2);
  int Batch = state.range(3);

  mini_jit::Brgemm brgemm;
  brgemm.generate(M, N, K, Batch, 0, 0, 0, mini_jit::Brgemm::dtype_t::fp32);
  auto kernel = brgemm.get_kernel();

  for (auto _ : state)
  {
    kernel(matrix_a.data(), matrix_b.data(), matrix_c.data(), M, K, M, M * K, K * N);
  }

  flops = M * N * K * Batch * 2 * state.iterations();
}

static void CustomArgumentsBatch(benchmark::internal::Benchmark *b)
{
  int Batch = 16;
  for (int M = 1; M <= 64; M += 1)
    for (int N = 1; N <= 64; N += 1)
      for (int K : {1, 16, 32, 64, 128})
        b->Args({M, N, K, Batch});
}

BENCHMARK_REGISTER_F(BrGemmFixture, BM_brMatmul)
  ->ArgNames({"M", "N", "K", "Batch"})
  ->DisplayAggregatesOnly(true)
  ->Apply(CustomArgumentsBatch)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds