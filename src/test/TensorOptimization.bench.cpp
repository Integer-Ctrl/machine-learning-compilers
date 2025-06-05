#include "../main/TensorOptimization.h"
#include "../main/TensorConfig.h"
#include "../main/TensorOperation.h"
#include "../main/release_assert.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <numeric>
#include <span>

class TensorFixture : public benchmark::Fixture
{
public:
  std::vector<float> matrix_a, matrix_b, matrix_c;
  double flops;
  uint64_t M, N, K, Batch, size_a, size_b, size_c;

  mini_jit::TensorConfig config;

  std::vector<mini_jit::TensorConfig> configs{
    {
      // config 0 (matrix multiplication)
      mini_jit::TensorConfig::prim_t::none,                                                                             // first_touch
      mini_jit::TensorConfig::prim_t::gemm,                                                                             // main
      mini_jit::TensorConfig::prim_t::none,                                                                             // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},           // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
      {1600, 1600, 1600},                                                                                               // dim_sizes
      {1, 0, 1600},                                                                                                     // strides_in0
      {0, 1600, 1},                                                                                                     // strides_in1
      {1, 1600, 0},                                                                                                     // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
    },
    {
      // config 1 (tensor contraction)
      mini_jit::TensorConfig::prim_t::none,  // first_touch
      mini_jit::TensorConfig::prim_t::gemm,  // main
      mini_jit::TensorConfig::prim_t::none,  // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n,
       mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::k},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
       mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
      {64, 25, 64, 25, 64, 25},                                                                                         // dim_sizes
      {25, 1, 0, 0, 40000, 1600},                                                                                       // strides_in0
      {0, 0, 40000, 1600, 25, 1},                                                                                       // strides_in1
      {25, 1, 40000, 1600, 0, 0},                                                                                       // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
    },
  };

  static void fill_random_matrix(float *matrix, uint32_t size)
  {
    std::srand(std::time(0));
    for (size_t i = 0; i < size; i++)
    {
      float denominator = 1;
      do
      {
        denominator = static_cast<float>(std::rand());
      } while (denominator == 0);

      float numerator = 1;
      do
      {
        numerator = static_cast<float>(std::rand());
      } while (numerator == 0);

      matrix[i] = numerator / denominator;
    }
  }

  void SetUp(::benchmark::State &state) override
  {
    size_a = state.range(0);
    size_b = state.range(1);
    size_c = state.range(2);
    config = configs[state.range(3)];

    matrix_a.resize(size_a);
    matrix_b.resize(size_b);
    matrix_c.resize(size_c);

    fill_random_matrix(matrix_a.data(), size_a);
    fill_random_matrix(matrix_b.data(), size_b);
    fill_random_matrix(matrix_c.data(), size_c);
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["FLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_DEFINE_F(TensorFixture, BM_tensor_optimization)(benchmark::State &state)
{
  mini_jit::TensorOperation tensor_op;
  mini_jit::TensorOperation::error_t err = tensor_op.setup(config);

  release_assert(err == mini_jit::TensorOperation::error_t::success, "Failed to generate the setup");

  for (auto _ : state)
  {
    tensor_op.execute(matrix_a.data(), matrix_b.data(), matrix_c.data());
  }

  flops = std::accumulate(config.dim_sizes.begin(), config.dim_sizes.end(), 1, std::multiplies<uint64_t>()) * 2 * state.iterations();
}

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_optimization)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    1600 * 1 * 1600,  // size_a
    1 * 1600 * 1600,  // size_b
    1600 * 1600 * 1,  // size_c
    0,                // Selected Config
  })
  ->Name("BM_optimized_tensor_GEMM")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_optimization)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    64 * 25 * 1 * 1 * 64 * 25,  // size_a
    1 * 1 * 64 * 25 * 64 * 25,  // size_b
    64 * 25 * 64 * 25 * 1 * 1,  // size_c
    1,                          // Selected Config
  })
  ->Name("BM_optimized_tensor_BRGEMM")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds