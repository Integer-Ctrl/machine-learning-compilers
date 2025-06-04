#include "../main/TensorOperation.h"
#include "../main/TensorConfig.h"
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
      // config 0
      mini_jit::TensorConfig::prim_t::none,  // first_touch
      mini_jit::TensorConfig::prim_t::gemm,  // main
      mini_jit::TensorConfig::prim_t::none,  // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k,
       mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
       mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                                                                                             // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
      {0, 8192, 1024, 0, 32, 1},                                                                                           // strides_in1
      {32768, 1024, 0, 1, 32, 0},                                                                                          // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
    },
    {
      // config 1
      mini_jit::TensorConfig::prim_t::none,    // first_touch
      mini_jit::TensorConfig::prim_t::brgemm,  // main
      mini_jit::TensorConfig::prim_t::none,    // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k,
       mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
       mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                                                                                             // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
      {0, 8192, 1024, 0, 32, 1},                                                                                           // strides_in1
      {32768, 1024, 0, 1, 32, 0},                                                                                          // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
    },
    {
      // config 2
      mini_jit::TensorConfig::prim_t::zero,    // first_touch
      mini_jit::TensorConfig::prim_t::brgemm,  // main
      mini_jit::TensorConfig::prim_t::relu,    // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k,
       mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
       mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                                                                                             // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
      {0, 8192, 1024, 0, 32, 1},                                                                                           // strides_in1
      {32768, 1024, 0, 1, 32, 0},                                                                                          // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
    },
    {
      // config 3
      mini_jit::TensorConfig::prim_t::zero,    // first_touch
      mini_jit::TensorConfig::prim_t::brgemm,  // main
      mini_jit::TensorConfig::prim_t::none,    // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k,
       mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
       mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                                                                                             // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
      {0, 8192, 1024, 0, 32, 1},                                                                                           // strides_in1
      {32768, 1024, 0, 1, 32, 0},                                                                                          // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
    },
    {
      // config 4
      mini_jit::TensorConfig::prim_t::none,  // first_touch
      mini_jit::TensorConfig::prim_t::relu,  // main
      mini_jit::TensorConfig::prim_t::none,  // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::m,
       mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
       mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32},                                                           // dim_sizes
      {32 * 32 * 8 * 32, 32 * 32 * 8, 32 * 32, 1, 32},                               // strides_in0
      {0, 8192, 1024, 0, 32},                                                        // strides_in1
      {32 * 32 * 8 * 32, 32 * 32 * 8, 32 * 32, 1, 32},                               // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                         // dtype_t
    },
    {
      // config 5
      mini_jit::TensorConfig::prim_t::none,    // first_touch
      mini_jit::TensorConfig::prim_t::brgemm,  // main
      mini_jit::TensorConfig::prim_t::relu,    // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k,
       mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
       mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                                                                                             // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
      {0, 8192, 1024, 0, 32, 1},                                                                                           // strides_in1
      {32768, 1024, 0, 1, 32, 0},                                                                                          // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
    },
    {
      // config 6
      mini_jit::TensorConfig::prim_t::none,    // first_touch
      mini_jit::TensorConfig::prim_t::brgemm,  // main
      mini_jit::TensorConfig::prim_t::relu,    // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k,
       mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
      {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
       mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {16, 16, 8, 64, 64, 64},                                                                                             // dim_sizes
      {8192, 0, 1024, 1, 0, 64},                                                                                           // strides_in0
      {0, 8192, 1024, 0, 64, 1},                                                                                           // strides_in1
      {32768, 1024, 0, 1, 64, 0},                                                                                          // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
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

BENCHMARK_DEFINE_F(TensorFixture, BM_tensor_variable_size)(benchmark::State &state)
{
  mini_jit::TensorOperation tensor_op;
  mini_jit::TensorOperation::error_t err =
    tensor_op.setup(mini_jit::TensorConfig::dtype_t::fp32, config.first_touch, config.main, config.last_touch, std::span{config.dim_types},
                    std::span{config.exec_types}, std::span{config.dim_sizes}, std::span{config.strides_in0}, std::span{config.strides_in1},
                    std::span{config.strides_out});

  release_assert(err == mini_jit::TensorOperation::error_t::success, "Failed to generate the setup");

  for (auto _ : state)
  {
    tensor_op.execute(matrix_a.data(), matrix_b.data(), matrix_c.data());
  }

  flops = std::accumulate(config.dim_sizes.begin(), config.dim_sizes.end(), 1, std::multiplies<uint64_t>()) * 2 * state.iterations();
}

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_variable_size)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    0,                          // Selected Config
  })
  ->Name("BM_tensor_GEMM")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_variable_size)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    1,                          // Selected Config
  })
  ->Name("BM_tensor_BRGEMM")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_variable_size)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    2,                          // Selected Config
  })
  ->Name("BM_tensor_Zero+BRGEMM+RELU")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_variable_size)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    3,                          // Selected Config
  })
  ->Name("BM_tensor_Zero+BRGEMM")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_variable_size)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 32 * 8 * 32 * 32,  // size_a
    1 * 32 * 8 * 1 * 32,    // size_b
    32 * 32 * 8 * 32 * 32,  // size_c
    4,                      // Selected Config
  })
  ->Name("BM_tensor_Relu")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_variable_size)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    5,                          // Selected Config
  })
  ->Name("BM_tensor_BRGEMM+RELU")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_variable_size)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    16 * 1 * 8 * 64 * 1 * 64,   // size_a
    1 * 16 * 8 * 1 * 64 * 64,   // size_b
    16 * 16 * 1 * 64 * 64 * 1,  // size_c
    6,                          // Selected Config
  })
  ->Name("BM_tensor_BRGEMM+RELU")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds