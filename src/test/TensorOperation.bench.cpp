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

  struct TensorConfig
  {
    mini_jit::TensorOperation::prim_t first_touch;
    mini_jit::TensorOperation::prim_t main;
    mini_jit::TensorOperation::prim_t last_touch;
    std::vector<mini_jit::TensorOperation::dim_t> dim_types;
    std::vector<mini_jit::TensorOperation::exec_t> exec_types;
    std::vector<int64_t> dim_sizes;
    std::vector<int64_t> strides_in0;
    std::vector<int64_t> strides_in1;
    std::vector<int64_t> strides_out;
  };

  TensorConfig config;

  std::vector<TensorConfig> configs{
    // ################
    // Serial execution
    // ################
    {
      // config 0
      mini_jit::TensorOperation::prim_t::none,  // first_touch
      mini_jit::TensorOperation::prim_t::gemm,  // main
      mini_jit::TensorOperation::prim_t::none,  // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 1
      mini_jit::TensorOperation::prim_t::none,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::none,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 2
      mini_jit::TensorOperation::prim_t::zero,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::relu,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 3
      mini_jit::TensorOperation::prim_t::zero,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::none,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 4
      mini_jit::TensorOperation::prim_t::none,  // first_touch
      mini_jit::TensorOperation::prim_t::relu,  // main
      mini_jit::TensorOperation::prim_t::none,  // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::m,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n},  // dim_types
      {mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32},                                                                 // dim_sizes
      {32 * 32 * 8 * 32, 32 * 32 * 8, 32 * 32, 1, 32},                                     // strides_in0
      {0, 8192, 1024, 0, 32},                                                              // strides_in1
      {32 * 32 * 8 * 32, 32 * 32 * 8, 32 * 32, 1, 32},                                     // strides_in2
    },
    {
      // config 5
      mini_jit::TensorOperation::prim_t::none,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::relu,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 6
      mini_jit::TensorOperation::prim_t::none,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::relu,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {16, 16, 8, 64, 64, 64},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 64},                  // strides_in0
      {0, 8192, 1024, 0, 64, 1},                  // strides_in1
      {32768, 1024, 0, 1, 64, 0},                 // strides_in2
    },

    // ##################
    // Parallel execution
    // ##################
    {
      // config 7
      mini_jit::TensorOperation::prim_t::none,  // first_touch
      mini_jit::TensorOperation::prim_t::gemm,  // main
      mini_jit::TensorOperation::prim_t::none,  // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 8
      mini_jit::TensorOperation::prim_t::none,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::none,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 9
      mini_jit::TensorOperation::prim_t::zero,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::relu,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 10
      mini_jit::TensorOperation::prim_t::zero,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::none,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 11
      mini_jit::TensorOperation::prim_t::none,  // first_touch
      mini_jit::TensorOperation::prim_t::relu,  // main
      mini_jit::TensorOperation::prim_t::none,  // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::m,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n},  // dim_types
      {mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::seq,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32},                                                                 // dim_sizes
      {32 * 32 * 8 * 32, 32 * 32 * 8, 32 * 32, 1, 32},                                     // strides_in0
      {0, 8192, 1024, 0, 32},                                                              // strides_in1
      {32 * 32 * 8 * 32, 32 * 32 * 8, 32 * 32, 1, 32},                                     // strides_in2
    },
    {
      // config 12
      mini_jit::TensorOperation::prim_t::none,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::relu,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {32, 32, 8, 32, 32, 32},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 32},                  // strides_in0
      {0, 8192, 1024, 0, 32, 1},                  // strides_in1
      {32768, 1024, 0, 1, 32, 0},                 // strides_in2
    },
    {
      // config 13
      mini_jit::TensorOperation::prim_t::none,    // first_touch
      mini_jit::TensorOperation::prim_t::brgemm,  // main
      mini_jit::TensorOperation::prim_t::relu,    // last touch
      {mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k,
       mini_jit::TensorOperation::dim_t::m, mini_jit::TensorOperation::dim_t::n, mini_jit::TensorOperation::dim_t::k},  // dim_types
      {mini_jit::TensorOperation::exec_t::shared, mini_jit::TensorOperation::exec_t::seq, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim, mini_jit::TensorOperation::exec_t::prim,
       mini_jit::TensorOperation::exec_t::prim},  // exec_types
      {16, 16, 8, 64, 64, 64},                    // dim_sizes
      {8192, 0, 1024, 1, 0, 64},                  // strides_in0
      {0, 8192, 1024, 0, 64, 1},                  // strides_in1
      {32768, 1024, 0, 1, 64, 0},                 // strides_in2
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

// ################
// Serial execution
// ################

BENCHMARK_DEFINE_F(TensorFixture, BM_tensor_operation)(benchmark::State &state)
{
  mini_jit::TensorOperation tensor_op;
  mini_jit::TensorOperation::error_t err =
    tensor_op.setup(mini_jit::TensorOperation::dtype_t::fp32, config.first_touch, config.main, config.last_touch,
                    std::span{config.dim_types}, std::span{config.exec_types}, std::span{config.dim_sizes}, std::span{config.strides_in0},
                    std::span{config.strides_in1}, std::span{config.strides_out});

  release_assert(err == mini_jit::TensorOperation::error_t::success, "Failed to generate the setup");

  for (auto _ : state)
  {
    tensor_op.execute(matrix_a.data(), matrix_b.data(), matrix_c.data());
  }

  flops = std::accumulate(config.dim_sizes.begin(), config.dim_sizes.end(), 1, std::multiplies<uint64_t>()) * 2 * state.iterations();
}

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_operation)
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

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_operation)
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

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_operation)
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

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_operation)
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

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_operation)
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

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_operation)
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

BENCHMARK_REGISTER_F(TensorFixture, BM_tensor_operation)
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

// ##################
// Parallel execution
// ##################

BENCHMARK_DEFINE_F(TensorFixture, BM_parallel_tensor_operation)(benchmark::State &state)
{
  mini_jit::TensorOperation tensor_op;
  mini_jit::TensorOperation::error_t err =
    tensor_op.setup(mini_jit::TensorOperation::dtype_t::fp32, config.first_touch, config.main, config.last_touch,
                    std::span{config.dim_types}, std::span{config.exec_types}, std::span{config.dim_sizes}, std::span{config.strides_in0},
                    std::span{config.strides_in1}, std::span{config.strides_out});

  release_assert(err == mini_jit::TensorOperation::error_t::success, "Failed to generate the setup");

  for (auto _ : state)
  {
    tensor_op.execute(matrix_a.data(), matrix_b.data(), matrix_c.data());
  }

  flops = std::accumulate(config.dim_sizes.begin(), config.dim_sizes.end(), 1, std::multiplies<uint64_t>()) * 2 * state.iterations();
}

BENCHMARK_REGISTER_F(TensorFixture, BM_parallel_tensor_operation)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    7,                          // Selected Config
  })
  ->Name("BM_parallel_tensor_GEMM")
  ->DisplayAggregatesOnly(true)
  ->Threads(4)           // Number of threads for parallel execution
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_parallel_tensor_operation)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    8,                          // Selected Config
  })
  ->Name("BM_parallel_tensor_BRGEMM")
  ->DisplayAggregatesOnly(true)
  ->Threads(4)           // Number of threads for parallel execution
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_parallel_tensor_operation)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    9,                          // Selected Config
  })
  ->Name("BM_parallel_tensor_Zero+BRGEMM+RELU")
  ->DisplayAggregatesOnly(true)
  ->Threads(4)           // Number of threads for parallel execution
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_parallel_tensor_operation)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    10,                         // Selected Config
  })
  ->Name("BM_parallel_tensor_Zero+BRGEMM")
  ->DisplayAggregatesOnly(true)
  ->Threads(4)           // Number of threads for parallel execution
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_parallel_tensor_operation)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 32 * 8 * 32 * 32,  // size_a
    1 * 32 * 8 * 1 * 32,    // size_b
    32 * 32 * 8 * 32 * 32,  // size_c
    11,                     // Selected Config
  })
  ->Name("BM_parallel_tensor_Relu")
  ->DisplayAggregatesOnly(true)
  ->Threads(4)           // Number of threads for parallel execution
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_parallel_tensor_operation)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    32 * 1 * 8 * 32 * 1 * 32,   // size_a
    1 * 32 * 8 * 1 * 32 * 32,   // size_b
    32 * 32 * 1 * 32 * 32 * 1,  // size_c
    12,                         // Selected Config
  })
  ->Name("BM_parallel_tensor_BRGEMM+RELU")
  ->DisplayAggregatesOnly(true)
  ->Threads(4)           // Number of threads for parallel execution
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(TensorFixture, BM_parallel_tensor_operation)
  ->ArgNames({"size_a", "size_b", "size_c", "config"})
  ->Args({
    16 * 1 * 8 * 64 * 1 * 64,   // size_a
    1 * 16 * 8 * 1 * 64 * 64,   // size_b
    16 * 16 * 1 * 64 * 64 * 1,  // size_c
    13,                         // Selected Config
  })
  ->Name("BM_parallel_tensor_BRGEMM+RELU")
  ->DisplayAggregatesOnly(true)
  ->Threads(4)           // Number of threads for parallel execution
  ->MinWarmUpTime(0.3);  // WarmUp in seconds