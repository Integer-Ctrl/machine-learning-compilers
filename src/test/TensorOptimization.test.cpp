#include "../main/TensorOptimization.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// ==================================================================
// Primitive Identification
// ==================================================================

TEST_CASE("Test tensor optimization primitive identification gemm", "[tensor_optimization][gemm][correctness]")
{
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                             // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                           // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                          // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_primitive_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(config, expected));
}

TEST_CASE("Test tensor optimization primitive identification brgemm", "[tensor_optimization][brgemm][correctness]")
{
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,    // first_touch
    mini_jit::TensorConfig::prim_t::brgemm,  // main
    mini_jit::TensorConfig::prim_t::none,    // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                             // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                           // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                          // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_primitive_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(config, expected));
}

TEST_CASE("Test tensor optimization primitive identification unary", "[tensor_optimization][unary][correctness]")
{
  auto type = GENERATE(mini_jit::TensorConfig::prim_t::zero, mini_jit::TensorConfig::prim_t::copy, mini_jit::TensorConfig::prim_t::relu);

  CAPTURE(type);

  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                            // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                          // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                          // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                         // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                              // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_primitive_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(config, expected));
}

// ==================================================================
// Shared Identification
// ==================================================================

TEST_CASE("Test tensor optimization shared identification gemm", "[tensor_optimization][gemm][correctness]")
{
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  omp_set_num_threads(4);
  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_shared_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(config, expected));
}

TEST_CASE("Test tensor optimization shared identification brgemm", "[tensor_optimization][brgemm][correctness]")
{
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,    // first_touch
    mini_jit::TensorConfig::prim_t::brgemm,  // main
    mini_jit::TensorConfig::prim_t::none,    // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  omp_set_num_threads(4);
  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_shared_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(config, expected));
}

TEST_CASE("Test tensor optimization shared identification unary", "[tensor_optimization][unary][correctness]")
{
  auto type = GENERATE(mini_jit::TensorConfig::prim_t::zero, mini_jit::TensorConfig::prim_t::copy, mini_jit::TensorConfig::prim_t::relu);

  CAPTURE(type);

  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  omp_set_num_threads(4);
  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_shared_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(config, expected));
}
