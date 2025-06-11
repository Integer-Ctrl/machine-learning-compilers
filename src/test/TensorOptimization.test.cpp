#include "../main/TensorOptimization.h"
#include "../main/TensorOperation.h"
#include "BaseGeneration.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <iostream>

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
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
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
    mini_jit::TensorConfig::prim_t::none,    // first_touch
    mini_jit::TensorConfig::prim_t::brgemm,  // main
    mini_jit::TensorConfig::prim_t::none,    // last touch
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
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
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
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

TEST_CASE("Test tensor optimization primitive identification unary all c dimensions", "[tensor_optimization][unary][correctness]")
{
  auto type = GENERATE(mini_jit::TensorConfig::prim_t::zero, mini_jit::TensorConfig::prim_t::copy, mini_jit::TensorConfig::prim_t::relu);

  CAPTURE(type);

  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c,
     mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 1, 32, 32 * 32},                                          // strides_in0
    {0, 0, 0, 0, 0, 0},                                                                                               // strides_in1
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 1, 32, 32 * 32},                                          // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::c},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                            // dim_sizes
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 1, 32, 32 * 32},                                            // strides_in0
    {0, 0, 0, 0, 0, 0},                                                                                                 // strides_in1
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 1, 32, 32 * 32},                                            // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                              // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_primitive_identification(config);

  INFO(new_config.to_string());
  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

TEST_CASE("Test tensor optimization primitive identification unary transpose all c dimensions",
          "[tensor_optimization][unary][transpose][correctness]")
{
  auto type = GENERATE(mini_jit::TensorConfig::prim_t::zero, mini_jit::TensorConfig::prim_t::copy, mini_jit::TensorConfig::prim_t::relu);

  CAPTURE(type);

  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c,
     mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 1, 32, 32 * 32},                                          // strides_in0
    {0, 0, 0, 0, 0, 0},                                                                                               // strides_in1
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 32, 1, 32 * 32},                                          // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::c},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                            // dim_sizes
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 1, 32, 32 * 32},                                            // strides_in0
    {0, 0, 0, 0, 0, 0},                                                                                                 // strides_in1
    {32 * 32 * 32 * 8 * 32, 32 * 32 * 32 * 8, 32 * 32 * 32, 32, 1, 32 * 32},                                            // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                              // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_primitive_identification(config);

  INFO(new_config.to_string());
  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

// ==================================================================
// Shared Identification
// ==================================================================

TEST_CASE("Test tensor optimization shared identification gemm 4 Threads", "[tensor_optimization][gemm][correctness]")
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
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

TEST_CASE("Test tensor optimization shared identification brgemm 4 Threads", "[tensor_optimization][brgemm][correctness]")
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
    mini_jit::TensorConfig::prim_t::none,    // first_touch
    mini_jit::TensorConfig::prim_t::brgemm,  // main
    mini_jit::TensorConfig::prim_t::none,    // last touch
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
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

TEST_CASE("Test tensor optimization shared identification unary 4 Threads", "[tensor_optimization][unary][correctness]")
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
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

TEST_CASE("Test tensor optimization shared identification gemm 3 Threads", "[tensor_optimization][gemm][correctness]")
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
    {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  omp_set_num_threads(3);
  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_shared_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

TEST_CASE("Test tensor optimization shared identification brgemm 3 Threads", "[tensor_optimization][brgemm][correctness]")
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
    mini_jit::TensorConfig::prim_t::none,    // first_touch
    mini_jit::TensorConfig::prim_t::brgemm,  // main
    mini_jit::TensorConfig::prim_t::none,    // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  omp_set_num_threads(3);
  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_shared_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

TEST_CASE("Test tensor optimization shared identification unary 3 Threads", "[tensor_optimization][unary][correctness]")
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
    {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 32, 8, 32, 32, 32},                                                                                          // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  omp_set_num_threads(3);
  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_shared_identification(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

// ==================================================================
// Dimension Reordering Shared
// ==================================================================

TEST_CASE("Test tensor optimization reordering shared", "[tensor_optimization][correctness]")
{
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
    {32, 16, 8, 32, 32, 32},                                                                                             // dim_sizes
    {1024, 0, 16384, 1, 0, 32},                                                                                          // strides_in0
    {0, 1024, 16384, 0, 32, 1},                                                                                          // strides_in1
    {1024, 32768, 65536, 1, 32, 0},                                                                                      // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::c, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
    {8, 16, 32, 32, 32, 32},                                                                                             // dim_sizes
    {16384, 0, 1024, 1, 0, 32},                                                                                          // strides_in0
    {16384, 1024, 0, 0, 32, 1},                                                                                          // strides_in1
    {65536, 32768, 1024, 1, 32, 0},                                                                                      // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_dimension_reordering_shared(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  CAPTURE(config.dim_sizes);
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

// ==================================================================
// Dimension Splitting
// ==================================================================

TEST_CASE("Test tensor optimization dimension splitting", "[tensor_optimization][correctness]")
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
    {500, 32, 8, 32, 32, 32},                                                                                         // dim_sizes
    {8192, 0, 1024, 1, 0, 32},                                                                                        // strides_in0
    {0, 8192, 1024, 0, 32, 1},                                                                                        // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k,
     mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq},   // exec_types
    {25, 20, 32, 8, 32, 32, 32},             // dim_sizes
    {8192 * 20, 8192, 0, 1024, 1, 0, 32},    // strides_in0
    {0, 0, 8192, 1024, 0, 32, 1},            // strides_in1
    {32768 * 20, 32768, 1024, 0, 1, 32, 0},  // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,   // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_dimension_splitting(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

// ==================================================================
// Dimension Fusing
// ==================================================================

TEST_CASE("Test tensor optimization dimension fusing", "[tensor_optimization][correctness]")
{
  auto type = GENERATE(mini_jit::TensorConfig::prim_t::zero, mini_jit::TensorConfig::prim_t::copy, mini_jit::TensorConfig::prim_t::relu);

  CAPTURE(type);

  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {5, 32, 8, 32, 32, 32},                                                                                           // dim_sizes
    {0, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
    {8192 * 32, 8192, 1024, 0, 32, 1},                                                                                // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n,
     mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {5 * 32, 8, 32, 32, 32},                                                     // dim_sizes
    {0, 1024, 1, 0, 32},                                                         // strides_in0
    {8192, 1024, 0, 32, 1},                                                      // strides_in1
    {1024, 0, 1, 32, 0},                                                         // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                       // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_dimension_fusing(config);

  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

// ==================================================================
// Dimension Reordering Fusing
// ==================================================================

TEST_CASE("Test tensor optimization dimension reordering fusing", "[tensor_optimization][correctness]")
{
  auto type = GENERATE(mini_jit::TensorConfig::prim_t::zero, mini_jit::TensorConfig::prim_t::copy, mini_jit::TensorConfig::prim_t::relu);

  CAPTURE(type);

  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 8, 32, 5, 32, 32},                                                                                           // dim_sizes
    {0, 1024, 1, 0, 0, 32},                                                                                           // strides_in0
    {8192, 1024, 0, 8192 * 32, 32, 1},                                                                                // strides_in1
    {1024, 0, 1, 32768, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    type,                                  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {5, 32, 8, 32, 32, 32},                                                                                           // dim_sizes
    {0, 0, 1024, 1, 0, 32},                                                                                           // strides_in0
    {8192 * 32, 8192, 1024, 0, 32, 1},                                                                                // strides_in1
    {32768, 1024, 0, 1, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorOptimization optimization;
  mini_jit::TensorConfig new_config = optimization.optimize_dimension_reordering_fusing(config);

  INFO(new_config.to_string());
  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, new_config));
  REQUIRE(mini_jit::TensorConfig::equals(expected, new_config));
}

// ==================================================================
// Full optimization pipeline
// ==================================================================

TEST_CASE("Test tensor operation with optimization with main kernel: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
{
  using namespace mini_jit;

  auto type = GENERATE(TensorConfig::prim_t::zero, TensorConfig::prim_t::copy, TensorConfig::prim_t::relu);

  CAPTURE(type);

  std::vector<TensorConfig::dim_t> dim_types = {TensorConfig::dim_t::m, TensorConfig::dim_t::n};
  std::vector<TensorConfig::exec_t> exec_types = {TensorConfig::exec_t::seq, TensorConfig::exec_t::seq};
  std::vector<int64_t> dim_sizes = {64, 64};
  std::vector<int64_t> strides_in0 = {1, 64};
  std::vector<int64_t> strides_in1 = {0, 0};
  std::vector<int64_t> strides_out = {1, 64};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Counting);

  mini_jit::TensorConfig config{
    TensorConfig::prim_t::none, type, TensorConfig::prim_t::none, dim_types, exec_types, dim_sizes, strides_in0, strides_in1, strides_out,
    TensorConfig::dtype_t::fp32};

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,                                          // first_touch
    type,                                                                          // main
    mini_jit::TensorConfig::prim_t::none,                                          // last touch
    dim_types,                                                                     // dim_types
    {mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
    dim_sizes,                                                                     // dim_sizes
    strides_in0,                                                                   // strides_in0
    strides_in1,                                                                   // strides_in1
    strides_out,                                                                   // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                         // dtype_t
  };

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(config);

  REQUIRE(err == TensorOperation::error_t::success);
  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, tensor_op.get_config()));
  REQUIRE(mini_jit::TensorConfig::equals(expected, tensor_op.get_config()));

  tensor_op.execute(test.matrix_a.data(), nullptr, test.matrix_c.data());

  UnaryType test_type = UnaryType::None;
  switch (type)
  {
  case TensorConfig::prim_t::zero:
    test_type = UnaryType::Zero;
    break;
  case TensorConfig::prim_t::copy:
    test_type = UnaryType::Identity;
    break;
  case TensorConfig::prim_t::relu:
    test_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  test.naive_unary_M_N(test.matrix_a.data(), test.matrix_c_verify.data(), 64, 64, false, test_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test parallel tensor operation with optimization with outer loop with first touch: unary (zero, relu, copy) & main kernel: "
          "brgemm & last touch: unary "
          "(zero, relu, copy)",
          "[tensor_operation][unary][brgemm][correctness]")
{
  using namespace mini_jit;

  auto first_type = GENERATE(TensorConfig::prim_t::zero, TensorConfig::prim_t::copy, TensorConfig::prim_t::relu);
  auto last_type = GENERATE(TensorConfig::prim_t::zero, TensorConfig::prim_t::copy, TensorConfig::prim_t::relu);

  CAPTURE(first_type, last_type);

  std::vector<TensorConfig::dim_t> dim_types{TensorConfig::dim_t::n, TensorConfig::dim_t::m, TensorConfig::dim_t::c, TensorConfig::dim_t::m,
                                             TensorConfig::dim_t::k, TensorConfig::dim_t::m, TensorConfig::dim_t::k, TensorConfig::dim_t::m,
                                             TensorConfig::dim_t::n, TensorConfig::dim_t::k};

  std::vector<TensorConfig::exec_t> exec_types{
    TensorConfig::exec_t::seq, TensorConfig::exec_t::seq, TensorConfig::exec_t::seq, TensorConfig::exec_t::seq, TensorConfig::exec_t::seq,
    TensorConfig::exec_t::seq, TensorConfig::exec_t::seq, TensorConfig::exec_t::seq, TensorConfig::exec_t::seq, TensorConfig::exec_t::seq};

  std::vector<int64_t> dim_sizes{2, 3, 5, 8, 13, 21, 3, 16, 16, 16};
  std::vector<int64_t> strides_in0{0,                              // n-dim
                                   16 * 16 * 3 * 21 * 13 * 8 * 5,  // m-dim
                                   16 * 16 * 3 * 21 * 13 * 8,      // c-dim
                                   16 * 16 * 3 * 21 * 13,          // m-dim
                                   16 * 16 * 3 * 21,               // k-dim
                                   16 * 16 * 3,                    // m-dim
                                   16 * 16,                        // k-dim-prim
                                   1,                              // m-dim-prim
                                   0,                              // n-dim-prim
                                   16};                            // k-dim-prim
  std::vector<int64_t> strides_in1{16 * 16 * 3 * 1 * 13 * 1 * 5 * 1,
                                   0,  // m-dim
                                   16 * 16 * 3 * 1 * 13 * 1,
                                   0,  // m-dim
                                   16 * 16 * 3 * 1,
                                   0,        // m-dim
                                   16 * 16,  // k-dim-prim
                                   0,
                                   16,
                                   1};
  std::vector<int64_t> strides_out{16 * 16 * 21 * 1 * 8 * 5 * 3,
                                   16 * 16 * 21 * 1 * 8 * 5,  // m-dim
                                   16 * 16 * 21 * 1 * 8,
                                   16 * 16 * 21 * 1,
                                   0,  // k-dim
                                   16 * 16,
                                   0,
                                   1,
                                   16,
                                   0};

  GenerationTest test(16, 16, 16, 3, 16 * 16 * 3 * 21 * 13 * 8 * 5 * 3 * 1, 16 * 16 * 3 * 1 * 13 * 1 * 5 * 1 * 2,
                      16 * 16 * 21 * 1 * 8 * 5 * 3 * 2);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorConfig config{
    first_type,  TensorConfig::prim_t::brgemm, last_type, dim_types, exec_types, dim_sizes, strides_in0, strides_in1,
    strides_out, TensorConfig::dtype_t::fp32};

  mini_jit::TensorConfig expected{
    first_type,                    // first_touch
    TensorConfig::prim_t::brgemm,  // main
    last_type,                     // last touch
    {TensorConfig::dim_t::m, TensorConfig::dim_t::c, TensorConfig::dim_t::n, TensorConfig::dim_t::m, TensorConfig::dim_t::k,
     TensorConfig::dim_t::m, TensorConfig::dim_t::k, TensorConfig::dim_t::m, TensorConfig::dim_t::n, TensorConfig::dim_t::k},  // dim_types
    {TensorConfig::exec_t::shared, TensorConfig::exec_t::shared, TensorConfig::exec_t::shared, TensorConfig::exec_t::shared,
     TensorConfig::exec_t::seq, TensorConfig::exec_t::seq, TensorConfig::exec_t::prim, TensorConfig::exec_t::prim,
     TensorConfig::exec_t::prim, TensorConfig::exec_t::prim},  // exec_types
    {3, 5, 2, 8, 13, 21, 3, 16, 16, 16},                       // dim_sizes
    {16 * 16 * 3 * 21 * 13 * 8 * 5,                            // m-dim
     16 * 16 * 3 * 21 * 13 * 8,                                // c-dim
     0,                                                        // n-dim
     16 * 16 * 3 * 21 * 13,                                    // m-dim
     16 * 16 * 3 * 21,                                         // k-dim
     16 * 16 * 3,                                              // m-dim
     16 * 16,                                                  // k-dim-prim
     1,                                                        // m-dim-prim
     0,                                                        // n-dim-prim
     16},                                                      // strides_in0
    {0,                                                        // m-dim
     16 * 16 * 3 * 1 * 13 * 1,                                 //
     16 * 16 * 3 * 1 * 13 * 1 * 5 * 1,
     0,  // m-dim
     16 * 16 * 3 * 1,
     0,                             // m-dim
     16 * 16,                       // k-dim-prim
     0, 16, 1},                     // strides_in1
    {16 * 16 * 21 * 1 * 8 * 5,      // m-dim
     16 * 16 * 21 * 1 * 8,          //
     16 * 16 * 21 * 1 * 8 * 5 * 3,  //
     16 * 16 * 21 * 1,
     0,                                     // k-dim
     16 * 16, 0, 1, 16, 0},                 // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,  // dtype_t
  };

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(config);

  INFO(tensor_op.get_config().to_string());

  REQUIRE(err == TensorOperation::error_t::success);
  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, tensor_op.get_config()));
  REQUIRE(mini_jit::TensorConfig::equals(expected, tensor_op.get_config()));

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // First touch operation
  UnaryType test_fist_type = UnaryType::None;
  switch (first_type)
  {
  case TensorConfig::prim_t::zero:
    test_fist_type = UnaryType::Zero;
    break;
  case TensorConfig::prim_t::copy:
    test_fist_type = UnaryType::Identity;
    break;
  case TensorConfig::prim_t::relu:
    test_fist_type = UnaryType::ReLu;
    break;

  default:
    break;
  }

  // Last touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (last_type)
  {
  case TensorConfig::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorConfig::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorConfig::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // Main kernel operation
  for (int64_t i0 = 0; i0 < dim_sizes[0]; i0++)
  {
    for (int64_t i1 = 0; i1 < dim_sizes[1]; i1++)
    {
      for (int64_t i2 = 0; i2 < dim_sizes[2]; i2++)
      {
        for (int64_t i3 = 0; i3 < dim_sizes[3]; i3++)
        {
          for (int64_t i4 = 0; i4 < dim_sizes[4]; i4++)
          {
            for (int64_t i5 = 0; i5 < dim_sizes[5]; i5++)
            {
              uint64_t offset_a = i0 * strides_in0[0] + i1 * strides_in0[1] + i2 * strides_in0[2] + i3 * strides_in0[3] +
                                  i4 * strides_in0[4] + i5 * strides_in0[5];
              uint64_t offset_b = i0 * strides_in1[0] + i1 * strides_in1[1] + i2 * strides_in1[2] + i3 * strides_in1[3] +
                                  i4 * strides_in1[4] + i5 * strides_in1[5];
              uint64_t offset_c = i0 * strides_out[0] + i1 * strides_out[1] + i2 * strides_out[2] + i3 * strides_out[3] +
                                  i4 * strides_out[4] + i5 * strides_out[5];
              if (i4 == 0)
              {
                // First touch
                test.naive_unary_M_N(test.matrix_c_verify.data() + offset_c, test.matrix_c_verify.data() + offset_c, 16, 16, false,
                                     test_fist_type);
              }
              test.naive_matmul_M_N_K_Batch(test.matrix_a.data() + offset_a, test.matrix_b.data() + offset_b,
                                            test.matrix_c_verify.data() + offset_c, 16, 16, 16, 16 * 16, 16 * 16);
              if (i4 == (dim_sizes[4] - 1))
              {
                // Last touch
                test.naive_unary_M_N(test.matrix_c_verify.data() + offset_c, test.matrix_c_verify.data() + offset_c, 16, 16, false,
                                     test_last_type);
              }
            }
          }
        }
      }
    }
  }

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with optimization dimension test reordering and fusing", "[tensor_optimization][gemm][correctness]")
{
  using namespace mini_jit;

  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n,
     mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
     mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
    {32, 8, 32, 5, 32, 32},                                                                                           // dim_sizes
    {0, 1024, 1, 0, 0, 32},                                                                                           // strides_in0
    {8192, 1024, 0, 8192 * 32, 32, 1},                                                                                // strides_in1
    {1024, 0, 1, 32768, 32, 0},                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
  };

  mini_jit::TensorConfig expected{
    mini_jit::TensorConfig::prim_t::none,  // first_touch
    mini_jit::TensorConfig::prim_t::gemm,  // main
    mini_jit::TensorConfig::prim_t::none,  // last touch
    {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n,
     mini_jit::TensorConfig::dim_t::k},  // dim_types
    {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
     mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
    {5 * 32, 8, 32, 32, 32},                                                       // dim_sizes
    {0, 1024, 1, 0, 32},                                                           // strides_in0
    {8192, 1024, 0, 32, 1},                                                        // strides_in1
    {1024, 0, 1, 32, 0},                                                           // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                         // dtype_t
  };

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(config);

  INFO(tensor_op.get_config().to_string());

  REQUIRE(err == TensorOperation::error_t::success);
  REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, tensor_op.get_config()));
  REQUIRE(mini_jit::TensorConfig::equals(expected, tensor_op.get_config()));

  GenerationTest test(32, 32, 32, 32 * 1 * 32 * 8 * 1 * 1, 32 * 32 * 1 * 8 * 32 * 5, 1 * 32 * 32 * 1 * 32 * 5);
  test.SetUp(TestInfill::Random);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  for (int64_t i0 = 0; i0 < expected.dim_sizes[0]; i0++)
  {
    for (int64_t i1 = 0; i1 < expected.dim_sizes[1]; i1++)
    {
      uint64_t offset_a = i0 * expected.strides_in0[0] + i1 * expected.strides_in0[1];
      uint64_t offset_b = i0 * expected.strides_in1[0] + i1 * expected.strides_in1[1];
      uint64_t offset_c = i0 * expected.strides_out[0] + i1 * expected.strides_out[1];
      test.naive_matmul_M_N_K_Batch(test.matrix_a.data() + offset_a, test.matrix_b.data() + offset_b,
                                    test.matrix_c_verify.data() + offset_c, 32, 32, 32, 32 * 32, 32 * 32);
    }
  }

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}