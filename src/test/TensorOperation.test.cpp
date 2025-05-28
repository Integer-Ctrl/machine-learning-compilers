#include "../main/TensorOperation.h"
#include "BaseGeneration.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>

/**
 * =================================================================================================
 * =================================================================================================
 *
 *                                  Testing without outer loop
 *
 * =================================================================================================
 * =================================================================================================
 */
TEST_CASE("Test tensor operation with main kernel: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
{
  using namespace mini_jit;

  auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64};
  constexpr int64_t strides_in0[]{1, 64};
  constexpr int64_t strides_in1[]{0, 0};
  constexpr int64_t strides_out[]{1, 64};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Counting);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, type, TensorOperation::prim_t::none, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), nullptr, test.matrix_c.data());

  UnaryType test_type = UnaryType::None;
  switch (type)
  {
  case TensorOperation::prim_t::zero:
    test_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  test.naive_unary_M_N(test.matrix_a.data(), test.matrix_c_verify.data(), 64, 64, false, test_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with main kernel: gemm", "[tensor_operation][gemm][correctness]")
{
  using namespace mini_jit;

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64};
  constexpr int64_t strides_in0[]{1, 0, 64};
  constexpr int64_t strides_in1[]{0, 64, 1};
  constexpr int64_t strides_out[]{1, 64, 0};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 1, 1);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with main kernel: brgemm", "[tensor_operation][brgemm][correctness]")
{
  using namespace mini_jit;

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n,
                                               TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64, 64};
  constexpr int64_t strides_in0[]{64 * 64, 1, 0, 64};
  constexpr int64_t strides_in1[]{64 * 64, 0, 64, 1};
  constexpr int64_t strides_out[]{0, 1, 64, 0};

  GenerationTest test(64, 64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::brgemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
{
  using namespace mini_jit;

  auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::relu, TensorOperation::prim_t::copy);

  CAPTURE(type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64};
  constexpr int64_t strides_in0[]{1, 64};
  constexpr int64_t strides_in1[]{1, 64};
  constexpr int64_t strides_out[]{1, 64};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, type, TensorOperation::prim_t::none, TensorOperation::prim_t::none, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), nullptr, test.matrix_c.data());

  // First touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // First touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with last touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
{
  using namespace mini_jit;

  auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::relu, TensorOperation::prim_t::copy);

  CAPTURE(type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64};
  constexpr int64_t strides_in0[]{1, 64};
  constexpr int64_t strides_in1[]{1, 64};
  constexpr int64_t strides_out[]{1, 64};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::none, type, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), nullptr, test.matrix_c.data());

  // First touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // First touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: gemm",
          "[tensor_operation][unary][gemm][correctness]")
{
  using namespace mini_jit;

  auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(first_type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64};
  constexpr int64_t strides_in0[]{1, 0, 64};
  constexpr int64_t strides_in1[]{0, 64, 1};
  constexpr int64_t strides_out[]{1, 64, 0};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::gemm, TensorOperation::prim_t::none, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // First touch operation
  UnaryType test_fist_type = UnaryType::None;
  switch (first_type)
  {
  case TensorOperation::prim_t::zero:
    test_fist_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_fist_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_fist_type = UnaryType::ReLu;
    break;

  default:
    break;
  }

  // First touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_fist_type);

  // Main kernel operation
  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with last touch: unary (zero, relu, copy) & main kernel: gemm",
          "[tensor_operation][unary][gemm][correctness]")
{
  using namespace mini_jit;

  auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(last_type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64};
  constexpr int64_t strides_in0[]{1, 0, 64};
  constexpr int64_t strides_in1[]{0, 64, 1};
  constexpr int64_t strides_out[]{1, 64, 0};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::gemm, last_type, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // Last touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (last_type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // Main kernel operation
  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  // Last touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: gemm & last touch: unary (zero, relu, copy)",
          "[tensor_operation][unary][gemm][correctness]")
{
  using namespace mini_jit;

  auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);
  auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(first_type, last_type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64};
  constexpr int64_t strides_in0[]{1, 0, 64};
  constexpr int64_t strides_in1[]{0, 64, 1};
  constexpr int64_t strides_out[]{1, 64, 0};

  GenerationTest test(64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::gemm, last_type, std::span{dim_types},
                    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // First touch operation
  UnaryType test_fist_type = UnaryType::None;
  switch (first_type)
  {
  case TensorOperation::prim_t::zero:
    test_fist_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_fist_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_fist_type = UnaryType::ReLu;
    break;

  default:
    break;
  }

  // Last touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (last_type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // First touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_fist_type);

  // Main kernel operation
  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  // Last touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

// Brgemm
TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: brgemm",
          "[tensor_operation][unary][brgemm][correctness]")
{
  using namespace mini_jit;

  auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(first_type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n,
                                               TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64, 64};
  constexpr int64_t strides_in0[]{64 * 64, 1, 0, 64};
  constexpr int64_t strides_in1[]{64 * 64, 0, 64, 1};
  constexpr int64_t strides_out[]{0, 1, 64, 0};

  GenerationTest test(64, 64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::brgemm, TensorOperation::prim_t::none, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // First touch operation
  UnaryType test_fist_type = UnaryType::None;
  switch (first_type)
  {
  case TensorOperation::prim_t::zero:
    test_fist_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_fist_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_fist_type = UnaryType::ReLu;
    break;

  default:
    break;
  }

  // First touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_fist_type);

  // Main kernel operation
  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with last touch: unary (zero, relu, copy) & main kernel: brgemm",
          "[tensor_operation][unary][brgemm][correctness]")
{
  using namespace mini_jit;

  auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(last_type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n,
                                               TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64, 64};
  constexpr int64_t strides_in0[]{64 * 64, 1, 0, 64};
  constexpr int64_t strides_in1[]{64 * 64, 0, 64, 1};
  constexpr int64_t strides_out[]{0, 1, 64, 0};

  GenerationTest test(64, 64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::brgemm, last_type, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // Last touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (last_type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // Main kernel operation
  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  // Last touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: brgemm & last touch: unary (zero, relu, copy)",
          "[tensor_operation][unary][brgemm][correctness]")
{
  using namespace mini_jit;

  auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);
  auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(first_type, last_type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n,
                                               TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64, 64};
  constexpr int64_t strides_in0[]{64 * 64, 1, 0, 64};
  constexpr int64_t strides_in1[]{64 * 64, 0, 64, 1};
  constexpr int64_t strides_out[]{0, 1, 64, 0};

  GenerationTest test(64, 64, 64, 64);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::brgemm, last_type, std::span{dim_types},
                    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // First touch operation
  UnaryType test_fist_type = UnaryType::None;
  switch (first_type)
  {
  case TensorOperation::prim_t::zero:
    test_fist_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_fist_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_fist_type = UnaryType::ReLu;
    break;

  default:
    break;
  }

  // Last touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (last_type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // First touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_fist_type);

  // Main kernel operation
  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  // Last touch
  test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

/**
 * =================================================================================================
 * =================================================================================================
 *
 *                                Testing with multiple outer loop
 *
 * =================================================================================================
 * =================================================================================================
 */
// TEST_CASE("Test tensor operation with outer loop with main kernel: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
// {
//   using namespace mini_jit;

//   auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::relu, TensorOperation::prim_t::copy);

//   CAPTURE(type);

//   constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::n, TensorOperation::dim_t::k, TensorOperation::dim_t::c,
//                                                TensorOperation::dim_t::m, TensorOperation::dim_t::k, TensorOperation::dim_t::m,
//                                                TensorOperation::dim_t::m, TensorOperation::dim_t::n};
//   constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
//   TensorOperation::exec_t::seq,
//                                                  TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
//                                                  TensorOperation::exec_t::seq, TensorOperation::exec_t::prim,
//                                                  TensorOperation::exec_t::prim};
//   constexpr int64_t dim_sizes[]{2, 3, 5, 8, 13, 21, 64, 64};
//   constexpr int64_t strides_in0[]{64 * 64 * 21 * 13 * 8 * 5 * 3 * 2,
//                                   64 * 64 * 21 * 13 * 8 * 5 * 1,
//                                   64 * 64 * 21 * 13 * 8 * 5,
//                                   64 * 64 * 21 * 13 * 8,
//                                   64 * 64 * 21 * 13,
//                                   64 * 64 * 21,
//                                   64 * 64,
//                                   1,
//                                   64};
//   constexpr int64_t strides_in1[]{0, 0, 0, 0, 0, 0, 0, 0};
//   constexpr int64_t strides_out[]{64 * 64 * 21 * 13 * 8 * 5 * 3 * 2,
//                                   64 * 64 * 21 * 13 * 8 * 5 * 1,
//                                   64 * 64 * 21 * 13 * 8 * 5,
//                                   64 * 64 * 21 * 13 * 8,
//                                   64 * 64 * 21 * 1,
//                                   64 * 64 * 21,
//                                   64 * 64,
//                                   1,
//                                   64};

//   GenerationTest test(64, 64, 64, 1, 64 * 64 * 21 * 13 * 8 * 5 * 3 * 2, 0, 64 * 64 * 21 * 13 * 8 * 5 * 3 * 2);

//   mini_jit::TensorOperation tensor_op;
//   TensorOperation::error_t err = tensor_op.setup(
//     TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, type, TensorOperation::prim_t::none, std::span{dim_types},
//     std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

//   REQUIRE(err == TensorOperation::error_t::success);

//   tensor_op.execute(test.matrix_a.data(), nullptr, test.matrix_c.data());

//   UnaryType test_type = UnaryType::None;
//   switch (type)
//   {
//   case TensorOperation::prim_t::zero:
//     test_type = UnaryType::Zero;
//     break;
//   case TensorOperation::prim_t::copy:
//     test_type = UnaryType::Identity;
//     break;
//   case TensorOperation::prim_t::relu:
//     test_type = UnaryType::ReLu;
//     break;
//   default:
//     FAIL("Could not parse the unary type!");
//     break;
//   }

//   for (size_t i0 = 0; i0 < dim_sizes[0]; i0++)
//   {
//     for (size_t i1 = 0; i1 < dim_sizes[1]; i1++)
//     {
//       for (size_t i2 = 0; i2 < dim_sizes[2]; i2++)
//       {
//         for (size_t i3 = 0; i3 < dim_sizes[3]; i3++)
//         {
//           for (size_t i4 = 0; i4 < dim_sizes[4]; i4++)
//           {
//             for (size_t i5 = 0; i5 < dim_sizes[5]; i5++)
//             {
//               uint64_t offset_a = i0 * strides_in0[0] + i1 * strides_in0[1] + i2 * strides_in0[2] + i3 * strides_in0[3] +
//                                   i4 * strides_in0[4] + i5 * strides_in0[5];
//               uint64_t offset_c = i0 * strides_out[0] + i1 * strides_out[1] + i2 * strides_out[2] + i3 * strides_out[3] +
//                                   i4 * strides_out[4] + i5 * strides_out[5];
//               test.naive_unary_M_N(test.matrix_a.data() + offset_a, test.matrix_c_verify.data() + offset_c, 64, 64, false, test_type);
//             }
//           }
//         }
//       }
//     }
//   }

//   test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
// }

TEST_CASE("Test tensor operation with outer loop with main kernel: gemm", "[tensor_operation][gemm][correctness]")
{
  using namespace mini_jit;

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::n, TensorOperation::dim_t::k, TensorOperation::dim_t::c,
                                               TensorOperation::dim_t::m, TensorOperation::dim_t::k, TensorOperation::dim_t::m,
                                               TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};

  constexpr TensorOperation::exec_t exec_types[]{
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
    TensorOperation::exec_t::prim, TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};

  constexpr int64_t dim_sizes[]{2, 3, 5, 8, 13, 21, 32, 32, 32};
  constexpr int64_t strides_in0[]{0,                          // n-dim
                                  32 * 32 * 21 * 13 * 8 * 5,  // k-dim
                                  32 * 32 * 21 * 13 * 8,      // c-dim
                                  32 * 32 * 21 * 13,          // m-dim
                                  32 * 32 * 21,               // k-dim
                                  32 * 32,                    // m-dim
                                  1,                          // m-dim-prim
                                  0,                          // n-dim-prim
                                  32};                        // k-dim-prim
  constexpr int64_t strides_in1[]{32 * 32 * 1 * 13 * 1 * 5 * 3,
                                  32 * 32 * 1 * 13 * 1 * 5,
                                  32 * 32 * 1 * 13 * 1,
                                  0,  // m-dim
                                  32 * 32 * 1,
                                  0,  // m-dim
                                  0,
                                  32,
                                  1};
  constexpr int64_t strides_out[]{32 * 32 * 21 * 1 * 8 * 5 * 1,
                                  0,  // k-dim
                                  32 * 32 * 21 * 1 * 8,
                                  32 * 32 * 21 * 1,
                                  0,  // k-dim
                                  32 * 32,
                                  1,
                                  32,
                                  0};

  GenerationTest test(32, 32, 32, 1, 32 * 32 * 21 * 13 * 8 * 5 * 3 * 1, 32 * 32 * 1 * 13 * 1 * 5 * 3 * 2, 32 * 32 * 21 * 1 * 8 * 5 * 1 * 2);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  for (size_t i0 = 0; i0 < dim_sizes[0]; i0++)
  {
    for (size_t i1 = 0; i1 < dim_sizes[1]; i1++)
    {
      for (size_t i2 = 0; i2 < dim_sizes[2]; i2++)
      {
        for (size_t i3 = 0; i3 < dim_sizes[3]; i3++)
        {
          for (size_t i4 = 0; i4 < dim_sizes[4]; i4++)
          {
            for (size_t i5 = 0; i5 < dim_sizes[5]; i5++)
            {
              uint64_t offset_a = i0 * strides_in0[0] + i1 * strides_in0[1] + i2 * strides_in0[2] + i3 * strides_in0[3] +
                                  i4 * strides_in0[4] + i5 * strides_in0[5];
              uint64_t offset_b = i0 * strides_in1[0] + i1 * strides_in1[1] + i2 * strides_in1[2] + i3 * strides_in1[3] +
                                  i4 * strides_in1[4] + i5 * strides_in1[5];
              uint64_t offset_c = i0 * strides_out[0] + i1 * strides_out[1] + i2 * strides_out[2] + i3 * strides_out[3] +
                                  i4 * strides_out[4] + i5 * strides_out[5];
              test.naive_matmul_M_N_K_Batch(test.matrix_a.data() + offset_a, test.matrix_b.data() + offset_b,
                                            test.matrix_c_verify.data() + offset_c, 32, 32, 32, 32 * 32, 32 * 32);
            }
          }
        }
      }
    }
  }

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with outer loop with main kernel: brgemm", "[tensor_operation][brgemm][correctness]")
{
  using namespace mini_jit;

  constexpr TensorOperation::dim_t dim_types[]{
    TensorOperation::dim_t::n, TensorOperation::dim_t::k, TensorOperation::dim_t::c, TensorOperation::dim_t::m, TensorOperation::dim_t::k,
    TensorOperation::dim_t::m, TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};

  constexpr TensorOperation::exec_t exec_types[]{
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
    TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};

  constexpr int64_t dim_sizes[]{2, 3, 5, 8, 13, 21, 3, 32, 32, 32};
  constexpr int64_t strides_in0[]{0,                              // n-dim
                                  32 * 32 * 3 * 21 * 13 * 8 * 5,  // k-dim
                                  32 * 32 * 3 * 21 * 13 * 8,      // c-dim
                                  32 * 32 * 3 * 21 * 13,          // m-dim
                                  32 * 32 * 3 * 21,               // k-dim
                                  32 * 32 * 3,                    // m-dim
                                  32 * 32,                        // k-dim-prim
                                  1,                              // m-dim-prim
                                  0,                              // n-dim-prim
                                  32};                            // k-dim-prim
  constexpr int64_t strides_in1[]{32 * 32 * 3 * 1 * 13 * 1 * 5 * 3,
                                  32 * 32 * 3 * 1 * 13 * 1 * 5,
                                  32 * 32 * 3 * 1 * 13 * 1,
                                  0,  // m-dim
                                  32 * 32 * 3 * 1,
                                  0,        // m-dim
                                  32 * 32,  // k-dim-prim
                                  0,
                                  32,
                                  1};
  constexpr int64_t strides_out[]{32 * 32 * 21 * 1 * 8 * 5 * 1,
                                  0,  // k-dim
                                  32 * 32 * 21 * 1 * 8,
                                  32 * 32 * 21 * 1,
                                  0,  // k-dim
                                  32 * 32,
                                  0,
                                  1,
                                  32,
                                  0};

  GenerationTest test(32, 32, 32, 3, 32 * 32 * 3 * 21 * 13 * 8 * 5 * 3 * 1, 32 * 32 * 3 * 1 * 13 * 1 * 5 * 3 * 2,
                      32 * 32 * 21 * 1 * 8 * 5 * 1 * 2);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::brgemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  for (size_t i0 = 0; i0 < dim_sizes[0]; i0++)
  {
    for (size_t i1 = 0; i1 < dim_sizes[1]; i1++)
    {
      for (size_t i2 = 0; i2 < dim_sizes[2]; i2++)
      {
        for (size_t i3 = 0; i3 < dim_sizes[3]; i3++)
        {
          for (size_t i4 = 0; i4 < dim_sizes[4]; i4++)
          {
            for (size_t i5 = 0; i5 < dim_sizes[5]; i5++)
            {
              uint64_t offset_a = i0 * strides_in0[0] + i1 * strides_in0[1] + i2 * strides_in0[2] + i3 * strides_in0[3] +
                                  i4 * strides_in0[4] + i5 * strides_in0[5];
              uint64_t offset_b = i0 * strides_in1[0] + i1 * strides_in1[1] + i2 * strides_in1[2] + i3 * strides_in1[3] +
                                  i4 * strides_in1[4] + i5 * strides_in1[5];
              uint64_t offset_c = i0 * strides_out[0] + i1 * strides_out[1] + i2 * strides_out[2] + i3 * strides_out[3] +
                                  i4 * strides_out[4] + i5 * strides_out[5];
              test.naive_matmul_M_N_K_Batch(test.matrix_a.data() + offset_a, test.matrix_b.data() + offset_b,
                                            test.matrix_c_verify.data() + offset_c, 32, 32, 32, 32 * 32, 32 * 32);
            }
          }
        }
      }
    }
  }

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
{
  using namespace mini_jit;

  auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::relu, TensorOperation::prim_t::copy);

  CAPTURE(type);

  constexpr TensorOperation::dim_t dim_types[]{
    TensorOperation::dim_t::n, TensorOperation::dim_t::k, TensorOperation::dim_t::c, TensorOperation::dim_t::m, TensorOperation::dim_t::k,
    TensorOperation::dim_t::m, TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};

  constexpr TensorOperation::exec_t exec_types[]{
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
    TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};

  constexpr int64_t dim_sizes[]{2, 3, 5, 8, 13, 21, 3, 32, 32, 32};
  constexpr int64_t strides_in0[]{0,                              // n-dim
                                  32 * 3 * 32 * 21 * 13 * 8 * 5,  // k-dim
                                  32 * 3 * 32 * 21 * 13 * 8,      // c-dim
                                  32 * 3 * 32 * 21 * 13,          // m-dim
                                  32 * 3 * 32 * 21,               // k-dim
                                  32 * 3 * 32,                    // m-dim
                                  32 * 3,                         // k-dim-prim
                                  1,                              // m-dim-prim
                                  0,                              // n-dim-prim
                                  32};                            // k-dim-prim
  constexpr int64_t strides_in1[]{32 * 3 * 32 * 1 * 13 * 1 * 5 * 3,
                                  32 * 3 * 32 * 1 * 13 * 1 * 5,
                                  32 * 3 * 32 * 1 * 13 * 1,
                                  0,  // m-dim
                                  32 * 3 * 32 * 1,
                                  0,       // m-dim
                                  32 * 3,  // k-dim-prim
                                  0,
                                  32,
                                  1};
  constexpr int64_t strides_out[]{32 * 32 * 21 * 1 * 8 * 5 * 1,
                                  0,  // k-dim
                                  32 * 32 * 21 * 1 * 8,
                                  32 * 32 * 21 * 1,
                                  0,  // k-dim
                                  32 * 32,
                                  0,
                                  1,
                                  32,
                                  0};

  GenerationTest test(32, 32, 32, 3, 32 * 32 * 3 * 21 * 13 * 8 * 5 * 3 * 1, 32 * 32 * 3 * 1 * 13 * 1 * 5 * 3 * 2,
                      32 * 32 * 21 * 1 * 8 * 5 * 1 * 2);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, type, TensorOperation::prim_t::none, TensorOperation::prim_t::none, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // First touch operation
  UnaryType test_first_type = UnaryType::None;
  switch (type)
  {
  case TensorOperation::prim_t::zero:
    test_first_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_first_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_first_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // Main kernel operation
  for (int64_t i0 = 0; i0 < dim_sizes[0]; i0++)
  {
    for (int64_t i1 = 0; i1 < dim_sizes[1]; i1++)  // k-dim
    {
      for (int64_t i2 = 0; i2 < dim_sizes[2]; i2++)
      {
        for (int64_t i3 = 0; i3 < dim_sizes[3]; i3++)
        {
          for (int64_t i4 = 0; i4 < dim_sizes[4]; i4++)  // k-dim
          {
            for (int64_t i5 = 0; i5 < dim_sizes[5]; i5++)
            {
              uint64_t offset_c = i0 * strides_out[0] + i1 * strides_out[1] + i2 * strides_out[2] + i3 * strides_out[3] +
                                  i4 * strides_out[4] + i5 * strides_out[5];
              if (i1 == 0 && i4 == 0)
              {
                std::cout << "FIRST TOUCH" << std::endl;
                // First touch
                test.naive_unary_M_N(test.matrix_c_verify.data() + offset_c, test.matrix_c_verify.data() + offset_c, 32, 32, false,
                                     test_first_type);
              }
            }
          }
        }
      }
    }
  }

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
{
  using namespace mini_jit;

  auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::relu, TensorOperation::prim_t::copy);

  CAPTURE(type);

  constexpr TensorOperation::dim_t dim_types[]{
    TensorOperation::dim_t::n, TensorOperation::dim_t::k, TensorOperation::dim_t::c, TensorOperation::dim_t::m, TensorOperation::dim_t::k,
    TensorOperation::dim_t::m, TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};

  constexpr TensorOperation::exec_t exec_types[]{
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
    TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};

  constexpr int64_t dim_sizes[]{2, 3, 5, 8, 13, 21, 3, 32, 32, 32};
  constexpr int64_t strides_in0[]{0,                              // n-dim
                                  32 * 3 * 32 * 21 * 13 * 8 * 5,  // k-dim
                                  32 * 3 * 32 * 21 * 13 * 8,      // c-dim
                                  32 * 3 * 32 * 21 * 13,          // m-dim
                                  32 * 3 * 32 * 21,               // k-dim
                                  32 * 3 * 32,                    // m-dim
                                  32 * 3,                         // k-dim-prim
                                  1,                              // m-dim-prim
                                  0,                              // n-dim-prim
                                  32};                            // k-dim-prim
  constexpr int64_t strides_in1[]{32 * 3 * 32 * 1 * 13 * 1 * 5 * 3,
                                  32 * 3 * 32 * 1 * 13 * 1 * 5,
                                  32 * 3 * 32 * 1 * 13 * 1,
                                  0,  // m-dim
                                  32 * 3 * 32 * 1,
                                  0,       // m-dim
                                  32 * 3,  // k-dim-prim
                                  0,
                                  32,
                                  1};
  constexpr int64_t strides_out[]{32 * 32 * 21 * 1 * 8 * 5 * 1,
                                  0,  // k-dim
                                  32 * 32 * 21 * 1 * 8,
                                  32 * 32 * 21 * 1,
                                  0,  // k-dim
                                  32 * 32,
                                  0,
                                  1,
                                  32,
                                  0};

  GenerationTest test(32, 32, 32, 3, 32 * 32 * 3 * 21 * 13 * 8 * 5 * 3 * 1, 32 * 32 * 3 * 1 * 13 * 1 * 5 * 3 * 2,
                      32 * 32 * 21 * 1 * 8 * 5 * 1 * 2);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::none, type, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // Last touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_last_type = UnaryType::ReLu;
    break;
  default:
    FAIL("Could not parse the unary type!");
    break;
  }

  // Main kernel operation
  for (int64_t i0 = 0; i0 < dim_sizes[0]; i0++)
  {
    for (int64_t i1 = 0; i1 < dim_sizes[1]; i1++)  // k-dim
    {
      for (int64_t i2 = 0; i2 < dim_sizes[2]; i2++)
      {
        for (int64_t i3 = 0; i3 < dim_sizes[3]; i3++)
        {
          for (int64_t i4 = 0; i4 < dim_sizes[4]; i4++)  // k-dim
          {
            for (int64_t i5 = 0; i5 < dim_sizes[5]; i5++)
            {
              uint64_t offset_c = i0 * strides_out[0] + i1 * strides_out[1] + i2 * strides_out[2] + i3 * strides_out[3] +
                                  i4 * strides_out[4] + i5 * strides_out[5];
              if (i1 == (dim_sizes[1] - 1) && i4 == (dim_sizes[4] - 1))
              {
                std::cout << "LAST TOUCH" << std::endl;
                // last touch
                test.naive_unary_M_N(test.matrix_c_verify.data() + offset_c, test.matrix_c_verify.data() + offset_c, 32, 32, false,
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

// TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: gemm",
//           "[tensor_operation][unary][gemm][correctness]")
// {
//   using namespace mini_jit;

//   auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

//   CAPTURE(first_type);

//   constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};
//   constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
//                                                  TensorOperation::exec_t::prim};
//   constexpr int64_t dim_sizes[]{64, 64, 64};
//   constexpr int64_t strides_in0[]{1, 0, 64};
//   constexpr int64_t strides_in1[]{0, 64, 1};
//   constexpr int64_t strides_out[]{1, 64, 0};

//   GenerationTest test(64, 64, 64);

//   mini_jit::TensorOperation tensor_op;
//   TensorOperation::error_t err = tensor_op.setup(
//     TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::gemm, TensorOperation::prim_t::none, std::span{dim_types},
//     std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

//   REQUIRE(err == TensorOperation::error_t::success);

//   tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

//   // First touch operation
//   UnaryType test_fist_type = UnaryType::None;
//   switch (first_type)
//   {
//   case TensorOperation::prim_t::zero:
//     test_fist_type = UnaryType::Zero;
//     break;
//   case TensorOperation::prim_t::copy:
//     test_fist_type = UnaryType::Identity;
//     break;
//   case TensorOperation::prim_t::relu:
//     test_fist_type = UnaryType::ReLu;
//     break;

//   default:
//     break;
//   }

//   // First touch
//   test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_fist_type);

//   // Main kernel operation
//   test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

//   test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
// }

// TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy) & main kernel: gemm",
//           "[tensor_operation][unary][gemm][correctness]")
// {
//   using namespace mini_jit;

//   auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

//   CAPTURE(last_type);

//   constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};
//   constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
//                                                  TensorOperation::exec_t::prim};
//   constexpr int64_t dim_sizes[]{64, 64, 64};
//   constexpr int64_t strides_in0[]{1, 0, 64};
//   constexpr int64_t strides_in1[]{0, 64, 1};
//   constexpr int64_t strides_out[]{1, 64, 0};

//   GenerationTest test(64, 64, 64);

//   mini_jit::TensorOperation tensor_op;
//   TensorOperation::error_t err = tensor_op.setup(
//     TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::gemm, last_type, std::span{dim_types},
//     std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

//   REQUIRE(err == TensorOperation::error_t::success);

//   tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

//   // Last touch operation
//   UnaryType test_last_type = UnaryType::None;
//   switch (last_type)
//   {
//   case TensorOperation::prim_t::zero:
//     test_last_type = UnaryType::Zero;
//     break;
//   case TensorOperation::prim_t::copy:
//     test_last_type = UnaryType::Identity;
//     break;
//   case TensorOperation::prim_t::relu:
//     test_last_type = UnaryType::ReLu;
//     break;
//   default:
//     FAIL("Could not parse the unary type!");
//     break;
//   }

//   // Main kernel operation
//   test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

//   // Last touch
//   test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

//   test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
// }

// TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: gemm & last touch: unary
// (zero, "
//           "relu, copy)",
//           "[tensor_operation][unary][gemm][correctness]")
// {
//   using namespace mini_jit;

//   auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);
//   auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

//   CAPTURE(first_type, last_type);

//   constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};
//   constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
//                                                  TensorOperation::exec_t::prim};
//   constexpr int64_t dim_sizes[]{64, 64, 64};
//   constexpr int64_t strides_in0[]{1, 0, 64};
//   constexpr int64_t strides_in1[]{0, 64, 1};
//   constexpr int64_t strides_out[]{1, 64, 0};

//   GenerationTest test(64, 64, 64);

//   mini_jit::TensorOperation tensor_op;
//   TensorOperation::error_t err =
//     tensor_op.setup(TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::gemm, last_type, std::span{dim_types},
//                     std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

//   REQUIRE(err == TensorOperation::error_t::success);

//   tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

//   // First touch operation
//   UnaryType test_fist_type = UnaryType::None;
//   switch (first_type)
//   {
//   case TensorOperation::prim_t::zero:
//     test_fist_type = UnaryType::Zero;
//     break;
//   case TensorOperation::prim_t::copy:
//     test_fist_type = UnaryType::Identity;
//     break;
//   case TensorOperation::prim_t::relu:
//     test_fist_type = UnaryType::ReLu;
//     break;

//   default:
//     break;
//   }

//   // Last touch operation
//   UnaryType test_last_type = UnaryType::None;
//   switch (last_type)
//   {
//   case TensorOperation::prim_t::zero:
//     test_last_type = UnaryType::Zero;
//     break;
//   case TensorOperation::prim_t::copy:
//     test_last_type = UnaryType::Identity;
//     break;
//   case TensorOperation::prim_t::relu:
//     test_last_type = UnaryType::ReLu;
//     break;
//   default:
//     FAIL("Could not parse the unary type!");
//     break;
//   }

//   // First touch
//   test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_fist_type);

//   // Main kernel operation
//   test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

//   // Last touch
//   test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

//   test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
// }

// // Brgemm
// TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: brgemm",
//           "[tensor_operation][unary][brgemm][correctness]")
// {
//   using namespace mini_jit;

//   auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

//   CAPTURE(first_type);

//   constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n,
//                                                TensorOperation::dim_t::k};
//   constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
//                                                  TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
//   constexpr int64_t dim_sizes[]{64, 64, 64, 64};
//   constexpr int64_t strides_in0[]{64 * 64, 1, 0, 64};
//   constexpr int64_t strides_in1[]{64 * 64, 0, 64, 1};
//   constexpr int64_t strides_out[]{0, 1, 64, 0};

//   GenerationTest test(64, 64, 64, 64);

//   mini_jit::TensorOperation tensor_op;
//   TensorOperation::error_t err = tensor_op.setup(
//     TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::brgemm, TensorOperation::prim_t::none, std::span{dim_types},
//     std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

//   REQUIRE(err == TensorOperation::error_t::success);

//   tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

//   // First touch operation
//   UnaryType test_fist_type = UnaryType::None;
//   switch (first_type)
//   {
//   case TensorOperation::prim_t::zero:
//     test_fist_type = UnaryType::Zero;
//     break;
//   case TensorOperation::prim_t::copy:
//     test_fist_type = UnaryType::Identity;
//     break;
//   case TensorOperation::prim_t::relu:
//     test_fist_type = UnaryType::ReLu;
//     break;

//   default:
//     break;
//   }

//   // First touch
//   test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_fist_type);

//   // Main kernel operation
//   test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

//   test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
// }

// TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy) & main kernel: brgemm",
//           "[tensor_operation][unary][brgemm][correctness]")
// {
//   using namespace mini_jit;

//   auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

//   CAPTURE(last_type);

//   constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n,
//                                                TensorOperation::dim_t::k};
//   constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
//                                                  TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
//   constexpr int64_t dim_sizes[]{64, 64, 64, 64};
//   constexpr int64_t strides_in0[]{64 * 64, 1, 0, 64};
//   constexpr int64_t strides_in1[]{64 * 64, 0, 64, 1};
//   constexpr int64_t strides_out[]{0, 1, 64, 0};

//   GenerationTest test(64, 64, 64, 64);

//   mini_jit::TensorOperation tensor_op;
//   TensorOperation::error_t err = tensor_op.setup(
//     TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::brgemm, last_type, std::span{dim_types},
//     std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

//   REQUIRE(err == TensorOperation::error_t::success);

//   tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

//   // Last touch operation
//   UnaryType test_last_type = UnaryType::None;
//   switch (last_type)
//   {
//   case TensorOperation::prim_t::zero:
//     test_last_type = UnaryType::Zero;
//     break;
//   case TensorOperation::prim_t::copy:
//     test_last_type = UnaryType::Identity;
//     break;
//   case TensorOperation::prim_t::relu:
//     test_last_type = UnaryType::ReLu;
//     break;
//   default:
//     FAIL("Could not parse the unary type!");
//     break;
//   }

//   // Main kernel operation
//   test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

//   // Last touch
//   test.naive_unary_M_N(test.matrix_c_verify.data(), test.matrix_c_verify.data(), 64, 64, false, test_last_type);

//   test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
// }

TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: brgemm & last touch: unary "
          "(zero, relu, copy)",
          "[tensor_operation][unary][brgemm][correctness]")
{
  using namespace mini_jit;

  auto first_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);
  auto last_type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::copy, TensorOperation::prim_t::relu);

  CAPTURE(first_type, last_type);

  using namespace mini_jit;

  constexpr TensorOperation::dim_t dim_types[]{
    TensorOperation::dim_t::n, TensorOperation::dim_t::k, TensorOperation::dim_t::c, TensorOperation::dim_t::m, TensorOperation::dim_t::k,
    TensorOperation::dim_t::m, TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n, TensorOperation::dim_t::k};

  constexpr TensorOperation::exec_t exec_types[]{
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq,
    TensorOperation::exec_t::seq,  TensorOperation::exec_t::seq, TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
    TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};

  constexpr int64_t dim_sizes[]{2, 3, 5, 8, 13, 21, 3, 16, 16, 16};
  constexpr int64_t strides_in0[]{0,                              // n-dim
                                  16 * 16 * 3 * 21 * 13 * 8 * 5,  // k-dim
                                  16 * 16 * 3 * 21 * 13 * 8,      // c-dim
                                  16 * 16 * 3 * 21 * 13,          // m-dim
                                  16 * 16 * 3 * 21,               // k-dim
                                  16 * 16 * 3,                    // m-dim
                                  16 * 16,                        // k-dim-prim
                                  1,                              // m-dim-prim
                                  0,                              // n-dim-prim
                                  16};                            // k-dim-prim
  constexpr int64_t strides_in1[]{16 * 16 * 3 * 1 * 13 * 1 * 5 * 3,
                                  16 * 16 * 3 * 1 * 13 * 1 * 5,
                                  16 * 16 * 3 * 1 * 13 * 1,
                                  0,  // m-dim
                                  16 * 16 * 3 * 1,
                                  0,        // m-dim
                                  16 * 16,  // k-dim-prim
                                  0,
                                  16,
                                  1};
  constexpr int64_t strides_out[]{16 * 16 * 21 * 1 * 8 * 5 * 1,
                                  0,  // k-dim
                                  16 * 16 * 21 * 1 * 8,
                                  16 * 16 * 21 * 1,
                                  0,  // k-dim
                                  16 * 16,
                                  0,
                                  1,
                                  16,
                                  0};

  GenerationTest test(16, 16, 16, 3, 16 * 16 * 3 * 21 * 13 * 8 * 5 * 3 * 1, 16 * 16 * 3 * 1 * 13 * 1 * 5 * 3 * 2,
                      16 * 16 * 21 * 1 * 8 * 5 * 1 * 2);
  test.SetUp(TestInfill::Random);

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, first_type, TensorOperation::prim_t::brgemm, last_type, std::span{dim_types},
                    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  // First touch operation
  UnaryType test_fist_type = UnaryType::None;
  switch (first_type)
  {
  case TensorOperation::prim_t::zero:
    test_fist_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_fist_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
    test_fist_type = UnaryType::ReLu;
    break;

  default:
    break;
  }

  // Last touch operation
  UnaryType test_last_type = UnaryType::None;
  switch (last_type)
  {
  case TensorOperation::prim_t::zero:
    test_last_type = UnaryType::Zero;
    break;
  case TensorOperation::prim_t::copy:
    test_last_type = UnaryType::Identity;
    break;
  case TensorOperation::prim_t::relu:
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
              if (i1 == 0 && i4 == 0)
              {
                // First touch
                test.naive_unary_M_N(test.matrix_c_verify.data() + offset_c, test.matrix_c_verify.data() + offset_c, 16, 16, false,
                                     test_fist_type);
              }
              test.naive_matmul_M_N_K_Batch(test.matrix_a.data() + offset_a, test.matrix_b.data() + offset_b,
                                            test.matrix_c_verify.data() + offset_c, 16, 16, 16, 16 * 16, 16 * 16);
              if (i1 == (dim_sizes[1] - 1) && i4 == (dim_sizes[4] - 1))
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