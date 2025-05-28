#include "../main/TensorOperation.h"
#include "BaseGeneration.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <span>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xrandom.hpp>

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

  auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::relu, TensorOperation::prim_t::copy);

  CAPTURE(type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64};
  constexpr int64_t strides_in0[]{1, 64};
  constexpr int64_t strides_in1[]{0, 0};
  constexpr int64_t strides_out[]{1, 64};

  GenerationTest test(64, 64, 64);

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

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::brgemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c_verify.data(), test.matrix_c.size());
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
TEST_CASE("Test tensor operation with outer loop with main kernel: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
{
  using namespace mini_jit;

  auto type = GENERATE(TensorOperation::prim_t::zero, TensorOperation::prim_t::relu, TensorOperation::prim_t::copy);

  CAPTURE(type);

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::m, TensorOperation::dim_t::n};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64};
  constexpr int64_t strides_in0[]{1, 64};
  constexpr int64_t strides_in1[]{0, 0};
  constexpr int64_t strides_out[]{1, 64};

  GenerationTest test(64, 64, 64);

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

TEST_CASE("Test tensor operation with outer loop with main kernel: gemm", "[tensor_operation][gemm][correctness]")
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

TEST_CASE("Test tensor operation with outer loop with main kernel: brgemm", "[tensor_operation][brgemm][correctness]")
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

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::brgemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

  test.naive_matmul_M_N_K_Batch(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c_verify.data(), 64, 64, 64, 64 * 64, 64 * 64);

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c_verify.data(), test.matrix_c.size());
}

TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
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

TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
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

TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: gemm",
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

TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy) & main kernel: gemm",
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

TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: gemm & last touch: unary (zero, "
          "relu, copy)",
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
TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: brgemm",
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

TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy) & main kernel: brgemm",
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

TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: brgemm & last touch: unary "
          "(zero, relu, copy)",
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