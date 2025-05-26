#include "../main/TensorOperation.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <span>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xrandom.hpp>

void verify_tensor(const float *__restrict__ expected, const float *__restrict__ result, uint32_t size)
{
  for (size_t i = 0; i < size; i++)
  {
    CAPTURE(i, result[i], expected[i]);

    if (std::isnan(expected[i]))
    {
      REQUIRE_THAT(result[i], Catch::Matchers::IsNaN());
    }
    else
    {
      REQUIRE_THAT(result[i], Catch::Matchers::WithinRel(expected[i]));
    }
  }
}

TEST_CASE("Test tensor operation with main kernel: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
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

  xt::random::seed(Catch::rngSeed());

  xt::xtensor<float, 2> tensorIn0({64, 64});
  tensorIn0 = xt::random::rand(tensorIn0.shape(), 0.f, 1.f);
  xt::xtensor<float, 2> tensorOut({64, 64});
  tensorOut = xt::random::rand(tensorIn0.shape(), 0.f, 1.f);
  xt::xtensor<float, 2> tensorOutVerify({64, 64});
  std::copy(tensorOut.begin(), tensorOut.end(), tensorOutVerify.begin());

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err = tensor_op.setup(
    TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, type, TensorOperation::prim_t::none, std::span{dim_types},
    std::span{exec_types}, std::span{dim_sizes}, std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(tensorIn0.data(), nullptr, tensorOut.data());

  switch (type)
  {
  case TensorOperation::prim_t::zero:
    tensorOutVerify.fill(0);
    break;
  case TensorOperation::prim_t::copy:
    std::copy(tensorIn0.begin(), tensorIn0.end(), tensorOutVerify.begin());
    break;
  case TensorOperation::prim_t::relu:
    std::transform(tensorIn0.begin(), tensorIn0.end(), tensorOutVerify.begin(), [](auto &&v) { return std::max(v, 0.f); });
    break;

  default:
    break;
  }

  verify_tensor(tensorOutVerify.data(), tensorOut.data(), tensorOut.size());
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

  xt::random::seed(Catch::rngSeed());

  xt::xtensor<float, 2> tensorIn0({64, 64});
  tensorIn0 = xt::random::rand(tensorIn0.shape(), 0.f, 1.f);
  xt::xtensor<float, 2> tensorIn1({64, 64});
  tensorIn1 = xt::random::rand(tensorIn1.shape(), 0.f, 1.f);
  xt::xtensor<float, 2> tensorOut({64, 64});
  tensorOut = xt::random::rand(tensorIn0.shape(), 0.f, 1.f);
  xt::xtensor<float, 2> tensorOutVerify({64, 64});
  std::copy(tensorOut.begin(), tensorOut.end(), tensorOutVerify.begin());

  std::cout << tensorIn0[0] << " " << tensorIn1[0] << " " << tensorOutVerify[0] << " " << tensorOut[0] << std::endl;

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(tensorIn0.data(), tensorIn1.data(), tensorOut.data());

  std::cout << tensorOutVerify[0] << " " << tensorOut[0] << std::endl;

  // TODO: Implement the verification logic for naive gemm operation

  verify_tensor(tensorOutVerify.data(), tensorOut.data(), tensorOut.size());
  FAIL();
}

TEST_CASE("Test tensor operation with main kernel: brgemm", "[tensor_operation][brgemm][correctness]")
{
  using namespace mini_jit;

  constexpr TensorOperation::dim_t dim_types[]{TensorOperation::dim_t::k, TensorOperation::dim_t::m, TensorOperation::dim_t::n,
                                               TensorOperation::dim_t::k};
  constexpr TensorOperation::exec_t exec_types[]{TensorOperation::exec_t::prim, TensorOperation::exec_t::prim,
                                                 TensorOperation::exec_t::prim, TensorOperation::exec_t::prim};
  constexpr int64_t dim_sizes[]{64, 64, 64, 64};
  constexpr int64_t strides_in0[]{64, 1, 0, 64};
  constexpr int64_t strides_in1[]{64, 0, 64, 1};
  constexpr int64_t strides_out[]{0, 1, 64, 0};

  xt::random::seed(Catch::rngSeed());

  xt::xtensor<float, 3> tensorIn0({64, 64, 64});
  tensorIn0 = xt::random::rand(tensorIn0.shape(), 0.f, 1.f);
  xt::xtensor<float, 3> tensorIn1({64, 64, 64});
  tensorIn1 = xt::random::rand(tensorIn1.shape(), 0.f, 1.f);
  xt::xtensor<float, 2> tensorOut({64, 64});
  tensorOut = xt::random::rand(tensorIn0.shape(), 0.f, 1.f);
  xt::xtensor<float, 2> tensorOutVerify({64, 64});
  std::copy(tensorOut.begin(), tensorOut.end(), tensorOutVerify.begin());

  mini_jit::TensorOperation tensor_op;
  TensorOperation::error_t err =
    tensor_op.setup(TensorOperation::dtype_t::fp32, TensorOperation::prim_t::none, TensorOperation::prim_t::brgemm,
                    TensorOperation::prim_t::none, std::span{dim_types}, std::span{exec_types}, std::span{dim_sizes},
                    std::span{strides_in0}, std::span{strides_in1}, std::span{strides_out});

  REQUIRE(err == TensorOperation::error_t::success);

  tensor_op.execute(tensorIn0.data(), tensorIn1.data(), tensorOut.data());

  // TODO: Implement the verification logic for naive brgemm operation

  verify_tensor(tensorOutVerify.data(), tensorOut.data(), tensorOut.size());
}
