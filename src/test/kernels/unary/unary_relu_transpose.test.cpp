#include "../../../main/kernels/unary/unary_relu_transpose.h"
#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test unary relu transpose symmetric jited correctness counting data", "[jit][correctness][unary]")
{
  auto M = GENERATE(range(1u, 73u + 1u, 1u));
  auto N = M;
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::unary_relu_transpose(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, M, UnaryType::Identity_Transpose);
}

TEST_CASE("Test unary relu transpose rest jited correctness random data", "[jit][correctness][unary]")
{
  //   auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  //   auto M = GENERATE(16u, 48u);
  //   auto N = M;
  //   CAPTURE(M, N, MRest);
  //   auto _M = M + MRest;
  //   UnaryTestFixture unaryTest(_M, N);
  //   unaryTest.SetUp(TestInfill::Random);
  //   mini_jit::kernels::unary_relu_transpose(unaryTest.native_kernel, _M, N);
  //   unaryTest.RunTest(_M, _M, UnaryType::Identity);
}
