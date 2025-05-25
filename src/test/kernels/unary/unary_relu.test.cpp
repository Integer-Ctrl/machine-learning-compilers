#include "../../../main/kernels/unary/unary_relu.h"
#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>
#include <iostream>

TEST_CASE("Test unary relu no rest jited correctness random data", "[jit][correctness][gemm]")
{
  auto M = GENERATE(64u, 512u, 2048u);
  auto N = GENERATE(50u, 64u, 512u, 2048u);
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_relu(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, M, UnaryType::ReLu);  // false = no transpose
}

TEST_CASE("Test unary relu rest jited correctness random data", "[jit][correctness][gemm]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto M = GENERATE(16u, 48u);
  auto N = M;
  CAPTURE(M, N, MRest);
  auto _M = M + MRest;
  UnaryTestFixture unaryTest(_M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_relu(unaryTest.native_kernel, _M, N);
  unaryTest.RunTest(_M, _M, UnaryType::ReLu);  // false = no transpose
}
