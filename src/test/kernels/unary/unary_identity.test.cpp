#include "../../../main/kernels/unary/unary_identity.h"
#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test unary identity no rest jited correctness random data", "[jit][correctness][unary]")
{
  auto M = GENERATE(64u, 512u, 2048u);
  auto N = GENERATE(50u, 64u, 512u, 2048u);
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_identity(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, M, false, UnaryType::Identity);  // false = no transpose
}

TEST_CASE("Test unary identity rest jited correctness random data", "[jit][correctness][unary]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto M = GENERATE(16u, 48u);
  auto N = M;
  CAPTURE(M, N, MRest);
  auto _M = M + MRest;
  UnaryTestFixture unaryTest(_M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_identity(unaryTest.native_kernel, _M, N);
  unaryTest.RunTest(_M, _M, false, UnaryType::Identity);  // false = no transpose
}

TEST_CASE("Test unary identity no rest jited correctness counting data", "[jit][correctness][unary]")
{
  auto M = GENERATE(64u, 512u, 2048u);
  auto N = GENERATE(50u, 64u, 512u, 2048u);
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::unary_identity(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, M, false, UnaryType::Identity);  // false = no transpose
}

TEST_CASE("Test unary identity rest jited correctness counting data", "[jit][correctness][unary]")
{
  auto MRest = GENERATE(range(1u, 15u + 1u, 1u));
  auto M = GENERATE(16u, 48u);
  auto N = M;
  CAPTURE(M, N, MRest);
  auto _M = M + MRest;
  UnaryTestFixture unaryTest(_M, N);
  unaryTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::unary_identity(unaryTest.native_kernel, _M, N);
  unaryTest.RunTest(_M, _M, false, UnaryType::Identity);  // false = no transpose
}
