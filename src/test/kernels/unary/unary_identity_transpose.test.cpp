#include "../../../main/kernels/unary/unary_identity_transpose.h"
#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test unary identity transpose symmetric jited correctness counting data", "[jit][correctness][unary]")
{
  auto M = GENERATE(range(1u, 73u + 1u, 1u));
  auto N = M;
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::unary_identity_transpose(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, N, true, UnaryType::Identity);  // true = transpose
}

TEST_CASE("Test unary identity transpose none symmetric jited correctness random data", "[jit][correctness][unary]")
{
  auto M = GENERATE(1);
  auto N = GENERATE(5);
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_identity_transpose(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, M, true, UnaryType::Identity);  // true = transpose
}
