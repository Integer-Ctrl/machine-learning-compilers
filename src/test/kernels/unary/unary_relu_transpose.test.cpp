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
  UnaryTestFixture unaryTest(M, N, M, N, true);
  unaryTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::unary_relu_transpose(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, N, UnaryType::ReLu);  // true = transpose
}

TEST_CASE("Test unary relu transpose none symmetric jited correctness counting data", "[jit][correctness][unary]")
{
  auto M = GENERATE(range(1u, 73u + 1u, 1u));
  auto N = GENERATE(range(1u, 73u + 1u, 1u));
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N, M, N, true);
  unaryTest.SetUp(TestInfill::Counting);
  mini_jit::kernels::unary_relu_transpose(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, N, UnaryType::ReLu);  // true = transpose
}

TEST_CASE("Test unary relu transpose symmetric jited correctness random data", "[jit][correctness][unary]")
{
  auto M = GENERATE(range(1u, 73u + 1u, 1u));
  auto N = M;
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N, M, N, true);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_relu_transpose(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, N, UnaryType::ReLu);  // true = transpose
}

TEST_CASE("Test unary relu transpose none symmetric jited correctness random data", "[jit][correctness][unary]")
{
  auto M = GENERATE(range(1u, 73u + 1u, 1u));
  auto N = GENERATE(range(1u, 73u + 1u, 1u));
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N, M, N, true);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_relu_transpose(unaryTest.native_kernel, M, N);
  unaryTest.RunTest(M, N, UnaryType::ReLu);
}
