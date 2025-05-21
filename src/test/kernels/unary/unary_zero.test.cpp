#include "../../../main/kernels/unary/unary_zero.h"
#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test unary zero jited correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 50, 64, 512, 2048);
  const uint32_t N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 50, 64, 512, 2048);
  CAPTURE(M, N);

  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_zero(unaryTest.native_kernel, M / 16, N, M % 16);
  unaryTest.RunTest(M, M, UnaryType::Zero);
}