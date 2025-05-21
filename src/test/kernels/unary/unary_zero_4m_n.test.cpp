#include "../../../main/kernels/unary/unary_zero_4m_n.h"
#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test unary zero_4m_n jited correctness random data", "[jit][correctness][gemm]")
{
  const uint32_t M = GENERATE(16, 64, 512, 2048);
  const uint32_t N = GENERATE(16, 50, 64, 512, 2048);
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_zero_4m_n(unaryTest.native_kernel, M / 4, N);
  unaryTest.RunTest(M, M, UnaryType::Zero);
}