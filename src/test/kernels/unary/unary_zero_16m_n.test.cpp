#include "../../../main/kernels/unary/unary_zero_16m_n.h"
#include "unary.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstdint>

TEST_CASE("Test unary zero_16m_n jited correctness random data", "[jit][correctness][unary]")
{
  const uint32_t M = GENERATE(64u, 512u, 2048u);
  const uint32_t N = GENERATE(50u, 64u, 512u, 2048u);
  CAPTURE(M, N);
  UnaryTestFixture unaryTest(M, N);
  unaryTest.SetUp(TestInfill::Random);
  mini_jit::kernels::unary_zero_16m_n(unaryTest.native_kernel, M / 16u, N);
  unaryTest.RunTest(M, M, UnaryType::Zero);
}