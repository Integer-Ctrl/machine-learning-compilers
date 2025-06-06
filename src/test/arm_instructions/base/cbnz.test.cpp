#include "../../../main/arm_instructions/base/cbnz.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test cbnz 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = cbnz(w25, 20*4);
  uint32_t expected = 0b0'0110101'0000000000000010100'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test cbnz 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = cbnz(x25, -36*4);
  uint32_t expected = 0b1'0110101'1111111111111011100'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test cbnz 64bit instruction with offset (-144*4)", "[codegen][64bit]")
{
  uint32_t value = cbnz(x25, -144*4);
  uint32_t expected = 0b1'0110101'1111111111101110000'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test cbnz 64bit instruction with offset (-143*4)", "[codegen][64bit]")
{
  uint32_t value = cbnz(x25, -143*4);
  uint32_t expected = 0b1'0110101'1111111111101110001'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test cbnz 64bit instruction with offset (-26*4)", "[codegen][64bit]")
{
  uint32_t value = cbnz(x25, -26*4);
  uint32_t expected = 0b1'0110101'1111111111111100110'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test cbnz internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::cbnz(25, -32*4, true);
  uint32_t expected = 0b1'0110101'1111111111111100000'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}