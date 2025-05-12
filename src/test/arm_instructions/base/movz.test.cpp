
#include "../../../main/arm_instructions/base/movz.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test movz 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = movz(w12, 1033);
  uint32_t expected = 0b0'10100101'00'0000010000001001'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movz 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = movz(x12, 1033);
  uint32_t expected = 0b1'10100101'00'0000010000001001'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movz shift 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = movz(w12, 1033, 16);
  uint32_t expected = 0b0'10100101'01'0000010000001001'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movz shift 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = movz(x12, 1033, 32);
  uint32_t expected = 0b1'10100101'10'0000010000001001'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movz immediate internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::movz(12, 1033, 32, true);
  uint32_t expected = 0b1'10100101'10'0000010000001001'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}