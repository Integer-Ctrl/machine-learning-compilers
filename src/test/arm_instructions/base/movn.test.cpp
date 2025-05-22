#include "../../../main/arm_instructions/base/movn.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test movn 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = movn(w12, 2);
  uint32_t expected = 0b0'00100101'00'0000000000000010'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movn 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = movn(x12, 2);
  uint32_t expected = 0b1'00100101'00'0000000000000010'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movn shift 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = movn(w12, 2, 16);
  uint32_t expected = 0b0'00100101'01'0000000000000010'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movn shift 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = movn(x12, 2, 32);
  uint32_t expected = 0b1'00100101'10'0000000000000010'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test movn immediate internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::movn(12, 2, 32, true);
  uint32_t expected = 0b1'00100101'10'0000000000000010'01100;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}