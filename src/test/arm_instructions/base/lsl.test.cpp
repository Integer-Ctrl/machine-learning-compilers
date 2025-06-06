#include "../../../main/arm_instructions/base/lsl.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test lsl immediate 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = lsl(w25, w5, 20);
  uint32_t expected = 0b0'101001100'101100'001011'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test lsl immediate 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = lsl(x25, x5, 35);
  uint32_t expected = 0b1'101001101'011101'011100'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test lsl immediate internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::lslImmediate(25, 5, 35, true);
  uint32_t expected = 0b1'101001101'011101'011100'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test lsl immediate hex 0xd37ef463 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = lsl(x3, x3, 2);
  uint32_t expected = 0xd37ef463;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}