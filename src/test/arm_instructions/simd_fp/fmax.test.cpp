#include "../../../main/arm_instructions/simd_fp/fmax.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test fmax (vector) two single-precision instruction", "[codegen][2s]")
{
  uint32_t value = fmax(v23, t2s, v19, t2s, v17, t2s);
  uint32_t expected = 0b00'0011100'0'1'10001'111101'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test fmax (vector) four single-precision instruction", "[codegen][4s]")
{
  uint32_t value = fmax(v23, t4s, v19, t4s, v17, t4s);
  uint32_t expected = 0b01'0011100'0'1'10001'111101'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test fmax (vector) two double-precision instruction", "[codegen][2d]")
{
  uint32_t value = fmax(v23, t2d, v19, t2d, v17, t2d);
  uint32_t expected = 0b01'0011100'1'1'10001'111101'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test fmax (scalar) two single-precision instruction", "[codegen][32bit]")
{
  uint32_t value = fmax(h23, h19, h17);
  uint32_t expected = 0b00011110'11'1'10001'010010'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test fmax (scalar) four double-precision instruction", "[codegen][64bit]")
{
  uint32_t value = fmax(s23, s19, s17);
  uint32_t expected = 0b00011110'00'1'10001'010010'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test fmax (scalar) two half-precision instruction", "[codegen][16bit]")
{
  uint32_t value = fmax(d23, d19, d17);
  uint32_t expected = 0b00011110'01'1'10001'010010'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}