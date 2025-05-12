#include "../../../main/arm_instructions/simd_fp/st1.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test st1 no offset one register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1(v5, t16b, x17);
  uint32_t expected = 0b0'1'00110000000000'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset one register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1(v5, t4h, x17);
  uint32_t expected = 0b0'0'00110000000000'0111'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset one register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1(v5, t4s, x17);
  uint32_t expected = 0b0'1'00110000000000'0111'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset one register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1(v5, t1d, x17);
  uint32_t expected = 0b0'0'00110000000000'0111'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset two register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1(v5, t16b, v6, t16b, x17);
  uint32_t expected = 0b0'1'00110000000000'1010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset two register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1(v5, t4h, v6, t4h, x17);
  uint32_t expected = 0b0'0'00110000000000'1010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset two register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1(v5, t4s, v6, t4s, x17);
  uint32_t expected = 0b0'1'00110000000000'1010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset two register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1(v5, t1d, v6, t1d, x17);
  uint32_t expected = 0b0'0'00110000000000'1010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset three register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1(v5, t16b, v6, t16b, v7, t16b, x17);
  uint32_t expected = 0b0'1'00110000000000'0110'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset three register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1(v5, t4h, v6, t4h, v7, t4h, x17);
  uint32_t expected = 0b0'0'00110000000000'0110'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset three register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1(v5, t4s, v6, t4s, v7, t4s, x17);
  uint32_t expected = 0b0'1'00110000000000'0110'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset three register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1(v5, t1d, v6, t1d, v7, t1d, x17);
  uint32_t expected = 0b0'0'00110000000000'0110'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset four register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1(v5, t16b, v6, t16b, v7, t16b, v8, t16b, x17);
  uint32_t expected = 0b0'1'00110000000000'0010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset four register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1(v5, t4h, v6, t4h, v7, t4h, v8, t4h, x17);
  uint32_t expected = 0b0'0'00110000000000'0010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset four register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x17);
  uint32_t expected = 0b0'1'00110000000000'0010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 no offset four register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1(v5, t1d, v6, t1d, v7, t1d, v8, t1d, x17);
  uint32_t expected = 0b0'0'00110000000000'0010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}
TEST_CASE("Test st1 no offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::st1MultipleStructures(5, internal::st1Types::t16b, 17, 1);
  uint32_t expected = 0b0'1'00110000000000'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset one register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, x17, 16);
  uint32_t expected = 0b0'1'001100100'11111'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset one register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, x17, 8);
  uint32_t expected = 0b0'0'001100100'11111'0111'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset one register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, x17, 16);
  uint32_t expected = 0b0'1'001100100'11111'0111'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset one register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, x17, 8);
  uint32_t expected = 0b0'0'001100100'11111'0111'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset two register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, v6, t16b, x17, 32);
  uint32_t expected = 0b0'1'001100100'11111'1010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset two register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, v6, t4h, x17, 16);
  uint32_t expected = 0b0'0'001100100'11111'1010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset two register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, v6, t4s, x17, 32);
  uint32_t expected = 0b0'1'001100100'11111'1010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset two register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, v6, t1d, x17, 16);
  uint32_t expected = 0b0'0'001100100'11111'1010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset three register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, v6, t16b, v7, t16b, x17, 48);
  uint32_t expected = 0b0'1'001100100'11111'0110'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset three register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, v6, t4h, v7, t4h, x17, 24);
  uint32_t expected = 0b0'0'001100100'11111'0110'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset three register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, v6, t4s, v7, t4s, x17, 48);
  uint32_t expected = 0b0'1'001100100'11111'0110'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset three register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, v6, t1d, v7, t1d, x17, 24);
  uint32_t expected = 0b0'0'001100100'11111'0110'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset four register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, v6, t16b, v7, t16b, v8, t16b, x17, 64);
  uint32_t expected = 0b0'1'001100100'11111'0010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset four register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, v6, t4h, v7, t4h, v8, t4h, x17, 32);
  uint32_t expected = 0b0'0'001100100'11111'0010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset four register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x17, 64);
  uint32_t expected = 0b0'1'001100100'11111'0010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 immediate offset four register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, v6, t1d, v7, t1d, v8, t1d, x17, 32);
  uint32_t expected = 0b0'0'001100100'11111'0010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}
TEST_CASE("Test st1 immediate offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::st1MultipleStructuresPost(5, internal::st1Types::t16b, 17, 16, internal::st1ImmediateRm, 1);
  uint32_t expected = 0b0'1'001100100'11111'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset one register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset one register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'0111'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset one register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'0111'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset one register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'0111'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset two register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, v6, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'1010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset two register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, v6, t4h, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'1010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset two register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, v6, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'1010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset two register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, v6, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'1010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset three register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, v6, t16b, v7, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'0110'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset three register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, v6, t4h, v7, t4h, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'0110'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset three register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, v6, t4s, v7, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'0110'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset three register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, v6, t1d, v7, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'0110'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset four register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = st1Post(v5, t16b, v6, t16b, v7, t16b, v8, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'0010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset four register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = st1Post(v5, t4h, v6, t4h, v7, t4h, v8, t4h, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'0010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset four register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = st1Post(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100100'11100'0010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test st1 register offset four register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = st1Post(v5, t1d, v6, t1d, v7, t1d, v8, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100100'11100'0010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}
TEST_CASE("Test st1 register offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::st1MultipleStructuresPost(5, internal::st1Types::t16b, 17, 0, 28, 1);
  uint32_t expected = 0b0'1'001100100'11100'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}