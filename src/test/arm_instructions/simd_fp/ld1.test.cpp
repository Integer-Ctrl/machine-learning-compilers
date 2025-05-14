#include "../../../main/arm_instructions/simd_fp/ld1.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test ld1 no offset one register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1(v5, t16b, x17);
  uint32_t expected = 0b0'1'00110001000000'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset one register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1(v5, t4h, x17);
  uint32_t expected = 0b0'0'00110001000000'0111'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset one register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1(v5, t4s, x17);
  uint32_t expected = 0b0'1'00110001000000'0111'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset one register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1(v5, t1d, x17);
  uint32_t expected = 0b0'0'00110001000000'0111'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset two register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1(v5, t16b, v6, t16b, x17);
  uint32_t expected = 0b0'1'00110001000000'1010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset two register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1(v5, t4h, v6, t4h, x17);
  uint32_t expected = 0b0'0'00110001000000'1010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset two register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1(v5, t4s, v6, t4s, x17);
  uint32_t expected = 0b0'1'00110001000000'1010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset two register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1(v5, t1d, v6, t1d, x17);
  uint32_t expected = 0b0'0'00110001000000'1010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset three register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1(v5, t16b, v6, t16b, v7, t16b, x17);
  uint32_t expected = 0b0'1'00110001000000'0110'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset three register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1(v5, t4h, v6, t4h, v7, t4h, x17);
  uint32_t expected = 0b0'0'00110001000000'0110'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset three register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1(v5, t4s, v6, t4s, v7, t4s, x17);
  uint32_t expected = 0b0'1'00110001000000'0110'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset three register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1(v5, t1d, v6, t1d, v7, t1d, x17);
  uint32_t expected = 0b0'0'00110001000000'0110'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset four register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1(v5, t16b, v6, t16b, v7, t16b, v8, t16b, x17);
  uint32_t expected = 0b0'1'00110001000000'0010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset four register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1(v5, t4h, v6, t4h, v7, t4h, v8, t4h, x17);
  uint32_t expected = 0b0'0'00110001000000'0010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset four register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x17);
  uint32_t expected = 0b0'1'00110001000000'0010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset four register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1(v5, t1d, v6, t1d, v7, t1d, v8, t1d, x17);
  uint32_t expected = 0b0'0'00110001000000'0010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}
TEST_CASE("Test ld1 no offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ld1MultipleStructures(5, internal::ld1Types::t16b, 17, 1);
  uint32_t expected = 0b0'1'00110001000000'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset one register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, x17, 16);
  uint32_t expected = 0b0'1'001100110'11111'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset one register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, x17, 8);
  uint32_t expected = 0b0'0'01100110'11111'0111'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset one register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, x17, 16);
  uint32_t expected = 0b0'1'001100110'11111'0111'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset one register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, x17, 8);
  uint32_t expected = 0b0'0'001100110'11111'0111'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset two register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, v6, t16b, x17, 32);
  uint32_t expected = 0b0'1'001100110'11111'1010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset two register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, v6, t4h, x17, 16);
  uint32_t expected = 0b0'0'001100110'11111'1010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset two register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, v6, t4s, x17, 32);
  uint32_t expected = 0b0'1'001100110'11111'1010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset two register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, v6, t1d, x17, 16);
  uint32_t expected = 0b0'0'001100110'11111'1010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset three register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, v6, t16b, v7, t16b, x17, 48);
  uint32_t expected = 0b0'1'001100110'11111'0110'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset three register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, v6, t4h, v7, t4h, x17, 24);
  uint32_t expected = 0b0'0'001100110'11111'0110'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset three register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, v6, t4s, v7, t4s, x17, 48);
  uint32_t expected = 0b0'1'001100110'11111'0110'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset three register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, v6, t1d, v7, t1d, x17, 24);
  uint32_t expected = 0b0'0'001100110'11111'0110'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset four register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, v6, t16b, v7, t16b, v8, t16b, x17, 64);
  uint32_t expected = 0b0'1'001100110'11111'0010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset four register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, v6, t4h, v7, t4h, v8, t4h, x17, 32);
  uint32_t expected = 0b0'0'001100110'11111'0010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset four register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x17, 64);
  uint32_t expected = 0b0'1'001100110'11111'0010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 immediate offset four register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, v6, t1d, v7, t1d, v8, t1d, x17, 32);
  uint32_t expected = 0b0'0'001100110'11111'0010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}
TEST_CASE("Test ld1 immediate offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ld1MultipleStructuresPost(5, internal::ld1Types::t16b, 17, 16, internal::ld1ImmediateRm, 1);
  uint32_t expected = 0b0'1'001100110'11111'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset one register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset one register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, x17, x28);
  uint32_t expected = 0b0'0'01100110'11100'0111'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset one register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'0111'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset one register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100110'11100'0111'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset two register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, v6, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'1010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset two register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, v6, t4h, x17, x28);
  uint32_t expected = 0b0'0'001100110'11100'1010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset two register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, v6, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'1010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset two register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, v6, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100110'11100'1010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset three register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, v6, t16b, v7, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'0110'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset three register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, v6, t4h, v7, t4h, x17, x28);
  uint32_t expected = 0b0'0'001100110'11100'0110'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset three register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, v6, t4s, v7, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'0110'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset three register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, v6, t1d, v7, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100110'11100'0110'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset four register 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(v5, t16b, v6, t16b, v7, t16b, v8, t16b, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'0010'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset four register 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(v5, t4h, v6, t4h, v7, t4h, v8, t4h, x17, x28);
  uint32_t expected = 0b0'0'001100110'11100'0010'01'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset four register 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(v5, t4s, v6, t4s, v7, t4s, v8, t4s, x17, x28);
  uint32_t expected = 0b0'1'001100110'11100'0010'10'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset four register 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(v5, t1d, v6, t1d, v7, t1d, v8, t1d, x17, x28);
  uint32_t expected = 0b0'0'001100110'11100'0010'11'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ld1MultipleStructuresPost(5, internal::ld1Types::t16b, 17, 0, 28, 1);
  uint32_t expected = 0b0'1'001100110'11100'0111'00'10001'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset single 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1(b5, 5, x8);
  uint32_t expected = 0b0'00110101000000'000'1'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset single 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1(h5, 2, x8);
  uint32_t expected = 0b0'00110101000000'010'1'00'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset single 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1(s5, 1, x8);
  uint32_t expected = 0b0'00110101000000'100'1'00'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 no offset single 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1(d5, 0, x8);
  uint32_t expected = 0b0'00110101000000'100'0'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 single internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ld1SingleStructures(5, internal::ld1DataTypes::v64bit, 0, 8);
  uint32_t expected = 0b0'00110101000000'100'0'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 offset single 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(b5, 5, x8, 1);
  uint32_t expected = 0b0'001101110'11111'000'1'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 offset single 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(h5, 2, x8, 2);
  uint32_t expected = 0b0'001101110'11111'010'1'00'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 offset single 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(s5, 1, x8, 4);
  uint32_t expected = 0b0'001101110'11111'100'1'00'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 offset single 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(d5, 0, x8, 8);
  uint32_t expected = 0b0'001101110'11111'100'0'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 single internal post imm instruction", "[codegen][internal]")
{
  uint32_t value = internal::ld1SingleStructuresPost(5, internal::ld1DataTypes::v64bit, 0, 8, 8, internal::ld1ImmediateRm);
  uint32_t expected = 0b0'001101110'11111'100'0'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register single 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = ld1Post(b5, 5, x8, x20);
  uint32_t expected = 0b0'001101110'10100'000'1'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register single 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = ld1Post(h5, 2, x8, x20);
  uint32_t expected = 0b0'001101110'10100'010'1'00'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register single 32bit instruction", "[codegen][32Bit]")
{
  uint32_t value = ld1Post(s5, 1, x8, x20);
  uint32_t expected = 0b0'001101110'10100'100'1'00'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 register single 64bit instruction", "[codegen][64Bit]")
{
  uint32_t value = ld1Post(d5, 0, x8, x20);
  uint32_t expected = 0b0'001101110'10100'100'0'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ld1 single internal post register instruction", "[codegen][internal]")
{
  uint32_t value = internal::ld1SingleStructuresPost(5, internal::ld1DataTypes::v64bit, 0, 8, 0, 20);
  uint32_t expected = 0b0'001101110'10100'100'0'01'01000'00101;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}