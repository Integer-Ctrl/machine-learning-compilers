#include "../../../main/arm_instructions/simd_fp/zip1.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test zip1 t8b instruction", "[codegen][t8b]")
{
  uint32_t value = zip1(v23, t8b, v19, t8b, v17, t8b);
  uint32_t expected = 0b00'001110'00'0'10001'001110'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test zip1 t16b instruction", "[codegen][t16b]")
{
  uint32_t value = zip1(v23, t16b, v19, t16b, v17, t16b);
  uint32_t expected = 0b01'001110'00'0'10001'001110'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test zip1 t4h instruction", "[codegen][t4h]")
{
  uint32_t value = zip1(v23, t4h, v19, t4h, v17, t4h);
  uint32_t expected = 0b00'001110'01'0'10001'001110'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test zip1 t8h instruction", "[codegen][t8h]")
{
  uint32_t value = zip1(v23, t8h, v19, t8h, v17, t8h);
  uint32_t expected = 0b01'001110'01'0'10001'001110'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test zip1 t2s instruction", "[codegen][t2s]")
{
  uint32_t value = zip1(v23, t2s, v19, t2s, v17, t2s);
  uint32_t expected = 0b00'001110'10'0'10001'001110'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test zip1 t4s instruction", "[codegen][t4s]")
{
  uint32_t value = zip1(v23, t4s, v19, t4s, v17, t4s);
  uint32_t expected = 0b01'001110'10'0'10001'001110'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test zip1 t2d instruction", "[codegen][t2d]")
{
  uint32_t value = zip1(v23, t2d, v19, t2d, v17, t2d);
  uint32_t expected = 0b01'001110'11'0'10001'001110'10011'10111;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}