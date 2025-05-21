#include "../../../main/arm_instructions/simd_fp/eor.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test eor 8bit instruction", "[codegen][8Bit]")
{
  uint32_t value = eor(v2, t8b, v10, t8b, v13, t8b);
  uint32_t expected = 0b0'0'101110001'01101'000111'01010'00010;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test eor 16bit instruction", "[codegen][16Bit]")
{
  uint32_t value = eor(v2, t16b, v10, t16b, v13, t16b);
  uint32_t expected = 0b0'1'101110001'01101'000111'01010'00010;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test eor vector internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::eorVector(2, 10, 13, internal::eorSimdTypes::t16b);
  uint32_t expected = 0b0'1'101110001'01101'000111'01010'00010;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}