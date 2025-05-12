#include "../../../main/arm_instructions/base/orr.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test orr plain 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = orr(w25, w5, w3);
  uint32_t expected = 0b0'0101010'00'0'00011'000000'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test orr plain 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = orr(x25, x5, x3);
  uint32_t expected = 0b1'0101010'00'0'00011'000000'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test orr shifted register 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = orr(w25, w5, w3, LSL, 1);
  uint32_t expected = 0b0'0101010'00'0'00011'000001'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test orr shifted register 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = orr(x25, x5, x3, LSR, 1);
  uint32_t expected = 0b1'0101010'01'0'00011'000001'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test orr shifted register internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::orrShiftedRegister(25, 5, 3, internal::orrShiftType::LSR, 1, true);
  uint32_t expected = 0b1'0101010'01'0'00011'000001'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}