#include "../../../main/arm_instructions/base/add.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test add plain 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = add(w25, w5, w3);
  uint32_t expected = 0b0'0001011'00'0'00011'000000'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add plain 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = add(x25, x5, x3);
  uint32_t expected = 0b1'0001011'00'0'00011'000000'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add shifted register 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = add(w25, w5, w3, LSL, 1);
  uint32_t expected = 0b0'0001011'00'0'00011'000001'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add shifted register 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = add(x25, x5, x3, LSR, 1);
  uint32_t expected = 0b1'0001011'01'0'00011'000001'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add shifted register internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::addShiftedRegister(25, 5, 3, internal::addShiftType::LSR, 1, true);
  uint32_t expected = 0b1'0001011'01'0'00011'000001'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add immediate 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = add(w25, w5, 2222);
  uint32_t expected = 0b0'00100010'0'100010101110'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add immediate 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = add(x25, x5, 2222);
  uint32_t expected = 0b1'00100010'0'100010101110'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add immediate shift 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = add(w25, w5, 2222, true);
  uint32_t expected = 0b0'00100010'1'100010101110'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add immediate shift 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = add(x25, x5, 2222, true);
  uint32_t expected = 0b1'00100010'1'100010101110'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test add immediate internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::addImmediate(25, 5, 2222, true, true);
  uint32_t expected = 0b1'00100010'1'100010101110'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}