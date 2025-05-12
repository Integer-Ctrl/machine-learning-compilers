#include "../../../main/arm_instructions/base/stp.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test stp immediate post 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = stpPost(w25, w10, x5, 88);
  uint32_t expected = 0b0'010100010'0010110'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp immediate post 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = stpPost(x25, x10, x5, 88);
  uint32_t expected = 0b1'010100010'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp immediate post internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::stpPost(25, 10, 5, 88, true);
  uint32_t expected = 0b1'010100010'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp immediate pre 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = stpPre(w25, w10, x5, 88);
  uint32_t expected = 0b0'010100110'0010110'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp immediate pre 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = stpPre(x25, x10, x5, 88);
  uint32_t expected = 0b1'010100110'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp immediate pre internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::stpPre(25, 10, 5, 88, true);
  uint32_t expected = 0b1'010100110'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp signed offset 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = stpOffset(w25, w10, x5, 128);
  uint32_t expected = 0b0'010100100'0100000'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp signed offset 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = stpOffset(x25, x10, x5, 252);
  uint32_t expected = 0b1'010100100'0011111'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test stp signed offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::stpOffset(25, 10, 5, 252, true);
  uint32_t expected = 0b1'010100100'0011111'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}