#include "../../../main/arm_instructions/base/ldp.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test ldp immediate post 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = ldpPost(w25, w10, x5, 88);
  uint32_t expected = 0b0'010100011'0010110'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp immediate post 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = ldpPost(x25, x10, x5, 88);
  uint32_t expected = 0b1'010100011'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp immediate post internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ldpPost(25, 10, 5, 88, true);
  uint32_t expected = 0b1'010100011'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp immediate pre 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = ldpPre(w25, w10, x5, 88);
  uint32_t expected = 0b0'010100111'0010110'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp immediate pre 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = ldpPre(x25, x10, x5, 88);
  uint32_t expected = 0b1'010100111'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp immediate pre internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ldpPre(25, 10, 5, 88, true);
  uint32_t expected = 0b1'010100111'0001011'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp signed offset 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = ldpOffset(w25, w10, x5, 128);
  uint32_t expected = 0b0'010100101'0100000'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp signed offset 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = ldpOffset(x25, x10, x5, 252);
  uint32_t expected = 0b1'010100101'0011111'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldp signed offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ldpOffset(25, 10, 5, 252, true);
  uint32_t expected = 0b1'010100101'0011111'01010'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}