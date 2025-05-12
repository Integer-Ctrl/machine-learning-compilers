#include "../../../main/arm_instructions/base/ldr.h"
#include <bitset>
#include <catch2/catch_test_macros.hpp>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test ldr immediate post 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = ldrPost(w25, x5, 89);
  uint32_t expected = 0b1'0'111000010'001011001'01'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldr immediate post 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = ldrPost(x25, x5, 89);
  uint32_t expected = 0b1'1'111000010'001011001'01'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldr immediate post internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ldrImmediatePost(25, 5, 89, true);
  uint32_t expected = 0b1'1'111000010'001011001'01'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test Ldr immediate pre 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = ldrPre(w25, x5, 89);
  uint32_t expected = 0b1'0'111000010'001011001'11'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldr immediate pre 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = ldrPre(x25, x5, 89);
  uint32_t expected = 0b1'1'111000010'001011001'11'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldr immediate pre internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ldrImmediatePre(25, 5, 89, true);
  uint32_t expected = 0b1'1'111000010'001011001'11'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test Ldr immediate unsigned offset 32bit instruction", "[codegen][32bit]")
{
  uint32_t value = ldrOffset(w25, x5, 128);
  uint32_t expected = 0b1'0'11100101'000000100000'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldr immediate unsigned offset 64bit instruction", "[codegen][64bit]")
{
  uint32_t value = ldrOffset(x25, x5, 256);
  uint32_t expected = 0b1'1'11100101'000000100000'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}

TEST_CASE("Test ldr immediate unsigned offset internal instruction", "[codegen][internal]")
{
  uint32_t value = internal::ldrImmediateOffset(25, 5, 256, true);
  uint32_t expected = 0b1'1'11100101'000000100000'00101'11001;

  INFO("value:    " << std::bitset<32>(value));
  INFO("expected: " << std::bitset<32>(expected));
  REQUIRE(value == expected);
}