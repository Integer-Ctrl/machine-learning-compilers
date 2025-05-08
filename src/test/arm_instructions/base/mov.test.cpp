#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/base/mov.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test mov 32bit instruction", "[codegen][32bit]")
{
    uint32_t value = mov(w12, w4);
    uint32_t expected = 0b0'0101010000'00100'00000011111'01100;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test mov 64bit instruction", "[codegen][64bit]")
{
    uint32_t value = mov(x12, x4);
    uint32_t expected = 0b1'0101010000'00100'00000011111'01100;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test mov immediate 32bit instruction", "[codegen][32bit]")
{
    uint32_t value = mov(w12, 1033);
    uint32_t expected = 0b0'10100101'00'0000010000001001'01100;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test mov immediate 64bit instruction", "[codegen][64bit]")
{
    uint32_t value = mov(x12, 1033);
    uint32_t expected = 0b1'10100101'00'0000010000001001'01100;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}