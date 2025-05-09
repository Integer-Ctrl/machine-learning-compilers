#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/base/madd.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test madd 32bit instruction", "[codegen][32bit]")
{
    uint32_t value = madd(w5, w12, w14, w1);
    uint32_t expected = 0b0'0011011000'01110'0'00001'01100'00101;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test madd 64bit instruction", "[codegen][64bit]")
{
    uint32_t value = madd(x5, x12, x14, x1);
    uint32_t expected = 0b1'0011011000'01110'0'00001'01100'00101;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}