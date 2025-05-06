
#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/base/lsl.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test lsl immediate 32bit instruction", "[codegen][32bit]")
{
    uint32_t value = lsl(w25, x5, 5);
    uint32_t expected = 0b0'10100110'0'111011'011111'00101'11001;

    INFO("lsl:      " << std::bitset<32>(lsl));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(lsl == expected);
}

TEST_CASE("Test lsl immediate 64bit instruction", "[codegen][64bit]")
{
    uint32_t lsl = lsl(x25, x5, 25);
    uint32_t expected = 0b1'10100110'1'100111'111111'00101'11001;

    INFO("lsl:      " << std::bitset<32>(lsl));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(lsl == expected);
}

TEST_CASE("Test lsl immediate internal instruction", "[codegen][internal]")
{
    uint32_t lsl = internal::lslImmediate(25, 5, 35, true);
    uint32_t expected = 0b1'10100110'1'011101'111111'00101'11001;

    INFO("lsl:      " << std::bitset<32>(lsl));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(lsl == expected);
}