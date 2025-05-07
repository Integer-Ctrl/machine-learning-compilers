#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/base/cbnz.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test cbnz 32bit instruction", "[codegen][32bit]")
{
    uint32_t value = cbnz(w25, 20);
    uint32_t expected = 0b0'0110101'0000000000000010100'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test cbnz 64bit instruction", "[codegen][64bit]")
{
    uint32_t value = cbnz(x25, -35);
    uint32_t expected = 0b1'0110101'1111111111111011101'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test lsl internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::cbnz(25, -35, true);
    uint32_t expected = 0b1'0110101'1111111111111011101'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}