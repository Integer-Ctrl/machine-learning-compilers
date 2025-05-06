#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/base/ldr.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test Ldr immediate post 32bit instruction", "[codegen][32bit]")
{
    uint32_t ldr = ldrPost(w25, x5, 89);
    uint32_t expected = 0b1'0'111000010'001011001'01'00101'11001;

    INFO("ldr:      " << std::bitset<32>(ldr));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(ldr == expected);
}

TEST_CASE("Test ldr immediate post 64bit instruction", "[codegen][64bit]")
{
    uint32_t ldr = ldrPost(x25, x5, 89);
    uint32_t expected = 0b1'1'111000010'001011001'01'00101'11001;

    INFO("ldr:      " << std::bitset<32>(ldr));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(ldr == expected);
}

TEST_CASE("Test ldr immediate post internal instruction", "[codegen][internal]")
{
    uint32_t ldr = internal::ldrImmediatePost(25, 5, 89, true);
    uint32_t expected = 0b1'1'111000010'001011001'01'00101'11001;

    INFO("ldr:      " << std::bitset<32>(ldr));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(ldr == expected);
}