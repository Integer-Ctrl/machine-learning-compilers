#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/base/ret.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test ret instruction", "[codegen][64bit]")
{
    uint32_t value = ret();
    uint32_t expected = 0b1101011001011111000000'11110'00000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ret with register instruction", "[codegen][64bit]")
{
    uint32_t value = ret(x5);
    uint32_t expected = 0b1101011001011111000000'00101'00000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ret internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::ret(5);
    uint32_t expected = 0b1101011001011111000000'00101'00000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}