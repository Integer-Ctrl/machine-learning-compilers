#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/simd-fp/stp.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test stp post 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = stpPost(s23, s19, x5, 12);
    uint32_t expected = 0b00'10110010'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp post 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = stpPost(d23, d19, x5, -64);
    uint32_t expected = 0b01'10110010'1111000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp post 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = stpPost(q23, q19, x5, -592);
    uint32_t expected = 0b10'10110010'1011011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp post internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::stpPost(23, 19, 5, 12, internal::stpSimdFpDataTypes::v32bit);
    uint32_t expected = 0b00'10110010'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp pre 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = stpPre(s23, s19, x5, 12);
    uint32_t expected = 0b00'10110110'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp pre 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = stpPre(d23, d19, x5, -64);
    uint32_t expected = 0b01'10110110'1111000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp pre 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = stpPre(q23, q19, x5, -592);
    uint32_t expected = 0b10'10110110'1011011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp pre internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::stpPre(23, 19, 5, 12, internal::stpSimdFpDataTypes::v32bit);
    uint32_t expected = 0b00'10110110'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp plain 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = stp(s23, s19, x5);
    uint32_t expected = 0b00'10110100'0000000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp plain 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = stp(d23, d19, x5);
    uint32_t expected = 0b01'10110100'0000000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp plain 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = stp(q23, q19, x5);
    uint32_t expected = 0b10'10110100'0000000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp offset 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = stpOffset(s23, s19, x5, 12);
    uint32_t expected = 0b00'10110100'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp offset 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = stpOffset(d23, d19, x5, -64);
    uint32_t expected = 0b01'10110100'1111000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp offset 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = stpOffset(q23, q19, x5, -592);
    uint32_t expected = 0b10'10110100'1011011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test stp offset internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::stpOffset(23, 19, 5, 12, internal::stpSimdFpDataTypes::v32bit);
    uint32_t expected = 0b00'10110100'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}