#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/simd-fp/ldp.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test ldp post 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldpPost(s23, s19, x5, 12);
    uint32_t expected = 0b00'10110011'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp post 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldpPost(d23, d19, x5, -64);
    uint32_t expected = 0b01'10110011'1111000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp post 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = ldpPost(q23, q19, x5, -592);
    uint32_t expected = 0b10'10110011'1011011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp post internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::ldpPost(23, 19, 5, 12, internal::ldpSimdFpDataTypes::v32bit);
    uint32_t expected = 0b00'10110011'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp pre 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldpPre(s23, s19, x5, 12);
    uint32_t expected = 0b00'10110111'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp pre 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldpPre(d23, d19, x5, -64);
    uint32_t expected = 0b01'10110111'1111000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp pre 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = ldpPre(q23, q19, x5, -592);
    uint32_t expected = 0b10'10110111'1011011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp pre internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::ldpPre(23, 19, 5, 12, internal::ldpSimdFpDataTypes::v32bit);
    uint32_t expected = 0b00'10110111'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp plain 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldp(s23, s19, x5);
    uint32_t expected = 0b00'10110101'0000000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp plain 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldp(d23, d19, x5);
    uint32_t expected = 0b01'10110101'0000000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp plain 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = ldp(q23, q19, x5);
    uint32_t expected = 0b10'10110101'0000000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp offset 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldpOffset(s23, s19, x5, 12);
    uint32_t expected = 0b00'10110101'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp offset 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldpOffset(d23, d19, x5, -64);
    uint32_t expected = 0b01'10110101'1111000'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp offset 128bit instruction", "[codegen][128bit]")
{
    uint32_t value = ldpOffset(q23, q19, x5, -592);
    uint32_t expected = 0b10'10110101'1011011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldp offset internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::ldpOffset(23, 19, 5, 12, internal::ldpSimdFpDataTypes::v32bit);
    uint32_t expected = 0b00'10110101'0000011'10011'00101'10111;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}