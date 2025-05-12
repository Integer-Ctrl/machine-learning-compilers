#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/simd_fp/str.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test str post 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = strPost(b8, x13, 89);
    uint32_t expected = 0b00'111100'00'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str post 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = strPost(h8, x13, -33);
    uint32_t expected = 0b01'111100'00'0'111011111'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str post 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = strPost(s8, x13, 89);
    uint32_t expected = 0b10'111100'00'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str post 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = strPost(d8, x13, -33);
    uint32_t expected = 0b11'111100'00'0'111011111'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str post 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = strPost(q8, x13, 89);
    uint32_t expected = 0b00'111100'10'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str post internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::strSimdFpImmediatePost(8, 13, 89, internal::strSimdFpDataTypes::v32bit);
    uint32_t expected = 0b10'111100'00'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str pre 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = strPre(b8, x13, 89);
    uint32_t expected = 0b00'111100'00'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str pre 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = strPre(h8, x13, -33);
    uint32_t expected = 0b01'111100'00'0'111011111'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str pre 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = strPre(s8, x13, 89);
    uint32_t expected = 0b10'111100'00'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str pre 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = strPre(d8, x13, -33);
    uint32_t expected = 0b11'111100'00'0'111011111'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str pre 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = strPre(q8, x13, 89);
    uint32_t expected = 0b00'111100'10'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str pre internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::strSimdFpImmediatePre(8, 13, 89, internal::strSimdFpDataTypes::v32bit);
    uint32_t expected = 0b10'111100'00'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str plain 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = str(b8, x13);
    uint32_t expected = 0b00'111101'00'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str plain 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = str(h8, x13);
    uint32_t expected = 0b01'111101'00'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str plain 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = str(s8, x13);
    uint32_t expected = 0b10'111101'00'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str plain 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = str(d8, x13);
    uint32_t expected = 0b11'111101'00'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str plain 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = str(q8, x13);
    uint32_t expected = 0b00'111101'10'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str offset 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = strOffset(b8, x13, 89);
    uint32_t expected = 0b00'111101'00'000001011001'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str offset 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = strOffset(h8, x13, 54);
    uint32_t expected = 0b01'111101'00'000000011011'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str offset 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = strOffset(s8, x13, 64);
    uint32_t expected = 0b10'111101'00'000000010000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str offset 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = strOffset(d8, x13, 184);
    uint32_t expected = 0b11'111101'00'000000010111'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str offset 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = strOffset(q8, x13, 864);
    uint32_t expected = 0b00'111101'10'000000110110'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test str offset internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::strSimdFpImmediateOffset(8, 13, 64, internal::strSimdFpDataTypes::v32bit);
    uint32_t expected = 0b10'111101'00'000000010000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}