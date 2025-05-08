#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/simd_fp/ldr.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test ldr post 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = ldrPost(b8, x13, 89);
    uint32_t expected = 0b00'111100'01'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr post 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = ldrPost(h8, x13, -33);
    uint32_t expected = 0b01'111100'01'0'111011111'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr post 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldrPost(s8, x13, 89);
    uint32_t expected = 0b10'111100'01'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr post 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldrPost(d8, x13, -33);
    uint32_t expected = 0b11'111100'01'0'111011111'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr post 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = ldrPost(q8, x13, 89);
    uint32_t expected = 0b00'111100'11'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr post internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::ldrSimdFpImmediatePost(8, 13, 89, internal::ldrSimdFpDataTypes::v32bit);
    uint32_t expected = 0b10'111100'01'0'001011001'01'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr pre 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = ldrPre(b8, x13, 89);
    uint32_t expected = 0b00'111100'01'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr pre 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = ldrPre(h8, x13, -33);
    uint32_t expected = 0b01'111100'01'0'111011111'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr pre 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldrPre(s8, x13, 89);
    uint32_t expected = 0b10'111100'01'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr pre 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldrPre(d8, x13, -33);
    uint32_t expected = 0b11'111100'01'0'111011111'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr pre 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = ldrPre(q8, x13, 89);
    uint32_t expected = 0b00'111100'11'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr pre internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::ldrSimdFpImmediatePre(8, 13, 89, internal::ldrSimdFpDataTypes::v32bit);
    uint32_t expected = 0b10'111100'01'0'001011001'11'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr plain 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = ldr(b8, x13);
    uint32_t expected = 0b00'111101'01'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr plain 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = ldr(h8, x13);
    uint32_t expected = 0b01'111101'01'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr plain 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldr(s8, x13);
    uint32_t expected = 0b10'111101'01'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr plain 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldr(d8, x13);
    uint32_t expected = 0b11'111101'01'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr plain 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = ldr(q8, x13);
    uint32_t expected = 0b00'111101'11'000000000000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr offset 8bit instruction", "[codegen][8Bit]")
{
    uint32_t value = ldrOffset(b8, x13, 89);
    uint32_t expected = 0b00'111101'01'000001011001'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr offset 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = ldrOffset(h8, x13, 54);
    uint32_t expected = 0b01'111101'01'000000011011'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr offset 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = ldrOffset(s8, x13, 64);
    uint32_t expected = 0b10'111101'01'000000010000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr offset 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = ldrOffset(d8, x13, 184);
    uint32_t expected = 0b11'111101'01'000000010111'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr offset 128bit instruction", "[codegen][128Bit]")
{
    uint32_t value = ldrOffset(q8, x13, 864);
    uint32_t expected = 0b00'111101'11'000000110110'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test ldr offset internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::ldrSimdFpImmediateOffset(8, 13, 64, internal::ldrSimdFpDataTypes::v32bit);
    uint32_t expected = 0b10'111101'01'000000010000'01101'01000;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}