#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/simd-fp/fmla.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test fmla scalar 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = fmla(h3, h5, v7, 0);
    uint32_t expected = 0b0101111100'0'0'0111'0001'0'0'00101'00011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla scalar 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = fmla(s3, s5, v7, 0);
    uint32_t expected = 0b010111111'0'000111'0001'0'0'00101'00011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla scalar 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = fmla(d3, d5, v7, 0);
    uint32_t expected = 0b010111111'1'000111'0001'0'0'00101'00011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla scalar half precision internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::fmlaByElementScalarHalfPrecision(3, 5, 7, 0);
    uint32_t expected = 0b0101111100'0'0'0111'0001'0'0'00101'00011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla scalar singel/double precision internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::fmlaByElementScalarSingleDoublePrecision(3, 5, 7, 0, false);
    uint32_t expected = 0b010111111'0'000111'0001'0'0'00101'00011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla vector 16bit instruction", "[codegen][16Bit]")
{
    uint32_t value = fmla(v11, t4h, v10, t4h, v14, 5);
    uint32_t expected = 0b0'0'00111100'0'1'1110'0001'1'0'01010'01011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla vector 32bit instruction", "[codegen][32Bit]")
{
    uint32_t value = fmla(v11, t4s, v10, t4s, v20, 1);
    uint32_t expected = 0b0'1'0011111'0'1'10100'0001'0'0'01010'01011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla vector 64bit instruction", "[codegen][64Bit]")
{
    uint32_t value = fmla(v11, t2d, v10, t2d, v20, 1);
    uint32_t expected = 0b0'1'0011111'1'0'10100'0001'1'0'01010'01011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla vector half precision internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::fmlaByElementVectorHalfPrecision(internal::fmlaHalfPrecisionTypes::t4H, 11, 10, 14, 5);
    uint32_t expected = 0b0'0'00111100'0'1'1110'0001'1'0'01010'01011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test fmla vector singel/double precision internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::fmlaByElementVectorSingleDoublePrecision(internal::fmlaSingleDoublePrecisionTypes::t2d,
        11, 10, 20, 1, true);
    uint32_t expected = 0b0'1'0011111'1'0'10100'0001'1'0'01010'01011;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}