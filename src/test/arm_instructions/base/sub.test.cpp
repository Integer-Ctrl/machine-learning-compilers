#include <catch2/catch_test_macros.hpp>
#include "../../../main/arm_instructions/base/sub.h"
#include <bitset>

using namespace mini_jit::arm_instructions;

TEST_CASE("Test sub immediate 32bit instruction", "[codegen][32bit]")
{
    uint32_t value = sub(w25, w10, 1053);
    uint32_t expected = 0b0'10100010'0'010000011101'01010'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test sub immediate 64bit instruction", "[codegen][64bit]")
{
    uint32_t value = sub(x25, x10, 33);
    uint32_t expected = 0b1'10100010'0'000000100001'01010'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test sub immediate internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::subImmediate(25, 10, 33, internal::subShiftType::DEFAULT, true);
    uint32_t expected = 0b1'10100010'0'000000100001'01010'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test sub immediate shift 32bit instruction", "[codegen][32bit]")
{
    uint32_t value = sub(w25, w10, 1053, false);
    uint32_t expected = 0b0'10100010'0'010000011101'01010'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test sub immediate shift 64bit instruction", "[codegen][64bit]")
{
    uint32_t value = sub(x25, x10, 33, true);
    uint32_t expected = 0b1'10100010'1'000000100001'01010'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}

TEST_CASE("Test sub immediate shift internal instruction", "[codegen][internal]")
{
    uint32_t value = internal::subImmediate(25, 10, 33, internal::subShiftType::LSL12, true);
    uint32_t expected = 0b1'10100010'1'000000100001'01010'11001;

    INFO("value:    " << std::bitset<32>(value));
    INFO("expected: " << std::bitset<32>(expected));
    REQUIRE(value == expected);
}
