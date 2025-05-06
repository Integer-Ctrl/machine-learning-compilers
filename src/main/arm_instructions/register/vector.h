#ifndef MINI_JIT_ARM_INSTRUCTIONS_VECTOR_H
#define MINI_JIT_ARM_INSTRUCTIONS_VECTOR_H

#include <cstdint>

namespace mini_jit {
namespace arm_instructions {

/// @brief Byte sized vector register B0 - B31
enum class V8Bit : uint32_t
{
    /// @brief 8 bit parameter/result register (caller-saved)
    b0 = 0,

    /// @brief 8 bit parameter/result register (caller-saved)
    b1 = 1,

    /// @brief 8 bit parameter/result register (caller-saved)
    b2 = 2,

    /// @brief 8 bit parameter/result register (caller-saved)
    b3 = 3,

    /// @brief 8 bit parameter/result register (caller-saved)
    b4 = 4,

    /// @brief 8 bit parameter/result register (caller-saved)
    b5 = 5,

    /// @brief 8 bit parameter/result register (caller-saved)
    b6 = 6,

    /// @brief 8 bit parameter/result register (caller-saved)
    b7 = 7,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b8 = 8,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b9 = 9,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b10 = 10,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b11 = 11,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b12 = 12,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b13 = 13,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b14 = 14,

    /// @brief 8 bit scratch register (callee-saved, lower 64 bits)
    b15 = 15,

    /// @brief 8 bit scratch register (caller-saved)
    b16 = 16,

    /// @brief 8 bit scratch register (caller-saved)
    b17 = 17,

    /// @brief 8 bit scratch register (caller-saved)
    b18 = 18,

    /// @brief 8 bit scratch register (caller-saved)
    b19 = 19,

    /// @brief 8 bit scratch register (caller-saved)
    b20 = 20,

    /// @brief 8 bit scratch register (caller-saved)
    b21 = 21,

    /// @brief 8 bit scratch register (caller-saved)
    b22 = 22,

    /// @brief 8 bit scratch register (caller-saved)
    b23 = 23,

    /// @brief 8 bit scratch register (caller-saved)
    b24 = 24,

    /// @brief 8 bit scratch register (caller-saved)
    b25 = 25,

    /// @brief 8 bit scratch register (caller-saved)
    b26 = 26,

    /// @brief 8 bit scratch register (caller-saved)
    b27 = 27,

    /// @brief 8 bit scratch register (caller-saved)
    b28 = 28,

    /// @brief 8 bit scratch register (caller-saved)
    b29 = 29,

    /// @brief 8 bit scratch register (caller-saved)
    b30 = 30,

    /// @brief 8 bit scratch register (caller-saved)
    b31 = 31,
};

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b0 = V8Bit::b0;

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b1 = V8Bit::b1;

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b2 = V8Bit::b2;

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b3 = V8Bit::b3;

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b4 = V8Bit::b4;

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b5 = V8Bit::b5;

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b6 = V8Bit::b6;

/// @brief 8 bit parameter/result register (caller-saved)
const V8Bit b7 = V8Bit::b7;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b8 = V8Bit::b8;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b9 = V8Bit::b9;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b10 = V8Bit::b10;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b11 = V8Bit::b11;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b12 = V8Bit::b12;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b13 = V8Bit::b13;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b14 = V8Bit::b14;

/// @brief 8 bit scratch register (callee-saved, lower 64 bit)
const V8Bit b15 = V8Bit::b15;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b16 = V8Bit::b16;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b17 = V8Bit::b17;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b18 = V8Bit::b18;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b19 = V8Bit::b19;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b20 = V8Bit::b20;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b21 = V8Bit::b21;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b22 = V8Bit::b22;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b23 = V8Bit::b23;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b24 = V8Bit::b24;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b25 = V8Bit::b25;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b26 = V8Bit::b26;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b27 = V8Bit::b27;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b28 = V8Bit::b28;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b29 = V8Bit::b29;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b30 = V8Bit::b30;

/// @brief 8 bit scratch register (caller-saved)
const V8Bit b31 = V8Bit::b31;

/// @brief Half word sized vector register H0 - H31
enum class V16Bit : uint32_t
{
    /// @brief 16 bit parameter/result register (caller-saved)
    h0 = 0,

    /// @brief 16 bit parameter/result register (caller-saved)
    h1 = 1,

    /// @brief 16 bit parameter/result register (caller-saved)
    h2 = 2,

    /// @brief 16 bit parameter/result register (caller-saved)
    h3 = 3,

    /// @brief 16 bit parameter/result register (caller-saved)
    h4 = 4,

    /// @brief 16 bit parameter/result register (caller-saved)
    h5 = 5,

    /// @brief 16 bit parameter/result register (caller-saved)
    h6 = 6,

    /// @brief 16 bit parameter/result register (caller-saved)
    h7 = 7,

    /// @brief 16 bit scratch register (caller-saved)
    h8 = 8,

    /// @brief 16 bit scratch register (callee-saved, lower 64 bits)
    h9 = 9,

    /// @brief 16 bit scratch register (callee-saved, lower 64 bits)
    h10 = 10,

    /// @brief 16 bit scratch register (callee-saved, lower 64 bits)
    h11 = 11,

    /// @brief 16 bit scratch register (callee-saved, lower 64 bits)
    h12 = 12,

    /// @brief 16 bit scratch register (callee-saved, lower 64 bits)
    h13 = 13,

    /// @brief 16 bit scratch register (callee-saved, lower 64 bits)
    h14 = 14,

    /// @brief 16 bit scratch register (callee-saved, lower 64 bits)
    h15 = 15,

    /// @brief 16 bit scratch register (caller-saved)
    h16 = 16,

    /// @brief 16 bit scratch register (caller-saved)
    h17 = 17,

    /// @brief 16 bit scratch register (caller-saved)
    h18 = 18,

    /// @brief 16 bit scratch register (caller-saved)
    h19 = 19,

    /// @brief 16 bit scratch register (caller-saved)
    h20 = 20,

    /// @brief 16 bit scratch register (caller-saved)
    h21 = 21,

    /// @brief 16 bit scratch register (caller-saved)
    h22 = 22,

    /// @brief 16 bit scratch register (caller-saved)
    h23 = 23,

    /// @brief 16 bit scratch register (caller-saved)
    h24 = 24,

    /// @brief 16 bit scratch register (caller-saved)
    h25 = 25,

    /// @brief 16 bit scratch register (caller-saved)
    h26 = 26,

    /// @brief 16 bit scratch register (caller-saved)
    h27 = 27,

    /// @brief 16 bit scratch register (caller-saved)
    h28 = 28,

    /// @brief 16 bit scratch register (caller-saved)
    h29 = 29,

    /// @brief 16 bit scratch register (caller-saved)
    h30 = 30,

    /// @brief 16 bit scratch register (caller-saved)
    h31 = 31,
};

// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h0 = V16Bit::h0;

/// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h1 = V16Bit::h1;

/// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h2 = V16Bit::h2;

/// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h3 = V16Bit::h3;

/// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h4 = V16Bit::h4;

/// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h5 = V16Bit::h5;

/// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h6 = V16Bit::h6;

/// @brief 16 bit parameter/result register (caller-saved)
const V16Bit h7 = V16Bit::h7;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h8 = V16Bit::h8;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h9 = V16Bit::h9;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h10 = V16Bit::h10;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h11 = V16Bit::h11;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h12 = V16Bit::h12;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h13 = V16Bit::h13;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h14 = V16Bit::h14;

/// @brief 16 bit scratch register (callee-saved, lower 64 bit)
const V16Bit h15 = V16Bit::h15;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h16 = V16Bit::h16;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h17 = V16Bit::h17;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h18 = V16Bit::h18;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h19 = V16Bit::h19;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h20 = V16Bit::h20;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h21 = V16Bit::h21;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h22 = V16Bit::h22;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h23 = V16Bit::h23;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h24 = V16Bit::h24;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h25 = V16Bit::h25;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h26 = V16Bit::h26;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h27 = V16Bit::h27;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h28 = V16Bit::h28;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h29 = V16Bit::h29;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h30 = V16Bit::h30;

/// @brief 16 bit scratch register (caller-saved)
const V16Bit h31 = V16Bit::h31;

/// @brief Word sized vector register S0 - S31
enum class V32Bit : uint32_t
{
    /// @brief 32 bit parameter/result register (caller-saved)
    s0 = 0,

    /// @brief 32 bit parameter/result register (caller-saved)
    s1 = 1,

    /// @brief 32 bit parameter/result register (caller-saved)
    s2 = 2,

    /// @brief 32 bit parameter/result register (caller-saved)
    s3 = 3,

    /// @brief 32 bit parameter/result register (caller-saved)
    s4 = 4,

    /// @brief 32 bit parameter/result register (caller-saved)
    s5 = 5,

    /// @brief 32 bit parameter/result register (caller-saved)
    s6 = 6,

    /// @brief 32 bit parameter/result register (caller-saved)
    s7 = 7,

    /// @brief 32 bit scratch register (caller-saved)
    s8 = 8,

    /// @brief 32 bit scratch register (callee-saved, lower 64 bits)
    s9 = 9,

    /// @brief 32 bit scratch register (callee-saved, lower 64 bits)
    s10 = 10,

    /// @brief 32 bit scratch register (callee-saved, lower 64 bits)
    s11 = 11,

    /// @brief 32 bit scratch register (callee-saved, lower 64 bits)
    s12 = 12,

    /// @brief 32 bit scratch register (callee-saved, lower 64 bits)
    s13 = 13,

    /// @brief 32 bit scratch register (callee-saved, lower 64 bits)
    s14 = 14,

    /// @brief 32 bit scratch register (callee-saved, lower 64 bits)
    s15 = 15,

    /// @brief 32 bit scratch register (caller-saved)
    s16 = 16,

    /// @brief 32 bit scratch register (caller-saved)
    s17 = 17,

    /// @brief 32 bit scratch register (caller-saved)
    s18 = 18,

    /// @brief 32 bit scratch register (caller-saved)
    s19 = 19,

    /// @brief 32 bit scratch register (caller-saved)
    s20 = 20,

    /// @brief 32 bit scratch register (caller-saved)
    s21 = 21,

    /// @brief 32 bit scratch register (caller-saved)
    s22 = 22,

    /// @brief 32 bit scratch register (caller-saved)
    s23 = 23,

    /// @brief 32 bit scratch register (caller-saved)
    s24 = 24,

    /// @brief 32 bit scratch register (caller-saved)
    s25 = 25,

    /// @brief 32 bit scratch register (caller-saved)
    s26 = 26,

    /// @brief 32 bit scratch register (caller-saved)
    s27 = 27,

    /// @brief 32 bit scratch register (caller-saved)
    s28 = 28,

    /// @brief 32 bit scratch register (caller-saved)
    s29 = 29,

    /// @brief 32 bit scratch register (caller-saved)
    s30 = 30,

    /// @brief 32 bit scratch register (caller-saved)
    s31 = 31,
};

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s0 = V32Bit::s0;

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s1 = V32Bit::s1;

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s2 = V32Bit::s2;

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s3 = V32Bit::s3;

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s4 = V32Bit::s4;

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s5 = V32Bit::s5;

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s6 = V32Bit::s6;

/// @brief 32 bit parameter/result register (caller-saved)
const V32Bit s7 = V32Bit::s7;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s8 = V32Bit::s8;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s9 = V32Bit::s9;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s10 = V32Bit::s10;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s11 = V32Bit::s11;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s12 = V32Bit::s12;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s13 = V32Bit::s13;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s14 = V32Bit::s14;

/// @brief 32 bit scratch register (callee-saved, lower 64 bit)
const V32Bit s15 = V32Bit::s15;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s16 = V32Bit::s16;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s17 = V32Bit::s17;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s18 = V32Bit::s18;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s19 = V32Bit::s19;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s20 = V32Bit::s20;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s21 = V32Bit::s21;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s22 = V32Bit::s22;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s23 = V32Bit::s23;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s24 = V32Bit::s24;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s25 = V32Bit::s25;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s26 = V32Bit::s26;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s27 = V32Bit::s27;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s28 = V32Bit::s28;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s29 = V32Bit::s29;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s30 = V32Bit::s30;

/// @brief 32 bit scratch register (caller-saved)
const V32Bit s31 = V32Bit::s31;

/// @brief Double word sized vector register D0 - D31
enum class V64Bit : uint32_t
{
    /// @brief 64 bit parameter/result register (caller-saved)
    d0 = 0,

    /// @brief 64 bit parameter/result register (caller-saved)
    d1 = 1,

    /// @brief 64 bit parameter/result register (caller-saved)
    d2 = 2,

    /// @brief 64 bit parameter/result register (caller-saved)
    d3 = 3,

    /// @brief 64 bit parameter/result register (caller-saved)
    d4 = 4,

    /// @brief 64 bit parameter/result register (caller-saved)
    d5 = 5,

    /// @brief 64 bit parameter/result register (caller-saved)
    d6 = 6,

    /// @brief 64 bit parameter/result register (caller-saved)
    d7 = 7,

    /// @brief 64 bit scratch register (caller-saved)
    d8 = 8,

    /// @brief 64 bit scratch register (callee-saved, lower 64 bits)
    d9 = 9,

    /// @brief 64 bit scratch register (callee-saved, lower 64 bits)
    d10 = 10,

    /// @brief 64 bit scratch register (callee-saved, lower 64 bits)
    d11 = 11,

    /// @brief 64 bit scratch register (callee-saved, lower 64 bits)
    d12 = 12,

    /// @brief 64 bit scratch register (callee-saved, lower 64 bits)
    d13 = 13,

    /// @brief 64 bit scratch register (callee-saved, lower 64 bits)
    d14 = 14,

    /// @brief 64 bit scratch register (callee-saved, lower 64 bits)
    d15 = 15,

    /// @brief 64 bit scratch register (caller-saved)
    d16 = 16,

    /// @brief 64 bit scratch register (caller-saved)
    d17 = 17,

    /// @brief 64 bit scratch register (caller-saved)
    d18 = 18,

    /// @brief 64 bit scratch register (caller-saved)
    d19 = 19,

    /// @brief 64 bit scratch register (caller-saved)
    d20 = 20,

    /// @brief 64 bit scratch register (caller-saved)
    d21 = 21,

    /// @brief 64 bit scratch register (caller-saved)
    d22 = 22,

    /// @brief 64 bit scratch register (caller-saved)
    d23 = 23,

    /// @brief 64 bit scratch register (caller-saved)
    d24 = 24,

    /// @brief 64 bit scratch register (caller-saved)
    d25 = 25,

    /// @brief 64 bit scratch register (caller-saved)
    d26 = 26,

    /// @brief 64 bit scratch register (caller-saved)
    d27 = 27,

    /// @brief 64 bit scratch register (caller-saved)
    d28 = 28,

    /// @brief 64 bit scratch register (caller-saved)
    d29 = 29,

    /// @brief 64 bit scratch register (caller-saved)
    d30 = 30,

    /// @brief 64 bit scratch register (caller-saved)
    d31 = 31,
};

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d0 = V64Bit::d0;

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d1 = V64Bit::d1;

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d2 = V64Bit::d2;

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d3 = V64Bit::d3;

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d4 = V64Bit::d4;

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d5 = V64Bit::d5;

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d6 = V64Bit::d6;

/// @brief 64 bit parameter/result register (caller-saved)
const V64Bit d7 = V64Bit::d7;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d8 = V64Bit::d8;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d9 = V64Bit::d9;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d10 = V64Bit::d10;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d11 = V64Bit::d11;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d12 = V64Bit::d12;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d13 = V64Bit::d13;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d14 = V64Bit::d14;

/// @brief 64 bit scratch register (callee-saved, lower 64 bit)
const V64Bit d15 = V64Bit::d15;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d16 = V64Bit::d16;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d17 = V64Bit::d17;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d18 = V64Bit::d18;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d19 = V64Bit::d19;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d20 = V64Bit::d20;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d21 = V64Bit::d21;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d22 = V64Bit::d22;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d23 = V64Bit::d23;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d24 = V64Bit::d24;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d25 = V64Bit::d25;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d26 = V64Bit::d26;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d27 = V64Bit::d27;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d28 = V64Bit::d28;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d29 = V64Bit::d29;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d30 = V64Bit::d30;

/// @brief 64 bit scratch register (caller-saved)
const V64Bit d31 = V64Bit::d31;

/// @brief Quad word sized vector register Q0 - Q31
enum class V128Bit : uint32_t
{
    /// @brief 128 bit parameter/result register (caller-saved)
    q0 = 0,

    /// @brief 128 bit parameter/result register (caller-saved)
    q1 = 1,

    /// @brief 128 bit parameter/result register (caller-saved)
    q2 = 2,

    /// @brief 128 bit parameter/result register (caller-saved)
    q3 = 3,

    /// @brief 128 bit parameter/result register (caller-saved)
    q4 = 4,

    /// @brief 128 bit parameter/result register (caller-saved)
    q5 = 5,

    /// @brief 128 bit parameter/result register (caller-saved)
    q6 = 6,

    /// @brief 128 bit parameter/result register (caller-saved)
    q7 = 7,

    /// @brief 128 bit scratch register (caller-saved)
    q8 = 8,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    q9 = 9,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    q10 = 10,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    q11 = 11,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    q12 = 12,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    q13 = 13,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    q14 = 14,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    q15 = 15,

    /// @brief 128 bit scratch register (caller-saved)
    q16 = 16,

    /// @brief 128 bit scratch register (caller-saved)
    q17 = 17,

    /// @brief 128 bit scratch register (caller-saved)
    q18 = 18,

    /// @brief 128 bit scratch register (caller-saved)
    q19 = 19,

    /// @brief 128 bit scratch register (caller-saved)
    q20 = 20,

    /// @brief 128 bit scratch register (caller-saved)
    q21 = 21,

    /// @brief 128 bit scratch register (caller-saved)
    q22 = 22,

    /// @brief 128 bit scratch register (caller-saved)
    q23 = 23,

    /// @brief 128 bit scratch register (caller-saved)
    q24 = 24,

    /// @brief 128 bit scratch register (caller-saved)
    q25 = 25,

    /// @brief 128 bit scratch register (caller-saved)
    q26 = 26,

    /// @brief 128 bit scratch register (caller-saved)
    q27 = 27,

    /// @brief 128 bit scratch register (caller-saved)
    q28 = 28,

    /// @brief 128 bit scratch register (caller-saved)
    q29 = 29,

    /// @brief 128 bit scratch register (caller-saved)
    q30 = 30,

    /// @brief 128 bit scratch register (caller-saved)
    q31 = 31,
};

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q0 = V128Bit::q0;

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q1 = V128Bit::q1;

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q2 = V128Bit::q2;

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q3 = V128Bit::q3;

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q4 = V128Bit::q4;

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q5 = V128Bit::q5;

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q6 = V128Bit::q6;

/// @brief 128 bit parameter/result register (caller-saved)
const V128Bit q7 = V128Bit::q7;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q8 = V128Bit::q8;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q9 = V128Bit::q9;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q10 = V128Bit::q10;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q11 = V128Bit::q11;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q12 = V128Bit::q12;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q13 = V128Bit::q13;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q14 = V128Bit::q14;

/// @brief 128 bit scratch register (callee-saved, lower 64 bit)
const V128Bit q15 = V128Bit::q15;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q16 = V128Bit::q16;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q17 = V128Bit::q17;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q18 = V128Bit::q18;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q19 = V128Bit::q19;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q20 = V128Bit::q20;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q21 = V128Bit::q21;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q22 = V128Bit::q22;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q23 = V128Bit::q23;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q24 = V128Bit::q24;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q25 = V128Bit::q25;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q26 = V128Bit::q26;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q27 = V128Bit::q27;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q28 = V128Bit::q28;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q29 = V128Bit::q29;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q30 = V128Bit::q30;

/// @brief 128 bit scratch register (caller-saved)
const V128Bit q31 = V128Bit::q31;

/// @brief General vector register of V0 - V31
enum class VGeneral
{
    /// @brief 128 bit parameter/result register (caller-saved)
    v0 = 0,

    /// @brief 128 bit parameter/result register (caller-saved)
    v1 = 1,

    /// @brief 128 bit parameter/result register (caller-saved)
    v2 = 2,

    /// @brief 128 bit parameter/result register (caller-saved)
    v3 = 3,

    /// @brief 128 bit parameter/result register (caller-saved)
    v4 = 4,

    /// @brief 128 bit parameter/result register (caller-saved)
    v5 = 5,

    /// @brief 128 bit parameter/result register (caller-saved)
    v6 = 6,

    /// @brief 128 bit parameter/result register (caller-saved)
    v7 = 7,

    /// @brief 128 bit scratch register (caller-saved)
    v8 = 8,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    v9 = 9,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    v10 = 10,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    v11 = 11,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    v12 = 12,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    v13 = 13,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    v14 = 14,

    /// @brief 128 bit scratch register (callee-saved, lower 128 bits)
    v15 = 15,

    /// @brief 128 bit scratch register (caller-saved)
    v16 = 16,

    /// @brief 128 bit scratch register (caller-saved)
    v17 = 17,

    /// @brief 128 bit scratch register (caller-saved)
    v18 = 18,

    /// @brief 128 bit scratch register (caller-saved)
    v19 = 19,

    /// @brief 128 bit scratch register (caller-saved)
    v20 = 20,

    /// @brief 128 bit scratch register (caller-saved)
    v21 = 21,

    /// @brief 128 bit scratch register (caller-saved)
    v22 = 22,

    /// @brief 128 bit scratch register (caller-saved)
    v23 = 23,

    /// @brief 128 bit scratch register (caller-saved)
    v24 = 24,

    /// @brief 128 bit scratch register (caller-saved)
    v25 = 25,

    /// @brief 128 bit scratch register (caller-saved)
    v26 = 26,

    /// @brief 128 bit scratch register (caller-saved)
    v27 = 27,

    /// @brief 128 bit scratch register (caller-saved)
    v28 = 28,

    /// @brief 128 bit scratch register (caller-saved)
    v29 = 29,

    /// @brief 128 bit scratch register (caller-saved)
    v30 = 30,

    /// @brief 128 bit scratch register (caller-saved)
    v31 = 31,
};

/// @brief general parameter/result register (caller-saved)
const VGeneral v0 = VGeneral::v0;

/// @brief general parameter/result register (caller-saved)
const VGeneral v1 = VGeneral::v1;

/// @brief general parameter/result register (caller-saved)
const VGeneral v2 = VGeneral::v2;

/// @brief general parameter/result register (caller-saved)
const VGeneral v3 = VGeneral::v3;

/// @brief general parameter/result register (caller-saved)
const VGeneral v4 = VGeneral::v4;

/// @brief general parameter/result register (caller-saved)
const VGeneral v5 = VGeneral::v5;

/// @brief general parameter/result register (caller-saved)
const VGeneral v6 = VGeneral::v6;

/// @brief general parameter/result register (caller-saved)
const VGeneral v7 = VGeneral::v7;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v8 = VGeneral::v8;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v9 = VGeneral::v9;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v10 = VGeneral::v10;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v11 = VGeneral::v11;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v12 = VGeneral::v12;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v13 = VGeneral::v13;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v14 = VGeneral::v14;

/// @brief general scratch register (callee-saved, lower 64 bit)
const VGeneral v15 = VGeneral::v15;

/// @brief general scratch register (caller-saved)
const VGeneral v16 = VGeneral::v16;

/// @brief general scratch register (caller-saved)
const VGeneral v17 = VGeneral::v17;

/// @brief general scratch register (caller-saved)
const VGeneral v18 = VGeneral::v18;

/// @brief general scratch register (caller-saved)
const VGeneral v19 = VGeneral::v19;

/// @brief general scratch register (caller-saved)
const VGeneral v20 = VGeneral::v20;

/// @brief general scratch register (caller-saved)
const VGeneral v21 = VGeneral::v21;

/// @brief general scratch register (caller-saved)
const VGeneral v22 = VGeneral::v22;

/// @brief general scratch register (caller-saved)
const VGeneral v23 = VGeneral::v23;

/// @brief general scratch register (caller-saved)
const VGeneral v24 = VGeneral::v24;

/// @brief general scratch register (caller-saved)
const VGeneral v25 = VGeneral::v25;

/// @brief general scratch register (caller-saved)
const VGeneral v26 = VGeneral::v26;

/// @brief general scratch register (caller-saved)
const VGeneral v27 = VGeneral::v27;

/// @brief general scratch register (caller-saved)
const VGeneral v28 = VGeneral::v28;

/// @brief general scratch register (caller-saved)
const VGeneral v29 = VGeneral::v29;

/// @brief general scratch register (caller-saved)
const VGeneral v30 = VGeneral::v30;

/// @brief general scratch register (caller-saved)
const VGeneral v31 = VGeneral::v31;

/// @brief Use 8 Byte sized vectors.
enum class VType8x8Bit : uint32_t
{
    /// @brief Use 8 Byte sized vectors.
    t8B,
};

/// @brief Use 16 Byte sized vectors.
enum class VType16x8Bit : uint32_t
{
    /// @brief Use 16 Byte sized vectors.
    t16B
};

/// @brief Use 8 Byte sized vectors.
const VType8x8Bit t8b = VType8x8Bit::t8B;

/// @brief Use 16 Byte sized vectors.
const VType16x8Bit t16b = VType16x8Bit::t16B;

/// @brief Use 4 half word (16 Bit) sized vectors.
enum class VType4x16Bit : uint32_t
{
    /// @brief Use 4 half word (16 Bit) sized vectors.
    t4H,
};

/// @brief Use 8 half word (16 Bit) sized vectors.
enum class VType8x16Bit : uint32_t
{
    /// @brief Use 8 half word (16 Bit) sized vectors.
    t8H
};

/// @brief Use 4 half word (16 Bit) sized vectors.
const VType4x16Bit t4h = VType4x16Bit::t4H;

/// @brief Use 8 half word (16 Bit) sized vectors.
const VType8x16Bit t8h = VType8x16Bit::t8H;

/// @brief Use 2 word (32 Bit) sized vectors.
enum class VType2x32Bit : uint32_t
{
    /// @brief Use 2 word (32 Bit) sized vectors.
    t2S,
};

/// @brief Use 4 word (32 Bit) sized vectors.
enum class VType4x32Bit : uint32_t
{
    /// @brief Use 4 word (32 Bit) sized vectors.
    t4S
};

/// @brief Use 2 word (32 Bit) sized vectors.
const VType2x32Bit t2s = VType2x32Bit::t2S;

/// @brief Use 4 word (32 Bit) sized vectors.
const VType4x32Bit t4s = VType4x32Bit::t4S;

/// @brief Use 1 double word (64 Bit) sized vector.
enum class VType1x64Bit : uint32_t
{
    /// @brief Use 1 double word (64 Bit) sized vector.
    t1D,
};

/// @brief Use 2 double word (64 Bit) sized vector.
enum class VType2x64Bit : uint32_t
{
    /// @brief Use 2 double word (64 Bit) sized vector.
    t2D
};

/// @brief Use 1 double word (64 Bit) sized vector.
const VType1x64Bit t1d = VType1x64Bit::t1D;

/// @brief Use 2 double word (64 Bit) sized vector.
const VType2x64Bit t2d = VType2x64Bit::t2D;

} // namespace arm_instructions
} // namespace mini_jit
#endif // MINI_JIT_ARM_INSTRUCTIONS_VECTOR_H