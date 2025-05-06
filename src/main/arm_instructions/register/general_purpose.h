#ifndef MINI_JIT_ARM_INSTRUCTIONS_GENERAL_PURPOSE_H
#define MINI_JIT_ARM_INSTRUCTIONS_GENERAL_PURPOSE_H

#include <cstdint>

namespace mini_jit {
namespace arm_instructions {

/// @brief 32 bit sized general purpose register
enum class R32Bit : uint32_t
{
    /// @brief 32 bit parameter/result register (caller-saved)
    w0 = 0,

    /// @brief 32 bit parameter/result register (caller-saved)
    w1 = 1,

    /// @brief 32 bit parameter/result register (caller-saved)
    w2 = 2,

    /// @brief 32 bit parameter/result register (caller-saved)
    w3 = 3,

    /// @brief 32 bit parameter/result register (caller-saved)
    w4 = 4,

    /// @brief 32 bit parameter/result register (caller-saved)
    w5 = 5,

    /// @brief 32 bit parameter/result register (caller-saved)
    w6 = 6,

    /// @brief 32 bit parameter/result register (caller-saved)
    w7 = 7,

    /// @brief 32 bit scratch register (caller-saved)
    w8 = 8,

    /// @brief 32 bit scratch register (caller-saved)
    w9 = 9,

    /// @brief 32 bit scratch register (caller-saved)
    w10 = 10,

    /// @brief 32 bit scratch register (caller-saved)
    w11 = 11,

    /// @brief 32 bit scratch register (caller-saved)
    w12 = 12,

    /// @brief 32 bit scratch register (caller-saved)
    w13 = 13,

    /// @brief 32 bit scratch register (caller-saved)
    w14 = 14,

    /// @brief 32 bit scratch register (caller-saved)
    w15 = 15,

    /// @brief 32 bit scratch register (caller-saved)
    w16 = 16,

    /// @brief 32 bit scratch register (caller-saved)
    w17 = 17,

    // w18 is platform register therefore not used

    /// @brief 32 bit scratch register (callee-saved)
    w19 = 19,

    /// @brief 32 bit scratch register (callee-saved)
    w20 = 20,

    /// @brief 32 bit scratch register (callee-saved)
    w21 = 21,

    /// @brief 32 bit scratch register (callee-saved)
    w22 = 22,

    /// @brief 32 bit scratch register (callee-saved)
    w23 = 23,

    /// @brief 32 bit scratch register (callee-saved)
    w24 = 24,

    /// @brief 32 bit scratch register (callee-saved)
    w25 = 25,

    /// @brief 32 bit scratch register (callee-saved)
    w26 = 26,

    /// @brief 32 bit scratch register (callee-saved)
    w27 = 27,

    /// @brief 32 bit scratch register (callee-saved)
    w28 = 28,

    /// @brief 32 bit scratch register (callee-saved)
    w29 = 29,

    /// @brief 32 bit scratch register (callee-saved)
    w30 = 30,

};

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w0 = R32Bit::w0;

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w1 = R32Bit::w1;

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w2 = R32Bit::w2;

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w3 = R32Bit::w3;

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w4 = R32Bit::w4;

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w5 = R32Bit::w5;

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w6 = R32Bit::w6;

/// @brief 32 bit parameter/result register (caller-saved)
const R32Bit w7 = R32Bit::w7;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w8 = R32Bit::w8;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w9 = R32Bit::w9;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w10 = R32Bit::w10;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w11 = R32Bit::w11;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w12 = R32Bit::w12;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w13 = R32Bit::w13;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w14 = R32Bit::w14;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w15 = R32Bit::w15;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w16 = R32Bit::w16;

/// @brief 32 bit scratch register (caller-saved)
const R32Bit w17 = R32Bit::w17;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w19 = R32Bit::w19;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w20 = R32Bit::w20;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w21 = R32Bit::w21;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w22 = R32Bit::w22;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w23 = R32Bit::w23;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w24 = R32Bit::w24;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w25 = R32Bit::w25;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w26 = R32Bit::w26;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w27 = R32Bit::w27;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w28 = R32Bit::w28;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w29 = R32Bit::w29;

/// @brief 32 bit scratch register (callee-saved)
const R32Bit w30 = R32Bit::w30;

/// @brief 64 bit sized general purpose register, including stack pointer
enum class R64Bit : uint32_t
{
    /// @brief 64 bit parameter/result register (caller-saved)
    x0 = 0,

    /// @brief 64 bit parameter/result register (caller-saved)
    x1 = 1,

    /// @brief 64 bit parameter/result register (caller-saved)
    x2 = 2,

    /// @brief 64 bit parameter/result register (caller-saved)
    x3 = 3,

    /// @brief 64 bit parameter/result register (caller-saved)
    x4 = 4,

    /// @brief 64 bit parameter/result register (caller-saved)
    x5 = 5,

    /// @brief 64 bit parameter/result register (caller-saved)
    x6 = 6,

    /// @brief 64 bit parameter/result register (caller-saved)
    x7 = 7,

    /// @brief 64 bit scratch register (caller-saved)
    x8 = 8,

    /// @brief 64 bit scratch register (caller-saved)
    x9 = 9,

    /// @brief 64 bit scratch register (caller-saved)
    x10 = 10,

    /// @brief 64 bit scratch register (caller-saved)
    x11 = 11,

    /// @brief 64 bit scratch register (caller-saved)
    x12 = 12,

    /// @brief 64 bit scratch register (caller-saved)
    x13 = 13,

    /// @brief 64 bit scratch register (caller-saved)
    x14 = 14,

    /// @brief 64 bit scratch register (caller-saved)
    x15 = 15,

    /// @brief 64 bit scratch register (caller-saved)
    x16 = 16,

    /// @brief 64 bit scratch register (caller-saved)
    x17 = 17,

    // x18 is platform register therefore not used

    /// @brief 64 bit scratch register (callee-saved)
    x19 = 19,

    /// @brief 64 bit scratch register (callee-saved)
    x20 = 20,

    /// @brief 64 bit scratch register (callee-saved)
    x21 = 21,

    /// @brief 64 bit scratch register (callee-saved)
    x22 = 22,

    /// @brief 64 bit scratch register (callee-saved)
    x23 = 23,

    /// @brief 64 bit scratch register (callee-saved)
    x24 = 24,

    /// @brief 64 bit scratch register (callee-saved)
    x25 = 25,

    /// @brief 64 bit scratch register (callee-saved)
    x26 = 26,

    /// @brief 64 bit scratch register (callee-saved)
    x27 = 27,

    /// @brief 64 bit scratch register (callee-saved)
    x28 = 28,

    /// @brief frame pointer register (callee-saved)
    x29 = 29,

    /// @brief link register (callee-saved)
    x30 = 30,

    /// @brief frame pointer register (callee-saved)
    fp = 29,

    /// @brief link register (callee-saved)
    lr = 30,

    /// @brief stack pointer register
    sp = 31,
};

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x0 = R64Bit::x0;

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x1 = R64Bit::x1;

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x2 = R64Bit::x2;

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x3 = R64Bit::x3;

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x4 = R64Bit::x4;

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x5 = R64Bit::x5;

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x6 = R64Bit::x6;

/// @brief 64 bit parameter/result register (caller-saved)
const R64Bit x7 = R64Bit::x7;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x8 = R64Bit::x8;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x9 = R64Bit::x9;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x10 = R64Bit::x10;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x11 = R64Bit::x11;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x12 = R64Bit::x12;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x13 = R64Bit::x13;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x14 = R64Bit::x14;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x15 = R64Bit::x15;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x16 = R64Bit::x16;

/// @brief 64 bit scratch register (caller-saved)
const R64Bit x17 = R64Bit::x17;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x19 = R64Bit::x19;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x20 = R64Bit::x20;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x21 = R64Bit::x21;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x22 = R64Bit::x22;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x23 = R64Bit::x23;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x24 = R64Bit::x24;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x25 = R64Bit::x25;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x26 = R64Bit::x26;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x27 = R64Bit::x27;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x28 = R64Bit::x28;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x29 = R64Bit::x29;

/// @brief 64 bit scratch register (callee-saved)
const R64Bit x30 = R64Bit::x30;

/// @brief frame pointer register (callee-saved)
const R64Bit fp = R64Bit::fp;

/// @brief link register (callee-saved)
const R64Bit lr = R64Bit::lr;

/// @brief stack pointer register
const R64Bit sp = R64Bit::sp;

/// @brief Represents the Logical Shift Left option
enum class ShiftLSL
{
    /// @brief Represents the Logical Shift Left option
    LSL
};

/// @brief Logical Shift Left
const ShiftLSL LSL = ShiftLSL::LSL;

/// @brief Represents the Logical Shift Right option
enum class ShiftLSR
{
    /// @brief Represents the Logical Shift Right option
    LSR
};

/// @brief Logical Shift Right
const ShiftLSR LSR = ShiftLSR::LSR;

/// @brief Represents the Arithmetic Shift Right option
enum class ShiftASR
{
    /// @brief Represents the Arithmetic Shift Right option
    ASR
};

/// @brief Arithmetic Shift Right
const ShiftASR ASR = ShiftASR::ASR;

/// @brief Represents the ROtate Right option
enum class ShiftROR
{
    /// @brief Represents the ROtate Right option
    ROR
};

/// @brief ROtate Right
const ShiftROR ROR = ShiftROR::ROR;


} // namespace arm_instructions
} // namespace mini_jit
#endif // MINI_JIT_ARM_INSTRUCTIONS_GENERAL_PURPOSE_H