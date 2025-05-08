#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_LDR_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_LDR_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

constexpr uint32_t ldrImmediatePost(const uint32_t Rt, const uint32_t Rn, const int32_t imm9, const bool is64bit)
{
    // Build in Release mode:
    // compiles into 2 mov instructions when all inputs are fixed:
    // mov	w1, #38073                      // =0x94b9
    // movk	w1, #63557, lsl #16
    //
    // compiles into 3 instruction when one input is runtime known:
    // mov	w1, #38048                      // =0x94a0
    // movk	w1, #63557, lsl #16
    // bfxil	x1, x8, #0, #5

    release_assert(((Rt & mask5) == Rt), "Rt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");
    release_assert(imm9 <= 255, "imm9 has a Maximum of 255");
    release_assert(imm9 >= -256, "imm9 has a Minimum of -256");

    uint32_t ldr = 0;
    ldr |= 0b1 << 31; // size bit 31
    ldr |= (is64bit & mask1) << 30;
    ldr |= 0b111000010 << 21; // opc 29 - 21
    ldr |= (imm9 & mask9) << 12;
    ldr |= 0b01 << 10; // opc 11 - 10
    ldr |= (Rn & mask5) << 5;
    ldr |= (Rt & mask5) << 0;
    return ldr;
}

constexpr uint32_t ldrImmediatePre(const uint32_t Rt, const uint32_t Rn, const int32_t imm9, const bool is64bit)
{
    release_assert(((Rt & mask5) == Rt), "Rt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");
    release_assert(imm9 <= 255, "imm9 has a Maximum of 255");
    release_assert(imm9 >= -256, "imm9 has a Minimum of -256");

    uint32_t ldr = 0;
    ldr |= 0b1 << 31; // size bit 31
    ldr |= (is64bit & mask1) << 30;
    ldr |= 0b111000010 << 21; // opc 29 - 21
    ldr |= (imm9 & mask9) << 12;
    ldr |= 0b11 << 10; // opc 11 - 10
    ldr |= (Rn & mask5) << 5;
    ldr |= (Rt & mask5) << 0;
    return ldr;
}

constexpr uint32_t ldrImmediateOffset(const uint32_t Rt, const uint32_t Rn, const uint32_t imm12, const bool is64bit)
{
    release_assert(((Rt & mask5) == Rt), "Rt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");

    if (is64bit)
    {
        release_assert((((imm12 >> 3) & mask12) == (imm12 >> 3)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 16380.");
        release_assert(((imm12 & mask3) == 0), "imm12 should be multiple of 8.");
    }
    else
    {
        release_assert((((imm12 >> 2) & mask12) == (imm12 >> 2)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 32760.");
        release_assert(((imm12 & mask2) == 0), "imm12 should be multiple of 4.");
    }

    uint32_t immShift = is64bit ? 3 : 2; // <pimm>/8 or <pimm>/4
    uint32_t ldr = 0;
    ldr |= 0b1 << 31; // size bit 31
    ldr |= (is64bit & mask1) << 30;
    ldr |= 0b11100101 << 22; // opc 29 - 22
    ldr |= ((imm12 >> immShift) & mask12) << 10;
    ldr |= (Rn & mask5) << 5;
    ldr |= (Rt & mask5) << 0;
    return ldr;
}

} // namespace internal

constexpr uint32_t ldrPost(const R32Bit Wt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrImmediatePost(static_cast<uint32_t>(Wt), static_cast<uint32_t>(Xn), imm9, false);
}

constexpr uint32_t ldrPost(const R64Bit Xt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrImmediatePost(static_cast<uint32_t>(Xt), static_cast<uint32_t>(Xn), imm9, true);
}

constexpr uint32_t ldrPre(const R32Bit Wt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrImmediatePre(static_cast<uint32_t>(Wt), static_cast<uint32_t>(Xn), imm9, false);
}

constexpr uint32_t ldrPre(const R64Bit Xt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrImmediatePre(static_cast<uint32_t>(Xt), static_cast<uint32_t>(Xn), imm9, true);
}

constexpr uint32_t ldrOffset(const R32Bit Wt, const R64Bit Xn, const int32_t imm12)
{
    return internal::ldrImmediateOffset(static_cast<uint32_t>(Wt), static_cast<uint32_t>(Xn), imm12, false);
}

constexpr uint32_t ldrOffset(const R64Bit Xt, const R64Bit Xn, const int32_t imm12)
{
    return internal::ldrImmediateOffset(static_cast<uint32_t>(Xt), static_cast<uint32_t>(Xn), imm12, true);
}


} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_LDR_H