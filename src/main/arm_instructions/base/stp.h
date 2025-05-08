#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_STP_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_STP_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

constexpr uint32_t _stpPostPreOffset(const uint32_t opcode, const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn,
    const int32_t imm7, const bool is64bit)
{
    release_assert((Rt1 & mask5) == Rt1, "Rt1 is only allowed to have a size of 5 bit.");
    release_assert((Rt2 & mask5) == Rt2, "Rt2 is only allowed to have a size of 5 bit.");
    release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");

    if (is64bit)
    {
        release_assert((imm7 & mask2) == 0, "imm7 should be a multiple of 4.");
        release_assert(imm7 >= -256, "imm7 minimum is -256.");
        release_assert(imm7 <= 252, "immm7 maximum is 252.");
    }
    else
    {
        release_assert((imm7 & mask3) == 0, "imm7 should be a multiple of 8.");
        release_assert(imm7 >= -512, "imm7 minimum is -512.");
        release_assert(imm7 <= 504, "immm7 maximum is 504.");
    }

    uint32_t immShift = is64bit ? 3 : 2;
    uint32_t stp = 0;
    stp |= (is64bit & mask1) << 31;
    stp |= (opcode & mask9) << 22;
    stp |= ((imm7 >> immShift) & mask7) << 15;
    stp |= (Rt2 & mask5) << 10;
    stp |= (Rn & mask5) << 5;
    stp |= (Rt1 & mask5) << 0;
    return stp;
}

constexpr uint32_t stpPost(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
    const bool is64bit)
{
    return _stpPostPreOffset(0b010100010, Rt1, Rt2, Rn, imm7, is64bit);
}

constexpr uint32_t stpPre(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
    const bool is64bit)
{
    return _stpPostPreOffset(0b010100110, Rt1, Rt2, Rn, imm7, is64bit);
}

constexpr uint32_t stpOffset(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
    const bool is64bit)
{
    return _stpPostPreOffset(0b010100100, Rt1, Rt2, Rn, imm7, is64bit);
}

} // namespace internal

constexpr uint32_t stpPost(const R32Bit Wt1, const R32Bit Wt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::stpPost(static_cast<uint32_t>(Wt1), static_cast<uint32_t>(Wt2), static_cast<uint32_t>(Xn),
        imm7, false);
}

constexpr uint32_t stpPost(const R64Bit Xt1, const R64Bit Xt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::stpPost(static_cast<uint32_t>(Xt1), static_cast<uint32_t>(Xt2), static_cast<uint32_t>(Xn),
        imm7, true);
}

constexpr uint32_t stpPre(const R32Bit Wt1, const R32Bit Wt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::stpPre(static_cast<uint32_t>(Wt1), static_cast<uint32_t>(Wt2), static_cast<uint32_t>(Xn),
        imm7, false);
}

constexpr uint32_t stpPre(const R64Bit Xt1, const R64Bit Xt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::stpPre(static_cast<uint32_t>(Xt1), static_cast<uint32_t>(Xt2), static_cast<uint32_t>(Xn),
        imm7, true);
}

constexpr uint32_t stp(const R32Bit Wt1, const R32Bit Wt2, const R64Bit Xn)
{
    return internal::stpOffset(static_cast<uint32_t>(Wt1), static_cast<uint32_t>(Wt2), static_cast<uint32_t>(Xn),
        0, false);
}

constexpr uint32_t stp(const R64Bit Xt1, const R64Bit Xt2, const R64Bit Xn)
{
    return internal::stpOffset(static_cast<uint32_t>(Xt1), static_cast<uint32_t>(Xt2), static_cast<uint32_t>(Xn),
        0, true);
}

constexpr uint32_t stpOffset(const R32Bit Wt1, const R32Bit Wt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::stpOffset(static_cast<uint32_t>(Wt1), static_cast<uint32_t>(Wt2), static_cast<uint32_t>(Xn),
        imm7, false);
}

constexpr uint32_t stpOffset(const R64Bit Xt1, const R64Bit Xt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::stpOffset(static_cast<uint32_t>(Xt1), static_cast<uint32_t>(Xt2), static_cast<uint32_t>(Xn),
        imm7, true);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_STP_H