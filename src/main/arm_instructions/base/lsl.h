#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_LSL_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_LSL_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

constexpr uint32_t lslImmediate(const uint32_t Rd, const uint32_t Rn, const uint32_t shift, const bool is64bit)
{

    release_assert((Rd & mask5) == Rd, "Rd is only allowed to have a size of 5 bit.");
    release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");
    release_assert((shift & mask6) == shift, "shift is only allowed to have a size of 6 bit.");

    if (is64bit)
    {
        release_assert(shift <= 63, "Shift should be in range of 0 to 63, for the 64-bit variant.");
    }
    else
    {
        release_assert(shift <= 31, "Shift should be in range of 0 to 31, for the 32-bit variant.");
    }

    int32_t immrMod = is64bit ? 64 : 32;
    int32_t immsMod = is64bit ? 63 : 31;
    uint32_t lsl = 0;
    lsl |= (is64bit & mask1) << 31; // sf
    lsl |= 0b10100110 << 23;
    lsl |= (is64bit & mask1) << 22; // N
    lsl |= (((-static_cast<int32_t>(shift)) % immrMod) & mask6) << 16; // immr
    lsl |= (immsMod - static_cast<int32_t>(shift)) << 10;
    lsl |= (Rn & mask5) << 5;
    lsl |= (Rd & mask5) << 0;
    return lsl;
}

} // namespace internal

constexpr uint32_t lsl(const R32Bit Wd, const R32Bit Wn, const uint32_t shift)
{
    return internal::lslImmediate(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), shift, false);
}

constexpr uint32_t lsl(const R64Bit Xd, const R64Bit Xn, const uint32_t shift)
{
    return internal::lslImmediate(static_cast<uint32_t>(Xd), static_cast<uint32_t>(Xn), shift, true);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_LSL_H