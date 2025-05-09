#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_MADD_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_MADD_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit
{
namespace arm_instructions
{
namespace internal
{

constexpr uint32_t madd(const uint32_t Rd, const uint32_t Rn, const uint32_t Rm, const uint32_t Ra,
    const bool is64bit)
{
    release_assert((Rd & mask5) == Rd, "Rd is only allowed to have a size of 5 bit.");
    release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");
    release_assert((Rm & mask5) == Rm, "Rm is only allowed to have a size of 5 bit.");
    release_assert((Ra & mask5) == Ra, "Ra is only allowed to have a size of 5 bit.");

    uint32_t add = 0;
    add |= (is64bit & mask1) << 31;
    add |= 0b0011011000 << 21;
    add |= (Rm & mask5) << 16;
    add |= 0b0 << 15;
    add |= (Ra & mask5) << 10;
    add |= (Rn & mask5) << 5;
    add |= (Rd & mask5) << 0;
    return add;
}

} // namespace internal

constexpr uint32_t madd(const R32Bit Wd, const R32Bit Wn, const R32Bit Wm, const R32Bit Wa)
{
    return internal::madd(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), static_cast<uint32_t>(Wm),
        static_cast<uint32_t>(Wa), false);
}

constexpr uint32_t madd(const R64Bit Xd, const R64Bit Xn, const R64Bit Xm, const R64Bit Xa)
{
    return internal::madd(static_cast<uint32_t>(Xd), static_cast<uint32_t>(Xn), static_cast<uint32_t>(Xm),
        static_cast<uint32_t>(Xa), true);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_MADD_H