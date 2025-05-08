#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_CBNZ_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_CBNZ_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

constexpr uint32_t cbnz(const uint32_t Rt, const int32_t imm19, bool is64bit)
{
    release_assert((Rt & mask5) == Rt, "Rt is only allowed to have a size of 5 bit.");
    release_assert((imm19 & mask2) == 0, "imm19 should be multiple of 4");
    release_assert(imm19 <= (1024 * 1024), "imm19 has a maximum of 1MB (= 1048576)");
    release_assert(imm19 >= (-1024 * 1024), "imm19 has a minimum of -1MB (= -1048576)");

    uint32_t cbnz = 0;
    cbnz |= (is64bit & mask1) << 31;
    cbnz |= 0b0110101 << 24;
    cbnz |= ((imm19 >> 2) & mask19) << 5;
    cbnz |= (Rt & mask5) << 0;
    return cbnz;
}

} // namespace internal

constexpr uint32_t cbnz(const R32Bit Wt, const int32_t offset)
{
    internal::cbnz(static_cast<uint32_t>(Wt), offset, false);
}

constexpr uint32_t cbnz(const R64Bit Xt, const int32_t offset)
{
    internal::cbnz(static_cast<uint32_t>(Xt), offset, true);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_CBNZ_H