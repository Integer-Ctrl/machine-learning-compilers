#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_RET_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_RET_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {

    namespace internal
    {

      constexpr uint32_t ret(const uint32_t Rn)
      {
        release_assert((Rn & mask5) == Rn, "Rn is only allowed ot have a size of 5 bit.");

        uint32_t ret = 0;
        ret |= 0b1101011001011111000000 << 10;
        ret |= (Rn & mask5) << 5;
        ret |= 0b00000 << 0;
        return ret;
      }

    }  // namespace internal

    constexpr uint32_t ret()
    {
      return internal::ret(static_cast<uint32_t>(R64Bit::lr));
    }

    constexpr uint32_t ret(const R64Bit Rn)
    {
      return internal::ret(static_cast<uint32_t>(Rn));
    }

  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_BASE_RET_H