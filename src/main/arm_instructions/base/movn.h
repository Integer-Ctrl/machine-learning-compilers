#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_MOVN_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_MOVN_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {

    namespace internal
    {
      constexpr uint32_t movn(const uint32_t Rd, const uint32_t imm16, const uint32_t shift, bool is64bit)
      {
        release_assert((Rd & mask5) == Rd, "Rd is only allowed to have a size of 5 bit.");
        release_assert((imm16 & mask16) == imm16, "imm16 is only allowed to have a size of 16 bit i.e. 65535.");

        if (is64bit)
        {
          release_assert((shift == 0 || shift == 16 || shift == 32 || shift == 48), "shift is only allowed to be 0, 16, 32, 48");
        }
        else
        {
          release_assert((shift == 0 || shift == 16), "shift is only allowed to be 0 or 16.");
        }

        uint32_t movn = 0;
        movn |= (is64bit & mask1) << 31;
        movn |= 0b00100101 << 23;
        movn |= ((shift / 16) & mask2) << 21;
        movn |= (imm16 & mask16) << 5;
        movn |= (Rd & mask5) << 0;
        return movn;
      }

    }  // namespace internal

    constexpr uint32_t movn(const R32Bit Wd, const uint32_t imm)
    {
      return internal::movn(static_cast<uint32_t>(Wd), imm, 0, false);
    }

    constexpr uint32_t movn(const R64Bit Xd, const uint32_t imm)
    {
      return internal::movn(static_cast<uint32_t>(Xd), imm, 0, true);
    }

    constexpr uint32_t movn(const R32Bit Wd, const uint32_t imm, const uint32_t lslShift)
    {
      return internal::movn(static_cast<uint32_t>(Wd), imm, lslShift, false);
    }

    constexpr uint32_t movn(const R64Bit Xd, const uint32_t imm, const uint32_t lslShift)
    {
      return internal::movn(static_cast<uint32_t>(Xd), imm, lslShift, true);
    }

  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_BASE_MOVN_H