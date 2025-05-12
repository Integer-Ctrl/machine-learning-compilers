#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_SUB_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_SUB_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {
    namespace internal
    {

      enum class subShiftType : uint32_t
      {
        DEFAULT = 0b0,  // LSL0
        LSL0 = 0b0,
        LSL12 = 0b1,
      };

      constexpr uint32_t subImmediate(const uint32_t Rd, const uint32_t Rn, const uint32_t imm12, const subShiftType shift, bool is64bit)
      {
        release_assert((Rd & mask5) == Rd, "Rd is only allowed to have a size of 5 bit.");
        release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");
        release_assert((imm12 & mask12) == imm12, "imm12 is only allowed to have a size of 12 bit.");
        release_assert(imm12 <= 4096, "imm12 has maximum of 4096.");

        uint32_t sub = 0;
        sub |= (is64bit & mask1) << 31;
        sub |= 0b10100010 << 23;
        sub |= (static_cast<uint32_t>(shift) & mask1) << 22;
        sub |= (imm12 & mask12) << 10;
        sub |= (Rn & mask5) << 5;
        sub |= (Rd & mask5) << 0;
        return sub;
      }

    }  // namespace internal

    constexpr uint32_t sub(const R32Bit Wd, const R32Bit Wn, const uint32_t imm12)
    {
      return internal::subImmediate(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), imm12, internal::subShiftType::DEFAULT, false);
    }

    constexpr uint32_t sub(const R64Bit Xd, const R64Bit Xn, const uint32_t imm12)
    {
      return internal::subImmediate(static_cast<uint32_t>(Xd), static_cast<uint32_t>(Xn), imm12, internal::subShiftType::DEFAULT, true);
    }

    constexpr uint32_t sub(const R32Bit Wd, const R32Bit Wn, const uint32_t imm12, const bool leftShift12)
    {
      internal::subShiftType shift = leftShift12 ? internal::subShiftType::LSL12 : internal::subShiftType::LSL0;
      return internal::subImmediate(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), imm12, shift, false);
    }

    constexpr uint32_t sub(const R64Bit Xd, const R64Bit Xn, const uint32_t imm12, const bool leftShift12)
    {
      internal::subShiftType shift = leftShift12 ? internal::subShiftType::LSL12 : internal::subShiftType::LSL0;
      return internal::subImmediate(static_cast<uint32_t>(Xd), static_cast<uint32_t>(Xn), imm12, shift, true);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_BASE_SUB_H