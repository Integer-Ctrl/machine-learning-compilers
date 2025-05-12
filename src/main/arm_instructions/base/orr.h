#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_ORR_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_ORR_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {

    namespace internal
    {

      enum class orrShiftType : uint32_t
      {
        DEFAULT = 0b00,  // LSL
        LSL = 0b00,
        LSR = 0b01,
        ASR = 0b10,
        ROR = 0b11,
      };

      template <typename T> constexpr orrShiftType _orrParseShiftType()
      {
        static_assert(false, "Not a valid orr shift option.");
        return orrShiftType::DEFAULT;
      }
      template <> constexpr orrShiftType _orrParseShiftType<ShiftLSL>()
      {
        return orrShiftType::LSL;
      }
      template <> constexpr orrShiftType _orrParseShiftType<ShiftLSR>()
      {
        return orrShiftType::LSR;
      }
      template <> constexpr orrShiftType _orrParseShiftType<ShiftASR>()
      {
        return orrShiftType::ASR;
      }
      template <> constexpr orrShiftType _orrParseShiftType<ShiftROR>()
      {
        return orrShiftType::ROR;
      }

      constexpr uint32_t orrShiftedRegister(uint32_t Rd, uint32_t Rn, uint32_t Rm, orrShiftType shift, uint32_t imm6, bool is64bit)
      {
        release_assert((Rd & mask5) == Rd, "Rd is only allowed to have a size of 5 bit.");
        release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");
        release_assert((Rm & mask5) == Rm, "Rm is only allowed to have a size of 5 bit.");
        release_assert((static_cast<uint32_t>(shift) & mask2) == static_cast<uint32_t>(shift),
                       "Rm is only allowed to have a size of 5 bit.");

        if (is64bit)
        {
          release_assert(imm6 <= 63, "Shift amount should be less equal than 63, for the 64-bit variant.");
        }
        else
        {
          release_assert(imm6 <= 31, "Shift amount should be less equal than 31, for the 32-bit variant.");
        }

        uint32_t orr = 0;
        orr |= (is64bit & mask1) << 31;
        orr |= 0b0101010 << 24;
        orr |= (static_cast<uint32_t>(shift) & mask2) << 22;
        orr |= 0b0 << 21;
        orr |= (Rm & mask5) << 16;
        orr |= (imm6 & mask6) << 10;
        orr |= (Rn & mask5) << 5;
        orr |= (Rd & mask5) << 0;
        return orr;
      }

    }  // namespace internal

    constexpr uint32_t orr(const R32Bit Wd, const R32Bit Wn, const R32Bit Wm)
    {
      return internal::orrShiftedRegister(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), static_cast<uint32_t>(Wm),
                                          internal::orrShiftType::DEFAULT, 0, false);
    }

    constexpr uint32_t orr(const R64Bit Rd, const R64Bit Rn, const R64Bit Rm)
    {
      return internal::orrShiftedRegister(static_cast<uint32_t>(Rd), static_cast<uint32_t>(Rn), static_cast<uint32_t>(Rm),
                                          internal::orrShiftType::DEFAULT, 0, true);
    }

    template <typename T> constexpr uint32_t orr(const R32Bit Wd, const R32Bit Wn, const R32Bit Wm, const T, uint32_t amount)
    {
      internal::orrShiftType shift = internal::_orrParseShiftType<T>();
      return internal::orrShiftedRegister(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), static_cast<uint32_t>(Wm), shift, amount,
                                          false);
    }

    template <typename T> constexpr uint32_t orr(const R64Bit Rd, const R64Bit Rn, const R64Bit Rm, const T, uint32_t amount)
    {
      internal::orrShiftType shift = internal::_orrParseShiftType<T>();
      return internal::orrShiftedRegister(static_cast<uint32_t>(Rd), static_cast<uint32_t>(Rn), static_cast<uint32_t>(Rm), shift, amount,
                                          true);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_BASE_ORR_H