#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_EOR_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_EOR_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {
    namespace internal
    {
      enum class eorSimdTypes : uint32_t
      {
        t8B = 0b0,
        t16b = 0b1
      };

      constexpr uint32_t eorVector(const uint32_t Vd, const uint32_t Vn, const uint32_t Vm, const eorSimdTypes type)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        uint32_t eor = 0;
        eor |= 0b0 << 31;
        eor |= (static_cast<uint32_t>(type) & mask1) << 30;
        eor |= 0b101110001 << 21;
        eor |= (Vm & mask5) << 16;
        eor |= 0b000111 << 10;
        eor |= (Vn & mask5) << 5;
        eor |= (Vd & mask5) << 0;
        return eor;
      }

    }  // namespace internal

    constexpr uint32_t eor(const VGeneral Vd, const VType16x8Bit, const VGeneral Vn, const VType16x8Bit, const VGeneral Vm,
                           const VType16x8Bit)
    {
      return internal::eorVector(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                 internal::eorSimdTypes::t16b);
    }

    constexpr uint32_t eor(const VGeneral Vd, const VType8x8Bit, const VGeneral Vn, const VType8x8Bit, const VGeneral Vm, const VType8x8Bit)
    {
      return internal::eorVector(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                 internal::eorSimdTypes::t8B);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_EOR_H