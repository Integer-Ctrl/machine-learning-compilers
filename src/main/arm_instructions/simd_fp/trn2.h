#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_TRN2_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_TRN2_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {
    namespace internal
    {
      enum class trn2SizeType : uint32_t
      {
        size00 = 0b00,
        size01 = 0b01,
        size10 = 0b10,
        size11 = 0b11
      };
      enum class trn2QType : uint32_t
      {
        q0 = 0b0,
        q1 = 0b1
      };

      constexpr uint32_t _trn2(const uint32_t Vd, const uint32_t Vn, const uint32_t Vm, const trn2SizeType size_type,
                               const trn2QType q_type)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        uint32_t trn2 = 0;
        trn2 |= 0b0 << 31;
        trn2 |= (static_cast<uint32_t>(q_type) & mask1) << 30;
        trn2 |= 0b001110'00'0 << 21;  // 0b001110ss0 s = size!
        trn2 |= (static_cast<uint32_t>(size_type) & mask2) << 22;
        trn2 |= (Vm & mask5) << 16;
        trn2 |= 0b011010 << 10;
        trn2 |= (Vn & mask5) << 5;
        trn2 |= (Vd & mask5) << 0;
        return trn2;
      }

    }  // namespace internal

    constexpr uint32_t trn2(const VGeneral Vd, const VType8x8Bit, const VGeneral Vn, const VType8x8Bit, const VGeneral Vm,
                            const VType8x8Bit)
    {
      return internal::_trn2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::trn2SizeType::size00, internal::trn2QType::q0);
    }

    constexpr uint32_t trn2(const VGeneral Vd, const VType16x8Bit, const VGeneral Vn, const VType16x8Bit, const VGeneral Vm,
                            const VType16x8Bit)
    {
      return internal::_trn2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::trn2SizeType::size00, internal::trn2QType::q1);
    }

    constexpr uint32_t trn2(const VGeneral Vd, const VType4x16Bit, const VGeneral Vn, const VType4x16Bit, const VGeneral Vm,
                            const VType4x16Bit)
    {
      return internal::_trn2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::trn2SizeType::size01, internal::trn2QType::q0);
    }

    constexpr uint32_t trn2(const VGeneral Vd, const VType8x16Bit, const VGeneral Vn, const VType8x16Bit, const VGeneral Vm,
                            const VType8x16Bit)
    {
      return internal::_trn2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::trn2SizeType::size01, internal::trn2QType::q1);
    }

    constexpr uint32_t trn2(const VGeneral Vd, const VType2x32Bit, const VGeneral Vn, const VType2x32Bit, const VGeneral Vm,
                            const VType2x32Bit)
    {
      return internal::_trn2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::trn2SizeType::size10, internal::trn2QType::q0);
    }

    constexpr uint32_t trn2(const VGeneral Vd, const VType4x32Bit, const VGeneral Vn, const VType4x32Bit, const VGeneral Vm,
                            const VType4x32Bit)
    {
      return internal::_trn2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::trn2SizeType::size10, internal::trn2QType::q1);
    }

    constexpr uint32_t trn2(const VGeneral Vd, const VType2x64Bit, const VGeneral Vn, const VType2x64Bit, const VGeneral Vm,
                            const VType2x64Bit)
    {
      return internal::_trn2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::trn2SizeType::size11, internal::trn2QType::q1);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_TRN2_H