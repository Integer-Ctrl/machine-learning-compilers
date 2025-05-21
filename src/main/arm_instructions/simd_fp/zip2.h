#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ZIP2_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ZIP2_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {
    namespace internal
    {
      enum class zip2SizeType : uint32_t
      {
        size00 = 0b00,
        size01 = 0b01,
        size10 = 0b10,
        size11 = 0b11
      };
      enum class zip2QType : uint32_t
      {
        q0 = 0b0,
        q1 = 0b1
      };

      constexpr uint32_t _zip2(const uint32_t Vd, const uint32_t Vn, const uint32_t Vm, const zip2SizeType size_type,
                               const zip2QType q_type)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        uint32_t zip2 = 0;
        zip2 |= 0b0 << 31;
        zip2 |= (static_cast<uint32_t>(q_type) & mask1) << 30;
        zip2 |= 0b001110'00'0 << 21;  // 0b001110ss0 s = size!
        zip2 |= (static_cast<uint32_t>(size_type) & mask2) << 22;
        zip2 |= (Vm & mask5) << 16;
        zip2 |= 0b011110 << 10;
        zip2 |= (Vn & mask5) << 5;
        zip2 |= (Vd & mask5) << 0;
        return zip2;
      }

    }  // namespace internal

    constexpr uint32_t zip2(const VGeneral Vd, const VType8x8Bit, const VGeneral Vn, const VType8x8Bit, const VGeneral Vm,
                            const VType8x8Bit)
    {
      return internal::_zip2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip2SizeType::size00, internal::zip2QType::q0);
    }

    constexpr uint32_t zip2(const VGeneral Vd, const VType16x8Bit, const VGeneral Vn, const VType16x8Bit, const VGeneral Vm,
                            const VType16x8Bit)
    {
      return internal::_zip2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip2SizeType::size00, internal::zip2QType::q1);
    }

    constexpr uint32_t zip2(const VGeneral Vd, const VType4x16Bit, const VGeneral Vn, const VType4x16Bit, const VGeneral Vm,
                            const VType4x16Bit)
    {
      return internal::_zip2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip2SizeType::size01, internal::zip2QType::q0);
    }

    constexpr uint32_t zip2(const VGeneral Vd, const VType8x16Bit, const VGeneral Vn, const VType8x16Bit, const VGeneral Vm,
                            const VType8x16Bit)
    {
      return internal::_zip2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip2SizeType::size01, internal::zip2QType::q1);
    }

    constexpr uint32_t zip2(const VGeneral Vd, const VType2x32Bit, const VGeneral Vn, const VType2x32Bit, const VGeneral Vm,
                            const VType2x32Bit)
    {
      return internal::_zip2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip2SizeType::size10, internal::zip2QType::q0);
    }

    constexpr uint32_t zip2(const VGeneral Vd, const VType4x32Bit, const VGeneral Vn, const VType4x32Bit, const VGeneral Vm,
                            const VType4x32Bit)
    {
      return internal::_zip2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip2SizeType::size10, internal::zip2QType::q1);
    }

    constexpr uint32_t zip2(const VGeneral Vd, const VType2x64Bit, const VGeneral Vn, const VType2x64Bit, const VGeneral Vm,
                            const VType2x64Bit)
    {
      return internal::_zip2(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip2SizeType::size11, internal::zip2QType::q1);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ZIP2_H