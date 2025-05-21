#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ZIP1_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ZIP1_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {
    namespace internal
    {
      enum class zip1SizeType : uint32_t
      {
        size00 = 0b00,
        size01 = 0b01,
        size10 = 0b10,
        size11 = 0b11
      };
      enum class zip1QType : uint32_t
      {
        q0 = 0b0,
        q1 = 0b1
      };

      constexpr uint32_t _zip1(const uint32_t Vd, const uint32_t Vn, const uint32_t Vm, const zip1SizeType size_type,
                               const zip1QType q_type)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        uint32_t zip1 = 0;
        zip1 |= 0b0 << 31;
        zip1 |= (static_cast<uint32_t>(q_type) & mask1) << 30;
        zip1 |= 0b001110'00'0 << 21;  // 0b001110ss0 s = size!
        zip1 |= (static_cast<uint32_t>(size_type) & mask2) << 22;
        zip1 |= (Vm & mask5) << 16;
        zip1 |= 0b001110 << 10;
        zip1 |= (Vn & mask5) << 5;
        zip1 |= (Vd & mask5) << 0;
        return zip1;
      }

    }  // namespace internal

    constexpr uint32_t zip1(const VGeneral Vd, const VType8x8Bit, const VGeneral Vn, const VType8x8Bit, const VGeneral Vm,
                            const VType8x8Bit)
    {
      return internal::_zip1(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip1SizeType::size00, internal::zip1QType::q0);
    }

    constexpr uint32_t zip1(const VGeneral Vd, const VType16x8Bit, const VGeneral Vn, const VType16x8Bit, const VGeneral Vm,
                            const VType16x8Bit)
    {
      return internal::_zip1(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip1SizeType::size00, internal::zip1QType::q1);
    }

    constexpr uint32_t zip1(const VGeneral Vd, const VType4x16Bit, const VGeneral Vn, const VType4x16Bit, const VGeneral Vm,
                            const VType4x16Bit)
    {
      return internal::_zip1(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip1SizeType::size01, internal::zip1QType::q0);
    }

    constexpr uint32_t zip1(const VGeneral Vd, const VType8x16Bit, const VGeneral Vn, const VType8x16Bit, const VGeneral Vm,
                            const VType8x16Bit)
    {
      return internal::_zip1(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip1SizeType::size01, internal::zip1QType::q1);
    }

    constexpr uint32_t zip1(const VGeneral Vd, const VType2x32Bit, const VGeneral Vn, const VType2x32Bit, const VGeneral Vm,
                            const VType2x32Bit)
    {
      return internal::_zip1(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip1SizeType::size10, internal::zip1QType::q0);
    }

    constexpr uint32_t zip1(const VGeneral Vd, const VType4x32Bit, const VGeneral Vn, const VType4x32Bit, const VGeneral Vm,
                            const VType4x32Bit)
    {
      return internal::_zip1(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip1SizeType::size10, internal::zip1QType::q1);
    }

    constexpr uint32_t zip1(const VGeneral Vd, const VType2x64Bit, const VGeneral Vn, const VType2x64Bit, const VGeneral Vm,
                            const VType2x64Bit)
    {
      return internal::_zip1(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                             internal::zip1SizeType::size11, internal::zip1QType::q1);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ZIP1_H