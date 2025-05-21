#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_FMAX_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_FMAX_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {
    namespace internal
    {
      enum class fmaxSzType : uint32_t
      {
        sz0 = 0b0,
        sz1 = 0b1
      };
      enum class fmaxQType : uint32_t
      {
        q0 = 0b0,
        q1 = 0b1
      };
      enum class fmaxFType : uint32_t
      {
        ftype00 = 0b00,
        ftype01 = 0b01,
        ftype11 = 0b11,
      };

      constexpr uint32_t fmaxVector(const uint32_t Vd, const uint32_t Vn, const uint32_t Vm, const fmaxSzType sz_type,
                                    const fmaxQType q_type)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        uint32_t fmax = 0;
        fmax |= 0b0 << 31;
        fmax |= (static_cast<uint32_t>(q_type) & mask1) << 30;
        fmax |= 0b001110001 << 21;  // 0011100x1 sz!
        fmax |= (static_cast<uint32_t>(sz_type) & mask1) << 22;
        fmax |= (Vm & mask5) << 16;
        fmax |= 0b111101 << 10;
        fmax |= (Vn & mask5) << 5;
        fmax |= (Vd & mask5) << 0;
        return fmax;
      }

      constexpr uint32_t fmaxScalar(const uint32_t Vd, const uint32_t Vn, const uint32_t Vm, const fmaxFType f_type)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        uint32_t fmax = 0;
        fmax |= 0b00011110001 << 21;  // 00011110ftype1 ftype (2 bits)!
        fmax |= (static_cast<uint32_t>(f_type) & mask2) << 22;
        fmax |= (Vm & mask5) << 16;
        fmax |= 0b010010 << 10;
        fmax |= (Vn & mask5) << 5;
        fmax |= (Vd & mask5) << 0;
        return fmax;
      }

    }  // namespace internal

    constexpr uint32_t fmax(const VGeneral Vd, const VType2x32Bit, const VGeneral Vn, const VType2x32Bit, const VGeneral Vm,
                            const VType2x32Bit)
    {
      return internal::fmaxVector(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                  internal::fmaxSzType::sz0, internal::fmaxQType::q0);
    }

    constexpr uint32_t fmax(const VGeneral Vd, const VType4x32Bit, const VGeneral Vn, const VType4x32Bit, const VGeneral Vm,
                            const VType4x32Bit)
    {
      return internal::fmaxVector(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                  internal::fmaxSzType::sz0, internal::fmaxQType::q1);
    }

    constexpr uint32_t fmax(const VGeneral Vd, const VType2x64Bit, const VGeneral Vn, const VType2x64Bit, const VGeneral Vm,
                            const VType2x64Bit)
    {
      return internal::fmaxVector(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                  internal::fmaxSzType::sz1, internal::fmaxQType::q1);
    }

    constexpr uint32_t fmax(const V16Bit Vd, const V16Bit Vn, const V16Bit Vm)
    {
      return internal::fmaxScalar(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                  internal::fmaxFType::ftype11);
    }

    constexpr uint32_t fmax(const V32Bit Vd, const V32Bit Vn, const V32Bit Vm)
    {
      return internal::fmaxScalar(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                  internal::fmaxFType::ftype00);
    }

    constexpr uint32_t fmax(const V64Bit Vd, const V64Bit Vn, const V64Bit Vm)
    {
      return internal::fmaxScalar(static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm),
                                  internal::fmaxFType::ftype01);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_FMAX_H