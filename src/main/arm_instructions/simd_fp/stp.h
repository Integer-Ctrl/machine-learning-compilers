#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_STP_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_STP_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {

    namespace internal
    {

      enum class stpSimdFpDataTypes : uint32_t
      {
        v32bit = 0b00,
        v64bit = 0b01,
        v128bit = 0b10
      };

      constexpr uint32_t _stpSimdFpPostPreOffset(const uint32_t opcode, const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn,
                                                 const int32_t imm7, const stpSimdFpDataTypes type)
      {
        release_assert((Rt1 & mask5) == Rt1, "Rt1 is only allowed to have a size of 5 bit.");
        release_assert((Rt2 & mask5) == Rt2, "Rt2 is only allowed to have a size of 5 bit.");
        release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");

        uint32_t immShift = 0;
        switch (type)
        {
        case stpSimdFpDataTypes::v32bit:
          immShift = 2;
          release_assert((imm7 & mask2) == 0, "imm7 should be a multiple of 4.");
          release_assert(imm7 >= -256, "imm7 minimum is -256.");
          release_assert(imm7 <= 252, "immm7 maximum is 252.");
          break;
        case stpSimdFpDataTypes::v64bit:
          immShift = 3;
          release_assert((imm7 & mask3) == 0, "imm7 should be a multiple of 8.");
          release_assert(imm7 >= -512, "imm7 minimum is -512.");
          release_assert(imm7 <= 504, "immm7 maximum is 504.");
          break;
        case stpSimdFpDataTypes::v128bit:
          immShift = 4;
          release_assert((imm7 & mask4) == 0, "imm7 should be a multiple of 16.");
          release_assert(imm7 >= -1024, "imm7 minimum is -1024.");
          release_assert(imm7 <= 1008, "immm7 maximum is 1008.");
          break;
        default:
          release_assert(false, "Undefined stp simd type found.");
          break;
        }

        uint32_t stp = 0;
        stp |= (static_cast<uint32_t>(type) & mask2) << 30;
        stp |= (opcode & mask8) << 22;
        stp |= ((imm7 >> immShift) & mask7) << 15;
        stp |= (Rt2 & mask5) << 10;
        stp |= (Rn & mask5) << 5;
        stp |= (Rt1 & mask5) << 0;
        return stp;
      }

      constexpr uint32_t stpPost(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
                                 const stpSimdFpDataTypes type)
      {
        return _stpSimdFpPostPreOffset(0b10110010, Rt1, Rt2, Rn, imm7, type);
      }

      constexpr uint32_t stpPre(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
                                const stpSimdFpDataTypes type)
      {
        return _stpSimdFpPostPreOffset(0b10110110, Rt1, Rt2, Rn, imm7, type);
      }

      constexpr uint32_t stpOffset(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
                                   const stpSimdFpDataTypes type)
      {
        return _stpSimdFpPostPreOffset(0b10110100, Rt1, Rt2, Rn, imm7, type);
      }

    }  // namespace internal

    constexpr uint32_t stpPost(const V32Bit St1, const V32Bit St2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpPost(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn), imm7,
                               internal::stpSimdFpDataTypes::v32bit);
    }

    constexpr uint32_t stpPost(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpPost(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn), imm7,
                               internal::stpSimdFpDataTypes::v64bit);
    }

    constexpr uint32_t stpPost(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpPost(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn), imm7,
                               internal::stpSimdFpDataTypes::v128bit);
    }

    constexpr uint32_t stpPre(const V32Bit St1, const V32Bit St2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpPre(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn), imm7,
                              internal::stpSimdFpDataTypes::v32bit);
    }

    constexpr uint32_t stpPre(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpPre(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn), imm7,
                              internal::stpSimdFpDataTypes::v64bit);
    }

    constexpr uint32_t stpPre(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpPre(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn), imm7,
                              internal::stpSimdFpDataTypes::v128bit);
    }

    constexpr uint32_t stp(const V32Bit St1, const V32Bit St2, const R64Bit Xn)
    {
      return internal::stpOffset(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn), 0,
                                 internal::stpSimdFpDataTypes::v32bit);
    }

    constexpr uint32_t stp(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn)
    {
      return internal::stpOffset(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn), 0,
                                 internal::stpSimdFpDataTypes::v64bit);
    }

    constexpr uint32_t stp(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn)
    {
      return internal::stpOffset(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn), 0,
                                 internal::stpSimdFpDataTypes::v128bit);
    }
    constexpr uint32_t stpOffset(const V32Bit St1, const V32Bit St2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpOffset(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn), imm7,
                                 internal::stpSimdFpDataTypes::v32bit);
    }

    constexpr uint32_t stpOffset(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpOffset(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn), imm7,
                                 internal::stpSimdFpDataTypes::v64bit);
    }

    constexpr uint32_t stpOffset(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn, const int32_t imm7)
    {
      return internal::stpOffset(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn), imm7,
                                 internal::stpSimdFpDataTypes::v128bit);
    }

  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_STP_H