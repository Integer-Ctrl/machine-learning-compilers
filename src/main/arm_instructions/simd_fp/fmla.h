#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_FMLA_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_FMLA_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {

    namespace internal
    {

      enum class fmlaHalfPrecisionTypes
      {
        t4H,
        t8H
      };

      enum class fmlaSingleDoublePrecisionTypes
      {
        t2s,
        t4s,
        t2d
      };

      template <typename T> constexpr bool _fmlaIsDouble()
      {
        static_assert(false, "Not a valid type to check.");
        return false;
      }
      template <> constexpr bool _fmlaIsDouble<VType2x32Bit>()
      {
        return false;
      }
      template <> constexpr bool _fmlaIsDouble<VType4x32Bit>()
      {
        return false;
      }
      template <> constexpr bool _fmlaIsDouble<VType2x64Bit>()
      {
        return true;
      }

      template <typename T> constexpr fmlaSingleDoublePrecisionTypes _fmlaParseSingleDoubleType()
      {
        static_assert(false, "Not a valid single or double precision type.");
        return fmlaSingleDoublePrecisionTypes::t2s;
      }
      template <> constexpr fmlaSingleDoublePrecisionTypes _fmlaParseSingleDoubleType<VType2x32Bit>()
      {
        return fmlaSingleDoublePrecisionTypes::t2s;
      }
      template <> constexpr fmlaSingleDoublePrecisionTypes _fmlaParseSingleDoubleType<VType4x32Bit>()
      {
        return fmlaSingleDoublePrecisionTypes::t4s;
      }
      template <> constexpr fmlaSingleDoublePrecisionTypes _fmlaParseSingleDoubleType<VType2x64Bit>()
      {
        return fmlaSingleDoublePrecisionTypes::t2d;
      }

      constexpr uint32_t fmlaByElementScalarHalfPrecision(const uint32_t Hd, const uint32_t Hn, const uint32_t Vm, const uint32_t index)
      {
        release_assert((Hd & mask5) == Hd, "Hd is only allowed to have a size of 5 bit.");
        release_assert((Hn & mask5) == Hn, "Hn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask4) == Vm, "Vm is only allowed to have a size of 4 bit, restricting to range of V0 to V15.");
        release_assert(index <= 7, "Index should be less equal than 7.");

        // index in the H:L:M fields
        uint32_t fmla = 0;
        fmla |= 0b0101111100 << 22;
        fmla |= (index & mask2) << 20;  // LM
        fmla |= (Vm & mask4) << 16;
        fmla |= 0b0001 << 12;
        fmla |= ((index >> 2) & mask1) << 11;  // H
        fmla |= 0b0 << 10;
        fmla |= (Hn & mask5) << 5;
        fmla |= (Hd & mask5) << 0;
        return fmla;
      }

      constexpr uint32_t fmlaByElementScalarSingleDoublePrecision(const uint32_t Vd, const uint32_t Vn, const uint32_t Vm,
                                                                  const uint32_t index, bool isDoublePrecision)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        // index encoded in sz:L:H, but sz encoded the type i.e Single or Double
        uint32_t L = 0;
        uint32_t H = 0;
        if (isDoublePrecision)
        {
          // L = 1 RESERVED
          L = 0b0;
          H = index & mask1;
          release_assert(index <= 1, "Index should be less equal than 1, for double precision.");
        }
        else
        {
          L = (index & mask1);
          H = (index >> 1) & mask1;
          release_assert(index <= 3, "Index should be less equal than 3, for single precision.");
        }

        // index in the H:L:M fields
        uint32_t fmla = 0;
        fmla |= 0b010111111 << 23;
        fmla |= (isDoublePrecision & mask1) << 22;
        fmla |= (L & mask1) << 21;  // L
        fmla |= (Vm & mask5) << 16;
        fmla |= 0b0001 << 12;
        fmla |= (H & mask1) << 11;  // H
        fmla |= 0b0 << 10;
        fmla |= (Vn & mask5) << 5;
        fmla |= (Vd & mask5) << 0;
        return fmla;
      }

      constexpr uint32_t fmlaByElementVectorHalfPrecision(const fmlaHalfPrecisionTypes T, const uint32_t Vd, const uint32_t Vn,
                                                          const uint32_t Vm, const uint32_t index)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask4) == Vm, "Vm is only allowed to have a size of 4 bit, restricting to range of V0 to V15.");
        release_assert(index <= 7, "Index should be less equal than 7.");

        uint32_t Q = 0;
        switch (T)
        {
        case fmlaHalfPrecisionTypes::t4H:
          Q = 0b0;
          break;
        case fmlaHalfPrecisionTypes::t8H:
          Q = 0b1;
          break;
        default:
          release_assert(false, "Undefined fmla half precision type found.");
          break;
        }

        // index in the H:L:M fields
        uint32_t fmla = 0;
        fmla |= 0b0 << 31;
        fmla |= (Q & mask1) << 30;
        fmla |= 0b00111100 << 22;
        fmla |= (index & mask2) << 20;  // LM
        fmla |= (Vm & mask4) << 16;
        fmla |= 0b0001 << 12;
        fmla |= ((index >> 2) & mask1) << 11;  // H
        fmla |= 0b0 << 10;
        fmla |= (Vn & mask5) << 5;
        fmla |= (Vd & mask5) << 0;
        return fmla;
      }

      constexpr uint32_t fmlaByElementVectorSingleDoublePrecision(const fmlaSingleDoublePrecisionTypes T, const uint32_t Vd,
                                                                  const uint32_t Vn, const uint32_t Vm, const uint32_t index,
                                                                  bool isDoublePrecision)
      {
        release_assert((Vd & mask5) == Vd, "Vd is only allowed to have a size of 5 bit.");
        release_assert((Vn & mask5) == Vn, "Vn is only allowed to have a size of 5 bit.");
        release_assert((Vm & mask5) == Vm, "Vm is only allowed to have a size of 5 bit.");

        // index encoded in sz:L:H, but sz encoded the type i.e Single or Double
        uint32_t L = 0;
        uint32_t H = 0;
        uint32_t Q = 0;
        if (isDoublePrecision)
        {
          // L = 1 RESERVED
          L = 0b0;
          H = index & mask1;
          release_assert(index <= 1, "Index should be less equal than 1, for double precision.");

          switch (T)
          {
          case fmlaSingleDoublePrecisionTypes::t2d:
            Q = 0b1;
            break;
          case fmlaSingleDoublePrecisionTypes::t2s:
          case fmlaSingleDoublePrecisionTypes::t4s:
            release_assert(false, "2S or 4S is not a valid type for double precision.");
            break;
          default:
            release_assert(false, "Undefined fmla single/double precision type found.");
            break;
          }
        }
        else
        {
          L = (index & mask1);
          H = (index >> 1) & mask1;
          release_assert(index <= 3, "Index should be less equal than 3, for single precision.");

          switch (T)
          {
          case fmlaSingleDoublePrecisionTypes::t2s:
            Q = 0b0;
            break;
          case fmlaSingleDoublePrecisionTypes::t4s:
            Q = 0b1;
            break;
          case fmlaSingleDoublePrecisionTypes::t2d:
            release_assert(false, "2D is not a valid type for single precision:");
            break;
          default:
            release_assert(false, "Undefined fmla single/double precision type found.");
            break;
          }
        }

        // index in the H:L:M fields
        uint32_t fmla = 0;
        fmla |= 0b0 << 31;
        fmla |= (Q & mask1) << 30;
        fmla |= 0b0011111 << 23;
        fmla |= (isDoublePrecision & mask1) << 22;
        fmla |= (L & mask1) << 21;
        fmla |= (Vm & mask5) << 16;
        fmla |= 0b0001 << 12;
        fmla |= (H & mask1) << 11;
        fmla |= 0b0 << 10;
        fmla |= (Vn & mask5) << 5;
        fmla |= (Vd & mask5) << 0;
        return fmla;
      }

    }  // namespace internal

    constexpr uint32_t fmla(const V16Bit Hd, const V16Bit Hn, const VGeneral Vm, const uint32_t index)
    {
      return internal::fmlaByElementScalarHalfPrecision(static_cast<uint32_t>(Hd), static_cast<uint32_t>(Hn), static_cast<uint32_t>(Vm),
                                                        index);
    }

    constexpr uint32_t fmla(const V32Bit Sd, const V32Bit Sn, const VGeneral Vm, const uint32_t index)
    {
      return internal::fmlaByElementScalarSingleDoublePrecision(static_cast<uint32_t>(Sd), static_cast<uint32_t>(Sn),
                                                                static_cast<uint32_t>(Vm), index, false);
    }

    constexpr uint32_t fmla(const V64Bit Dd, const V64Bit Dn, const VGeneral Vm, const uint32_t index)
    {
      return internal::fmlaByElementScalarSingleDoublePrecision(static_cast<uint32_t>(Dd), static_cast<uint32_t>(Dn),
                                                                static_cast<uint32_t>(Vm), index, true);
    }

    template <typename T>
    constexpr uint32_t fmla(const VGeneral Vd, const T, const VGeneral Vn, const T, const VGeneral Vm, const uint32_t index)
    {
      internal::fmlaSingleDoublePrecisionTypes type = internal::_fmlaParseSingleDoubleType<T>();
      return internal::fmlaByElementVectorSingleDoublePrecision(type, static_cast<uint32_t>(Vd), static_cast<uint32_t>(Vn),
                                                                static_cast<uint32_t>(Vm), index, internal::_fmlaIsDouble<T>());
    }

    template <>
    constexpr uint32_t fmla<VType4x16Bit>(const VGeneral Vd, const VType4x16Bit, const VGeneral Vn, const VType4x16Bit, const VGeneral Vm,
                                          const uint32_t index)
    {
      return internal::fmlaByElementVectorHalfPrecision(internal::fmlaHalfPrecisionTypes::t4H, static_cast<uint32_t>(Vd),
                                                        static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm), index);
    }

    template <>
    constexpr uint32_t fmla<VType8x16Bit>(const VGeneral Vd, const VType8x16Bit, const VGeneral Vn, const VType8x16Bit, const VGeneral Vm,
                                          const uint32_t index)
    {
      return internal::fmlaByElementVectorHalfPrecision(internal::fmlaHalfPrecisionTypes::t8H, static_cast<uint32_t>(Vd),
                                                        static_cast<uint32_t>(Vn), static_cast<uint32_t>(Vm), index);
    }

  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_FMLA_H