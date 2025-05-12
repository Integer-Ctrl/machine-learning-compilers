#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ST1_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ST1_H

#include "../../release_assert.h"
#include "../register.h"
#include <cstdint>

namespace mini_jit
{
  namespace arm_instructions
  {

    namespace internal
    {

      const uint32_t st1ImmediateRm = 0b11111;

      enum class st1Types
      {
        t8b,
        t16b,
        t4h,
        t8h,
        t2s,
        t4s,
        t1d,
        t2d
      };

      template <typename T> constexpr st1Types _st1ParseType()
      {
        static_assert(false, "Not a valid st1 vector type.");
        return st1Types::t8b;
      }
      template <> constexpr st1Types _st1ParseType<VType8x8Bit>()
      {
        return st1Types::t8b;
      }
      template <> constexpr st1Types _st1ParseType<VType16x8Bit>()
      {
        return st1Types::t16b;
      }
      template <> constexpr st1Types _st1ParseType<VType4x16Bit>()
      {
        return st1Types::t4h;
      }
      template <> constexpr st1Types _st1ParseType<VType8x16Bit>()
      {
        return st1Types::t8h;
      }
      template <> constexpr st1Types _st1ParseType<VType2x32Bit>()
      {
        return st1Types::t2s;
      }
      template <> constexpr st1Types _st1ParseType<VType4x32Bit>()
      {
        return st1Types::t4s;
      }
      template <> constexpr st1Types _st1ParseType<VType1x64Bit>()
      {
        return st1Types::t1d;
      }
      template <> constexpr st1Types _st1ParseType<VType2x64Bit>()
      {
        return st1Types::t2d;
      }

      constexpr void _st1GetQAndSize(const st1Types type, uint32_t &out_q, uint32_t &out_size)
      {
        switch (type)
        {
        case st1Types::t8b:
          out_q = 0b0;
          out_size = 0b00;
          break;
        case st1Types::t16b:
          out_q = 0b1;
          out_size = 0b00;
          break;
        case st1Types::t4h:
          out_q = 0b0;
          out_size = 0b01;
          break;
        case st1Types::t8h:
          out_q = 0b1;
          out_size = 0b01;
          break;
        case st1Types::t2s:
          out_q = 0b0;
          out_size = 0b10;
          break;
        case st1Types::t4s:
          out_q = 0b1;
          out_size = 0b10;
          break;
        case st1Types::t1d:
          out_q = 0b0;
          out_size = 0b11;
          break;
        case st1Types::t2d:
          out_q = 0b1;
          out_size = 0b11;
          break;
        default:
          release_assert(false, "Undefined st1 type found.");
          break;
        }
      }

      constexpr uint32_t _st1GetOpCode(uint32_t registerAmount)
      {
        release_assert(registerAmount >= 1, "Minimum of 1 register must be used.");
        release_assert(registerAmount <= 4, "Maximum of 4 register can be used.");

        const uint32_t opcodes[4] = {
          0b0111,  // 1
          0b1010,  // 2
          0b0110,  // 3
          0b0010   // 4
        };

        return opcodes[registerAmount - 1];
      }

      constexpr uint32_t st1MultipleStructures(const uint32_t Vt, const st1Types Tt, const uint32_t Xn, const uint32_t registerAmount)
      {
        release_assert((Vt & mask5) == Vt, "Vt is only allowed to have a size of 5 bit.");
        release_assert((Xn & mask5) == Xn, "Xn is only allowed to have a size of 5 bit.");

        uint32_t q = 0xff;     // should change
        uint32_t size = 0xff;  // should change
        _st1GetQAndSize(Tt, q, size);
        release_assert(q != 0xff, "Q should be retrieved from a type.");
        release_assert(size != 0xff, "Size should be retrieved from a type.");
        release_assert((q & mask1) == q, "Q is only allowed to have a size of 1 bit.");
        release_assert((size & mask2) == size, "Size is only allowed to have a size of 2 bit.");

        uint32_t opcode = _st1GetOpCode(registerAmount);
        release_assert((opcode & mask4) == opcode, "Opcode is only allowed to have a size of 4 bit.");

        uint32_t st1 = 0;
        st1 |= 0b0 << 31;
        st1 |= (q & mask1) << 30;
        st1 |= 0b00110000000000 << 16;
        st1 |= (opcode & mask4) << 12;
        st1 |= (size & mask2) << 10;
        st1 |= (Xn & mask5) << 5;
        st1 |= (Vt & mask5) << 0;
        return st1;
      }

      constexpr uint32_t st1MultipleStructuresPost(const uint32_t Vt, const st1Types Tt, const uint32_t Xn, const uint32_t imm,
                                                   const uint32_t Xm, const uint32_t registerAmount)
      {
        release_assert((Vt & mask5) == Vt, "Vt is only allowed to have a size of 5 bit.");
        release_assert((Xn & mask5) == Xn, "Xn is only allowed to have a size of 5 bit.");
        release_assert((Xm & mask5) == Xm, "Xm is only allowed to have a size of 5 bit.");

        uint32_t q = 0xff;     // should change
        uint32_t size = 0xff;  // should change
        _st1GetQAndSize(Tt, q, size);
        release_assert(q != 0xff, "Q should be retrieved from a type.");
        release_assert(size != 0xff, "Size should be retrieved from a type.");
        release_assert((q & mask1) == q, "Q is only allowed to have a size of 1 bit.");
        release_assert((size & mask2) == size, "Size is only allowed to have a size of 2 bit.");

        uint32_t opcode = _st1GetOpCode(registerAmount);
        release_assert((opcode & mask4) == opcode, "Opcode is only allowed to have a size of 4 bit.");

        if (Xm == st1ImmediateRm)
        {
          switch (registerAmount)
          {
          case 1:
            release_assert(imm == (8 + q * 8), "imm should be same number as size of used vectors i.e. 8 or 16");
            break;
          case 2:
            release_assert(imm == (16 + q * 16), "imm should be same number as size of used vectors i.e. 16 or 32");
            break;
          case 3:
            release_assert(imm == (24 + q * 24), "imm should be same number as size of used vectors i.e. 24 or 48");
            break;
          case 4:
            release_assert(imm == (32 + q * 32), "imm should be same number as size of used vectors i.e. 32 or 64");
            break;
          default:
            release_assert(false, "Register amount of ouf range.");
            break;
          }
        }

        uint32_t st1 = 0;
        st1 |= 0b0 << 31;
        st1 |= (q & mask1) << 30;
        st1 |= 0b001100100 << 21;
        st1 |= (Xm & mask5) << 16;
        st1 |= (opcode & mask4) << 12;
        st1 |= (size & mask2) << 10;
        st1 |= (Xn & mask5) << 5;
        st1 |= (Vt & mask5) << 0;
        return st1;
      }

    }  // namespace internal

    template <typename T> constexpr uint32_t st1(const VGeneral Vt, const T, const R64Bit Xn)
    {
      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructures(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 1);
    }

    template <typename T> constexpr uint32_t st1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const R64Bit Xn)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructures(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 2);
    }

    template <typename T>
    constexpr uint32_t st1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T, const R64Bit Xn)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3), "Vt3 should be a consecutive vector register.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructures(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 3);
    }

    template <typename T>
    constexpr uint32_t st1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T, const VGeneral Vt4,
                           const T, const R64Bit Xn)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3), "Vt3 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 3) % 32) == static_cast<uint32_t>(Vt4), "Vt4 should be a consecutive vector register.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructures(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 4);
    }

    template <typename T> constexpr uint32_t st1Post(const VGeneral Vt, const T, const R64Bit Xn, const uint32_t imm)
    {
      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), imm, internal::st1ImmediateRm,
                                                 1);
    }

    template <typename T>
    constexpr uint32_t st1Post(const VGeneral Vt, const T, const VGeneral Vt2, const T, const R64Bit Xn, const uint32_t imm)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), imm, internal::st1ImmediateRm,
                                                 2);
    }

    template <typename T>
    constexpr uint32_t st1Post(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T, const R64Bit Xn,
                               const uint32_t imm)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3), "Vt3 should be a consecutive vector register.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), imm, internal::st1ImmediateRm,
                                                 3);
    }

    template <typename T>
    constexpr uint32_t st1Post(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T, const VGeneral Vt4,
                               const T, const R64Bit Xn, const uint32_t imm)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3), "Vt3 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 3) % 32) == static_cast<uint32_t>(Vt4), "Vt4 should be a consecutive vector register.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), imm, internal::st1ImmediateRm,
                                                 4);
    }

    template <typename T> constexpr uint32_t st1Post(const VGeneral Vt, const T, const R64Bit Xn, const R64Bit Xm)
    {
      release_assert((static_cast<uint32_t>(Xm) & mask5) != internal::st1ImmediateRm,
                     "The offset register should not have the same value as the Rm for the immediate.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 0, static_cast<uint32_t>(Xm),
                                                 1);
    }

    template <typename T>
    constexpr uint32_t st1Post(const VGeneral Vt, const T, const VGeneral Vt2, const T, const R64Bit Xn, const R64Bit Xm)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");
      release_assert((static_cast<uint32_t>(Xm) & mask5) != internal::st1ImmediateRm,
                     "The offset register should not have the same value as the Rm for the immediate.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 0, static_cast<uint32_t>(Xm),
                                                 2);
    }

    template <typename T>
    constexpr uint32_t st1Post(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T, const R64Bit Xn,
                               const R64Bit Xm)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3), "Vt3 should be a consecutive vector register.");
      release_assert((static_cast<uint32_t>(Xm) & mask5) != internal::st1ImmediateRm,
                     "The offset register should not have the same value as the Rm for the immediate.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 0, static_cast<uint32_t>(Xm),
                                                 3);
    }

    template <typename T>
    constexpr uint32_t st1Post(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T, const VGeneral Vt4,
                               const T, const R64Bit Xn, const R64Bit Xm)
    {
      release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2), "Vt2 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3), "Vt3 should be a consecutive vector register.");
      release_assert(((static_cast<uint32_t>(Vt) + 3) % 32) == static_cast<uint32_t>(Vt4), "Vt4 should be a consecutive vector register.");
      release_assert((static_cast<uint32_t>(Xm) & mask5) != internal::st1ImmediateRm,
                     "The offset register should not have the same value as the Rm for the immediate.");

      internal::st1Types type = internal::_st1ParseType<T>();
      return internal::st1MultipleStructuresPost(static_cast<uint32_t>(Vt), type, static_cast<uint32_t>(Xn), 0, static_cast<uint32_t>(Xm),
                                                 4);
    }
  }  // namespace arm_instructions
}  // namespace mini_jit

#endif  // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_ST1_H