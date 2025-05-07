#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_LD1_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_LD1_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

const uint32_t ld1ImmediateRm = 0b11111;

enum class ld1Types {
    t8b,
    t16b,
    t4h,
    t8h,
    t2s,
    t4s,
    t1d,
    t2d
};

template<typename T> constexpr ld1Types _ld1ParseType()
{
    static_assert(false, "Not a valid ld1 vector type.");
    return ld1Types::t8b;
}
template<> constexpr ld1Types _ld1ParseType<VType8x8Bit>() { return ld1Types::t8b; }
template<> constexpr ld1Types _ld1ParseType<VType16x8Bit>() { return ld1Types::t16b; }
template<> constexpr ld1Types _ld1ParseType<VType4x16Bit>() { return ld1Types::t4h; }
template<> constexpr ld1Types _ld1ParseType<VType4x16Bit>() { return ld1Types::t8h; }
template<> constexpr ld1Types _ld1ParseType<VType2x32Bit>() { return ld1Types::t2s; }
template<> constexpr ld1Types _ld1ParseType<VType4x32Bit>() { return ld1Types::t4s; }
template<> constexpr ld1Types _ld1ParseType<VType1x64Bit>() { return ld1Types::t1d; }
template<> constexpr ld1Types _ld1ParseType<VType2x64Bit>() { return ld1Types::t2d; }


constexpr void _ld1GetQAndSize(const ld1Types type, uint32_t& out_q, uint32_t& out_size) {
    switch (type)
    {
    case ld1Types::t8b:
        out_q = 0b0;
        out_size = 0b00;
        break;
    case ld1Types::t16b:
        out_q = 0b1;
        out_size = 0b00;
        break;
    case ld1Types::t4h:
        out_q = 0b0;
        out_size = 0b01;
        break;
    case ld1Types::t8h:
        out_q = 0b1;
        out_size = 0b01;
        break;
    case ld1Types::t2s:
        out_q = 0b0;
        out_size = 0b10;
        break;
    case ld1Types::t4s:
        out_q = 0b1;
        out_size = 0b10;
        break;
    case ld1Types::t1d:
        out_q = 0b0;
        out_size = 0b11;
        break;
    case ld1Types::t2d:
        out_q = 0b1;
        out_size = 0b11;
        break;
    default:
        release_assert(false, "Undefined ld1 type found.");
    }
}

constexpr uint32_t _ld1GetOpCode(uint32_t registerAmount)
{
    release_assert(registerAmount >= 1, "Minimum of 1 register must be used.");
    release_assert(registerAmount <= 4, "Maximum of 4 register can be used.");

    const uint32_t opcodes[4] = {
        0b0111, // 1
        0b1010, // 2
        0b0110, // 3
        0b0010  // 4
    };

    return opcodes[registerAmount - 1];
}

constexpr uint32_t ld1MultipleStructures(const uint32_t Vt, const ld1Types Tt, const uint32_t Xn,
    const uint32_t registerAmount)
{
    release_assert((Vt & mask5) == Vt, "Vt is only allowed to have a size of 5 bit.");
    release_assert((Xn & mask5) == Xn, "Xn is only allowed to have a size of 5 bit.");

    uint32_t q = 0xff; // should change
    uint32_t size = 0xff; // should change
    _ld1GetQAndSize(Tt, q, size);
    release_assert(q != 0xff, "Q should be retrieved from a type.");
    release_assert(size != 0xff, "Size should be retrieved from a type.");
    release_assert((q & mask1) == q, "Q is only allowed to have a size of 1 bit.");
    release_assert((size & mask2) == size, "Size is only allowed to have a size of 2 bit.");

    const uint32_t opcode = _ld1GetOpCode(registerAmount);
    release_assert((opcode & mask4) == opcode, "Opcode is only allowed to have a size of 4 bit.");

    uint32_t ld1 = 0;
    ld1 |= 0b0 << 31;
    ld1 |= (q & mask1) << 30;
    ld1 |= 0b00110001000000 << 16;
    ld1 |= (opcode & mask4) << 12;
    ld1 |= (size & mask2) << 10;
    ld1 |= (Xn & mask5) << 5;
    ld1 |= (Vt & mask5) << 0;
    return ld1;
}

constexpr uint32_t ld1MultipleStructuresPost(const uint32_t Vt, const ld1Types Tt, const uint32_t Xn,
    const uint32_t imm, const uint32_t Xm, const uint32_t registerAmount)
{
    release_assert((Vt & mask5) == Vt, "Vt is only allowed to have a size of 5 bit.");
    release_assert((Xn & mask5) == Xn, "Xn is only allowed to have a size of 5 bit.");
    release_assert((Xm & mask5) == Xm, "Xm is only allowed to have a size of 5 bit.");

    uint32_t q = 0xff; // should change
    uint32_t size = 0xff; // should change
    _ld1GetQAndSize(Tt, q, size);
    release_assert(q != 0xff, "Q should be retrieved from a type.");
    release_assert(size != 0xff, "Size should be retrieved from a type.");
    release_assert((q & mask1) == q, "Q is only allowed to have a size of 1 bit.");
    release_assert((size & mask2) == size, "Size is only allowed to have a size of 2 bit.");

    const uint32_t opcode = _ld1GetOpCode(registerAmount);
    release_assert((opcode & mask4) == opcode, "Opcode is only allowed to have a size of 4 bit.");

    if (Xm == ld1ImmediateRm)
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

    uint32_t ld1 = 0;
    ld1 |= 0b0 << 31;
    ld1 |= (q & mask1) << 30;
    ld1 |= 0b001100110 << 21;
    ld1 |= (Xm & mask5) << 16;
    ld1 |= (opcode & mask4) << 12;
    ld1 |= (size & mask2) << 10;
    ld1 |= (Xn & mask5) << 5;
    ld1 |= (Vt & mask5) << 0;
    return ld1;
}

} // namespace internal

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const R64Bit Xn)
{
    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructures(Vt, type, Xn, 1);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const R64Bit Xn)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructures(Vt, type, Xn, 2);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T,
    const R64Bit Xn)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3),
        "Vt3 should be a consecutive vector register.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructures(Vt, type, Xn, 3);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T,
    const VGeneral Vt4, const T, const R64Bit Xn)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3),
        "Vt3 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 3) % 32) == static_cast<uint32_t>(Vt4),
        "Vt4 should be a consecutive vector register.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructures(Vt, type, Xn, 4);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const R64Bit Xn, const uint32_t imm)
{
    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, imm, internal::ld1ImmediateRm, 1);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const R64Bit Xn, const uint32_t imm)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, imm, internal::ld1ImmediateRm, 2);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T,
    const R64Bit Xn, const uint32_t imm)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3),
        "Vt3 should be a consecutive vector register.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, imm, internal::ld1ImmediateRm, 3);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T,
    const VGeneral Vt4, const T, const R64Bit Xn, const uint32_t imm)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3),
        "Vt3 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 3) % 32) == static_cast<uint32_t>(Vt4),
        "Vt4 should be a consecutive vector register.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, imm, internal::ld1ImmediateRm, 4);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const R64Bit Xn, const R64Bit Xm)
{
    release_assert((static_cast<uint32_t>(Xm) & mask5) == internal::ld1ImmediateRm,
        "The offset register should not have the same value as the Rm for the immediate.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, 0, Xm, 1);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const R64Bit Xn, const R64Bit Xm)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");
    release_assert((static_cast<uint32_t>(Xm) & mask5) == internal::ld1ImmediateRm,
        "The offset register should not have the same value as the Rm for the immediate.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, 0, Xm, 2);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T,
    const R64Bit Xn, const R64Bit Xm)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3),
        "Vt3 should be a consecutive vector register.");
    release_assert((static_cast<uint32_t>(Xm) & mask5) == internal::ld1ImmediateRm,
        "The offset register should not have the same value as the Rm for the immediate.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, 0, Xm, 3);
}

template<typename T>
constexpr uint32_t ld1(const VGeneral Vt, const T, const VGeneral Vt2, const T, const VGeneral Vt3, const T,
    const VGeneral Vt4, const T, const R64Bit Xn, const R64Bit Xm)
{
    release_assert(((static_cast<uint32_t>(Vt) + 1) % 32) == static_cast<uint32_t>(Vt2),
        "Vt2 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 2) % 32) == static_cast<uint32_t>(Vt3),
        "Vt3 should be a consecutive vector register.");
    release_assert(((static_cast<uint32_t>(Vt) + 3) % 32) == static_cast<uint32_t>(Vt4),
        "Vt4 should be a consecutive vector register.");
    release_assert((static_cast<uint32_t>(Xm) & mask5) == internal::ld1ImmediateRm,
        "The offset register should not have the same value as the Rm for the immediate.");

    internal::ld1Types type = internal::_ld1ParseType<T>();
    return internal::ld1MultipleStructuresPost(Vt, type, Xn, 0, Xm, 4);
}
} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_LD1_H