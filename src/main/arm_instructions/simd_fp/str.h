#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_STR_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_STR_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

enum class strSimdFpDataTypes : uint32_t
{
    v8bit = 0b00,
    v16bit = 0b01,
    v32bit = 0b10,
    v64bit = 0b11,
    v128bit = 0b100 // used as 0b00
};

constexpr uint32_t _strSimdFpGetOpCode(const strSimdFpDataTypes type)
{
    switch (type)
    {
    case strSimdFpDataTypes::v8bit:
    case strSimdFpDataTypes::v16bit:
    case strSimdFpDataTypes::v32bit:
    case strSimdFpDataTypes::v64bit:
        return 0b00;
        break;
    case strSimdFpDataTypes::v128bit:
        return 0b10;
        break;
    default:
        release_assert(false, "Undefined simd str type found.");
        break;
    }
    return 0;
}

constexpr uint32_t strSimdFpImmediatePost(const uint32_t Vt, const uint32_t Rn, const int32_t imm9,
    const strSimdFpDataTypes type)
{
    release_assert(((Vt & mask5) == Vt), "Vt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");
    release_assert(imm9 <= 255, "imm9 has a Maximum of 255");
    release_assert(imm9 >= -256, "imm9 has a Minimum of -256");

    uint32_t opcode = _strSimdFpGetOpCode(type);
    uint32_t str = 0;
    str |= (static_cast<uint32_t>(type) & mask2) << 30;
    str |= 0b111100 << 24;
    str |= (opcode & mask2) << 22;
    str |= 0b0 << 21;
    str |= (imm9 & mask9) << 12;
    str |= 0b01 << 10;
    str |= (Rn & mask5) << 5;
    str |= (Vt & mask5) << 0;
    return str;
}

constexpr uint32_t strSimdFpImmediatePre(const uint32_t Vt, const uint32_t Rn, const int32_t imm9,
    const strSimdFpDataTypes type)
{
    release_assert(((Vt & mask5) == Vt), "Vt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");
    release_assert(imm9 <= 255, "imm9 has a Maximum of 255");
    release_assert(imm9 >= -256, "imm9 has a Minimum of -256");

    uint32_t opcode = _strSimdFpGetOpCode(type);
    uint32_t str = 0;
    str |= (static_cast<uint32_t>(type) & mask2) << 30;
    str |= 0b111100 << 24;
    str |= (opcode & mask2) << 22;
    str |= 0b0 << 21;
    str |= (imm9 & mask9) << 12;
    str |= 0b11 << 10;
    str |= (Rn & mask5) << 5;
    str |= (Vt & mask5) << 0;
    return str;
}

constexpr uint32_t strSimdFpImmediateOffset(const uint32_t Vt, const uint32_t Rn, const uint32_t imm12,
    const strSimdFpDataTypes type)
{
    release_assert(((Vt & mask5) == Vt), "Vt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");

    uint32_t immShift = 0;
    switch (type)
    {
    case strSimdFpDataTypes::v8bit:
        immShift = 0;
        release_assert((((imm12 >> 0) & mask12) == (imm12 >> 0)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 4096.");
        break;
    case strSimdFpDataTypes::v16bit:
        immShift = 1;
        release_assert((((imm12 >> 1) & mask12) == (imm12 >> 1)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 8190.");
        release_assert(((imm12 & mask1) == 0), "imm12 should be multiple of 2.");
        break;
    case strSimdFpDataTypes::v32bit:
        immShift = 2;
        release_assert((((imm12 >> 2) & mask12) == (imm12 >> 2)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 16380.");
        release_assert(((imm12 & mask2) == 0), "imm12 should be multiple of 4.");
        break;
    case strSimdFpDataTypes::v64bit:
        immShift = 3;
        release_assert((((imm12 >> 3) & mask12) == (imm12 >> 3)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 32760.");
        release_assert(((imm12 & mask3) == 0), "imm12 should be multiple of 8.");
        break;
    case strSimdFpDataTypes::v128bit:
        immShift = 4;
        release_assert((((imm12 >> 4) & mask12) == (imm12 >> 4)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 65520.");
        release_assert(((imm12 & mask4) == 0), "imm12 should be multiple of 16.");
        break;
    default:
        release_assert(false, "Undefined simd str type found.");
        break;
    }

    uint32_t opcode = _strSimdFpGetOpCode(type);
    uint32_t str = 0;
    str |= (static_cast<uint32_t>(type) & mask2) << 30;
    str |= 0b111101 << 24;
    str |= (opcode & mask2) << 22;
    str |= ((imm12 >> immShift) & mask12) << 10;
    str |= (Rn & mask5) << 5;
    str |= (Vt & mask5) << 0;
    return str;
}

} // namespace internal

constexpr uint32_t strPost(const V8Bit Bt, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePost(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v8bit);
}

constexpr uint32_t strPost(const V16Bit Ht, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePost(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v16bit);
}

constexpr uint32_t strPost(const V32Bit St, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePost(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v32bit);
}

constexpr uint32_t strPost(const V64Bit Dt, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePost(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v64bit);
}

constexpr uint32_t strPost(const V128Bit Qt, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePost(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v128bit);
}

constexpr uint32_t strPre(const V8Bit Bt, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePre(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v8bit);
}

constexpr uint32_t strPre(const V16Bit Ht, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePre(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v16bit);
}

constexpr uint32_t strPre(const V32Bit St, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePre(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v32bit);
}

constexpr uint32_t strPre(const V64Bit Dt, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePre(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v64bit);
}

constexpr uint32_t strPre(const V128Bit Qt, const R64Bit Xn, const int32_t imm9)
{
    return internal::strSimdFpImmediatePre(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), imm9,
        internal::strSimdFpDataTypes::v128bit);
}

constexpr uint32_t str(const V8Bit Bt, const R64Bit Xn)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), 0,
        internal::strSimdFpDataTypes::v8bit);
}

constexpr uint32_t str(const V16Bit Ht, const R64Bit Xn)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), 0,
        internal::strSimdFpDataTypes::v16bit);
}

constexpr uint32_t str(const V32Bit St, const R64Bit Xn)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), 0,
        internal::strSimdFpDataTypes::v32bit);
}

constexpr uint32_t str(const V64Bit Dt, const R64Bit Xn)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), 0,
        internal::strSimdFpDataTypes::v64bit);
}

constexpr uint32_t str(const V128Bit Qt, const R64Bit Xn)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), 0,
        internal::strSimdFpDataTypes::v128bit);
}

constexpr uint32_t strOffset(const V8Bit Bt, const R64Bit Xn, const uint32_t imm12)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), imm12,
        internal::strSimdFpDataTypes::v8bit);
}

constexpr uint32_t strOffset(const V16Bit Ht, const R64Bit Xn, const uint32_t imm12)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), imm12,
        internal::strSimdFpDataTypes::v16bit);
}

constexpr uint32_t strOffset(const V32Bit St, const R64Bit Xn, const uint32_t imm12)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), imm12,
        internal::strSimdFpDataTypes::v32bit);
}

constexpr uint32_t strOffset(const V64Bit Dt, const R64Bit Xn, const uint32_t imm12)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), imm12,
        internal::strSimdFpDataTypes::v64bit);
}

constexpr uint32_t strOffset(const V128Bit Qt, const R64Bit Xn, const uint32_t imm12)
{
    return internal::strSimdFpImmediateOffset(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), imm12,
        internal::strSimdFpDataTypes::v128bit);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_STR_H