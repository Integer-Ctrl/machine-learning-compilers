#ifndef MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_LDR_H
#define MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_LDR_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

enum class ldrSimdFpDataTypes : uint32_t
{
    v8bit = 0b00,
    v16bit = 0b01,
    v32bit = 0b10,
    v64bit = 0b11,
    v128bit = 0b100 // used as 0b00
};

constexpr uint32_t _ldrSimdFpGetOpCode(const ldrSimdFpDataTypes type)
{
    switch (type)
    {
    case ldrSimdFpDataTypes::v8bit:
    case ldrSimdFpDataTypes::v16bit:
    case ldrSimdFpDataTypes::v32bit:
    case ldrSimdFpDataTypes::v64bit:
        return 0b01;
        break;
    case ldrSimdFpDataTypes::v128bit:
        return 0b11;
        break;
    default:
        release_assert(false, "Undefined simd ldr type found.");
        break;
    }
    return 0;
}

constexpr uint32_t ldrSimdFpImmediatePost(const uint32_t Vt, const uint32_t Rn, const int32_t imm9,
    const ldrSimdFpDataTypes type)
{
    release_assert(((Vt & mask5) == Vt), "Vt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");
    release_assert(imm9 <= 255, "imm9 has a Maximum of 255");
    release_assert(imm9 >= -256, "imm9 has a Minimum of -256");

    uint32_t opcode = _ldrSimdFpGetOpCode(type);
    uint32_t ldr = 0;
    ldr |= (static_cast<uint32_t>(type) & mask2) << 30;
    ldr |= 0b111100 << 24;
    ldr |= (opcode & mask2) << 22;
    ldr |= 0b0 << 21;
    ldr |= (imm9 & mask9) << 12;
    ldr |= 0b01 << 10;
    ldr |= (Rn & mask5) << 5;
    ldr |= (Vt & mask5) << 0;
    return ldr;
}

constexpr uint32_t ldrSimdFpImmediatePre(const uint32_t Vt, const uint32_t Rn, const int32_t imm9,
    const ldrSimdFpDataTypes type)
{
    release_assert(((Vt & mask5) == Vt), "Vt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");
    release_assert(imm9 <= 255, "imm9 has a Maximum of 255");
    release_assert(imm9 >= -256, "imm9 has a Minimum of -256");

    uint32_t opcode = _ldrSimdFpGetOpCode(type);
    uint32_t ldr = 0;
    ldr |= (static_cast<uint32_t>(type) & mask2) << 30;
    ldr |= 0b111100 << 24;
    ldr |= (opcode & mask2) << 22;
    ldr |= 0b0 << 21;
    ldr |= (imm9 & mask9) << 12;
    ldr |= 0b11 << 10;
    ldr |= (Rn & mask5) << 5;
    ldr |= (Vt & mask5) << 0;
    return ldr;
}

constexpr uint32_t ldrSimdFpImmediateOffset(const uint32_t Vt, const uint32_t Rn, const uint32_t imm12,
    const ldrSimdFpDataTypes type)
{
    release_assert(((Vt & mask5) == Vt), "Vt is only allowed to have a size of 5 bit.");
    release_assert(((Rn & mask5) == Rn), "Rn is only allowed to have a size of 5 bit.");

    uint32_t immShift = 0;
    switch (type)
    {
    case ldrSimdFpDataTypes::v8bit:
        immShift = 0;
        release_assert((((imm12 >> 0) & mask12) == (imm12 >> 0)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 4096.");
        break;
    case ldrSimdFpDataTypes::v16bit:
        immShift = 1;
        release_assert((((imm12 >> 1) & mask12) == (imm12 >> 1)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 8190.");
        release_assert(((imm12 & mask1) == 0), "imm12 should be multiple of 2.");
        break;
    case ldrSimdFpDataTypes::v32bit:
        immShift = 2;
        release_assert((((imm12 >> 2) & mask12) == (imm12 >> 2)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 16380.");
        release_assert(((imm12 & mask2) == 0), "imm12 should be multiple of 4.");
        break;
    case ldrSimdFpDataTypes::v64bit:
        immShift = 3;
        release_assert((((imm12 >> 3) & mask12) == (imm12 >> 3)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 32760.");
        release_assert(((imm12 & mask3) == 0), "imm12 should be multiple of 8.");
        break;
    case ldrSimdFpDataTypes::v128bit:
        immShift = 4;
        release_assert((((imm12 >> 4) & mask12) == (imm12 >> 4)),
            "imm12 is only allowed to have a size of 12 bit after shift i.e. 65520.");
        release_assert(((imm12 & mask4) == 0), "imm12 should be multiple of 16.");
        break;
    default:
        release_assert(false, "Undefined simd ldr type found.");
        break;
    }

    uint32_t opcode = _ldrSimdFpGetOpCode(type);
    uint32_t ldr = 0;
    ldr |= (static_cast<uint32_t>(type) & mask2) << 30;
    ldr |= 0b111101 << 24;
    ldr |= (opcode & mask2) << 22;
    ldr |= ((imm12 >> immShift) & mask12) << 10;
    ldr |= (Rn & mask5) << 5;
    ldr |= (Vt & mask5) << 0;
    return ldr;
}

} // namespace internal

constexpr uint32_t ldrPost(const V8Bit Bt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePost(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v8bit);
}

constexpr uint32_t ldrPost(const V16Bit Ht, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePost(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v16bit);
}

constexpr uint32_t ldrPost(const V32Bit St, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePost(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldrPost(const V64Bit Dt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePost(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldrPost(const V128Bit Qt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePost(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v128bit);
}

constexpr uint32_t ldrPre(const V8Bit Bt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePre(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v8bit);
}

constexpr uint32_t ldrPre(const V16Bit Ht, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePre(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v16bit);
}

constexpr uint32_t ldrPre(const V32Bit St, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePre(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldrPre(const V64Bit Dt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePre(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldrPre(const V128Bit Qt, const R64Bit Xn, const int32_t imm9)
{
    return internal::ldrSimdFpImmediatePre(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), imm9,
        internal::ldrSimdFpDataTypes::v128bit);
}

constexpr uint32_t ldr(const V8Bit Bt, const R64Bit Xn)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), 0,
        internal::ldrSimdFpDataTypes::v8bit);
}

constexpr uint32_t ldr(const V16Bit Ht, const R64Bit Xn)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), 0,
        internal::ldrSimdFpDataTypes::v16bit);
}

constexpr uint32_t ldr(const V32Bit St, const R64Bit Xn)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), 0,
        internal::ldrSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldr(const V64Bit Dt, const R64Bit Xn)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), 0,
        internal::ldrSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldr(const V128Bit Qt, const R64Bit Xn)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), 0,
        internal::ldrSimdFpDataTypes::v128bit);
}

constexpr uint32_t ldrOffset(const V8Bit Bt, const R64Bit Xn, const int32_t imm12)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Bt), static_cast<uint32_t>(Xn), imm12,
        internal::ldrSimdFpDataTypes::v8bit);
}

constexpr uint32_t ldrOffset(const V16Bit Ht, const R64Bit Xn, const int32_t imm12)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Ht), static_cast<uint32_t>(Xn), imm12,
        internal::ldrSimdFpDataTypes::v16bit);
}

constexpr uint32_t ldrOffset(const V32Bit St, const R64Bit Xn, const int32_t imm12)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(St), static_cast<uint32_t>(Xn), imm12,
        internal::ldrSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldrOffset(const V64Bit Dt, const R64Bit Xn, const int32_t imm12)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Dt), static_cast<uint32_t>(Xn), imm12,
        internal::ldrSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldrOffset(const V128Bit Qt, const R64Bit Xn, const int32_t imm12)
{
    return internal::ldrSimdFpImmediateOffset(static_cast<uint32_t>(Qt), static_cast<uint32_t>(Xn), imm12,
        internal::ldrSimdFpDataTypes::v128bit);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_SIMD_FP_LDR_H