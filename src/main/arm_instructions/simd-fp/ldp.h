#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_LDP_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_LDP_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

enum class ldpSimdFpDataTypes : uint32_t
{
    v32bit = 0b00,
    v64bit = 0b01,
    v128bit = 0b10
};

constexpr uint32_t _ldpSimdFpPostPreOffset(const uint32_t opcode, const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn,
    const int32_t imm7, const ldpSimdFpDataTypes type)
{
    release_assert((Rt1 & mask5) == Rt1, "Rt1 is only allowed to have a size of 5 bit.");
    release_assert((Rt2 & mask5) == Rt2, "Rt2 is only allowed to have a size of 5 bit.");
    release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");

    uint32_t immShift = 0;
    switch (type)
    {
    case ldpSimdFpDataTypes::v32bit:
        immShift = 2;
        release_assert((imm7 & mask2) == 0, "imm7 should be a multiple of 4.");
        release_assert(imm7 >= -256, "imm7 minimum is -256.");
        release_assert(imm7 <= 252, "immm7 maximum is 252.");
        break;
    case ldpSimdFpDataTypes::v64bit:
        immShift = 3;
        release_assert((imm7 & mask3) == 0, "imm7 should be a multiple of 8.");
        release_assert(imm7 >= -512, "imm7 minimum is -512.");
        release_assert(imm7 <= 504, "immm7 maximum is 504.");
        break;
    case ldpSimdFpDataTypes::v128bit:
        immShift = 4;
        release_assert((imm7 & mask4) == 0, "imm7 should be a multiple of 16.");
        release_assert(imm7 >= -1024, "imm7 minimum is -1024.");
        release_assert(imm7 <= 1008, "immm7 maximum is 1008.");
        break;
    default:
        release_assert(false, "Undefined ldp simd type found.");
        break;
    }


    uint32_t ldp = 0;
    ldp |= (static_cast<uint32_t>(type) & mask2) << 30;
    ldp |= (opcode & mask8) << 22;
    ldp |= ((imm7 >> immShift) & mask7) << 15;
    ldp |= (Rt2 & mask5) << 10;
    ldp |= (Rn & mask5) << 5;
    ldp |= (Rt1 & mask5) << 0;
    return ldp;
}

constexpr uint32_t ldpPost(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
    const ldpSimdFpDataTypes type)
{
    return _ldpSimdFpPostPreOffset(0b10110011, Rt1, Rt2, Rn, imm7, type);
}

constexpr uint32_t ldpPre(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
    const ldpSimdFpDataTypes type)
{
    return _ldpSimdFpPostPreOffset(0b10110111, Rt1, Rt2, Rn, imm7, type);
}

constexpr uint32_t ldpOffset(const uint32_t Rt1, const uint32_t Rt2, const uint32_t Rn, const int32_t imm7,
    const ldpSimdFpDataTypes type)
{
    return _ldpSimdFpPostPreOffset(0b10110101, Rt1, Rt2, Rn, imm7, type);
}

} // namespace internal

constexpr uint32_t ldpPost(const V32Bit St1, const V32Bit St2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpPost(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldpPost(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpPost(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldpPost(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpPost(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v128bit);
}

constexpr uint32_t ldpPre(const V32Bit St1, const V32Bit St2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpPre(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldpPre(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpPre(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldpPre(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpPre(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v128bit);
}

constexpr uint32_t ldp(const V32Bit St1, const V32Bit St2, const R64Bit Xn)
{
    return internal::ldpOffset(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn),
        0, internal::ldpSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldp(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn)
{
    return internal::ldpOffset(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn),
        0, internal::ldpSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldp(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn)
{
    return internal::ldpOffset(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn),
        0, internal::ldpSimdFpDataTypes::v128bit);
}
constexpr uint32_t ldpOffset(const V32Bit St1, const V32Bit St2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpOffset(static_cast<uint32_t>(St1), static_cast<uint32_t>(St2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v32bit);
}

constexpr uint32_t ldpOffset(const V64Bit Dt1, const V64Bit Dt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpOffset(static_cast<uint32_t>(Dt1), static_cast<uint32_t>(Dt2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v64bit);
}

constexpr uint32_t ldpOffset(const V128Bit Qt1, const V128Bit Qt2, const R64Bit Xn, const int32_t imm7)
{
    return internal::ldpOffset(static_cast<uint32_t>(Qt1), static_cast<uint32_t>(Qt2), static_cast<uint32_t>(Xn),
        imm7, internal::ldpSimdFpDataTypes::v128bit);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_LDP_H