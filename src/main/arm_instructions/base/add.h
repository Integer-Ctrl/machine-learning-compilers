
#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_ADD_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_ADD_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit {
namespace arm_instructions {

namespace internal {

enum class addShiftType : uint32_t
{
    DEFAULT = 0b00, // LSL
    LSL = 0b00,
    LSR = 0b01,
    ASR = 0b10,
    // 0b11 RESERVED
};

template<typename T> constexpr addShiftType _addParseShiftType() { static_assert(false, "Not a valid add shift option.") }
template<> constexpr addShiftType _addParseShiftType<ShiftLSL>() { return addShiftType::LSL; }
template<> constexpr addShiftType _addParseShiftType<ShiftLSR>() { return addShiftType::LSR; }
template<> constexpr addShiftType _addParseShiftType<ShiftASR>() { return addShiftType::ASR; }

constexpr uint32_t addShiftedRegister(uint32_t Rd, uint32_t Rn, uint32_t Rm, addShiftType shift, uint32_t imm6,
    bool is64bit)
{
    release_assert((Rd & mask5) == Rd, "Rd is only allowed to have a size of 5 bit.");
    release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");
    release_assert((Rm & mask5) == Rm, "Rm is only allowed to have a size of 5 bit.");
    release_assert((static_cast<uint32_t>(shift) & mask2) == static_cast<uint32_t>(shift),
        "Rm is only allowed to have a size of 5 bit.");
    release_assert((imm6 & mask6) == imm6, "imm6 is only allowed to have a size of 6 bit.");
    release_assert(imm6 >= 0, "Shift amount should be greater equal than 0.");

    if (is64bit)
    {
        release_assert(imm6 <= 63, "Shift amount should be less equal than 63, for the 64-bit variant.");
    }
    else
    {
        release_assert(imm6 <= 31, "Shift amount should be less equal than 31, for the 32-bit variant.");
    }

    uint32_t add = 0;
    add |= (is64bit & mask1) << 31;
    add |= 0b0001011 << 24;
    add |= (static_cast<uint32_t>(shift) & mask2) << 22;
    add |= 0b0 << 21;
    add |= (Rm & mask5) << 16;
    add |= (imm6 & mask6) << 10;
    add |= (Rn & mask5) << 5;
    add |= (Rd & mask5) << 0;
    return add;
}

} // namespace internal

constexpr uint32_t add(const R32Bit Wd, const R32Bit Wn, const R32Bit Wm)
{
    return internal::addShiftedRegister(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), static_cast<uint32_t>(Wm),
        internal::addShiftType::DEFAULT, 0, false);
}

constexpr uint32_t add(const R64Bit Rd, const R64Bit Rn, const R64Bit Rm)
{
    return internal::addShiftedRegister(static_cast<uint32_t>(Rd), static_cast<uint32_t>(Rn), static_cast<uint32_t>(Rm),
        internal::addShiftType::DEFAULT, 0, true);
}

template <typename T>
constexpr uint32_t add(const R32Bit Wd, const R32Bit Wn, const R32Bit Wm, const T, uint32_t amount)
{
    internal::addShiftType shift = internal::_addParseShiftType<T>();
    return internal::addShiftedRegister(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn), static_cast<uint32_t>(Wm),
        shift, amount, false);
}

template <typename T>
constexpr uint32_t add(const R64Bit Rd, const R64Bit Rn, const R64Bit Rm, const T, uint32_t amount)
{
    internal::addShiftType shift = internal::_addParseShiftType<T>();
    return internal::addShiftedRegister(static_cast<uint32_t>(Rd), static_cast<uint32_t>(Rn), static_cast<uint32_t>(Rm),
        shift, amount, true);
}
} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_ADD_H