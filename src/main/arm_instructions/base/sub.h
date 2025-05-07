
#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_SUB_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_SUB_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"

namespace mini_jit
{
	namespace arm_instructions
	{
		namespace internal
		{

			enum class subShiftType : uint32_t
			{
					DEFAULT = 0b0, // LSL0
					LSL0 = 0b0,
					LSL12 = 0b1,
			};

			template<typename T> constexpr subShiftType _subParseShiftType() { static_assert(false, "Not a valid sub shift option.") }
			template<> constexpr subShiftType _subParseShiftType<ShiftLSL>() { return subShiftType::LSL12; }

			constexpr uint32_t subImmediate(uint32_t Rd, uint32_t Rn, subShiftType shift, uint32_t imm12,
					bool is64bit)
			{
					release_assert((Rd & mask5) == Rd, "Rd is only allowed to have a size of 5 bit.");
					release_assert((Rn & mask5) == Rn, "Rn is only allowed to have a size of 5 bit.");
					release_assert((imm12 & mask12) == imm12, "imm12 is only allowed to have a size of 12 bit.");

					uint32_t sub = 0;
					sub |= (is64bit & mask1) << 31;
					sub |= 0b10100010 << 23;
					sub |= (static_cast<uint32_t>(shift) & mask1) << 22;
					sub |= (imm12 & mask12) << 10;
					sub |= (Rn & mask5) << 5;
					sub |= (Rd & mask5) << 0;
					return sub;
			}

		} // namespace internal

		constexpr uint32_t sub(const R32Bit Wd, const R32Bit Wn, const R32Bit Wm)
		{
				return internal::subImmediate(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn),
						internal::subShiftType::DEFAULT, 0, false);
		}

		constexpr uint32_t sub(const R64Bit Rd, const R64Bit Rn, const R64Bit Rm)
		{
				return internal::subImmediate(static_cast<uint32_t>(Rd), static_cast<uint32_t>(Rn),
						internal::subShiftType::DEFAULT, 0, true);
		}

		template <typename T>
		constexpr uint32_t sub(const R32Bit Wd, const R32Bit Wn, const R32Bit Wm, const T)
		{
				internal::subShiftType shift = internal::_subParseShiftType<T>();
				return internal::subImmediate(static_cast<uint32_t>(Wd), static_cast<uint32_t>(Wn),
						shift, false);
		}

		template <typename T>
		constexpr uint32_t sub(const R64Bit Rd, const R64Bit Rn, const R64Bit Rm, const T)
		{
				internal::subShiftType shift = internal::_subParseShiftType<T>();
				return internal::subImmediate(static_cast<uint32_t>(Rd), static_cast<uint32_t>(Rn),
						shift, true);
		}
	} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_SUB_H