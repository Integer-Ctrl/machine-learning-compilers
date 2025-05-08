#ifndef MINI_JIT_ARM_INSTRUCTIONS_BASE_MOV_H
#define MINI_JIT_ARM_INSTRUCTIONS_BASE_MOV_H

#include <cstdint>
#include "../../release_assert.h"
#include "../register.h"
#include "orr.h"
#include "movz.h"

namespace mini_jit {
namespace arm_instructions {

constexpr uint32_t mov(const R32Bit Wd, const R32Bit Wm)
{
    return orr(Wd, wzr, Wm);
}

constexpr uint32_t mov(const R64Bit Xd, const R64Bit Xm)
{
    return orr(Xd, xzr, Xm);
}

constexpr uint32_t mov(const R32Bit Wd, const uint32_t imm)
{
    return movz(Wd, imm);
}

constexpr uint32_t mov(const R64Bit Xd, const uint32_t imm)
{
    return movz(Xd, imm);
}

} // namespace arm_instructions
} // namespace mini_jit

#endif // MINI_JIT_ARM_INSTRUCTIONS_BASE_MOV_H