#ifndef MINI_JIT_ARM_INSTRUCTIONS_REGISTER_H
#define MINI_JIT_ARM_INSTRUCTIONS_REGISTER_H

namespace mini_jit {
namespace arm_instructions {

const uint32_t mask1 = 0b1;
const uint32_t mask2 = 0b11;
const uint32_t mask3 = 0b111;
const uint32_t mask4 = 0b1111;
const uint32_t mask5 = 0b1'1111;
const uint32_t mask6 = 0b11'1111;
const uint32_t mask7 = 0b111'1111;
const uint32_t mask8 = 0b1111'1111;
const uint32_t mask9 = 0b1'1111'1111;
const uint32_t mask10 = 0b11'1111'1111;
const uint32_t mask11 = 0b111'1111'1111;
const uint32_t mask12 = 0b1111'1111'1111;
const uint32_t mask13 = 0b1'1111'1111'1111;
const uint32_t mask14 = 0b11'1111'1111'1111;
const uint32_t mask15 = 0b111'1111'1111'1111;

} // namespace arm_instructions
} // namespace mini_jit

#include "register/general_purpose.h"
#include "register/vector.h"

#endif // MINI_JIT_ARM_INSTRUCTIONS_REGISTER_H