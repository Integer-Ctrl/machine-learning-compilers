reg_name = "v"
enum_name = "VGeneral"
bit_name = "general"
for i in range(8):
    print(f"/// @brief {bit_name} parameter/result register (caller-saved)")
    print(f"const {enum_name} {reg_name}{i} = {enum_name}::{reg_name}{i};")
    print()

for i in range(8, 16):
    print(f"/// @brief {bit_name} scratch register (callee-saved, lower 64 bit)")
    print(f"const {enum_name} {reg_name}{i} = {enum_name}::{reg_name}{i};")
    print()

for i in range(16, 32):
    print(f"/// @brief {bit_name} scratch register (caller-saved)")
    print(f"const {enum_name} {reg_name}{i} = {enum_name}::{reg_name}{i};")
    print()