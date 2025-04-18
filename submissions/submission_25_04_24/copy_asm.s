    .text
    .type copy_asm_0, %function
    .global copy_asm_0
 copy_asm_0:
    ldp w2, w3, [x0]
    stp w2, w3, [x1]

    ldp w4, w5, [x0, #8]
    stp w4, w5, [x1, #8]
    
    ldp w6, w7, [x0, #16]
    stp w6, w7, [x1, #16]

    ldr w8, [x0, #24]
    str w8, [x1, #24]

    ret


    .text
    .type copy_asm_1, %function
    .global copy_asm_1
copy_asm_1:
    mov x3, #0  // counter

start_loop:
    cmp x3, x0  // compare value in x3 and x0
    b.ge end_loop  // conditions: counter x3 greater equal n/x0 (value in [x0])

    ldr w4, [x1, x3, lsl #2]  // adress = x1 + (x3 << 2)
    str w4, [x2, x3, lsl #2]  // x3 << 2 = x3 * 4

    add x3, x3, #1
    b start_loop  // unconditional branch

end_loop:
    ret
