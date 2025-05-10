#include "matmul_16_6_1.h"
#include "../arm_instructions/arm_all.h"
#include "../Kernel.h"


void mini_jit::kernels::matmul_16_6_1(mini_jit::Kernel& kernel)
{
    using namespace mini_jit::arm_instructions;

    kernel.add({
        // Offset the used leading dimension by the size of floats
        lsl(x3, x3, 2), // lsl x3, x3, #2
        lsl(x4, x4, 2), // lsl x4, x4, #2
        lsl(x5, x5, 2), // lsl x5, x5, #2

        // Load all data from the 16x1 matrix a
        ld1(v0, t4s, v1, t4s, v2, t4s, v3, t4s, x0) // ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]
        });

    for (int i = 0; i < 2; i++)
    {
        kernel.add({
            // Load first element from the 1x6 matrix b
            ldr(s4, x1),     // ldr s4, [x1] WARNING
            add(x1, x1, x4), // add x1, x1, x4
            // Load first column from the 16x6 matrix c
            ld1(v25, t4s, v26, t4s, v27, t4s, v28, t4s, x2), // ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2]

            // Calculate first column of c
            fmla(v25, t4s, v0, t4s, v4, 0), // fmla v25.4s, v0.4s, v4.s[0]
            fmla(v26, t4s, v1, t4s, v4, 0), // fmla v26.4s, v1.4s, v4.s[0]
            fmla(v27, t4s, v2, t4s, v4, 0), // fmla v27.4s, v2.4s, v4.s[0]
            fmla(v28, t4s, v3, t4s, v4, 0), // fmla v28.4s, v3.4s, v4.s[0]

            // Store first column back to memory
            st1Post(v25, t4s, v26, t4s, v27, t4s, v28, t4s, x2, x5), // st1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x2], x5

            // Load second element from the 1x6 matrix b
            ldr(s4, x1),     // ldr s4, [x1]
            add(x1, x1, x4), // add x1, x1, x4
            // Load second column from the 16x6 matrix c
            ld1(v17, t4s, v18, t4s, v19, t4s, v20, t4s, x2), // ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2]

            // Calculate second column of c
            fmla(v17, t4s, v0, t4s, v4, 0), // fmla v17.4s, v0.4s, v4.s[0]
            fmla(v18, t4s, v1, t4s, v4, 0), // fmla v18.4s, v1.4s, v4.s[0]
            fmla(v19, t4s, v2, t4s, v4, 0), // fmla v19.4s, v2.4s, v4.s[0]
            fmla(v20, t4s, v3, t4s, v4, 0), // fmla v20.4s, v3.4s, v4.s[0]

            // Store second column back to memory
            st1Post(v17, t4s, v18, t4s, v19, t4s, v20, t4s, x2, x5), // st1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x2], x5

            // Load third element from the 1x6 matrix b
            ldr(s4, x1),     // ldr s4, [x1]
            add(x1, x1, x4), // add x1, x1, x4
            // Load third column from the 16x6 matrix c
            ld1(v21, t4s, v22, t4s, v23, t4s, v24, t4s, x2), // ld1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2]

            // Calculated third column of c
            fmla(v21, t4s, v0, t4s, v4, 0), // fmla v21.4s, v0.4s, v4.s[0]
            fmla(v22, t4s, v1, t4s, v4, 0), // fmla v22.4s, v1.4s, v4.s[0]
            fmla(v23, t4s, v2, t4s, v4, 0), // fmla v23.4s, v2.4s, v4.s[0]
            fmla(v24, t4s, v3, t4s, v4, 0), // fmla v24.4s, v3.4s, v4.s[0]

            // Store third column back to memory
            st1Post(v21, t4s, v22, t4s, v23, t4s, v24, t4s, x2, x5), // st1 {v21.4s, v22.4s, v23.4s, v24.4s}, [x2], x5
            });
    }

    kernel.add(ret()); // ret

#ifdef SAVE_JITS_TO_FILE
    kernel.write("matmul_16_6_1.bin");
#endif // SAVE_JITS_TO_FILE

}
