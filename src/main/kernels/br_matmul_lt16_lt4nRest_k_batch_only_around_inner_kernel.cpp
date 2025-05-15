#include "../Kernel.h"
#include "../arm_instructions/arm_all.h"
#include "../release_assert.h"
#include "br_matmul_lt16_4n_k.h"
#include "br_matmul_lt16_lt4nRest_k.h"
#include <iostream>

void mini_jit::kernels::br_matmul_lt16_lt4nRest_k(mini_jit::Kernel &kernel, const uint32_t n_loop_4, const uint32_t k_loop,
                                                  const uint32_t br_size, const uint32_t m_loop_rest, const uint32_t n_loop_rest)
{
  using namespace mini_jit::arm_instructions;

  release_assert(k_loop != 0, "Cannot proccess matrix with k loop of 0.");
  release_assert(m_loop_rest != 0, "Cannot create a matrix with a rest of m equal to 0!");
  release_assert(m_loop_rest <= 15, "Cannot create a matrix with a rest of m larger than 15!");
  release_assert(n_loop_rest != 0, "Cannot create a matrix with a rest of n equal to 0!");
  release_assert(n_loop_rest <= 4, "Cannot create a matrix with a rest of n larger than 4!");
  release_assert(br_size != 0, "Cannot proccess batch dimension of 0.");

  // Hold the number of instruction to jump for each loop
  int32_t kernel_size_stamp;
  int32_t jump_batch_loop = 5;

  // Idea: Division of the matrix into sub-matrices and calculated in the following order.
  // 1. matmul_lt16_4n_k is omitted if n is less than 4;
  //
  //                    N dimension
  // ←-------------------------------------------→
  // ============================================= ↑
  // |                        |                  | |
  // |  1. matmul_lt16_4n_k   | 2. m < 16, n < 4 | | M < 16 dimension
  // |                        |                  | |
  // ============================================= ↓

  kernel.add({
    // /**
    //     * @param x0 = a pointer to column-major 64x64 matrix A.
    //     * @param x1 = b pointer to column-major 64x64 matrix B.
    //     * @param x2 = c pointer to column-major 64x64 matrix C.
    //     * @param x3 = lda leading dimension of A.
    //     * @param x4 = ldb leading dimension of B.
    //     * @param x5 = ldc leading dimension of C.
    // **/
    // .type matmul_64_48_64, %function
    // .global matmul_64_48_64
    // matmul_64_48_64:

    //     // Procedural Call Standard
    //     // save frame pointer and link register
    //     // stp fp, lr, [sp, #-16]!
    //     // update frame pointer to current stack pointer
    //     // mov fp, sp

    //     // save callee-saved registers
    stpPre(x19, x20, sp, -16),  //     // stp x19, x20, [sp, #-16]!
    stpPre(x21, x22, sp, -16),  //     // stp x21, x22, [sp, #-16]!
    stpPre(x23, x24, sp, -16),  //     // stp x23, x24, [sp, #-16]!
    //     // stp x25, x26, [sp, #-16]!
    stpPre(x27, x28, sp, -16),  //     // stp x27, x28, [sp, #-16]!

    stpPre(d8, d9, sp, -16),  //     stp  d8,  d9, [sp, #-16]!
    //     // stp d10, d11, [sp, #-16]!
    //     // stp d12, d13, [sp, #-16]!
    //     // stp d14, d15, [sp, #-16]!

    //     // Offset the used leading dimension by the size of floats
    lsl(x3, x3, 2),  //     lsl x3, x3, #2 // x3 * 4 = x3 * sizeof(float)
    lsl(x4, x4, 2),  //     lsl x4, x4, #2 // x4 * 4 = x4 * sizeof(float)
    lsl(x5, x5, 2),  //     lsl x5, x5, #2 // x5 * 4 = x5 * sizeof(float)
    lsl(x6, x6, 2),  //     lsl x6, x6, #2 // x6 * 4 = x6 * sizeof(float)
    lsl(x7, x7, 2),  //     lsl x7, x7, #2 // x7 * 4 = x7 * sizeof(float)

    mov(x27, x1),  //     mov x27, x1 // Store the initial value of x1, to be restored in the K loop iteration
    mov(x28, x2),  //     mov x28, x2 // Store the initial value of x2, to be restored in the K loop iteration

    mov(x8, x0),  //     mov x8, x0 // Store the initial value of x0, to be restored in the M loop iteration
    mov(x9, x1),  //     mov x9, x1 // Store the initial value of x1, to be restored in the M loop iteration

    mov(x10, x0),  //     mov x10, x0 // Store the initial value of x0, to be restored in the N loop iteration
    mov(x11, x2),  //     mov x11, x2 // Store the initial value of x2, to bes restored in the N loop iteration
    mov(x12, 4),   //     mov x12, #4 // hold the size of N that are processed in one loop, needed for offset calculation

    mov(x20, x1),  // Store the initial value of x1, to be restored in the Batch loop iteration
    mov(x21, x2),  // Store the initial value of x2, to be restored in the Batch loop iteration

    mov(x22, x0),  //     mov x0 // Store the initial value of x0, to be restored in the N loop iteration
    mov(x23, x1),  //     mov x1 // Store the initial value of x0, to be restored in the N loop iteration
    mov(x24, x2),  //     mov x2 // Store the initial value of x2, to bes restored in the N loop iteration
  });

  // ========================================================================================
  // Calculation of the n loop
  // ========================================================================================

  if (n_loop_4 != 0)
  {
    br_matmul_lt16_4n_k(kernel, n_loop_4, k_loop, br_size, m_loop_rest, false);
  }

  kernel.add({
    // Restore for batch loop
    mov(x27, x23),  //     mov x27, x1 // Store the initial value of x1, to be restored in the K loop iteration
    mov(x28, x24),  //     mov x28, x2 // Store the initial value of x2, to be restored in the K loop iteration

    mov(x8, x22),  //     mov x8, x0 // Store the initial value of x0, to be restored in the M loop iteration
    mov(x9, x23),  //     mov x9, x1 // Store the initial value of x1, to be restored in the M loop iteration

    mov(x10, x22),           //     mov x10, x0 // Store the initial value of x0, to be restored in the N loop iteration
    mov(x11, x24),           //     mov x11, x2 // Store the initial value of x2, to bes restored in the N loop iteration
    mov(x12, n_loop_4 * 4),  //     mov x12, #4 // hold the size of N that are processed in one loop, needed for offset calculation

    mov(x20, x23),  // Store the initial value of x1, to be restored in the Batch loop iteration
    mov(x21, x24),  // Store the initial value of x2, to be restored in the Batch loop iteration

    madd(x20, x4, x12, x20),
    madd(x21, x5, x12, x21),
  });

  // ========================================================================================
  // Rest Calculation of m and n loop
  // ========================================================================================

  kernel.add({
    mov(x19, br_size),  // mov x19, #16 // x19 iterator for the batch dimension
    // matmul_loop_batch_dimension:
    sub(x19, x19, 1),  // sub x19, x19, #1

    // Restore for the loop jumps
    // Update for the matrix b
    mov(x9, x20),  // mov x9, x20 // Update the restore register of x1 for the N loop

    // Update for the matrix c
    mov(x11, x21),  // mov x11, x21 // Update the restore register of x2 for the N loop
  });

  kernel_size_stamp = kernel.get_instruction_count();

  kernel.add({
    //     // Restore for the loop jumps
    //     // Update for the matrix a
    mov(x8, x10),  //     mov x8, x10 // Update the restore register for x0 for the M loop

    //     // Updates for the matrix c
    mov(x28, x11),  //     mov x28, x11 // Update the restore register of x2 for the K loop

    // =================================================================
    // We do not loop through M as we currently processing the rest of M
    // =================================================================
    // mov(x16, m_loop_16),  //     mov x16, #4 // x16 iterator for M loop
    // matmul_loop_over_M:
    // sub(x16, x16, 1),  //     sub x16, x16, #1

    //     // Restore for the loop jumps
    //     // Updates for the matrix c
    mov(x2, x28),  //     mov x2, x28 // also apply offset to x2

    //     // Updates for the matrix a
    mov(x0, x8),  //     mov x0, x8 // also apply offset to x0

    //     // Updates for the matrix b
    mov(x27, x9),  //     mov x27, x9 // Update the restore register for x1 for the K loop
    mov(x1, x9),   //     mov x1, x9 // Update the x1 register itself

  });

  std::vector<uint32_t> instruction1_fp_loads_less_than_4;
  std::vector<uint32_t> instruction2_fp_loads_less_than_4;
  std::vector<uint32_t> instruction3_fp_loads_less_than_4;
  std::vector<uint32_t> instructions_fp_load_column_of_matrix_a;
  const uint32_t m_loop_full_4s = m_loop_rest / 4;
  const uint32_t m_loop_remainder = m_loop_rest % 4;
  const uint32_t offset_from_full_loads = (m_loop_full_4s) * 4 * 4;  // count * number of floats * sizeof(float)

  // For usage reason see case 3 of the below swich statement
  const uint32_t revert_offset_x2_gt1full =
    offset_from_full_loads >= 4 ? sub(x2, x2, offset_from_full_loads - 4) : add(x2, x2, 4 - offset_from_full_loads);
  const uint32_t revert_offset_x0_gt1full =
    offset_from_full_loads >= 4 ? sub(x0, x0, offset_from_full_loads - 4) : add(x0, x0, 4 - offset_from_full_loads);
  const uint32_t apply_offset_x2 = add(x2, x2, offset_from_full_loads + 4);
  const uint32_t revert_offset_x2 = sub(x2, x2, offset_from_full_loads);  // we ldrPost with a -4 to be at the correct position for next ld1
  const uint32_t apply_offset_x0 = add(x0, x0, offset_from_full_loads + 4);
  const uint32_t revert_offset_x0 = sub(x0, x0, offset_from_full_loads);  // we ldrPost with a -4 to be at the correct position for next ld1

  switch (m_loop_remainder)
  {
  case 0:
    // noting to do
    break;
  case 1:  // 1s
    // Using the always last register that are unused
    instruction1_fp_loads_less_than_4 = {ldrOffset(s28, x2, offset_from_full_loads)};
    instruction2_fp_loads_less_than_4 = {ldrOffset(s20, x2, offset_from_full_loads)};
    instruction3_fp_loads_less_than_4 = {ldrOffset(s24, x2, offset_from_full_loads)};
    instructions_fp_load_column_of_matrix_a.push_back(ldrOffset(s3, x0, offset_from_full_loads));
    break;
  case 2:  // 2s
    instruction1_fp_loads_less_than_4 = {ldrOffset(d28, x2, offset_from_full_loads)};
    instruction2_fp_loads_less_than_4 = {ldrOffset(d20, x2, offset_from_full_loads)};
    instruction3_fp_loads_less_than_4 = {ldrOffset(d24, x2, offset_from_full_loads)};
    instructions_fp_load_column_of_matrix_a.push_back(ldrOffset(d3, x0, offset_from_full_loads));
    break;
  case 3:  // 3s
    if (m_loop_full_4s == 0)
    {
      // We need to load 3 float elements in the same register we can load the first two using ldr and the 3. using ld1 single
      // structure i.e. |       | 2. ld1 | 1. ldr | 1. ldr |
      // We can not load more than 3 elements (in reverser direction), as we can not guarantee that we look out of memory.

      // ldrOffset cannot be used as it only supports positive and multiple of 16 numbers.
      instruction1_fp_loads_less_than_4 = {apply_offset_x2, ldrPost(d28, x2, -4), ld1(s28, 2, x2), revert_offset_x2};
      instruction2_fp_loads_less_than_4 = {apply_offset_x2, ldrPost(d20, x2, -4), ld1(s20, 2, x2), revert_offset_x2};
      instruction3_fp_loads_less_than_4 = {apply_offset_x2, ldrPost(d24, x2, -4), ld1(s24, 2, x2), revert_offset_x2};
      instructions_fp_load_column_of_matrix_a.push_back(apply_offset_x0);
      instructions_fp_load_column_of_matrix_a.push_back(ldrPost(d3, x0, -4));
      instructions_fp_load_column_of_matrix_a.push_back(ld1(s3, 2, x0));
      instructions_fp_load_column_of_matrix_a.push_back(revert_offset_x0);
    }
    else
    {
      // Idea load 4 float elements, but only output the three last
      // Note we need to be carful when we store back so that the already calculated 1. element does not get overwritten
      // Solution load single already calculated 1. element to unused register. Save the whole 4s of this calculation
      // Overwrite the first 1. element from the freshly load register.

      // ldrOffset cannot be used as it only supports positive and multiple of 16 numbers.
      instruction1_fp_loads_less_than_4 = {ldrPre(q28, x2, offset_from_full_loads - 4), revert_offset_x2_gt1full};
      instruction2_fp_loads_less_than_4 = {ldrPre(q20, x2, offset_from_full_loads - 4), revert_offset_x2_gt1full};
      instruction3_fp_loads_less_than_4 = {ldrPre(q24, x2, offset_from_full_loads - 4), revert_offset_x2_gt1full};
      instructions_fp_load_column_of_matrix_a.push_back(ldrPre(q3, x0, offset_from_full_loads - 4));
      instructions_fp_load_column_of_matrix_a.push_back(revert_offset_x0_gt1full);
    }

    break;
  default:
    release_assert(false, "Out of range loop rest detected to load with ldr.");
    break;
  }

  switch (m_loop_full_4s)
  {
  case 0:
    if (m_loop_remainder != 0)
    {
      // We only have loads less then 4 fp elements to do.
      if (n_loop_rest >= 1)
      {
        kernel.add(instruction1_fp_loads_less_than_4);
        kernel.add(add(x2, x2, x5));
      }

      if (n_loop_rest >= 2)
      {
        kernel.add(instruction2_fp_loads_less_than_4);
        kernel.add(add(x2, x2, x5));
      }

      if (n_loop_rest >= 3)
      {
        kernel.add(instruction3_fp_loads_less_than_4);
        kernel.add(add(x2, x2, x5));
      }
    }

    instructions_fp_load_column_of_matrix_a.push_back(add(x0, x0, x3));
    break;

  case 1:
    if (m_loop_remainder != 0)
    {
      if (n_loop_rest >= 1)
      {
        //     // Load first column from the 16x6 matrix c
        kernel.add(instruction1_fp_loads_less_than_4);
        kernel.add(ld1Post(v25, t4s, x2, x5));  //     ld1 {v25.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Load second column from the 16x6 matrix c
        kernel.add(instruction2_fp_loads_less_than_4);
        kernel.add(ld1Post(v17, t4s, x2, x5));  //     ld1 {v17.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Load third column from the 16x6 matrix c
        kernel.add(instruction3_fp_loads_less_than_4);
        kernel.add(ld1Post(v21, t4s, x2, x5));  //     ld1 {v21.4s}, [x2], x5
      }
    }
    else
    {
      if (n_loop_rest >= 1)
      {
        //     // Load first column from the 16x6 matrix c
        kernel.add(ld1Post(v25, t4s, x2, x5));  //     ld1 {v25.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Load second column from the 16x6 matrix c
        kernel.add(ld1Post(v17, t4s, x2, x5));  //     ld1 {v17.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Load third column from the 16x6 matrix c
        kernel.add(ld1Post(v21, t4s, x2, x5));  //     ld1 {v21.4s}, [x2], x5
      }
    }

    instructions_fp_load_column_of_matrix_a.push_back(
      //     // Load first column data from the 16x1 matrix a
      ld1Post(v0, t4s, x0, x3));  //     ld1 {v0.4s}, [x0], x3

    break;

  case 2:
    if (m_loop_remainder != 0)
    {
      if (n_loop_rest >= 1)
      {
        //     // Load first column from the 16x6 matrix c
        kernel.add(instruction1_fp_loads_less_than_4);
        kernel.add(ld1Post(v25, t4s, v26, t4s, x2, x5));  //     ld1 {v25.4s, v26.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Load second column from the 16x6 matrix c
        kernel.add(instruction2_fp_loads_less_than_4);
        kernel.add(ld1Post(v17, t4s, v18, t4s, x2, x5));  //     ld1 {v17.4s, v18.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Load third column from the 16x6 matrix c
        kernel.add(instruction3_fp_loads_less_than_4);
        kernel.add(ld1Post(v21, t4s, v22, t4s, x2, x5));  //     ld1 {v21.4s, v22.4s}, [x2], x5
      }
    }
    else
    {
      if (n_loop_rest >= 1)
      {
        //     // Load first column from the 16x6 matrix c
        kernel.add(ld1Post(v25, t4s, v26, t4s, x2, x5));  //     ld1 {v25.4s, v26.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Load second column from the 16x6 matrix c
        kernel.add(ld1Post(v17, t4s, v18, t4s, x2, x5));  //     ld1 {v17.4s, v18.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Load third column from the 16x6 matrix c
        kernel.add(ld1Post(v21, t4s, v22, t4s, x2, x5));  //     ld1 {v21.4s, v22.4s}, [x2], x5
      }
    }

    instructions_fp_load_column_of_matrix_a.push_back(
      //     // Load first column data from the 16x1 matrix a
      ld1Post(v0, t4s, v1, t4s, x0, x3));  //     ld1 {v0.4s, v1.4s}, [x0], x3
    break;

  case 3:
    if (m_loop_remainder != 0)
    {
      if (n_loop_rest >= 1)
      {
        //     // Load first column from the 16x6 matrix c
        kernel.add(instruction1_fp_loads_less_than_4);
        kernel.add(ld1Post(v25, t4s, v26, t4s, v27, t4s, x2, x5));  //     ld1 {v25.4s, v26.4s, v27.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Load second column from the 16x6 matrix c
        kernel.add(instruction2_fp_loads_less_than_4);
        kernel.add(ld1Post(v17, t4s, v18, t4s, v19, t4s, x2, x5));  //     ld1 {v17.4s, v18.4s, v19.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Load third column from the 16x6 matrix c
        kernel.add(instruction3_fp_loads_less_than_4);
        kernel.add(ld1Post(v21, t4s, v22, t4s, v23, t4s, x2, x5));  //     ld1 {v21.4s, v22.4s, v23.4s}, [x2], x5
      }
    }
    else
    {
      if (n_loop_rest >= 1)
      {
        //     // Load first column from the 16x6 matrix c
        kernel.add(ld1Post(v25, t4s, v26, t4s, v27, t4s, x2, x5));  //     ld1 {v25.4s, v26.4s, v27.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Load second column from the 16x6 matrix c
        kernel.add(ld1Post(v17, t4s, v18, t4s, v19, t4s, x2, x5));  //     ld1 {v17.4s, v18.4s, v19.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Load third column from the 16x6 matrix c
        kernel.add(ld1Post(v21, t4s, v22, t4s, v23, t4s, x2, x5));  //     ld1 {v21.4s, v22.4s, v23.4s}, [x2], x5
      }
    }

    instructions_fp_load_column_of_matrix_a.push_back(
      //     // Load first column data from the 16x1 matrix a
      ld1Post(v0, t4s, v1, t4s, v2, t4s, x0, x3));  //     ld1 {v0.4s, v1.4s, v2.4s}, [x0], x3

    break;

    // 4 and further does not exist as 4 = 16 and his handled by the 16x4x1 kernel
  default:
    release_assert(false, "Out of range loop rest detected to load with ld1Post.");
    break;
  }

  // Hold the number of instruction to jump for each loop
  int32_t jump_K_loop = 3;  // start value = amount of instructions outside of control flow

  kernel.add({
    mov(x15, k_loop),  //     mov x15, #64 // x15 iterator for K loop
    // matmul_loop_over_K:
    sub(x15, x15, 1),  //     sub x15, x15, #1

  });

  kernel.add(instructions_fp_load_column_of_matrix_a);
  jump_K_loop += instructions_fp_load_column_of_matrix_a.size();

  if (n_loop_rest >= 1)
  {
    kernel.add({
      //     // run the matmul_16_4_1_unrolled kernel
      //     // Load first element from the 1x4 matrix b
      ldr(s4, x1),      //     ldr s4, [x1]
      add(x1, x1, x4),  //     add x1, x1, x4
    });
    // jump_K_loop is increase for these 2 fixed instructions at the end of the 3 if checks

    //     // Calculate first column of c
    if (m_loop_full_4s >= 1)
    {
      kernel.add(fmla(v25, t4s, v0, t4s, v4, 0));  //     fmla v25.4s, v0.4s, v4.s[0]
    }

    if (m_loop_full_4s >= 2)
    {
      kernel.add(fmla(v26, t4s, v1, t4s, v4, 0));  //     fmla v26.4s, v1.4s, v4.s[0]
    }

    if (m_loop_full_4s >= 3)
    {
      kernel.add(fmla(v27, t4s, v2, t4s, v4, 0));  //     fmla v27.4s, v2.4s, v4.s[0]
    }

    jump_K_loop += m_loop_full_4s;

    if (m_loop_remainder != 0)
    {
      // We calculate all 4s but we dont store them all back, only the valid ones are stored
      kernel.add(fmla(v28, t4s, v3, t4s, v4, 0));  //     fmla v28.4s, v3.4s, v4.s[0]
      jump_K_loop += 1;
    }
  }

  if (n_loop_rest >= 2)
  {
    kernel.add({
      //     // Load second element from the 1x4 matrix b
      ldr(s4, x1),      //     ldr s4, [x1]
      add(x1, x1, x4),  //     add x1, x1, x4
    });
    // jump_K_loop is increase for these 2 fixed instructions at the end of the 3 if checks

    //     // Calculate second column of c
    if (m_loop_full_4s >= 1)
    {
      kernel.add(fmla(v17, t4s, v0, t4s, v4, 0));  //     fmla v17.4s, v0.4s, v4.s[0]
    }

    if (m_loop_full_4s >= 2)
    {
      kernel.add(fmla(v18, t4s, v1, t4s, v4, 0));  //     fmla v18.4s, v1.4s, v4.s[0]
    }

    if (m_loop_full_4s >= 3)
    {
      kernel.add(fmla(v19, t4s, v2, t4s, v4, 0));  //     fmla v19.4s, v2.4s, v4.s[0]
    }

    jump_K_loop += m_loop_full_4s;

    if (m_loop_remainder != 0)
    {
      // We calculate all 4s but we dont store them all back, only the valid ones are stored
      kernel.add(fmla(v20, t4s, v3, t4s, v4, 0));  //     fmla v20.4s, v3.4s, v4.s[0]
      jump_K_loop += 1;
    }
  }

  if (n_loop_rest >= 3)
  {
    kernel.add({
      //     // Load third element from the 1x4 matrix b
      ldr(s4, x1),      //     ldr s4, [x1]
      add(x1, x1, x4),  //     add x1, x1, x4
    });
    // jump_K_loop is increase for these 2 fixed instructions at the end of the 3 if checks

    //     // Calculated third column of c
    if (m_loop_full_4s >= 1)
    {
      kernel.add(fmla(v21, t4s, v0, t4s, v4, 0));  //     fmla v21.4s, v0.4s, v4.s[0]
    }

    if (m_loop_full_4s >= 2)
    {
      kernel.add(fmla(v22, t4s, v1, t4s, v4, 0));  //     fmla v22.4s, v1.4s, v4.s[0]
    }

    if (m_loop_full_4s >= 3)
    {
      kernel.add(fmla(v23, t4s, v2, t4s, v4, 0));  //     fmla v23.4s, v2.4s, v4.s[0]
    }

    jump_K_loop += m_loop_full_4s;

    if (m_loop_remainder != 0)
    {
      // We calculate all 4s but we dont store them all back, only the valid ones are stored
      kernel.add(fmla(v24, t4s, v3, t4s, v4, 0));  //     fmla v24.4s, v3.4s, v4.s[0]
      jump_K_loop += 1;
    }
  }

  jump_K_loop += n_loop_rest * 2;

  kernel.add({
    //     // offset x27 to the next element in the column
    add(x27, x27, 4),  //     add x27, x27, #4 // #4 = sizeof(float)

    //     // Restore x1 to be incremented again
    mov(x1, x27),  //     mov x1, x27

    //     // Loop back to K
    cbnz(x15, -jump_K_loop * 4),  //     cbnz x15, matmul_loop_over_K

    //     // Restore initial value of x2 that was changed by the loads
    mov(x2, x28),  //     mov x2, x28
  });

  std::vector<uint32_t> instruction1_fp_store_less_than_4;
  std::vector<uint32_t> instruction2_fp_store_less_than_4;
  std::vector<uint32_t> instruction3_fp_store_less_than_4;

  // Dynamically saving back to memory
  // jump_batch_loop += jump_K_loop;
  // jump_batch_loop += 1;  // line 533: mov(x2, x28),  //     mov x2, x28

  switch (m_loop_remainder)
  {
  case 0:
    // noting to do
    break;
  case 1:  // 1s
    // Using the always last register that are unused
    instruction1_fp_store_less_than_4 = {strOffset(s28, x2, offset_from_full_loads)};
    instruction2_fp_store_less_than_4 = {strOffset(s20, x2, offset_from_full_loads)};
    instruction3_fp_store_less_than_4 = {strOffset(s24, x2, offset_from_full_loads)};
    break;
  case 2:  // 2s
    instruction1_fp_store_less_than_4 = {strOffset(d28, x2, offset_from_full_loads)};
    instruction2_fp_store_less_than_4 = {strOffset(d20, x2, offset_from_full_loads)};
    instruction3_fp_store_less_than_4 = {strOffset(d24, x2, offset_from_full_loads)};
    break;
  case 3:  // 3s
    // Remember  we need to be carful when we store back so that the already calculated 1. element does not get overwritten
    // The Solution was to load single already calculated 1. element to unused register. Save the whole 4s of this calculation
    // Overwrite the first 1. element from the freshly load register.
    // Here we use v29 as temp store register
    // Addition: In the cases when st1Post is used we stores these values before the post calls, therefore we dont
    // need to restore the first element, as it is not written at this time
    if (m_loop_full_4s == 0)
    {
      // Remember: We need to load 3 float elements in the same register we can load the first two using ldr and the 3. using ld1 single
      // structure i.e. |       | 2. ld1 | 1. ldr | 1. ldr |
      // We can not load more than 3 elements (in reverser direction), as we can not guarantee that we look out of memory.

      // ldrOffset cannot be used as it only supports positive and multiple of 16 numbers.
      instruction1_fp_store_less_than_4 = {apply_offset_x2, strPost(d28, x2, -4), st1(s28, 2, x2), revert_offset_x2};
      instruction2_fp_store_less_than_4 = {apply_offset_x2, strPost(d20, x2, -4), st1(s20, 2, x2), revert_offset_x2};
      instruction3_fp_store_less_than_4 = {apply_offset_x2, strPost(d24, x2, -4), st1(s24, 2, x2), revert_offset_x2};
    }
    else
    {
      // ldrOffset cannot be used as it only supports positive and multiple of 16 numbers.
      instruction1_fp_store_less_than_4 = {strPre(q28, x2, offset_from_full_loads - 4), revert_offset_x2_gt1full};
      instruction2_fp_store_less_than_4 = {strPre(q20, x2, offset_from_full_loads - 4), revert_offset_x2_gt1full};
      instruction3_fp_store_less_than_4 = {strPre(q24, x2, offset_from_full_loads - 4), revert_offset_x2_gt1full};
    }

    break;
  default:
    release_assert(false, "Out of range loop rest detected to load with ldr.");
    break;
  }

  switch (m_loop_full_4s)
  {
  case 0:
    if (m_loop_remainder != 0)
    {
      // We only have store less then 4 fp elements to do.
      if (n_loop_rest >= 1)
      {
        kernel.add(instruction1_fp_store_less_than_4);
        kernel.add(add(x2, x2, x5));
      }

      if (n_loop_rest >= 2)
      {
        kernel.add(instruction2_fp_store_less_than_4);
        kernel.add(add(x2, x2, x5));
      }

      if (n_loop_rest >= 3)
      {
        kernel.add(instruction3_fp_store_less_than_4);
        kernel.add(add(x2, x2, x5));
      }
    }
    break;

  case 1:
    if (m_loop_remainder != 0)
    {
      if (n_loop_rest >= 1)
      {
        kernel.add(instruction1_fp_store_less_than_4);
        kernel.add(
          //     // Store first column back to memory
          st1Post(v25, t4s, x2, x5));  //     st1 {v25.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        kernel.add(instruction2_fp_store_less_than_4);
        kernel.add(
          //     // Store second column back to memory
          st1Post(v17, t4s, x2, x5));  //     st1 {v17.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        kernel.add(instruction3_fp_store_less_than_4);
        kernel.add(
          //     // Store third column back to memory
          st1Post(v21, t4s, x2, x5));  //     st1 {v21.4s}, [x2], x5
      }
    }
    else
    {
      if (n_loop_rest >= 1)
      {
        //     // Store first column back to memory
        kernel.add(st1Post(v25, t4s, x2, x5));  //     st1 {v25.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Store second column back to memory
        kernel.add(st1Post(v17, t4s, x2, x5));  //     st1 {v17.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Store third column back to memory
        kernel.add(st1Post(v21, t4s, x2, x5));  //     st1 {v21.4s}, [x2], x5
      }
    }

    break;

  case 2:
    if (m_loop_remainder != 0)
    {
      if (n_loop_rest >= 1)
      {
        kernel.add(instruction1_fp_store_less_than_4);
        kernel.add(
          //     // Store first column back to memory
          st1Post(v25, t4s, v26, t4s, x2, x5));  //     st1 {v25.4s, v26.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        kernel.add(instruction2_fp_store_less_than_4);
        kernel.add(
          //     // Store second column back to memory
          st1Post(v17, t4s, v18, t4s, x2, x5));  //     st1 {v17.4s, v18.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        kernel.add(instruction3_fp_store_less_than_4);
        kernel.add(
          //     // Store third column back to memory
          st1Post(v21, t4s, v22, t4s, x2, x5));  //     st1 {v21.4s, v22.4s}, [x2], x5
      }
    }
    else
    {
      if (n_loop_rest >= 1)
      {
        //     // Store first column back to memory
        kernel.add(st1Post(v25, t4s, v26, t4s, x2, x5));  //     st1 {v25.4s, v26.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Store second column back to memory
        kernel.add(st1Post(v17, t4s, v18, t4s, x2, x5));  //     st1 {v17.4s, v18.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Store third column back to memory
        kernel.add(st1Post(v21, t4s, v22, t4s, x2, x5));  //     st1 {v21.4s, v22.4s}, [x2], x5
      }
    }

    break;

  case 3:
    if (m_loop_remainder != 0)
    {
      if (n_loop_rest >= 1)
      {
        kernel.add(instruction1_fp_store_less_than_4);
        kernel.add(
          //     // Store first column back to memory
          st1Post(v25, t4s, v26, t4s, v27, t4s, x2, x5));  //     st1 {v25.4s, v26.4s, v27.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        kernel.add(instruction2_fp_store_less_than_4);
        kernel.add(
          //     // Store second column back to memory
          st1Post(v17, t4s, v18, t4s, v19, t4s, x2, x5));  //     st1 {v17.4s, v18.4s, v19.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        kernel.add(instruction3_fp_store_less_than_4);
        kernel.add(
          //     // Store third column back to memory
          st1Post(v21, t4s, v22, t4s, v23, t4s, x2, x5));  //     st1 {v21.4s, v22.4s, v23.4s}, [x2], x5
      }
    }
    else
    {
      if (n_loop_rest >= 1)
      {
        //     // Store first column back to memory
        kernel.add(st1Post(v25, t4s, v26, t4s, v27, t4s, x2, x5));  //     st1 {v25.4s, v26.4s, v27.4s}, [x2], x5
      }

      if (n_loop_rest >= 2)
      {
        //     // Store second column back to memory
        kernel.add(st1Post(v17, t4s, v18, t4s, v19, t4s, x2, x5));  //     st1 {v17.4s, v18.4s, v19.4s}, [x2], x5
      }

      if (n_loop_rest >= 3)
      {
        //     // Store third column back to memory
        kernel.add(st1Post(v21, t4s, v22, t4s, v23, t4s, x2, x5));  //     st1 {v21.4s, v22.4s, v23.4s}, [x2], x5
      }
    }

    break;

    // 4 and further does not exist as 4 = 16 and his handled by the 16x4x1 kernel
  default:
    release_assert(false, "Out of range loop rest detected to store with st1Post.");
    break;
  }

  // ======================================================================
  // No need to loop back to M as we are currently processing the rest of M
  // ======================================================================
  //     // next M iteration on the matrix c and matrix a, both need offset about 16 values
  //     // also matrix b needs to start at the initial location again
  //     // Updates for the matrix c
  // add(x28, x28, 16 * 4), //     add x28, x28, #16*4 // column height * sizeof(float)

  //     // Updates for the matrix a
  // add(x8, x8, 16 * 4), //     add x8, x8, #16*4 // column height * sizeof(float)

  //     // Loop back to M
  // cbnz(x16, -46 * 4), //     cbnz x16, matmul_loop_over_M

  jump_batch_loop += kernel.get_instruction_count() - kernel_size_stamp;

  kernel.add({
    // next batch iteration the matrix a and b need to offset about the batch_stride_a (x6) and batch_stride_b (x7)
    // the matrix c need to start at the initial location again
    // Update matrix a
    add(x10, x10, x6),  // add x10, x10, x6 // Offset to next matrix a in batch

    // Update matrix b
    add(x20, x20, x7),  // add x20, x20, x7 // Offset to next matrix b in batch

    // Loop back to batch dimension
    cbnz(x19, -jump_batch_loop * 4),  // cbnz x19, matmul_loop_batch_dimension
  });

  kernel.add({
    // ==============================================================================
    // No need to loop back to N as we currently processing the rest
    // ==============================================================================
    //     // next M iteration on the matrix b and matrix c, both need offset about 4*ldb/ldc values
    //     // also matrix a needs to start at the initial location again

    //     // Updates for the matrix b
    // madd(x9, x4, x12, x9),  //     madd x9, x4, x12, x9 // ldb * 4 + initial position

    //     // Updates for the matrix c
    // madd(x11, x5, x12, x11),  //     madd x11, x5, x12, x11 // ldc * 4 + initial position

    //     // Loop back to N
    // cbnz(x17, -jump_N_loop * 4),  //     cbnz x17, matmul_loop_over_N

    //     // Procedural Call Standard
    //     // restore callee-saved registers
    //     // ldp d14, d15, [sp], #16
    //     // ldp d12, d13, [sp], #16
    //     // ldp d10, d11, [sp], #16
    ldpPost(d8, d9, sp, 16),  //     ldp  d8,  d9, [sp], #16

    ldpPost(x27, x28, sp, 16),  //     // ldp x27, x28, [sp], #16
    //     // ldp x25, x26, [sp], #16
    ldpPost(x23, x24, sp, 16),  //     // ldp x23, x24, [sp], #16
    ldpPost(x21, x22, sp, 16),  //     // ldp x21, x22, [sp], #16
    ldpPost(x19, x20, sp, 16),  //     // ldp x19, x20, [sp], #16

    //     // restore frame pointer and link register
    //     // ldp fp, lr, [sp], #16

    ret()  //     ret
    //     .size matmul_64_48_64, (. - matmul_64_48_64)

  });

#ifdef SAVE_JITS_TO_FILE
  kernel.write("br_matmul_lt16_lt4nRest_k.bin");
#endif  // SAVE_JITS_TO_FILE
}