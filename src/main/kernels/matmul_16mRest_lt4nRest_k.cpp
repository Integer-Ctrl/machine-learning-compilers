#include "matmul_16mRest_lt4nRest_k.h"
#include "../Kernel.h"
#include "../arm_instructions/arm_all.h"
#include "../release_assert.h"
#include "matmul_16mRest_4n_k.h"
#include "matmul_16m_lt4nRest_k.h"

void mini_jit::kernels::matmul_16mRest_lt4nRest_k(mini_jit::Kernel &kernel, const uint32_t m_loop_16, const uint32_t n_loop_4,
                                                  const uint32_t k_loop, const uint32_t m_loop_rest, const uint32_t n_loop_rest)
{
  using namespace mini_jit::arm_instructions;

  release_assert(m_loop_16 != 0, "Cannot proccess matrix with m loop of 0.");
  release_assert(k_loop != 0, "Cannot proccess matrix with k loop of 0.");
  release_assert(m_loop_rest != 0, "Cannot create a matrix with a rest of m equal to 0!");
  release_assert(m_loop_rest <= 15, "Cannot create a matrix with a rest of m larger than 15!");
  release_assert(n_loop_rest != 0, "Cannot create a matrix with a rest of n equal to 0!");
  release_assert(n_loop_rest <= 3, "Cannot create a matrix with a rest of n larger than 3!");

  // Idea: Division of the matrix into sub-matrices and calculated in the following order.
  // 1. matmul_lt16_4n_k is omitted if n is less than 4;
  //
  //                       N dimension
  // ←---------------------------------------------------→
  // ===================================================== ↑
  // |                        |                          | |
  // |                        | 2. matmul_16m_lt4nRest_k | |
  // | 1. matmul_16mRest_4n_k |                          | | M dimension
  // |                        |--------------------------| |
  // |                        | 3. Rest of m and n       | |
  // ===================================================== ↓

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
    //     // stp x19, x20, [sp, #-16]!
    //     // stp x21, x22, [sp, #-16]!
    //     // stp x23, x24, [sp, #-16]!
    //     // stp x25, x26, [sp, #-16]!
    //     // stp x27, x28, [sp, #-16]!

    stpPre(d8, d9, sp, -16),  //     stp  d8,  d9, [sp, #-16]!
    //     // stp d10, d11, [sp, #-16]!
    //     // stp d12, d13, [sp, #-16]!
    //     // stp d14, d15, [sp, #-16]!

    //     // Offset the used leading dimension by the size of floats
    lsl(x3, x3, 2),  //     lsl x3, x3, #2 // x3 * 4 = x3 * sizeof(float)
    lsl(x4, x4, 2),  //     lsl x4, x4, #2 // x4 * 4 = x4 * sizeof(float)
    lsl(x5, x5, 2),  //     lsl x5, x5, #2 // x5 * 4 = x5 * sizeof(float)

    mov(x6, x1),  //     mov x6, x1 // Store the initial value of x1, to be restored in the K loop iteration
    mov(x7, x2),  //     mov x7, x2 // Store the initial value of x2, to be restored in the K loop iteration

    mov(x8, x0),  //     mov x8, x0 // Store the initial value of x0, to be restored in the M loop iteration
    mov(x9, x1),  //     mov x9, x1 // Store the initial value of x1, to be restored in the M loop iteration

    mov(x10, x0),  //     mov x10, x0 // Store the initial value of x0, to be restored in the N loop iteration
    mov(x11, x2),  //     mov x11, x2 // Store the initial value of x2, to bes restored in the N loop iteration
    mov(x12, 4),   //     mov x12, #4 // hold the size of N that are processed in one loop, needed for offset calculation
  });

  // ========================================================================================
  // Calculate m + rest but n is multiple of 4
  // ========================================================================================
  if (n_loop_4 != 0)
  {
    matmul_16mRest_4n_k(kernel, m_loop_16, n_loop_4, k_loop, m_loop_rest, false);
  }

  // Offset to the next matrix block
  // Here we want to start with the initial m value but n should be offset by the already calculated amount.
  // we should be at the right position calculated through 16mRest_4n_k

  // ========================================================================================
  // Rest Calculation of n loop i.e. n < 4 but m > 16
  // ========================================================================================
  matmul_16m_lt4nRest_k(kernel, m_loop_16, 0, k_loop, n_loop_rest, false);

  // Now we want to make sure to not restore the position of the m as it is in the right position.
  // Therefore we should restore the register below the m_loop

  // ========================================================================================
  // Rest Calculation of m and n loop
  // ========================================================================================

  kernel.add({
    //     // Restore for the loop jumps
    //     // Update for the matrix a
    // mov(x8, x10),  //     mov x8, x10 // Update the restore register for x0 for the M loop

    //     // Updates for the matrix c
    // mov(x7, x11),  //     mov x7, x11 // Update the restore register of x2 for the K loop

    // =================================================================
    // We do not loop through M as we currently processing the rest of M
    // =================================================================
    // mov(x16, m_loop_16),  //     mov x16, #4 // x16 iterator for M loop
    // matmul_loop_over_M:
    // sub(x16, x16, 1),  //     sub x16, x16, #1

    //     // Restore for the loop jumps
    //     // Updates for the matrix c
    mov(x2, x7),  //     mov x2, x7 // also apply offset to x2

    //     // Updates for the matrix a
    mov(x0, x8),  //     mov x0, x8 // also apply offset to x0

    //     // Updates for the matrix b
    mov(x6, x9),  //     mov x6, x9 // Update the restore register for x1 for the K loop
    mov(x1, x9),  //     mov x1, x9 // Update the x1 register itself

  });

  std::vector<uint32_t> instruction1_fp_loads_less_than_4;
  std::vector<uint32_t> instruction2_fp_loads_less_than_4;
  std::vector<uint32_t> instruction3_fp_loads_less_than_4;
  std::vector<uint32_t> instructions_fp_load_column_of_matrix_a;
  const uint32_t m_loop_full_4s = m_loop_rest / 4;
  const uint32_t m_loop_remainder = m_loop_rest % 4;
  const uint32_t offset_from_full_loads = (m_loop_full_4s) * 4 * 4;  // count * number of floats * sizeof(float)

  // For usage reason see case 3 of the below swich statement
  const uint32_t revert_offset_x2 =
    offset_from_full_loads >= 4 ? sub(x2, x2, offset_from_full_loads - 4) : add(x2, x2, 4 - offset_from_full_loads);
  const uint32_t revert_offset_x0 =
    offset_from_full_loads >= 4 ? sub(x0, x0, offset_from_full_loads - 4) : add(x0, x0, 4 - offset_from_full_loads);

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
    // Idea load 4 float elements, but only output the three last
    // Note we need to be carful when we store back so that the already calculated 1. element does not get overwritten
    // Solution load single already calculated 1. element to unused register. Save the whole 4s of this calculation
    // Overwrite the first 1. element from the freshly load register.

    // ldrOffset cannot be used as it only supports positive and multiple of 16 numbers.
    instruction1_fp_loads_less_than_4 = {ldrPre(q28, x2, offset_from_full_loads - 4), revert_offset_x2};
    instruction2_fp_loads_less_than_4 = {ldrPre(q20, x2, offset_from_full_loads - 4), revert_offset_x2};
    instruction3_fp_loads_less_than_4 = {ldrPre(q24, x2, offset_from_full_loads - 4), revert_offset_x2};
    instructions_fp_load_column_of_matrix_a.push_back(ldrPre(q3, x0, offset_from_full_loads - 4));
    instructions_fp_load_column_of_matrix_a.push_back(revert_offset_x0);

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
    //     // offset x6 to the next element in the column
    add(x6, x6, 4),  //     add x6, x6, #4 // #4 = sizeof(float)

    //     // Restore x1 to be incremented again
    mov(x1, x6),  //     mov x1, x6

    //     // Loop back to K
    cbnz(x15, -jump_K_loop * 4),  //     cbnz x15, matmul_loop_over_K

    //     // Restore initial value of x2 that was changed by the loads
    mov(x2, x7),  //     mov x2, x7
  });

  std::vector<uint32_t> instruction1_fp_store_less_than_4;
  std::vector<uint32_t> instruction2_fp_store_less_than_4;
  std::vector<uint32_t> instruction3_fp_store_less_than_4;

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
      // ldrOffset cannot be used as it only supports positive and multiple of 16 numbers.
      instruction1_fp_store_less_than_4 = {ldrPre(s29, x2, offset_from_full_loads - 4), str(q28, x2),
                                           strPost(s29, x2, 4 - offset_from_full_loads)};
      instruction2_fp_store_less_than_4 = {ldrPre(s29, x2, offset_from_full_loads - 4), str(q20, x2),
                                           strPost(s29, x2, 4 - offset_from_full_loads)};
      instruction3_fp_store_less_than_4 = {ldrPre(s29, x2, offset_from_full_loads - 4), str(q24, x2),
                                           strPost(s29, x2, 4 - offset_from_full_loads)};
    }
    else
    {
      // ldrOffset cannot be used as it only supports positive and multiple of 16 numbers.
      instruction1_fp_store_less_than_4 = {strPre(q28, x2, offset_from_full_loads - 4), revert_offset_x2};
      instruction2_fp_store_less_than_4 = {strPre(q20, x2, offset_from_full_loads - 4), revert_offset_x2};
      instruction3_fp_store_less_than_4 = {strPre(q24, x2, offset_from_full_loads - 4), revert_offset_x2};
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
  // add(x7, x7, 16 * 4), //     add x7, x7, #16*4 // column height * sizeof(float)

  //     // Updates for the matrix a
  // add(x8, x8, 16 * 4), //     add x8, x8, #16*4 // column height * sizeof(float)

  //     // Loop back to M
  // cbnz(x16, -46 * 4), //     cbnz x16, matmul_loop_over_M

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

    //     // ldp x27, x28, [sp], #16
    //     // ldp x25, x26, [sp], #16
    //     // ldp x23, x24, [sp], #16
    //     // ldp x21, x22, [sp], #16
    //     // ldp x19, x20, [sp], #16

    //     // restore frame pointer and link register
    //     // ldp fp, lr, [sp], #16

    ret()  //     ret
    //     .size matmul_64_48_64, (. - matmul_64_48_64)

  });

#ifdef SAVE_JITS_TO_FILE
  kernel.write("matmul_16mRest_6nRest_k.bin");
#endif  // SAVE_JITS_TO_FILE
}