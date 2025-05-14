#include "Brgemm.h"
#include "Kernel.h"
#include "kernels/matmuls_all.h"
#include <format>
#include <stdexcept>

mini_jit::Brgemm::error_t mini_jit::Brgemm::generate(uint32_t m, uint32_t n, uint32_t k, uint32_t br_size, uint32_t trans_a,
                                                     uint32_t trans_b, uint32_t trans_c, dtype_t dtype)
{
  if (dtype != dtype_t::fp32)
  {
    return error_t::err_wrong_dtype;
  }
  if (m == 0 || n == 0 || k == 0)
  {
    return error_t::err_wrong_dimension;
  }
  if ((trans_a + trans_b + trans_c) != 0)
  {
    return error_t::err_row_major_order_not_supported;
  }
  if (br_size != 1)
  {
    return error_t::err_batch_reduce_size_not_supported;
  }
  if (br_size == 1 && (trans_a + trans_b + trans_c) == 0 && dtype == dtype_t::fp32)
  {
    fill_with_matmuls_no_batch_dim_column_major_fp32(m, n, k);
  }
  if (br_size > 1 && (trans_a + trans_b + trans_c) == 0 && dtype == dtype_t::fp32)
  {
    fill_with_matmuls_batch_dim_column_major_fp32(m, n, k, br_size);
  }
  else
  {
    throw std::logic_error(
      std::format("Unhandled parameter combination found: m='{}', n='{}', k='{}', br_size='{}', trans_a='{}', trans_b='{}', "
                  "trans_c = '{}', dtype = '{}'",
                  m, n, k, br_size, trans_a, trans_b, trans_c, static_cast<int32_t>(dtype)));
  }

  native_kernel.set_kernel();
  kernel = reinterpret_cast<kernel_t>(const_cast<void *>(native_kernel.get_kernel()));  // Properly cast from const void* to kernel_t

  return error_t::success;
}

mini_jit::Brgemm::kernel_t mini_jit::Brgemm::get_kernel() const
{
  return kernel;
}

void mini_jit::Brgemm::fill_with_matmuls_no_batch_dim_column_major_fp32(uint32_t m, uint32_t n, uint32_t k)
{
  // Always sort from the specific to the more general case

  if (m == 16 && n == 6 && k == 1)
  {
    kernels::matmul_16_6_1(native_kernel);
    return;
  }

  if (m == 16 && n == 6)
  {
    kernels::matmul_16_6_k(native_kernel, k);
    return;
  }

  if (m >= 16 && m % 16 == 0 && n >= 4 && n % 4 == 0)
  {
    kernels::matmul_16m_4n_k(native_kernel, m / 16, n / 4, k);
    return;
  }

  if (m >= 16 && m % 16 == 0)
  {
    // At this point n % 4 != 0
    kernels::matmul_16m_lt4nRest_k(native_kernel, m / 16, n / 4, k, n % 4);
    return;
  }

  if (m >= 16 && n >= 4 && n % 4 == 0)
  {
    // At this point m % 16 != 0
    kernels::matmul_16mRest_4n_k(native_kernel, m / 16, n / 4, k, m % 16);
    return;
  }

  if (m < 16 && n >= 4 && n % 4 == 0)
  {
    kernels::matmul_lt16_4n_k(native_kernel, n / 4, k, m % 16);
    return;
  }

  if (m >= 16)
  {
    // At this point m % 16 != 0 and n % 4 != 0
    kernels::matmul_16mRest_lt4nRest_k(native_kernel, m / 16, n / 4, k, m % 16, n % 4);
    return;
  }

  if (m < 16)
  {
    // At this point m % 16 != 0 and n % 4 != 0
    kernels::matmul_lt16_lt4nRest_k(native_kernel, n / 4, k, m % 16, n % 4);
    return;
  }

  throw std::logic_error(std::format("Unhandled combination found for MxNxK matmul: m='{}', n='{}', k='{}'", m, n, k));
}

void mini_jit::Brgemm::fill_with_matmuls_batch_dim_column_major_fp32(uint32_t m, uint32_t n, uint32_t k, uint32_t br_size)
{
  // Always sort from the specific to the more general case

  if (m >= 16 && m % 16 == 0 && n >= 4 && n % 4 == 0)
  {
    kernels::br_matmul_16m_4n_k(native_kernel, m / 16, n / 4, k, br_size);
    return;
  }

  if (m >= 16 && m % 16 == 0)
  {
    // At this point n % 4 != 0
    kernels::br_matmul_16m_lt4nRest_k(native_kernel, m / 16, n / 4, k, br_size, n % 4);
    return;
  }

  if (m >= 16 && n >= 4 && n % 4 == 0)
  {
    // At this point m % 16 != 0
    kernels::br_matmul_16mRest_4n_k(native_kernel, m / 16, n / 4, k, br_size, m % 16);
    return;
  }

  if (m < 16 && n >= 4 && n % 4 == 0)
  {
    kernels::br_matmul_lt16_4n_k(native_kernel, n / 4, k, br_size, m % 16);
    return;
  }

  if (m >= 16)
  {
    // At this point m % 16 != 0 and n % 4 != 0
    kernels::br_matmul_16mRest_lt4nRest_k(native_kernel, m / 16, n / 4, k, br_size, m % 16, n % 4);
    return;
  }

  if (m < 16)
  {
    // At this point m % 16 != 0 and n % 4 != 0
    kernels::br_matmul_lt16_lt4nRest_k(native_kernel, n / 4, k, br_size, m % 16, n % 4);
    return;
  }
}