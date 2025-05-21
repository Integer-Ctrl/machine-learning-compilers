#include "Unary.h"
#include "kernels/unary/unary_all.h"
#include "release_assert.h"
#include <format>

mini_jit::Unary::error_t mini_jit::Unary::generate(uint32_t m, uint32_t n, uint32_t trans_b, dtype_t dtype, ptype_t ptype)
{
  if (dtype != dtype_t::fp32)
  {
    return error_t::err_wrong_dtype;
  }
  if (m == 0 || n == 0)
  {
    return error_t::err_wrong_dimension;
  }

  switch (ptype)
  {
  case ptype_t::zero:
    if (trans_b == 0)  // Column major format
    {
      fill_with_zero_unary_column_major_fp32(m, n);
    }
    else if (trans_b == 1)  // Row major format
    {
      fill_with_zero_unary_column_major_fp32(n, m);
    }
    else
    {
      throw std::logic_error(std::format("Unhandled parameter combination found: m='{}', n='{}', trans_b='{}', dtype = '{}', ptype = '{}'",
                                         m, n, trans_b, static_cast<int32_t>(dtype), static_cast<int32_t>(ptype)));
    }
    break;

  case ptype_t::identity:
    throw std::logic_error(std::format("Unhandled parameter combination found: m='{}', n='{}', trans_b='{}', dtype = '{}', ptype = '{}'", m,
                                       n, trans_b, static_cast<int32_t>(dtype), static_cast<int32_t>(ptype)));
    break;

  case ptype_t::relu:
    throw std::logic_error(std::format("Unhandled parameter combination found: m='{}', n='{}', trans_b='{}', dtype = '{}', ptype = '{}'", m,
                                       n, trans_b, static_cast<int32_t>(dtype), static_cast<int32_t>(ptype)));
    break;

  default:
    release_assert(false, "Found unhandled ptype_t");
    break;
  }

  native_kernel.set_kernel();
  kernel = reinterpret_cast<kernel_t>(const_cast<void *>(native_kernel.get_kernel()));

  return error_t::success;
}

void mini_jit::Unary::fill_with_zero_unary_column_major_fp32(uint32_t m, uint32_t n)
{
  if (m == 16 && n == 4)
  {
  }
}