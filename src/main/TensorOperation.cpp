#include "TensorOperation.h"

mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup(dtype_t dtype, prim_t prim_first_touch, prim_t prim_main,
                                                                    prim_t prim_last_touch, std::span<const dim_t> dim_types,
                                                                    std::span<const exec_t> exec_types, std::span<const int64_t> dim_sizes,
                                                                    std::span<const int64_t> strides_in0,
                                                                    std::span<const int64_t> strides_in1,
                                                                    std::span<const int64_t> strides_out)
{
  // Validate dimensions
  if (dim_sizes.size() != dim_types.size() || dim_sizes.empty() || dim_types.empty())
  {
    return error_t::err_wrong_dimension;
  }
  if (strides_in0.size() != dim_sizes.size() || strides_out.size() != dim_sizes.size() ||
      (strides_in1.size() != dim_sizes.size() && strides_in1.size() != 0))  // strides_in1 can be empty for unary operations
  {
    return error_t::err_wrong_dimension;  // Strides must match the number of dimensions
  }

  // Validate dtype types - currently only fp32 is supported
  if (dtype != dtype_t::fp32)
  {
    return error_t::err_wrong_dtype;
  }

  mini_jit::Unary l_unary_first_touch;
  mini_jit::Brgemm l_brgemm_main;
  mini_jit::Unary l_unary_last_touch;

  if (prim_first_touch != prim_t::none)
  {
    // TODO: call generate for the first touch primitive
  }
  if (prim_main != prim_t::none)
  {
    if (prim_main == prim_t::brgemm || prim_main == prim_t::gemm)
    {
      // TODO: call generate for the brgemm primitive
    }
    else
    {
      // TODO: call generate for the unary primitive
    }
  }
  if (prim_last_touch != prim_t::none)
  {
    // TODO: call generate for the last touch primitive
  }

  return error_t::success;
}

void mini_jit::TensorOperation::execute(void const *tensor_in0, void const *tensor_in1, void *tensor_out)
{
}

void mini_jit::TensorOperation::execute_iter(int64_t id_loop, char const *ptr_in0, char const *ptr_in1, char *ptr_out, bool first_access,
                                             bool last_access)
{
}
