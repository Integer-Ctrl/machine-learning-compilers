#include "TensorOperation.h"

mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup(dtype_t dtype, prim_t prim_first_touch, prim_t prim_main,
                                                                    prim_t prim_last_touch, std::span<const dim_t> dim_types,
                                                                    std::span<const exec_t> exec_types, std::span<const int64_t> dim_sizes,
                                                                    std::span<const int64_t> strides_in0,
                                                                    std::span<const int64_t> strides_in1,
                                                                    std::span<const int64_t> strides_out)
{
  return error_t::success;
}

void mini_jit::TensorOperation::execute(void const *tensor_in0, void const *tensor_in1, void *tensor_out)
{
}

void mini_jit::TensorOperation::execute_iter(int64_t id_loop, char const *ptr_in0, char const *ptr_in1, char *ptr_out, bool first_access,
                                             bool last_access)
{
}
