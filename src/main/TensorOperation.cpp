#include "TensorOperation.h"
#include "release_assert.h"
#include <ranges>
#include <tuple>

mini_jit::TensorOperation::~TensorOperation()
{
  cleanup();
}

void mini_jit::TensorOperation::cleanup()
{
  if (isUnary(prim_first))
  {
    delete first_touch.unary;
    prim_first = prim_t::none;
  }

  if (isUnary(prim_main))
  {
    delete main.unary;
    prim_main = prim_t::none;
  }
  else if (isBrgemm(prim_main))
  {
    delete main.brgemm;
    prim_main = prim_t::none;
  }

  if (isUnary(prim_last))
  {
    delete last_touch.unary;
    prim_last = prim_t::none;
  }

  release_assert(prim_first == prim_t::none, "Expected prim_first to be none after cleanup.");
  release_assert(prim_main == prim_t::none, "Expected prim_main to be none after cleanup.");
  release_assert(prim_last == prim_t::none, "Expected prim_last to be none after cleanup.");
}

bool mini_jit::TensorOperation::isUnary(prim_t prim)
{
  return prim == prim_t::copy || prim == prim_t::relu || prim == prim_t::relu;
}

bool mini_jit::TensorOperation::isBrgemm(prim_t prim)
{
  return prim == prim_t::brgemm || prim == prim_t::gemm;
}

int32_t mini_jit::TensorOperation::findMatch(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec, dim_t searchDim,
                                             exec_t searchExec, uint32_t startIndex)
{
  release_assert(dim.size() == exec.size(), "Expected the dimension types size to match the execution types size.");
  release_assert(startIndex < dim.size(), "Expected the start index to be less than the dimension types size.");

  for (auto [iDim, iExec] = std::tuple{dim.begin() + startIndex, exec.begin() + startIndex}; iDim != dim.end(); ++iDim, ++iExec)
  {
    if (*iDim == searchDim && *iExec == searchExec)
    {
      return std::distance(iDim, dim.begin());
    }
  }

  return -1;
}

bool mini_jit::TensorOperation::isValidPrimConfig(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec)
{
  int32_t indexM = findMatch(dim, exec, dim_t::m, exec_t::prim);
  int32_t indexN = findMatch(dim, exec, dim_t::n, exec_t::prim);
  if (indexM == -1 || indexN == -1)
  {
    return false;
  }

  // Search for new that fits the configuration, both should return -1
  int32_t indexM = findMatch(dim, exec, dim_t::m, exec_t::prim);
  int32_t indexN = findMatch(dim, exec, dim_t::n, exec_t::prim);
  if (indexM != -1 || indexN != -1)
  {
    return false;
  }

  return true;
}

mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup(dtype_t dtype, prim_t prim_first_touch, prim_t prim_main,
                                                                    prim_t prim_last_touch, std::span<const dim_t> dim_types,
                                                                    std::span<const exec_t> exec_types, std::span<const int64_t> dim_sizes,
                                                                    std::span<const int64_t> strides_in0,
                                                                    std::span<const int64_t> strides_in1,
                                                                    std::span<const int64_t> strides_out)
{
  // clear all old used resources
  cleanup();

  TensorOperation::prim_first = prim_t::none;  // Not yet generated, correctness of cleanup
  TensorOperation::prim_main = prim_t::none;   // Not yet generated, correctness of cleanup
  TensorOperation::prim_last = prim_t::none;   // Not yet generated, correctness of cleanup

  // Validate dimensions
  if (dim_sizes.size() != dim_types.size() || dim_sizes.empty() || dim_types.empty())
  {
    return error_t::err_wrong_dimension;
  }

  if (!(strides_in0.size() == dim_sizes.size() && strides_out.size() == dim_sizes.size() &&
        (strides_in1.size() == dim_sizes.size()
         // strides_in1 can be empty for unary operations
         || ((isUnary(prim_first_touch) || prim_first_touch == prim_t::none) && (isUnary(prim_main) || prim_main == prim_t::none) &&
             (isUnary(prim_last_touch) || prim_last_touch == prim_t::none) && strides_in1.empty()))))
  {
    return error_t::err_wrong_dimension;  // Strides must match the number of dimensions
  }

  for (exec_t exec : exec_types)
  {
    if (exec == exec_t::shared)
    {
      return error_t::err_execution_type_not_supported;
    }
  }

  // Validate dtype types - currently only fp32 is supported
  if (dtype != dtype_t::fp32)
  {
    return error_t::err_wrong_dtype;
  }

  if (!isValidPrimConfig(dim_types, exec_types))
  {
    return error_t::err_invalid_primitive_configuration;
  }

  if (prim_first_touch != prim_t::none)
  {
    if (prim_first_touch == prim_t::zero || prim_first_touch == prim_t::copy || prim_first_touch == prim_t::relu)
    {
      int32_t indexM = findMatch(dim_types, exec_types, dim_t::m, exec_t::prim);
      int32_t indexN = findMatch(dim_types, exec_types, dim_t::n, exec_t::prim);

      release_assert(indexM != -1, "Expected a match for the m primitive dimension");
      release_assert(indexN != -1, "Expected a match for the n primitive dimension");

      Unary::ptype_t type;
      switch (prim_first_touch)
      {
      case prim_t::zero:
        type = Unary::ptype_t::zero;
        break;

      case prim_t::copy:
        type = Unary::ptype_t::identity;
        break;

      case prim_t::relu:
        type = Unary::ptype_t::relu;
        break;

      default:
        release_assert(false, "Found a invalid type for the unary first touch.");
        break;
      }

      first_touch.unary = new Unary();
      TensorOperation::prim_first = prim_first_touch;

      // TODO find out what to input into the trans_b parameter
      Unary::error_t error = first_touch.unary->generate(dim_sizes[indexM], dim_sizes[indexN], 0, Unary::dtype_t::fp32, type);

      if (error != Unary::error_t::success)
      {
        return error_t::err_invalid_first_touch_configuration;
      }
    }
    else
    {
      return error_t::err_wrong_first_touch_primitive;
    }
  }

  if (prim_main != prim_t::none)
  {
    if (isBrgemm(prim_main))
    {
      // TODO: call generate for the brgemm primitive

      TensorOperation::prim_main = prim_main;
    }
    else if (isUnary(prim_main))
    {
      // TODO: call generate for the unary primitive

      TensorOperation::prim_main = prim_main;
    }
    else
    {
      return error_t::err_wrong_main_primitive;
    }
  }

  if (prim_last_touch != prim_t::none)
  {
    if (isUnary(prim_last_touch))
    {
      // TODO: call generate for the unary primitive

      TensorOperation::prim_last = prim_last_touch;
    }
    else
    {
      return error_t::err_wrong_last_touch_primitive;
    }
  }

  TensorOperation::dtype = dtype;
  TensorOperation::dim_types = dim_types;
  TensorOperation::exec_types = exec_types;
  TensorOperation::dim_sizes = dim_sizes;
  TensorOperation::strides_in0 = strides_in0;
  TensorOperation::strides_in1 = strides_in1;
  TensorOperation::strides_out = strides_out;

  return error_t::success;
}

void mini_jit::TensorOperation::execute(void const *tensor_in0, void const *tensor_in1, void *tensor_out)
{
}

void mini_jit::TensorOperation::execute_iter(int64_t id_loop, char const *ptr_in0, char const *ptr_in1, char *ptr_out, bool first_access,
                                             bool last_access)
{
}
