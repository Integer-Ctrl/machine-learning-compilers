#include "TensorOperation.h"
#include "release_assert.h"
#include <ranges>
#include <tuple>

bool mini_jit::TensorOperation::isUnary(prim_t prim)
{
  return prim == prim_t::copy || prim == prim_t::relu || prim == prim_t::zero;
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

bool mini_jit::TensorOperation::isValidPrimConfig(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec,
                                                  const std::span<const int64_t> &strides_in0, const std::span<const int64_t> &strides_out)
{
  int32_t indexM = findMatch(dim, exec, dim_t::m, exec_t::prim);
  int32_t indexN = findMatch(dim, exec, dim_t::n, exec_t::prim);
  if (indexM == -1 || indexN == -1)
  {
    return false;
  }

  if (!(isExpectedStride(1, indexM, strides_in0) && isExpectedStride(1, indexM, strides_out)))
  {
    return false;
  }

  // Search for new that fits the configuration, both should return -1
  indexM = findMatch(dim, exec, dim_t::m, exec_t::prim, indexM);
  indexN = findMatch(dim, exec, dim_t::n, exec_t::prim, indexN);
  if (indexM != -1 || indexN != -1)
  {
    return false;
  }

  return true;
}

bool mini_jit::TensorOperation::isValidKDim(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec,
                                            const std::span<const int64_t> &strides_in1, prim_t prim)
{
  if (isBrgemm(prim))
  {
    int32_t indexK = findMatch(dim, exec, dim_t::k, exec_t::prim);

    if (indexK == -1)
    {
      return false;
    }

    if (!isExpectedStride(1, indexK, strides_in1))
    {
      return false;
    }

    if (prim == prim_t::brgemm)
    {
      // Another k dim should exists
      indexK = findMatch(dim, exec, dim_t::k, exec_t::prim, indexK);

      if (indexK == -1)
      {
        return false;
      }
    }

    // No other k dim should exists
    indexK = findMatch(dim, exec, dim_t::k, exec_t::prim, indexK);
    return indexK == -1;
  }
  else if (isUnary(prim))
  {
    // Expected to find not K dim
    int32_t indexK = findMatch(dim, exec, dim_t::k, exec_t::prim);

    return indexK == -1;
  }
  else
  {
    return false;
  }
}
bool mini_jit::TensorOperation::isSortedConfiguration(const std::span<const exec_t> &exec)
{
  bool foundPrimitive = false;
  for (auto type = exec.begin(); type != exec.end(); ++type)
  {
    if (!foundPrimitive && *type == exec_t::prim)
    {
      foundPrimitive = true;
      indexPrimitiveLoop = std::distance(type, exec.begin());
    }

    if (foundPrimitive && *type != exec_t::prim)
    {
      return false;
    }
  }

  return true;
}

bool mini_jit::TensorOperation::isExpectedStride(int64_t expected, int index, const std::span<const int64_t> &strides)
{
  if (index == -1)
  {
    return false;
  }

  return strides[index] == expected;
}

mini_jit::Unary::error_t mini_jit::TensorOperation::generateUnary(Unary &unary, prim_t prim, const std::span<const dim_t> &dim_types,
                                                                  const std::span<const exec_t> &exec_types,
                                                                  const std::span<const int64_t> &dim_sizes)
{
  release_assert(indexPrimM != -1, "Expected a match for the m primitive dimension");
  release_assert(indexPrimN != -1, "Expected a match for the n primitive dimension");

  Unary::ptype_t type;
  switch (prim)
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
  return unary.generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], 0, Unary::dtype_t::fp32, type);
}

mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup(dtype_t dtype, prim_t prim_first_touch, prim_t prim_main,
                                                                    prim_t prim_last_touch, std::span<const dim_t> dim_types,
                                                                    std::span<const exec_t> exec_types, std::span<const int64_t> dim_sizes,
                                                                    std::span<const int64_t> strides_in0,
                                                                    std::span<const int64_t> strides_in1,
                                                                    std::span<const int64_t> strides_out)
{
  indexPrimBatch = -1;
  indexPrimK = -1;
  indexPrimM = -1;
  indexPrimN = -1;
  indexPrimitiveLoop = -1;

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

  if (!isSortedConfiguration(exec_types))
  {
    return error_t::err_invalid_execution_order;
  }

  if (!isValidPrimConfig(dim_types, exec_types, strides_in0, strides_out))
  {
    return error_t::err_invalid_primitive_configuration;
  }

  if (!isValidKDim(dim_types, exec_types, strides_in1, prim_main))
  {
    return error_t::err_invalid_primitive_configuration;
  }

  // Validated through isValidPrimConfig that these indices exists
  indexPrimM = findMatch(dim_types, exec_types, dim_t::m, exec_t::prim);
  indexPrimN = findMatch(dim_types, exec_types, dim_t::n, exec_t::prim);

  release_assert(indexPrimM != -1, "Expected a valid index for the M dimension but found none.");
  release_assert(indexPrimN != -1, "Expected a valid index for the N dimension but found none.");
  release_assert(indexPrimitiveLoop != -1, "Expected a valid start of the primitive loop but found none.");

  if (prim_first_touch != prim_t::none)
  {
    if (isUnary(prim_first_touch))
    {
      first_touch.emplace<Unary>();
      TensorOperation::prim_first = prim_first_touch;

      Unary::error_t error = generateUnary(std::get<Unary>(first_touch), prim_first_touch, dim_types, exec_types, dim_sizes);

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
      main_kernel.emplace<Brgemm>();
      TensorOperation::prim_main = prim_main;

      if (prim_main == prim_t::brgemm)
      {
        indexPrimBatch = findMatch(dim_types, exec_types, dim_t::k, exec_t::prim);
        indexPrimK = findMatch(dim_types, exec_types, dim_t::k, exec_t::prim, indexPrimBatch);

        release_assert(indexPrimBatch != -1, "Expected a valid index for the Batch dimension but found none.");
        release_assert(indexPrimK != -1, "Expected a valid index for the Batch dimension but found none.");

        std::get<Brgemm>(main_kernel)
          .generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], dim_sizes[indexPrimK], dim_sizes[indexPrimBatch], 0, 0, 0,
                    Brgemm::dtype_t::fp32);
      }
      else if (prim_main == prim_t::gemm)
      {
        indexPrimK = findMatch(dim_types, exec_types, dim_t::k, exec_t::prim);

        release_assert(indexPrimK != -1, "Expected a valid index for the K dimension but found none.");

        std::get<Brgemm>(main_kernel)
          .generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], dim_sizes[indexPrimK], 1, 0, 0, 0, Brgemm::dtype_t::fp32);
      }
      else
      {
        release_assert(false, "Found missing brgemm configuration.");
      }
    }
    else if (isUnary(prim_main))
    {
      main_kernel.emplace<Unary>();

      Unary::error_t error = generateUnary(std::get<Unary>(main_kernel), prim_main, dim_types, exec_types, dim_sizes);

      if (error != Unary::error_t::success)
      {
        return error_t::err_invalid_main_configuration;
      }
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
      last_touch.emplace<Unary>();
      TensorOperation::prim_last = prim_last_touch;

      Unary::error_t error = generateUnary(std::get<Unary>(last_touch), prim_last_touch, dim_types, exec_types, dim_sizes);

      if (error != Unary::error_t::success)
      {
        return error_t::err_invalid_main_configuration;
      }
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
  release_assert(tensor_in0 != nullptr, "The tensor_in0 parameter is a nullptr, but should be a valid pointer to memory.");
  release_assert(tensor_out != nullptr, "The tensor_out parameter is a nullptr, but should be a valid pointer to memory.");

  char const *ptr_in0 = static_cast<char const *>(tensor_in0);
  char const *ptr_in1 = static_cast<char const *>(tensor_in1);
  char *ptr_out = static_cast<char *>(tensor_out);

  execute_dimension(0, ptr_in0, ptr_in1, ptr_out, false, false);
}

void mini_jit::TensorOperation::execute_dimension(int64_t index_dim, char const *ptr_in0, char const *ptr_in1, char *ptr_out,
                                                  bool first_access, bool last_access)
{
  release_assert(exec_types[index_dim] != exec_t::seq, "Expected a sequential loop");

  uint32_t dtype_bytes = 4;
  int64_t dim_size = dim_sizes[index_dim];
  int64_t stride_in0 = strides_in0[index_dim];
  int64_t stride_in1 = strides_in1[index_dim];
  int64_t stride_out = strides_out[index_dim];

  for (int64_t iDim = 0; iDim < dim_size; iDim++)
  {
    // TODO derive if this is first or last access to the output block
    // first_access = first_access || (index_dim == 0);
    // last_access = last_access || (index_dim == dim_size - 1);

    char const *rec_ptr_in0 = ptr_in0 + iDim * stride_in0 * dtype_bytes;
    char const *rec_ptr_in1 = ptr_in1 + iDim * stride_in1 * dtype_bytes;
    char *rec_ptr_out = ptr_out + iDim * stride_out * dtype_bytes;

    if (index_dim + 1 < indexPrimitiveLoop)
    {
      execute_dimension(index_dim + 1, rec_ptr_in0, rec_ptr_in1, rec_ptr_out, first_access, last_access);
    }
    else
    {
      // call first touch kernel if necessary
      if (prim_first != prim_t::none)
      {
        if (std::holds_alternative<Unary>(first_touch))
        {
          Unary::kernel_t kernel = std::get<Unary>(first_touch).get_kernel();
          kernel(rec_ptr_in0, rec_ptr_out, strides_in0[indexPrimN], strides_out[indexPrimN]);
        }
        else
        {
          release_assert(false, "Unexpected first touch primitive");
        }
      }

      // call main_kernel kernel
      if (prim_main != prim_t::none)
      {
        if (std::holds_alternative<Unary>(main_kernel))
        {
          Unary::kernel_t kernel = std::get<Unary>(main_kernel).get_kernel();
          kernel(rec_ptr_in0, rec_ptr_out, strides_in0[indexPrimN], strides_out[indexPrimN]);
        }
        else if (std::holds_alternative<Brgemm>(main_kernel))
        {
          Brgemm::kernel_t kernel = std::get<Brgemm>(main_kernel).get_kernel();
          kernel(rec_ptr_in0, rec_ptr_in1, rec_ptr_out, strides_in0[indexPrimK], strides_in1[indexPrimN], strides_out[indexPrimN],
                 strides_in0[indexPrimBatch], strides_in1[indexPrimBatch]);
        }
        else
        {
          release_assert(false, "Unexpected main primitive");
        }
      }

      // call last touch kernel if necessary
      if (prim_last != prim_t::none)
      {
        if (std::holds_alternative<Unary>(last_touch))
        {
          Unary::kernel_t kernel = std::get<Unary>(last_touch).get_kernel();
          kernel(rec_ptr_in0, rec_ptr_out, strides_in0[indexPrimN], strides_out[indexPrimN]);
        }
        else
        {
          release_assert(false, "Unexpected last touch primitive");
        }
      }
    }
  }
}