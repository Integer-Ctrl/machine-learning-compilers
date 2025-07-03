#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/TensorOperation.h"
#include "TensorUtils.h"

mlc::Error mlc::unary_zero(Tensor &input)
{
  int64_t stride = 1;
  std::vector<int64_t> dimSizes(input.dim_sizes.size());
  std::vector<int64_t> strides(input.dim_sizes.size());

  for (int64_t i = input.dim_sizes.size() - 1; i >= 0; i--)
  {
    strides[i] = stride;
    dimSizes[i] = static_cast<int64_t>(input.dim_sizes[i]);
    stride *= input.dim_sizes[i];
  }

  mini_jit::TensorOperation op;
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,                                      // first_touch
    mini_jit::TensorConfig::prim_t::zero,                                      // main
    mini_jit::TensorConfig::prim_t::none,                                      // last touch
    std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::dim_t::c),     // dim_types
    std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::exec_t::seq),  // exec_types
    dimSizes,                                                                  // dim_sizes
    strides,                                                                   // strides_in0
    std::vector<int64_t>(input.dim_sizes.size(), 0),                           // strides_in1
    strides,                                                                   // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                     // dtype_t
  };

  mini_jit::TensorOperation::error_t error = op.setup(config);
  mlc::ErrorType errorType = internal::convertTensorOperationError(error);
  if (errorType != mlc::ErrorType::None)
  {
    return {errorType, "Could not generate the kernels for the gemm operation."};
  }

  op.execute(input.data, nullptr, input.data);
  return {ErrorType::None, "Success"};
}

mlc::Error mlc::unary_relu(const Tensor &input, Tensor &output)
{
  if (output.dim_sizes.size() != input.dim_sizes.size())
  {
    return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same number of dimension as the input."};
  }

  for (size_t i = 0; i < input.dim_sizes.size(); i++)
  {
    if (output.dim_sizes[i] != input.dim_sizes[i])
    {
      return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same number of dimension as the input."};
    }
  }

  int64_t stride = 1;
  std::vector<int64_t> dimSizes(input.dim_sizes.size());
  std::vector<int64_t> strides(input.dim_sizes.size());

  for (int64_t i = input.dim_sizes.size() - 1; i >= 0; i--)
  {
    strides[i] = stride;
    dimSizes[i] = static_cast<int64_t>(input.dim_sizes[i]);
    stride *= input.dim_sizes[i];
  }

  mini_jit::TensorOperation op;
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,                                      // first_touch
    mini_jit::TensorConfig::prim_t::relu,                                      // main
    mini_jit::TensorConfig::prim_t::none,                                      // last touch
    std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::dim_t::c),     // dim_types
    std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::exec_t::seq),  // exec_types
    dimSizes,                                                                  // dim_sizes
    strides,                                                                   // strides_in0
    std::vector<int64_t>(input.dim_sizes.size(), 0),                           // strides_in1
    strides,                                                                   // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                     // dtype_t
  };

  mini_jit::TensorOperation::error_t error = op.setup(config);
  mlc::ErrorType errorType = internal::convertTensorOperationError(error);
  if (errorType != mlc::ErrorType::None)
  {
    return {errorType, "Could not generate the kernels for the gemm operation."};
  }

  op.execute(input.data, nullptr, output.data);
  return {ErrorType::None, "Success"};
}

mlc::Error mlc::unary_identity(const Tensor &input, Tensor &output)
{
  if (output.dim_sizes.size() != input.dim_sizes.size())
  {
    return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same number of dimension as the input."};
  }

  for (size_t i = 0; i < input.dim_sizes.size(); i++)
  {
    if (output.dim_sizes[i] != input.dim_sizes[i])
    {
      return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same number of dimension as the input."};
    }
  }

  int64_t stride = 1;
  std::vector<int64_t> dimSizes(input.dim_sizes.size());
  std::vector<int64_t> strides(input.dim_sizes.size());

  for (int64_t i = input.dim_sizes.size() - 1; i >= 0; i--)
  {
    strides[i] = stride;
    dimSizes[i] = static_cast<int64_t>(input.dim_sizes[i]);
    stride *= input.dim_sizes[i];
  }

  mini_jit::TensorOperation op;
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,                                      // first_touch
    mini_jit::TensorConfig::prim_t::copy,                                      // main
    mini_jit::TensorConfig::prim_t::none,                                      // last touch
    std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::dim_t::c),     // dim_types
    std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::exec_t::seq),  // exec_types
    dimSizes,                                                                  // dim_sizes
    strides,                                                                   // strides_in0
    std::vector<int64_t>(input.dim_sizes.size(), 0),                           // strides_in1
    strides,                                                                   // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                     // dtype_t
  };

  mini_jit::TensorOperation::error_t error = op.setup(config);
  mlc::ErrorType errorType = internal::convertTensorOperationError(error);
  if (errorType != mlc::ErrorType::None)
  {
    return {errorType, "Could not generate the kernels for the gemm operation."};
  }

  op.execute(input.data, nullptr, output.data);
  return {ErrorType::None, "Success"};
}