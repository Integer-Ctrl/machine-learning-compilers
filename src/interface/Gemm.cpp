#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/TensorOperation.h"
#include "TensorUtils.h"

mlc::Error mlc::gemm(const Tensor &input0, const Tensor &input1, Tensor &output)
{
  if (input0.dim_sizes.size() != 2 || input1.dim_sizes.size() != 2 || output.dim_sizes.size() != 2)
  {
    return {ErrorType::TensorExpected2DTensor, "GEMM requires input0 and input1 to be 2D tensors and output to be a 2D tensor."};
  }

  int64_t mSize = static_cast<int64_t>(input0.dim_sizes[1]);
  int64_t nSize = static_cast<int64_t>(input1.dim_sizes[0]);
  int64_t kSize = static_cast<int64_t>(input0.dim_sizes[0]);

  if (static_cast<int64_t>(output.dim_sizes[1]) != mSize)
  {
    return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same m dimension size as the input0."};
  }

  if (static_cast<int64_t>(output.dim_sizes[0]) != nSize)
  {
    return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same n dimension size as the input1."};
  }

  if (static_cast<int64_t>(input1.dim_sizes[1]) != kSize)
  {
    return {ErrorType::ExecuteWrongDimension, "Expected the input1 tensor to have the same k dimension size as the input0."};
  }

  mini_jit::TensorOperation op;
  mini_jit::TensorConfig config{
    mini_jit::TensorConfig::prim_t::none,                                                                                // first_touch
    mini_jit::TensorConfig::prim_t::gemm,                                                                                // main
    mini_jit::TensorConfig::prim_t::none,                                                                                // last touch
    {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},              // dim_types
    {mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
    {mSize, nSize, kSize},                                                                                               // dim_sizes
    {1, 0, mSize},                                                                                                       // strides_in0
    {0, kSize, 1},                                                                                                       // strides_in1
    {1, mSize, 0},                                                                                                       // strides_out
    mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
  };

  mini_jit::TensorOperation::error_t error = op.setup(config);
  mlc::ErrorType errorType = internal::convertTensorOperationError(error);
  if (errorType != mlc::ErrorType::None)
  {
    return {errorType, "Could not generate the kernels for the gemm operation."};
  }

  op.execute(input0.data, input1.data, output.data);
  return {ErrorType::None, "Success"};
}