#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include "../main/TensorOperation.h"
#include "TensorUtils.h"
#include <iostream>

void mlc::fill_random(Tensor &tensor)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  uint64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; ++i)
  {
    float denominator = 1;
    denominator = static_cast<float>(std::rand());
    if (denominator == 0)
    {
      denominator = 1;
    }

    float numerator = 1;
    numerator = static_cast<float>(std::rand());

    float random = numerator / denominator;

    tensor.data[i] = random;
  }
}

void mlc::fill_number(Tensor &tensor, float number)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  uint64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++)
  {
    tensor.data[i] = number;
  }
}

void mlc::fill_lambda(Tensor &tensor, std::function<float(const Tensor &, size_t)> function)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  uint64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++)
  {
    tensor.data[i] = function(tensor, i);
  }
}

mlc::Error mlc::einsum(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output, const std::string &tree)
{
  return internal::einsum<std::reference_wrapper<const Tensor>>(inputs, output, tree);
}

mlc::Error mlc::einsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree)
{
  return internal::einsum<Tensor *>(inputs, output, tree);
}

mlc::Error mlc::contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction)
{
  return internal::einsum<std::reference_wrapper<const Tensor>>({input0, input1}, output, contraction);
}

mlc::Error mlc::contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction,
                            const UnaryType firstTouch, const UnaryType lastTouch)
{
  mini_jit::EinsumTree einsumTree(contraction);
  mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
  if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
  {
    mlc::ErrorType type = internal::convertParseError(errorParse);
    return {type, "Failed during parsing the given einsum tree."};
  }
  if (einsumTree.get_root()->left->type != mini_jit::EinsumTree::NodeType::Leaf ||
      einsumTree.get_root()->right->type != mini_jit::EinsumTree::NodeType::Leaf)
  {
    return {mlc::ErrorType::ExpectedSingleContraction, "Expected the given einsum string to be a single string."};
  }

  std::vector<int64_t> sorted_dim_sizes;
  internal::get_sorted_dimensions_sizes(einsumTree.get_root(), {input0, input1}, sorted_dim_sizes);
  einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

  mini_jit::TensorOperation op;
  mini_jit::TensorConfig config = einsumTree.lower_node(einsumTree.get_root());
  config.first_touch = internal::convertPrimitiveType(firstTouch);
  config.last_touch = internal::convertPrimitiveType(lastTouch);

  mini_jit::TensorOperation::error_t error = op.setup(config);
  mlc::ErrorType errorType = internal::convertTensorOperationError(error);
  if (errorType != mlc::ErrorType::None)
  {
    return {errorType, "Could not generate the kernels for the gemm operation."};
  }

  op.execute(input0.data, input1.data, output.data);
  return {ErrorType::None, "Success"};
}

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
