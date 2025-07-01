#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include "TensorUtils.h"
#include <iostream>

void mlc::fill_random(Tensor &tensor)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  uint64_t size = getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for simd
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

  uint64_t size = getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for simd
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

  uint64_t size = getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for simd
#endif
  for (size_t i = 0; i < size; i++)
  {
    tensor.data[i] = function(tensor, i);
  }
}

mlc::Error mlc::einsum(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output, const std::string &tree)
{
  return einsum<std::reference_wrapper<const Tensor>>(inputs, output, tree);
}

mlc::Error mlc::einsum(const std::vector<const Tensor *> &inputs, Tensor &output, const std::string &tree)
{
  return einsum<const Tensor *>(inputs, output, tree);
}

mlc::Error mlc::contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction)
{
  return einsum<const Tensor *>({&input0, &input1}, output, contraction);
}

mlc::Error mlc::unary_zero(const Tensor &input, Tensor &output)
{
  (void)input;
  (void)output;
  return {ErrorType::None, ""};
}

mlc::Error mlc::unary_relu(const Tensor &input, Tensor &output)
{
  (void)input;
  (void)output;
  return {ErrorType::None, ""};
}

mlc::Error mlc::unary_identity(const Tensor &input, Tensor output)
{
  (void)input;
  (void)output;
  return {ErrorType::None, ""};
}
