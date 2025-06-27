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

  uint64_t size = 1;
  for (auto dim : tensor.dim_sizes)
  {
    size *= dim;
  }

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

mlc::Error mlc::einsum(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree)
{
  return ::einsum<std::reference_wrapper<Tensor>>(inputs, output, tree);
}

mlc::Error mlc::einsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree)
{
  return ::einsum<Tensor *>(inputs, output, tree);
}