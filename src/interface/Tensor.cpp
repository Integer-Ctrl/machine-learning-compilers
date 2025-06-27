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

void mlc::einsum(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree)
{
  mini_jit::EinsumTree einsumTree(tree);
  mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
  (void)(errorParse);

  std::vector<int64_t> sorted_dim_sizes;
  ::get_sorted_dimensions_sizes(einsumTree.get_root(), inputs, sorted_dim_sizes);
  einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

  std::vector<void *> tensors(inputs.size() + 1);
  for (size_t i = 0; i < inputs.size(); i++)
  {
    tensors[i] = inputs[i].get().data;
  }
  tensors[inputs.size()] = output.data;

  einsumTree.execute(tensors);
}
