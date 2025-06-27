#include "SetupEinsum.h"
#include "TensorUtils.h"
#include <utility>

mlc::SetupEinsum::SetupEinsum(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree)
    : einsumTree(tree)
{
  std::vector<Tensor *> tensors(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    tensors[i] = &(inputs[i].get());
  }

  mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
  if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
  {
    mlc::ErrorType type = ::convertParseError(errorParse);
    error = {type, ""};  // TODO add error message
  }

  std::vector<int64_t> sorted_dim_sizes;
  ::get_sorted_dimensions_sizes(einsumTree.get_root(), tensors, sorted_dim_sizes);
  einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

  error = {mlc::ErrorType::None, "Success"};
}

mlc::SetupEinsum::SetupEinsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree) : einsumTree(tree)
{
  mini_jit::EinsumTree einsumTree(tree);
  mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
  if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
  {
    mlc::ErrorType type = ::convertParseError(errorParse);
    error = {type, ""};  // TODO add error message
  }

  std::vector<int64_t> sorted_dim_sizes;
  ::get_sorted_dimensions_sizes(einsumTree.get_root(), inputs, sorted_dim_sizes);
  einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

  error = {mlc::ErrorType::None, "Success"};
}