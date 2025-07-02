#include "Einsum.h"
#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include "utility"

mlc::Error mlc::einsum(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output, const std::string &tree)
{
  return internal::einsum<std::reference_wrapper<const Tensor>>(inputs, output, tree);
}

mlc::Error mlc::einsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree)
{
  return internal::einsum<Tensor *>(inputs, output, tree);
}

mlc::EinsumOperation::EinsumOperation(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &, const std::string &tree)
    : einsumTree(tree)
{
  mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
  if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
  {
    mlc::ErrorType type = internal::convertParseError(errorParse);
    error = {type, "Failed to parse the tree."};
  }

  std::vector<int64_t> sorted_dim_sizes;
  internal::get_sorted_dimensions_sizes<std::reference_wrapper<const Tensor>>(einsumTree.get_root(), inputs, sorted_dim_sizes);
  einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

  error = {mlc::ErrorType::None, "Success"};
}

mlc::Error mlc::EinsumOperation::getSetupError() const
{
  return error;
}

mlc::Error mlc::EinsumOperation::execute(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output)
{
  if (error.type != ErrorType::None)
  {
    return error;
  }

  Error checkError = hasSameDimensions<std::reference_wrapper<const Tensor>>(inputs);
  if (checkError.type != ErrorType::None)
  {
    return checkError;
  }

  return execute<std::reference_wrapper<const Tensor>>(inputs, output);
}

mlc::Error mlc::EinsumOperation::execute(const std::vector<const Tensor *> &inputs, Tensor &output)
{
  if (error.type != ErrorType::None)
  {
    return error;
  }

  Error checkError = hasSameDimensions<const Tensor *>(inputs);
  if (checkError.type != ErrorType::None)
  {
    return checkError;
  }

  return execute<const Tensor *>(inputs, output);
}

mlc::TensorOperation *mlc::einsum_operation(const std::vector<std::vector<uint64_t>> &inputs, const std::vector<uint64_t> &output,
                                            const std::string &tree)
{
  std::vector<Tensor> rawTensor;
  std::vector<std::reference_wrapper<const Tensor>> inputTensors;
  rawTensor.reserve(inputs.size());
  inputTensors.reserve(inputs.size());
  for (const auto &shape : inputs)
  {
    // Create a dummy tensor with the given shape
    rawTensor.emplace_back(nullptr, shape);
    inputTensors.push_back(rawTensor.back());
  }

  Tensor outputTensor(output);
  EinsumOperation *operation = new EinsumOperation(inputTensors, outputTensor, tree);
  return operation;
}