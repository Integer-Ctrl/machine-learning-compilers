#include "SetupEinsum.h"
#include "TensorUtils.h"
#include <utility>

mlc::SetupEinsum::SetupEinsum(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree)
    : einsumTree(tree)
{
  setup<std::reference_wrapper<Tensor>>(inputs, output, tree);
}

mlc::SetupEinsum::SetupEinsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree) : einsumTree(tree)
{
  setup<Tensor *>(inputs, output, tree);
}

mlc::Error mlc::SetupEinsum::getSetupError() const
{
  return error;
}

mlc::Error mlc::SetupEinsum::execute(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output)
{
  if (error.type != ErrorType::None)
  {
    return error;
  }

  Error checkError = hasSameDimensions(inputs);
  if (checkError.type != ErrorType::None)
  {
    return checkError;
  }

  return execute<std::reference_wrapper<Tensor>>(inputs, output);
}

mlc::Error mlc::SetupEinsum::execute(const std::vector<Tensor *> &inputs, Tensor &output)
{
  if (error.type != ErrorType::None)
  {
    return error;
  }

  Error checkError = hasSameDimensions(inputs);
  if (checkError.type != ErrorType::None)
  {
    return checkError;
  }

  return execute<Tensor *>(inputs, output);
}