#include "../../include/MachineLearningCompiler/Setup.h"
#include "SetupEinsum.h"

mlc::Setup &mlc::einsum_setup(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree)
{
  mlc::SetupEinsum setup(inputs, output, tree);
  return setup;
}

mlc::Setup &mlc::einsum_setup(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree)
{
  mlc::SetupEinsum setup(inputs, output, tree);
  return setup;
}
