#ifndef MLC_SETUPEINSUM_H
#define MLC_SETUPEINSUM_H

#include "../../include/MachineLearningCompiler/Setup.h"
#include "../main/EinsumTree.h"

namespace mlc
{
  class SetupEinsum : public Setup
  {
  public:
    SetupEinsum(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree);
    SetupEinsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree);

    virtual Error execute(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output) override;
    virtual Error execute(const std::vector<Tensor *> &inputs, Tensor &output) override;
    virtual Error getSetupError() const override;

  private:
    template <typename T> void setup(const std::vector<T> &inputs, Tensor &output, const std::string &tree);
    template <typename T> Error execute(const std::vector<T> &inputs, Tensor &output);

    std::vector<uint64_t> sortedDimSizes;
    Error error;
    mini_jit::EinsumTree einsumTree;
  };

  template <typename T> inline void SetupEinsum::setup(const std::vector<T> &inputs, Tensor &output, const std::string &tree)
  {
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

  template <typename T> inline Error SetupEinsum::execute(const std::vector<T> &inputs, Tensor &output)
  {
    std::vector<void *> tensors(inputs.size() + 1);
    for (size_t i = 0; i < inputs.size(); i++)
    {
      tensors[i] = getTensor(inputs[i])->data;
    }
    tensors[inputs.size()] = output.data;

    mini_jit::EinsumTree::ErrorExecute errorExecute = einsumTree.execute(tensors);
    if (errorExecute != mini_jit::EinsumTree::ErrorExecute::None)
    {
      mlc::ErrorType type = ::convertErrorExecute(errorExecute);
      return {type, ""};  // TODO add error message
    }

    return {mlc::ErrorType::None, "Success"};
  }

}  // namespace mlc

#endif  // MLC_SETUPEINSUM_H