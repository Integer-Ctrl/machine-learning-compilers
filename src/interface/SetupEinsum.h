#ifndef MLC_SETUPEINSUM_H
#define MLC_SETUPEINSUM_H
#include "../../include/MachineLearningCompiler/Setup.h"
#include "../main/EinsumTree.h"
#include <vector>

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
    template <typename T> Error hasSameDimensions(const std::vector<T> &inputs);

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
    get_sorted_dimensions_sizes(einsumTree.get_root(), inputs, sorted_dim_sizes);
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

  template <typename T> inline Error SetupEinsum::hasSameDimensions(const std::vector<T> &inputs)
  {
    std::vector<mini_jit::EinsumTree::EinsumNode *> nodesToProcess = {einsumTree.get_root()};
    auto &sortedDimSizes = einsumTree.get_sorted_dim_sizes();
    uint32_t processedInputs = 0;
    while (nodesToProcess.size() > 0)
    {
      mini_jit::EinsumTree::EinsumNode *node = nodesToProcess.back();
      nodesToProcess.pop_back();

      if (node->type == mini_jit::EinsumTree::NodeType::Leaf)
      {
        if (!(node->input_tensor_index < inputs.size()))
        {
          return {ErrorType::EinsumTooManyInputTensors, "The was more input tensors than the original setup used."}
        }

        Tensor *tensor = getTensor(inputs[node->input_tensor_index]);

        if (tensor->dim_sizes.size() != node->output_dim_ids.size())
        {
          return {ErrorType::ExecuteWrongDimension, "The count of dimensions do not match."};
        }

        for (size_t i = 0; i < node->output_dim_ids.size(); i++)
        {
          if (tensor->dim_sizes[i] != static_cast<uint64_t>(sortedDimSizes[node->output_dim_ids[i]]))
          {
            return {ErrorType::ExecuteWrongDimension,
                    "The input tensor dimension has a different size than the size than the tensor it was setup up with."};
          }
        }

        processedInputs++;
        continue;
      }

      if (node->left != nullptr)
      {
        nodesToProcess.push_back(node->left);
      }

      if (node->right != nullptr)
      {
        nodesToProcess.push_back(node->right);
      }
    }

    if (processedInputs < inputs.size())
    {
      return {mlc::ErrorType::EinsumNotEnoughInputTensors, "There was less input tensors than the original setups used."};
    }

    return {mlc::ErrorType::None, "Success"};
  }

}  // namespace mlc

#endif  // MLC_SETUPEINSUM_H