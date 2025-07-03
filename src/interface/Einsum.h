#ifndef MLC_EINSUM_H
#define MLC_EINSUM_H

#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include "TensorUtils.h"
#include <vector>

namespace mlc
{
  namespace internal
  {
    /**
     * @brief Executes the einsum expression with the given inputs to output based on the given einsum tree.
     *
     * @tparam T The type how the input tensors are passed to the einsum expression.
     * @param inputs All inputs of the einsum expression.
     * @param output The single output tensor of the einsum calculation.
     * @param tree The tree how two tensors are contracted.
     * @return mlc::Error The error code or ErrorType::None on success.
     */
    template <typename T> mlc::Error einsum(const std::vector<T> &inputs, mlc::Tensor &output, const std::string &tree)
    {
      mini_jit::EinsumTree einsumTree(tree);
      mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
      if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
      {
        mlc::ErrorType type = convertParseError(errorParse);
        return {type, "Failed during parsing the given einsum tree."};
      }

      std::vector<int64_t> sorted_dim_sizes;
      get_sorted_dimensions_sizes(einsumTree.get_root(), inputs, sorted_dim_sizes);
      einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

      std::vector<void *> tensors(inputs.size() + 1);
      for (size_t i = 0; i < inputs.size(); i++)
      {
        tensors[i] = getTensor<T>(inputs[i])->data;
        assert(tensors[i] != nullptr);
      }
      tensors[inputs.size()] = output.data;

      mini_jit::EinsumTree::ErrorExecute errorExecute = einsumTree.execute(tensors);
      if (errorExecute != mini_jit::EinsumTree::ErrorExecute::None)
      {
        mlc::ErrorType type = convertErrorExecute(errorExecute);
        return {type, "Failed during calculation of the einsum tree."};
      }

      return {mlc::ErrorType::None, "Success"};
    }
  }  // namespace internal

  class EinsumOperation : public TensorOperation
  {
  public:
    EinsumOperation(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output, const std::string &tree);

    //! @copydoc mlc::TensorOperation::execute(const std::vector<std::reference_wrapper<const Tensor>> &, Tensor &)
    virtual Error execute(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output) override;
    virtual Error execute(const std::vector<const Tensor *> &inputs, Tensor &output) override;
    virtual Error getSetupError() const override;

  private:
    /**
     * @brief Executes the Einsum operation with the given inputs and output tensor.
     */
    template <typename T> Error execute(const std::vector<T> &inputs, Tensor &output);
    template <typename T> Error hasSameDimensions(const std::vector<T> &inputs);

    Error error;
    mini_jit::EinsumTree einsumTree;
  };

  template <typename T> inline Error EinsumOperation::execute(const std::vector<T> &inputs, Tensor &output)
  {
    std::vector<void *> tensors(inputs.size() + 1);
    for (size_t i = 0; i < inputs.size(); i++)
    {
      tensors[i] = internal::getTensor<T>(inputs[i])->data;
    }
    tensors[inputs.size()] = output.data;

    mini_jit::EinsumTree::ErrorExecute errorExecute = einsumTree.execute(tensors);
    if (errorExecute != mini_jit::EinsumTree::ErrorExecute::None)
    {
      mlc::ErrorType type = internal::convertErrorExecute(errorExecute);
      return {type, "Failed to execute the einsum operation."};
    }

    return {mlc::ErrorType::None, "Success"};
  }

  template <typename T> inline Error EinsumOperation::hasSameDimensions(const std::vector<T> &inputs, const Tensor &output)
  {
    auto &sortedDimSizes = einsumTree.get_sorted_dim_sizes();
    const mini_jit::EinsumTree::EinsumNode *root = einsumTree.getRoot();

    if (output->dim_sizes.size() != root->output_dim_ids.size())
    {
      return {ErrorType::ExecuteWrongDimension, "The count of dimensions do not match in the output tensor."};
    }

    for (size_t i = 0; i < root->output_dim_ids.size(); i++)
    {
      if (output->dim_sizes[i] != static_cast<uint64_t>(sortedDimSizes[root->output_dim_ids[i]]))
      {
        return {ErrorType::ExecuteWrongDimension,
                "The output tensor dimension has a different size than the size than the tensor it was setup up with."};
      }
    }

    std::vector<mini_jit::EinsumTree::EinsumNode *> nodesToProcess = {einsumTree.get_root()};
    uint32_t processedInputs = 0;
    while (nodesToProcess.size() > 0)
    {
      mini_jit::EinsumTree::EinsumNode *node = nodesToProcess.back();
      nodesToProcess.pop_back();

      if (node->type == mini_jit::EinsumTree::NodeType::Leaf)
      {
        if (!(node->input_tensor_index < static_cast<int32_t>(inputs.size())))
        {
          return {ErrorType::EinsumTooManyInputTensors, "The was more input tensors than the original setup used."};
        }

        const Tensor *tensor = internal::getTensor<T>(inputs[node->input_tensor_index]);

        if (tensor->dim_sizes.size() != node->output_dim_ids.size())
        {
          return {ErrorType::ExecuteWrongDimension, "The count of dimensions do not match in an input tensor."};
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
#endif  // MLC_EINSUM_H