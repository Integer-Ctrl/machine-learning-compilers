#ifndef MLC_TENSORUTILS_H
#define MLC_TENSORUTILS_H
#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>

namespace mlc
{
  namespace internal
  {
    /**
     * @brief Function definition for converting a generic type to a pointer to a mlc::Tensor.
     *
     * @param T The type to convert.
     * @return mlc::Tensor
     */
    template <typename T> constexpr const mlc::Tensor *getTensor(const T &)
    {
      static_assert(false, "No generic conversion of tensor possible.");
      return nullptr;
    }

    /**
     * @brief Gets the pointer to the mlc::Tensor.
     *
     * @param tensor The tensor to get the pointer from.
     * @return Pointer to the mlc::Tensor.
     */
    template <> constexpr const mlc::Tensor *getTensor<mlc::Tensor *>(mlc::Tensor *const &tensor)
    {
      std::cout << tensor << std::endl;
      return tensor;
    }

    // TODO: doc
    template <>
    constexpr const mlc::Tensor *
    getTensor<std::reference_wrapper<const mlc::Tensor>>(const std::reference_wrapper<const mlc::Tensor> &tensor)
    {
      return &(tensor.get());
    }

    // TODO: doc
    template <typename T>
    constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root, const std::vector<T> &inputs,
                                               std::vector<int64_t> &sorted_dim_sizes)
    {
      if (root->left != nullptr)
      {
        if (root->left->type == mini_jit::EinsumTree::NodeType::Leaf)
        {
          const auto &dim_sizes = getTensor<T>(inputs[root->left->input_tensor_index])->dim_sizes;
          uint i = 0;
          for (int64_t id : root->left->output_dim_ids)
          {
            sorted_dim_sizes.resize(std::max(static_cast<int64_t>(sorted_dim_sizes.size()), id + 1));
            sorted_dim_sizes[id] = dim_sizes[i++];
          }
        }
        else
        {
          get_sorted_dimensions_sizes<T>(root->left, inputs, sorted_dim_sizes);
        }
      }

      if (root->right != nullptr)
      {
        if (root->right->type == mini_jit::EinsumTree::NodeType::Leaf)
        {
          const auto &dim_sizes = getTensor<T>(inputs[root->right->input_tensor_index])->dim_sizes;
          uint i = 0;
          for (int64_t id : root->right->output_dim_ids)
          {
            sorted_dim_sizes.resize(std::max(static_cast<int64_t>(sorted_dim_sizes.size()), id + 1));
            sorted_dim_sizes[id] = dim_sizes[i++];
          }
        }
        else
        {
          get_sorted_dimensions_sizes<T>(root->right, inputs, sorted_dim_sizes);
        }
      }
    }

    // TODO: doc
    constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root,
                                               const std::vector<std::reference_wrapper<const mlc::Tensor>> &inputs,
                                               std::vector<int64_t> &sorted_dim_sizes)
    {
      get_sorted_dimensions_sizes<std::reference_wrapper<const mlc::Tensor>>(root, inputs, sorted_dim_sizes);
    }

    // TODO: doc
    constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root, const std::vector<mlc::Tensor *> &inputs,
                                               std::vector<int64_t> &sorted_dim_sizes)
    {
      get_sorted_dimensions_sizes<mlc::Tensor *>(root, inputs, sorted_dim_sizes);
    }

    // TODO: doc
    constexpr mlc::ErrorType convertParseError(mini_jit::EinsumTree::ErrorParse error)
    {
      switch (error)
      {
      case mini_jit::EinsumTree::ErrorParse::None:
        return mlc::ErrorType::None;
      case mini_jit::EinsumTree::ErrorParse::ExpectedLeftBracket:
        return mlc::ErrorType::ParseExpectedLeftBracket;
      case mini_jit::EinsumTree::ErrorParse::ExpectedRightBracket:
        return mlc::ErrorType::ParseExpectedRightBracket;
      case mini_jit::EinsumTree::ErrorParse::ExpectedArrow:
        return mlc::ErrorType::ParseExpectedArrow;
      case mini_jit::EinsumTree::ErrorParse::ExpectedComma:
        return mlc::ErrorType::ParseExpectedComma;
      case mini_jit::EinsumTree::ErrorParse::ExpectedDimensionList:
        return mlc::ErrorType::ParseExpectedDimensionList;
      case mini_jit::EinsumTree::ErrorParse::NotAllowedToParseAgain:
        return mlc::ErrorType::ParseNotAllowedToParseAgain;
      case mini_jit::EinsumTree::ErrorParse::UndefinedNode:
        return mlc::ErrorType::ParseUndefinedNode;
      default:
        return mlc::ErrorType::Undefined;
      }
    }

    // TODO: doc
    constexpr mlc::ErrorType convertErrorExecute(mini_jit::EinsumTree::ErrorExecute error)
    {
      if (static_cast<int64_t>(error) > 100)
      {
        return static_cast<mlc::ErrorType>(static_cast<int64_t>(error));
      }

      switch (error)
      {
      case mini_jit::EinsumTree::ErrorExecute::None:
        return mlc::ErrorType::None;
      case mini_jit::EinsumTree::ErrorExecute::InvalidRoot:
        return mlc::ErrorType::EinsumInvalidRoot;
      case mini_jit::EinsumTree::ErrorExecute::NotEnoughInputTensors:
        return mlc::ErrorType::EinsumNotEnoughInputTensors;
      case mini_jit::EinsumTree::ErrorExecute::TooManyInputTensors:
        return mlc::ErrorType::EinsumTooManyInputTensors;
      case mini_jit::EinsumTree::ErrorExecute::NullPtrAsInputTensor:
        return mlc::ErrorType::EinsumNullPtrAsInputTensor;
      default:
        return mlc::ErrorType::Undefined;
      }
    }

    // TODO: doc
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

    // TODO: doc
    constexpr uint64_t getTensorSize(const mlc::Tensor *tensor)
    {
      uint64_t size = 1;
      for (auto dim : tensor->dim_sizes)
      {
        size *= dim;
      }
      return size;
    }

  }  // namespace internal
}  // namespace mlc
#endif  // MLC_TENSORUTILS_H