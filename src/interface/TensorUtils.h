#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include <functional>
#include <iostream>
#include <string>

namespace
{

  template <typename T> constexpr mlc::Tensor *getTensor(T &)
  {
    static_assert("No generic conversion of tensor possible");
    return nullptr;
  }

  template <> constexpr mlc::Tensor *getTensor<mlc::Tensor *>(mlc::Tensor *&tensor)
  {
    return tensor;
  }

  template <> constexpr mlc::Tensor *getTensor<std::reference_wrapper<mlc::Tensor>>(std::reference_wrapper<mlc::Tensor> &tensor)
  {
    return &(tensor.get());
  }

  template <typename T>
  constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root, const std::vector<T> &inputs,
                                             std::vector<int64_t> &sorted_dim_sizes)
  {
    if (root->left != nullptr)
    {
      if (root->left->type == mini_jit::EinsumTree::NodeType::Leaf)
      {
        const auto &dim_sizes = getTensor(inputs[root->left->input_tensor_index])->dim_sizes;
        uint i = 0;
        for (int64_t id : root->left->output_dim_ids)
        {
          sorted_dim_sizes.resize(std::max(static_cast<int64_t>(sorted_dim_sizes.size()), id + 1));
          sorted_dim_sizes[id] = dim_sizes[i++];
        }
      }
      else
      {
        get_sorted_dimensions_sizes(root->left, inputs, sorted_dim_sizes);
      }
    }

    if (root->right != nullptr)
    {
      if (root->right->type == mini_jit::EinsumTree::NodeType::Leaf)
      {
        const auto &dim_sizes = getTensor(inputs[root->right->input_tensor_index])->dim_sizes;
        uint i = 0;
        for (int64_t id : root->right->output_dim_ids)
        {
          sorted_dim_sizes.resize(std::max(static_cast<int64_t>(sorted_dim_sizes.size()), id + 1));
          sorted_dim_sizes[id] = dim_sizes[i++];
        }
      }
      else
      {
        get_sorted_dimensions_sizes(root->right, inputs, sorted_dim_sizes);
      }
    }
  }

  constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root,
                                             const std::vector<std::reference_wrapper<mlc::Tensor>> &inputs,
                                             std::vector<int64_t> &sorted_dim_sizes)
  {
    get_sorted_dimensions_sizes<std::reference_wrapper<mlc::Tensor>>(root, inputs, sorted_dim_sizes);
  }

  constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root, const std::vector<mlc::Tensor *> &inputs,
                                             std::vector<int64_t> &sorted_dim_sizes)
  {
    get_sorted_dimensions_sizes<mlc::Tensor *>(root, inputs, sorted_dim_sizes);
  }

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

  template <typename T> mlc::Error einsum(const std::vector<T> &inputs, mlc::Tensor &output, const std::string &tree)
  {
    mini_jit::EinsumTree einsumTree(tree);
    mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
    if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
    {
      mlc::ErrorType type = ::convertParseError(errorParse);
      return {type, ""};  // TODO add error message
    }

    std::vector<int64_t> sorted_dim_sizes;
    ::get_sorted_dimensions_sizes(einsumTree.get_root(), inputs, sorted_dim_sizes);
    einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

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

}  // namespace