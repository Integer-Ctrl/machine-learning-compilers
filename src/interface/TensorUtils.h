#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include <iostream>

constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root, const std::vector<mlc::Tensor *> &inputs,
                                           std::vector<int64_t> &sorted_dim_sizes)
{
  if (root->left != nullptr)
  {
    if (root->left->type == mini_jit::EinsumTree::NodeType::Leaf)
    {
      const auto &dim_sizes = inputs[root->left->input_tensor_index]->dim_sizes;
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
      const auto &dim_sizes = inputs[root->right->input_tensor_index]->dim_sizes;
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

// constexpr void fill_random(mlc::Tensor &tensor, uint64_t index, uint64_t offset)
// {
//   if (index < (tensor.dim_sizes.size() - 1))
//   {
//     for (uint64_t i = 0; i < tensor.dim_sizes[index]; i++)
//     {
//       fill_random(tensor, index + 1, offset + tensor.strides[index] * i);
//     }
//   }
//   else
//   {
//     for (uint64_t i = 0; i < tensor.dim_sizes[index]; i++)
//     {
//       float denominator = 1;
//       denominator = static_cast<float>(std::rand());
//       if (denominator == 0)
//       {
//         denominator = 1;
//       }

//       float numerator = 1;
//       numerator = static_cast<float>(std::rand());

//       float random = numerator / denominator;

//       tensor.data[offset + tensor.strides[index] * i] = random;
//     }
//   }
// }