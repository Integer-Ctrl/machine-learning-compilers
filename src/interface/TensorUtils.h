#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include <iostream>

constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root, const std::vector<mlc::Tensor> &inputs,
                                           std::vector<int64_t> &sorted_dim_sizes)
{
  if (root->left != nullptr)
  {
    if (root->left->type == mini_jit::EinsumTree::NodeType::Leaf)
    {
      const auto &dim_sizes = inputs[root->left->input_tensor_index].dim_sizes;
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
      const auto &dim_sizes = inputs[root->right->input_tensor_index].dim_sizes;
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
