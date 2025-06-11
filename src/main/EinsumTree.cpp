#include "EinsumTree.h"
#include "TensorOperation.h"
#include "release_assert.h"
#include <format>
#include <iostream>
#include <utility>

mini_jit::EinsumTree::EinsumTree(const std::string &tree_str, const std::vector<int> &sorted_dim_sizes) : tree_str(tree_str)
{
  for (size_t i = 0; i < sorted_dim_sizes.size(); ++i)
  {
    dim_sizes[i] = sorted_dim_sizes[i];
  }
  EinsumTree::error_parse = EinsumTree::ErrorParse::None;
}

mini_jit::EinsumTree::ErrorParse mini_jit::EinsumTree::parse_tree()
{
  if (root != nullptr)
  {
    return ErrorParse::NotAllowedToParseAgain;
  }

  size_t pos = 0;
  root = parse_node(pos, tree_str);
  return error_parse;
}

mini_jit::EinsumTree::EinsumNode *mini_jit::EinsumTree::parse_node(size_t &pos, const std::string &str)
{
  // Skip leading whitespace
  while (str[pos] == ' ')
    ++pos;

  // Base case: leaf node like 1,2,3]
  if (str[pos] != '[')
  {
    std::vector<int64_t> dims = parse_dim_list(pos, str);
    EinsumNode *node = new EinsumNode();
    node->type = NodeType::Leaf;
    node->output_dims = dims;
    node->input_dims0 = dims;
    return node;
  }

  // Otherwise it’s a nested structure: [...],[...]->[...]
  if (str[pos] != '[')
  {
    EinsumTree::error_parse = ErrorParse::ExpectedLeftBracket;
    return nullptr;
  }

  ++pos;  // skip outer '['

  // Parse first child
  std::cout << "Parsing left node at pos " << pos << ": " << str.substr(pos) << std::endl;
  EinsumNode *left = parse_node(pos, str);
  pos++;  // skip ']'

  while (str[pos] == ' ')
    ++pos;

  // Right Child
  EinsumNode *right = nullptr;
  if (str[pos] == ',')
  {
    ++pos;

    while (str[pos] == ' ')
      ++pos;

    if (str[pos] != '[')
    {
      EinsumTree::error_parse = ErrorParse::ExpectedLeftBracket;
      return nullptr;
    }
    ++pos;

    std::cout << "Parsing right dim list at pos " << pos << ": " << str.substr(pos) << std::endl;
    right = parse_node(pos, str);
    pos++;  // skip ']'
  }

  while (str[pos] == ' ')
    ++pos;

  std::cout << "String after parsing children: " << str.substr(pos) << std::endl;
  if (str[pos] != '-')
  {
    EinsumTree::error_parse = ErrorParse::ExpectedArrow;
    return nullptr;
  }
  ++pos;

  if (str[pos] != '>')
  {
    EinsumTree::error_parse = ErrorParse::ExpectedArrow;
    return nullptr;
  }
  ++pos;

  if (str[pos] != '[')
  {
    EinsumTree::error_parse = ErrorParse::ExpectedLeftBracket;
    return nullptr;
  }
  ++pos;

  std::cout << "Parsing out dim list at pos " << pos << ": " << str.substr(pos) << std::endl;
  EinsumNode *node = parse_node(pos, str);
  pos++;  // skip ']'

  node->left = left;
  node->right = right;

  // Infer if this is a contraction or transpose
  if (left && right)
  {
    node->type = NodeType::Contraction;
    node->input_dims0 = left->output_dims;
    node->input_dims1 = right->output_dims;
  }
  else
  {
    node->type = NodeType::Transposition;
    node->input_dims0 = left->output_dims;
  }

  return node;
}

std::vector<int64_t> mini_jit::EinsumTree::parse_dim_list(size_t &pos, const std::string &str)
{
  std::vector<int64_t> dims;

  while (str[pos] != ']')
  {
    while (str[pos] == ' ')
      ++pos;
    int num = 0;
    // Needed if digit is greater than 9
    while (isdigit(str[pos]))
    {
      // ASCII to integer conversion, e.g. '3' - '0' = 51 - 48 = 3
      num = num * 10 + (str[pos++] - '0');
    }
    dims.push_back(num);
    while (str[pos] == ' ')
      ++pos;
    if (str[pos] == ',')
      ++pos;
  }

  return dims;
}

mini_jit::EinsumTree::EinsumNode *mini_jit::EinsumTree::get_root() const
{
  return root;
}

mini_jit::EinsumTree::~EinsumTree()
{
  delete_tree(root);
}

void mini_jit::EinsumTree::delete_tree(EinsumNode *node)
{
  if (!node)
    return;
  delete_tree(node->left);
  delete_tree(node->right);
  if (node->type != NodeType::Leaf && node->tensor != nullptr && node != get_root())
  {
    delete node->tensor;
  }
  delete node;
}

std::string mini_jit::EinsumTree::EinsumNode::_to_string(uint depth, std::string connection, std::string depthString) const
{
  std::string output;

  if (depth == 0)
  {
    if (output_dims.size() > 0)
    {
      output += std::format("{}", output_dims[0]);
      for (auto iDim = output_dims.begin() + 1; iDim != output_dims.end(); iDim++)
      {
        output += std::format(",{}", *iDim);
      }
      output += "\n";
    }
  }
  else
  {
    if (output_dims.size() > 0)
    {
      output += std::format("{}{}─ {}", depthString, connection, output_dims[0]);
      for (auto iDim = output_dims.begin() + 1; iDim != output_dims.end(); iDim++)
      {
        output += std::format(",{}", *iDim);
      }
      output += "\n";
    }
  }

  if (left != nullptr)
  {
    if (right == nullptr)
    {
      output += left->_to_string(depth + 1, "└", depthString + "| ");
    }
    else
    {
      output += left->_to_string(depth + 1, "├", depthString + "| ");
    }
  }

  if (right != nullptr)
  {
    output += right->_to_string(depth + 1, "└", depthString + "  ");
  }

  return output;
}

std::string mini_jit::EinsumTree::EinsumNode::to_string() const
{
  return mini_jit::EinsumTree::EinsumNode::_to_string(0, "", "");
}

mini_jit::TensorConfig mini_jit::EinsumTree::lower_node(const EinsumNode *node)
{
  // Node has two children -> contraction
  if (node->type == NodeType::Contraction)
  {
    TensorConfig config{
      TensorConfig::prim_t::none,                                                                        // first_touch
      TensorConfig::prim_t::brgemm,                                                                      // main
      TensorConfig::prim_t::none,                                                                        // last touch
      infer_dim_types(node->input_dims0),                                                                // dim_types
      std::vector<TensorConfig::exec_t>(size, TensorConfig::exec_t::seq),                                // exec_types
      {static_cast<int64_t>(node->input_dims0.size()), static_cast<int64_t>(node->input_dims1.size())},  // dim_sizes
      compute_strides(node->input_dims0),                                                                // strides_in0
      compute_strides(node->input_dims1),                                                                // strides_in1
      compute_strides(node->output_dims),                                                                // strides_out
      TensorConfig::dtype_t::fp32                                                                        // dtype_t
    };
    return config;
  }

  // Node has only left child -> transposition
  if (node->type == NodeType::Transposition)
  {
    release_assert(node->input_dims0.size() == node->output_dims.size(),
                   "Expected input and output to have same dimensions for copy operation.");
    release_assert(node->get_size() == node->left->get_size(), "Expected the accumulated size to be the same.");

    TensorConfig config{
      TensorConfig::prim_t::none,                                                          // first_touch
      TensorConfig::prim_t::copy,                                                          // main
      TensorConfig::prim_t::none,                                                          // last touch
      std::vector<TensorConfig::dim_t>(node->input_dims0.size(), TensorConfig::dim_t::c),  // dim_types
      {TensorConfig::exec_t::seq, TensorConfig::exec_t::seq},                              // exec_types
      node->input_dims0,                                                                   // dim_sizes
      compute_strides(node->input_dims0),                                                  // strides_in0
      std::vector<int64_t>(node->input_dims0.size(), 0),                                   // strides_in1 (not used for transposition)
      compute_strides(node->output_dims),                                                  // strides_out
      TensorConfig::dtype_t::fp32                                                          // dtype_t
    };
    return config;
  }
}

void mini_jit::EinsumTree::assign_tensor(std::vector<void *> tensors, EinsumNode *node)
{
  if (node == nullptr)
  {
    return;
  }

  assign_tensor(tensors, node->left);
  assign_tensor(tensors, node->right);

  if (node->type == NodeType::Leaf)
  {
    if (tensorIndex >= (tensors.size() - 1))  // Last is reserved for output tensor
    {
      error_execute = mini_jit::EinsumTree::ErrorExecute::NotEnoughInputTensors;
      return;
    }

    // user input tensor
    node->tensor = static_cast<float *>(tensors[tensorIndex++]);
  }
}

mini_jit::EinsumTree::ErrorExecute mini_jit::EinsumTree::execute(const std::vector<void *> tensors)
{
  if (root == nullptr)
  {
    std::cerr << "EinsumTree: Cannot execute, root is null." << std::endl;
    return ErrorExecute::InvalidRoot;
  }

  tensorIndex = 0;
  assign_tensor(tensors, root);
  root->tensor = static_cast<float *>(tensors[tensors.size() - 1]);

  if (tensorIndex < (tensors.size() - 1))  // Last is reserved for output tensor
  {
    return ErrorExecute::TooManyInputTensors;
  }

  if (error_execute != ErrorExecute::None)
  {
    return error_execute;
  }

  // Recursive execution of the tree

  execute_node(root);

  root->tensor = nullptr;

  return ErrorExecute::None;
}

void *mini_jit::EinsumTree::execute_node(EinsumNode *node)
{
  if (node->type == NodeType::Leaf)
  {
    release_assert(node->tensor != nullptr, "Expected a pointer to be a valid pointer.");
    return node->tensor;
  }
  else if (node->type == NodeType::Transposition)
  {
    release_assert(node->left->tensor != nullptr, "Expected the left child tensor of the transposition to be a valid pointer.");
    release_assert(node->right->tensor == nullptr, "Expected the right child tensor of transposition to be a nullptr.");

    mini_jit::TensorOperation tensor_op;
    tensor_op.setup(lower_node(node));

    if (node->tensor == nullptr)
    {
      // Generate intermediate tensor.
      node->tensor = new float[node->get_size()];
    }

    tensor_op.execute(node->left->tensor, nullptr, node->tensor);
  }
  else if (node->type == NodeType::Contraction)
  {
    release_assert(node->left->tensor != nullptr, "Expected the right child tensor of contraction to be a valid pointer.");
    release_assert(node->right->tensor != nullptr, "Expected the right child tensor of contraction to be a valid pointer.");

    if (node->tensor == nullptr)
    {
      // Generate intermediate tensor.
      node->tensor = new float[node->get_size()];
    }

    mini_jit::TensorOperation tensor_op;
    tensor_op.setup(lower_node(node));
    tensor_op.execute(node->left->tensor, node->right->tensor, node->tensor);
  }
  else
  {
    release_assert(false, "Found unhandled einsum tree node type.");
  }
}

int64_t mini_jit::EinsumTree::EinsumNode::get_size() const
{
  int64_t size = 1;
  for (const auto &dim : output_dims)
  {
    size *= dim;
  }
  return size;
}

std::vector<int64_t> mini_jit::EinsumTree::compute_strides(const std::vector<int64_t> &dims)
{
  std::vector<int64_t> strides(dims.size());
  int64_t stride_size = 1;

  for (int i = dims.size(); i > 0; i--)
  {
    stride_size *= dims[i];
    strides[i] = stride_size;
  }

  return strides;
}