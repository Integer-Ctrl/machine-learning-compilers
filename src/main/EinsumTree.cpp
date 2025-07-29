#include "EinsumTree.h"
#include "TensorOperation.h"
#include "release_assert.h"
#include <format>
#include <iostream>
#include <utility>

mini_jit::EinsumTree::EinsumTree(const std::string &tree_str) : tree_str(tree_str)
{
}

mini_jit::EinsumTree::EinsumTree(const std::string &tree_str, const std::vector<int64_t> &sorted_dim_sizes)
    : tree_str(tree_str), dim_sizes(sorted_dim_sizes)
{
}

mini_jit::EinsumTree::ErrorParse mini_jit::EinsumTree::parse_tree_no_optimization(bool build_operators)
{
  if (root != nullptr)
  {
    return ErrorParse::NotAllowedToParseAgain;
  }

  size_t pos = 0;
  root = parse_node(pos, tree_str);

  tensorIndex = 0;
  assign_tensor_indices(root);

  if (error_parse != ErrorParse::None)
  {
    return error_parse;
  }

  if (build_operators)
  {
    error_parse = generate_operators();
  }

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
    node->output_dim_ids = dims;
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

    right = parse_node(pos, str);
    pos++;  // skip ']'
  }

  while (str[pos] == ' ')
    ++pos;

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

  EinsumNode *node = parse_node(pos, str);
  pos++;  // skip ']'

  node->left = left;
  node->right = right;

  // Infer if this is a contraction or transpose
  if (left && right)
  {
    node->type = NodeType::Contraction;
  }
  else
  {
    node->type = NodeType::Transposition;
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

void mini_jit::EinsumTree::set_sorted_dim_sizes(const std::vector<int64_t> &sorted_dim_sizes)
{
  EinsumTree::dim_sizes = sorted_dim_sizes;
}

const std::vector<int64_t> &mini_jit::EinsumTree::get_sorted_dim_sizes()
{
  return dim_sizes;
}

void mini_jit::EinsumTree::delete_tree(EinsumNode *node)
{
  if (node == nullptr)
  {
    return;
  }

  delete_tree(node->left);
  delete_tree(node->right);
  node->left = nullptr;
  node->right = nullptr;

  if (node->type != NodeType::Leaf && node->tensor != nullptr && node != get_root())
  {
    delete[] node->tensor;
  }
  delete node;
}

std::string mini_jit::EinsumTree::EinsumNode::_to_string(uint depth, std::string connection, std::string depthString) const
{
  std::string output;

  if (depth == 0)
  {
    if (output_dim_ids.size() > 0)
    {
      output += std::format("{}", output_dim_ids[0]);
      for (auto iDim = output_dim_ids.begin() + 1; iDim != output_dim_ids.end(); iDim++)
      {
        output += std::format(",{}", *iDim);
      }
      output += "\n";
    }
  }
  else
  {
    if (output_dim_ids.size() > 0)
    {
      output += std::format("{}{}─ {}", depthString.substr(0, depthString.size() - 3), connection, output_dim_ids[0]);
      for (auto iDim = output_dim_ids.begin() + 1; iDim != output_dim_ids.end(); iDim++)
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
      output += left->_to_string(depth + 1, "└", depthString + "|  ");
    }
    else
    {
      output += left->_to_string(depth + 1, "├", depthString + "|  ");
    }
  }

  if (right != nullptr)
  {
    output += right->_to_string(depth + 1, "└", depthString + "   ");
  }

  return output;
}

std::string mini_jit::EinsumTree::EinsumNode::to_string() const
{
  return mini_jit::EinsumTree::EinsumNode::_to_string(0, "", "");
}

std::string mini_jit::EinsumTree::EinsumNode::name() const
{
  std::string output = std::format("{}", output_dim_ids[0]);
  for (auto iDim = output_dim_ids.begin() + 1; iDim != output_dim_ids.end(); iDim++)
  {
    output += std::format("_{}", *iDim);
  }
  return output;
}

mini_jit::TensorConfig mini_jit::EinsumTree::lower_node(const EinsumNode *node)
{
  // Node has two children -> contraction
  if (node->type == NodeType::Contraction)
  {
    release_assert(node->left != nullptr, "Expected the left child tensor to be a valid pointer.");
    release_assert(node->right != nullptr, "Expected the right child tensor to be a valid pointer.");

    std::vector<TensorConfig::dim_t> dim_types;
    std::vector<int64_t> dim_sizes;
    std::map<int64_t, size_t> id_map;
    uint32_t number_of_k = 0;
    dim_types.reserve(node->output_dim_ids.size());
    dim_sizes.reserve(node->output_dim_ids.size());

    get_config_dim_types_and_sizes(node, id_map, dim_types, dim_sizes, number_of_k);

    std::vector<int64_t> stridesIn0 = get_config_strides(node->left, id_map);
    std::vector<int64_t> stridesIn1 = get_config_strides(node->right, id_map);
    std::vector<int64_t> stridesOut = get_config_strides(node, id_map);

    TensorConfig config{
      node == root ? TensorConfig::prim_t::none : TensorConfig::prim_t::zero,       // first_touch
      number_of_k > 1 ? TensorConfig::prim_t::brgemm : TensorConfig::prim_t::gemm,  // main
      TensorConfig::prim_t::none,                                                   // last touch
      dim_types,                                                                    // dim_types
      std::vector<TensorConfig::exec_t>(id_map.size(), TensorConfig::exec_t::seq),  // exec_types
      dim_sizes,                                                                    // dim_sizes
      stridesIn0,                                                                   // strides_in0
      stridesIn1,                                                                   // strides_in1
      stridesOut,                                                                   // strides_out
      TensorConfig::dtype_t::fp32                                                   // dtype_t
    };
    return config;
  }

  // Node has only left child -> transposition
  if (node->type == NodeType::Transposition)
  {
    release_assert(node->left->output_dim_ids.size() == node->output_dim_ids.size(),
                   "Expected input and output to have same dimensions for copy operation.");
    release_assert(node->get_size(dim_sizes) == node->left->get_size(dim_sizes), "Expected the accumulated size to be the same.");

    std::vector<int64_t> stridesIn0 = compute_strides(node->left->output_dim_ids);
    stridesIn0 = swap_strides_id_based(stridesIn0, node->left->output_dim_ids, node->output_dim_ids);

    TensorConfig config{
      TensorConfig::prim_t::none,                                                                 // first_touch
      TensorConfig::prim_t::copy,                                                                 // main
      TensorConfig::prim_t::none,                                                                 // last touch
      std::vector<TensorConfig::dim_t>(node->output_dim_ids.size(), TensorConfig::dim_t::c),      // dim_types
      std::vector<TensorConfig::exec_t>(node->output_dim_ids.size(), TensorConfig::exec_t::seq),  // exec_types
      get_output_dims(node->output_dim_ids),                                                      // dim_sizes
      stridesIn0,                                                                                 // strides_in0
      std::vector<int64_t>(node->output_dim_ids.size(), 0),  // strides_in1 (not used for transposition)
      compute_strides(node->output_dim_ids),                 // strides_out
      TensorConfig::dtype_t::fp32                            // dtype_t
    };
    return config;
  }

  release_assert(false, "Found no matching node type to lower to a tensor config.");
  return {};
}

void mini_jit::EinsumTree::get_config_dim_types_and_sizes(const mini_jit::EinsumTree::EinsumNode *node, std::map<int64_t, size_t> &id_map,
                                                          std::vector<mini_jit::TensorConfig::dim_t> &dim_types,
                                                          std::vector<int64_t> &dim_sizes, uint32_t &number_of_k)
{
  // Left tensor (tensor0)
  for (int64_t id : node->left->output_dim_ids)
  {
    id_map[id] = id_map.size();
    dim_types.push_back(TensorConfig::dim_t::undefined);  // Not defined whether dim_t::k, dim_t::m or dim_t::c
    dim_sizes.push_back(EinsumTree::dim_sizes[id]);
  }

  // Right tensor (tensor1)
  for (int64_t id : node->right->output_dim_ids)
  {
    auto iPair = id_map.find(id);
    if (iPair == id_map.end())
    {
      // Not found
      dim_types.push_back(TensorConfig::dim_t::undefined);  // Not defined whether dim_t::k, dim_t::n or dim_t::c
      dim_sizes.push_back(EinsumTree::dim_sizes[id]);
      id_map[id] = id_map.size();
    }
    else
    {
      // Found id index
      size_t index = iPair->second;
      if (dim_types[index] == TensorConfig::dim_t::undefined)
      {
        dim_types[index] = TensorConfig::dim_t::k;
        number_of_k += 1;
      }
      else
      {
        release_assert(false, "Found unexpected value while dim type of adding the second input to config.");
      }
    }
  }

  // Output tensor
  for (int64_t id : node->output_dim_ids)
  {
    auto iPair = id_map.find(id);
    if (iPair == id_map.end())
    {
      // Not found
      dim_types.push_back(TensorConfig::dim_t::undefined);  // Must not occur -> setup will throw an error
      dim_sizes.push_back(EinsumTree::dim_sizes[id]);
      id_map[id] = id_map.size();
    }
    else
    {
      // Found id index
      size_t index = iPair->second;
      switch (dim_types[index])
      {
      case TensorConfig::dim_t::undefined:  // Tensor0 or Tensor1 defined this value
        dim_types[index] = (index < node->left->output_dim_ids.size()) ? TensorConfig::dim_t::m : TensorConfig::dim_t::n;
        break;

      case TensorConfig::dim_t::k:
        dim_types[index] = TensorConfig::dim_t::c;
        number_of_k -= 1;
        break;

        // Should never be one of the following three dim types
      case TensorConfig::dim_t::c:
      case TensorConfig::dim_t::m:
      case TensorConfig::dim_t::n:
      default:
        release_assert(false, "Found unexpected value while dim type of adding the output to config.");
        break;
      }
    }
  }
}

std::vector<int64_t> mini_jit::EinsumTree::get_config_strides(const EinsumNode *node, std::map<int64_t, size_t> &id_map)
{
  std::vector<int64_t> stridesConfig(id_map.size(), 0);
  std::vector<int64_t> strides = compute_strides(node->output_dim_ids);
  for (size_t i = 0; i < strides.size(); i++)
  {
    size_t index = id_map[node->output_dim_ids[i]];
    stridesConfig[index] = strides[i];
  }
  return stridesConfig;
}

void mini_jit::EinsumTree::assign_tensor_indices(EinsumNode *node)
{
  if (node == nullptr)
  {
    error_parse = EinsumTree::ErrorParse::UndefinedNode;
    return;
  }

  if (node->type == NodeType::Leaf)
  {
    // user input tensor
    node->input_tensor_index = tensorIndex++;
  }
  else if (node->type == NodeType::Transposition)
  {
    assign_tensor_indices(node->left);
  }
  else if (node->type == NodeType::Contraction)
  {
    assign_tensor_indices(node->left);
    assign_tensor_indices(node->right);
  }
  else
  {
    release_assert(false, "Found unhandled node type in assign tensor.");
  }
}

mini_jit::EinsumTree::ErrorExecute mini_jit::EinsumTree::execute(const std::vector<void *> &tensors)
{
  if (root == nullptr)
  {
    std::cerr << "EinsumTree: Cannot execute, root is null." << std::endl;
    return ErrorExecute::InvalidRoot;
  }

  if (tensorIndex >= tensors.size())  // Last is reserved for output tensor
  {
    return EinsumTree::ErrorExecute::NotEnoughInputTensors;
  }

  if (tensorIndex < (tensors.size() - 1))  // Last is reserved for output tensor
  {
    return ErrorExecute::TooManyInputTensors;
  }

  root->tensor = static_cast<float *>(tensors[tensors.size() - 1]);

  // Recursive execution of the tree
  ErrorExecute error_execute = execute_node(tensors, root);

  if (error_execute != ErrorExecute::None)
  {
    return error_execute;
  }

  // Do not delete user memory attached to the root node
  root->tensor = nullptr;

  return ErrorExecute::None;
}

mini_jit::EinsumTree::ErrorExecute mini_jit::EinsumTree::execute_node(const std::vector<void *> &input_tensors, EinsumNode *node)
{
  if (node->type == NodeType::Leaf)
  {
    release_assert(node->input_tensor_index != -1, "Expected a input_tensor_index to be a valid index.");
    node->tensor = static_cast<float *>(input_tensors[node->input_tensor_index]);

    if (node->tensor == nullptr)
    {
      return ErrorExecute::NullPtrAsInputTensor;
    }
  }
  else if (node->type == NodeType::Transposition)
  {
    release_assert(node->left != nullptr, "Expected the left child of contraction to be a valid pointer.");
    release_assert(node->right == nullptr, "Expected the right child of contraction to be a nullptr.");

    ErrorExecute error = execute_node(input_tensors, node->left);

    if (error != ErrorExecute::None)
    {
      return error;
    }

    release_assert(node->left->tensor != nullptr, "Expected the left child tensor of the transposition to be a valid pointer.");

    if (node->tensor == nullptr)
    {
      // Generate intermediate tensor.
      node->tensor = new float[node->get_size(dim_sizes)]();
    }

    if (node->tensor_op.getHasSetupError() == true)
    {
      return ErrorExecute::SetupHasError;
    }

    node->tensor_op.execute(node->left->tensor, nullptr, node->tensor);
  }
  else if (node->type == NodeType::Contraction)
  {
    release_assert(node->left != nullptr, "Expected the left child of contraction to be a valid pointer.");
    release_assert(node->right != nullptr, "Expected the right child of contraction to be a valid pointer.");

    ErrorExecute error = execute_node(input_tensors, node->left);

    if (error != ErrorExecute::None)
    {
      return error;
    }

    error = execute_node(input_tensors, node->right);

    if (error != ErrorExecute::None)
    {
      return error;
    }

    release_assert(node->left->tensor != nullptr, "Expected the left child tensor of contraction to be a valid pointer.");
    release_assert(node->right->tensor != nullptr, "Expected the right child tensor of contraction to be a valid pointer.");

    if (node->tensor == nullptr)
    {
      // Generate intermediate tensor.
      node->tensor = new float[node->get_size(dim_sizes)]();
    }

    if (node->tensor_op.getHasSetupError() == true)
    {
      return ErrorExecute::SetupHasError;
    }

    node->tensor_op.execute(node->left->tensor, node->right->tensor, node->tensor);
  }
  else
  {
    release_assert(false, "Found unhandled einsum tree node type.");
  }

  return ErrorExecute::None;
}

int64_t mini_jit::EinsumTree::EinsumNode::get_size(const std::vector<int64_t> dim_sizes) const
{
  int64_t size = 1;
  for (const auto &dim : output_dim_ids)
  {
    size *= dim_sizes[dim];
  }
  return size;
}

std::vector<int64_t> mini_jit::EinsumTree::compute_strides(const std::vector<int64_t> &dim_ids)
{
  std::vector<int64_t> strides(dim_ids.size());
  int64_t stride_size = 1;

  for (int i = dim_ids.size() - 1; i >= 0; i--)
  {
    strides[i] = stride_size;
    stride_size *= dim_sizes[dim_ids[i]];
  }

  return strides;
}

std::vector<int64_t> mini_jit::EinsumTree::swap_strides_id_based(const std::vector<int64_t> &strides, const std::vector<int64_t> &inIds,
                                                                 const std::vector<int64_t> &outIds)
{
  release_assert(inIds.size() == outIds.size(), "Expected inIds to have the same size as the outIds.");
  release_assert(inIds.size() == strides.size(), "Expected the inIds to have the same size as the outIds.");

  std::vector<int64_t> outStrides(strides.size());

  for (size_t i = 0; i < inIds.size(); i++)
  {
    auto outPtr = std::find(outIds.begin(), outIds.end(), inIds[i]);
    release_assert(outPtr != outIds.end(), "Expected to have the same elements as the inIds.");

    auto outIndex = std::distance(outIds.begin(), outPtr);
    outStrides[outIndex] = strides[i];
  }

  return outStrides;
}

std::vector<int64_t> mini_jit::EinsumTree::get_output_dims(const std::vector<int64_t> &dim_ids)
{
  std::vector<int64_t> dims(dim_ids.size());

  for (size_t i = 0; i < dim_ids.size(); ++i)
  {
    dims[i] = dim_sizes[dim_ids[i]];
  }

  return dims;
}

mini_jit::EinsumTree::ErrorParse mini_jit::EinsumTree::parse_setup_error(TensorOperation::error_t error)
{
  if (error == TensorOperation::error_t::success)
  {
    return ErrorParse::None;
  }

  uint32_t error_num = static_cast<uint32_t>(error) + 100;

  release_assert(error_num >= 101, "Expected error_num to be larger equal than 101.");
  release_assert(error_num <= 115, "Expected error_num to be less equal than 115.");

  return static_cast<ErrorParse>(error_num);
}

bool mini_jit::EinsumTree::is_unit_stride_n(EinsumNode *node)
{
  release_assert(node->left != nullptr, "Expected a valid pointer.");

  int64_t last_dim_id = node->output_dim_ids.back();
  bool isMDim = false;

  for (int64_t dim_id : node->left->output_dim_ids)
  {
    if (dim_id == last_dim_id)
    {
      // Found same dimension in left child and parent
      isMDim = true;
      break;
    }
  }

  // Conclusion that 'n' dimension is unit stride
  if (isMDim == false)
  {
    return true;
  }

  // Check that last dimension is not a batch dimension 'c'
  for (int64_t dim_id : node->right->output_dim_ids)
  {
    release_assert(dim_id != last_dim_id, "Found a C dimension as unit stride.");
  }

  // unit stride is 'm' dimension
  return false;
}

void mini_jit::EinsumTree::optimize(EinsumNode *node)
{
  if (node->type != NodeType::Contraction)
  {
    return;
  }

  conditional_swap(node);

  reorder_left_node(node);
  reorder_right_node(node);

  optimize(node->left);
  optimize(node->right);
}

void mini_jit::EinsumTree::conditional_swap(mini_jit::EinsumTree::EinsumNode *node)
{
  // Ensure that 'm' dimension has unit stride
  if (is_unit_stride_n(node))
  {
    std::swap(node->left, node->right);
  }
}

mini_jit::EinsumTree::ErrorParse mini_jit::EinsumTree::parse_tree(bool build_operators)
{
  ErrorParse error = parse_tree_no_optimization(false);

  if (error != ErrorParse::None)
  {
    return error;
  }

  optimize(root);

  if (build_operators)
  {
    error = generate_operators();
  }

  return error;
}

mini_jit::EinsumTree::ErrorParse mini_jit::EinsumTree::generate_operators()
{
  if (root == nullptr)
  {
    std::cerr << "EinsumTree: Cannot execute, root is null." << std::endl;
    return ErrorParse::InvalidRoot;
  }

  return generate_operator_node(root);
}

mini_jit::EinsumTree::ErrorParse mini_jit::EinsumTree::generate_operator_node(EinsumNode *node)
{
  if (node->type == NodeType::Leaf)
  {
    return ErrorParse::None;
  }
  else if (node->type == NodeType::Transposition)
  {
    release_assert(node->left != nullptr, "Expected the left child of contraction to be a valid pointer.");
    release_assert(node->right == nullptr, "Expected the right child of contraction to be a nullptr.");

    ErrorParse error = generate_operator_node(node->left);

    if (error != ErrorParse::None)
    {
      return error;
    }

    TensorConfig config = lower_node(node);
    TensorOperation::error_t error_setup = node->tensor_op.setup(config);
    error = parse_setup_error(error_setup);

    if (error != ErrorParse::None)
    {
      return error;
    }

#ifdef SAVE_JITS_TO_FILE
    node->tensor_op.write_kernel_to_file(node->name());
#endif  // SAVE_JITS_TO_FILE
  }
  else if (node->type == NodeType::Contraction)
  {
    release_assert(node->left != nullptr, "Expected the left child of contraction to be a valid pointer.");
    release_assert(node->right != nullptr, "Expected the right child of contraction to be a valid pointer.");

    ErrorParse error = generate_operator_node(node->left);
    if (error != ErrorParse::None)
    {
      return error;
    }

    error = generate_operator_node(node->right);
    if (error != ErrorParse::None)
    {
      return error;
    }

    TensorConfig config = lower_node(node);
    TensorOperation::error_t error_setup = node->tensor_op.setup(config);
    error = parse_setup_error(error_setup);

    if (error != ErrorParse::None)
    {
      return error;
    }

#ifdef SAVE_JITS_TO_FILE
    node->tensor_op.write_kernel_to_file(node->name());
#endif  // SAVE_JITS_TO_FILE
  }
  else
  {
    release_assert(false, "Found unhandled einsum tree node type.");
  }
  return ErrorParse::None;
}

void mini_jit::EinsumTree::reorder_left_node(EinsumNode *node)
{
  release_assert(node->left != nullptr, "Expected a valid pointer.");

  int32_t indexLeftMDim = findMDim(node);
  int32_t indexLeftKDim = findKDim(node, true);

  release_assert(indexLeftKDim != -1, "Did not find a 'k' dimension in left child.");
  release_assert(indexLeftMDim != -1, "Did not find a 'm' dimension in left child.");

  if (indexLeftKDim == static_cast<int32_t>(node->left->output_dim_ids.size()) - 2 &&
      indexLeftMDim == static_cast<int32_t>(node->left->output_dim_ids.size()) - 1)
  {
    // Already ordered
    return;
  }

  std::vector<int64_t> reorderDimIds = node->left->output_dim_ids;  // copy
  // iter_swap -> swap values between two indices
  std::iter_swap(reorderDimIds.begin() + indexLeftMDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 1);
  if (indexLeftKDim != static_cast<int32_t>(node->left->output_dim_ids.size()) - 1)
  {
    std::iter_swap(reorderDimIds.begin() + indexLeftKDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 2);
  }
  else  // Swapped mDim with kDim -> kDim was placed at indexLeftMDim
  {
    std::iter_swap(reorderDimIds.begin() + indexLeftMDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 2);
  }

  if (node->left->type == NodeType::Leaf)
  {
    // Add additional Permutation Node
    EinsumNode *reorderNode = new EinsumNode();
    reorderNode->type = NodeType::Transposition;
    reorderNode->output_dim_ids = std::move(reorderDimIds);

    reorderNode->left = node->left;
    node->left = reorderNode;
  }
  else
  {
    // Only reorder the output of the left operation
    node->left->output_dim_ids = std::move(reorderDimIds);
  }
}

void mini_jit::EinsumTree::reorder_right_node(EinsumNode *node)
{
  release_assert(node->right != nullptr, "Expected a valid pointer.");

  int32_t indexRightNDim = findNDim(node);
  int32_t indexRightKDim = findKDim(node, false);

  release_assert(indexRightKDim != -1, "Did not find a 'k' dimension in right child.");
  release_assert(indexRightNDim != -1, "Did not find a 'm' dimension in right child.");

  if (indexRightNDim == static_cast<int32_t>(node->right->output_dim_ids.size()) - 2 &&
      indexRightKDim == static_cast<int32_t>(node->right->output_dim_ids.size()) - 1)
  {
    // Already ordered
    return;
  }

  std::vector<int64_t> reorderDimIds = node->right->output_dim_ids;  // copy
  // iter_swap -> swap values between two indices
  std::iter_swap(reorderDimIds.begin() + indexRightKDim, reorderDimIds.begin() + node->right->output_dim_ids.size() - 1);
  if (indexRightNDim != static_cast<int32_t>(node->right->output_dim_ids.size()) - 1)
  {
    std::iter_swap(reorderDimIds.begin() + indexRightNDim, reorderDimIds.begin() + node->right->output_dim_ids.size() - 2);
  }
  else  // Swapped kDim with nDim -> nDim was placed at indexRightKDim
  {
    std::iter_swap(reorderDimIds.begin() + indexRightKDim, reorderDimIds.begin() + node->right->output_dim_ids.size() - 2);
  }

  if (node->right->type == NodeType::Leaf)
  {
    // Add additional Permutation Node
    EinsumNode *reorderNode = new EinsumNode();
    reorderNode->type = NodeType::Transposition;
    reorderNode->output_dim_ids = std::move(reorderDimIds);

    reorderNode->left = node->right;
    node->right = reorderNode;
  }
  else
  {
    // Only reorder the output of the right operation
    node->right->output_dim_ids = std::move(reorderDimIds);
  }
}

int32_t mini_jit::EinsumTree::findMDim(EinsumNode *node)
{
  int64_t mDim = node->output_dim_ids.back();
  for (int32_t i = node->left->output_dim_ids.size() - 1; i >= 0; i--)
  {
    if (node->left->output_dim_ids[i] == mDim)
    {
      return i;
    }
  }

  return -1;
}

int32_t mini_jit::EinsumTree::findNDim(EinsumNode *node)
{
  release_assert(node != nullptr, "Expected a valid pointer");
  release_assert(node->left != nullptr, "Expected a valid left child pointer");
  release_assert(node->right != nullptr, "Expected a valid right child pointer");

  for (int32_t iParent = node->output_dim_ids.size() - 1; iParent >= 0; iParent--)
  {
    for (int32_t iLeft = node->left->output_dim_ids.size() - 1; iLeft >= 0; iLeft--)
    {
      // M or C dimension found
      if (node->output_dim_ids[iParent] == node->left->output_dim_ids[iLeft])
      {
        break;
      }

      for (int32_t iRight = node->right->output_dim_ids.size() - 1; iRight >= 0; iRight--)
      {
        // N dimension found
        if (node->right->output_dim_ids[iRight] == node->output_dim_ids[iParent])
        {
          return iRight;
        }
      }
    }
  }

  return -1;
}

int32_t mini_jit::EinsumTree::findKDim(EinsumNode *node, bool getLeftIndex)
{
  release_assert(node != nullptr, "Expected a valid pointer");
  release_assert(node->left != nullptr, "Expected a valid left child pointer");
  release_assert(node->right != nullptr, "Expected a valid right child pointer");

  for (int32_t iParent = node->output_dim_ids.size() - 1; iParent >= 0; iParent--)
  {
    for (int32_t iLeft = node->left->output_dim_ids.size() - 1; iLeft >= 0; iLeft--)
    {
      // M or C dimension found
      if (node->output_dim_ids[iParent] == node->left->output_dim_ids[iLeft])
      {
        break;
      }

      for (int32_t iRight = node->right->output_dim_ids.size() - 1; iRight >= 0; iRight--)
      {
        // K dimension found
        if (node->right->output_dim_ids[iRight] == node->left->output_dim_ids[iLeft])
        {
          if (getLeftIndex)
          {
            return iLeft;
          }
          else
          {
            return iRight;
          }
        }
      }
    }
  }

  return -1;
}