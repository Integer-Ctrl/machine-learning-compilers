#ifndef MINI_JIT_EINSUM_TREE_H
#define MINI_JIT_EINSUM_TREE_H

#include "TensorConfig.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mini_jit
{
  class EinsumTree
  {

  public:
    enum class ErrorParse
    {
      None = 0,
      ExpectedLeftBracket = 1,
      ExpectedRightBracket = 2,
      ExpectedArrow = 3,
      ExpectedComma = 4,
      ExpectedDimensionList = 5,
      NotAllowedToParseAgain = 6,
    };

    enum class ErrorExecute
    {
      None = 0,
      InvalidRoot = 1,
      NotEnoughInputTensors = 2,
      TooManyInputTensors = 3,
    };

    enum class NodeType
    {
      Leaf,
      Contraction,
      Transposition
    };

    struct EinsumNode
    {
      NodeType type;
      float *tensor;

      // For leaf and contraction nodes
      std::vector<int64_t> input_dims0;
      std::vector<int64_t> input_dims1;  // Only used for contraction

      // Always filled — dims of the output tensor
      std::vector<int64_t> output_dims;

      // Pointers to children
      EinsumNode *left = nullptr;
      EinsumNode *right = nullptr;

      std::string to_string() const;

      int64_t get_size() const;

    private:
      std::string _to_string(uint depth, std::string connection, std::string depthString) const;
    };

  private:
    uint32_t tensorIndex = 0;
    EinsumNode *root = nullptr;
    const std::string tree_str;
    ErrorParse error_parse;
    ErrorExecute error_execute;
    std::map<int, int> dim_sizes;  // Maps dim ID to actual size

    // Parser
    EinsumNode *parse_node(size_t &pos, const std::string &str);

    // Lowering
    TensorConfig lower_node(const EinsumNode *node);
    void *execute_node(EinsumNode *node);
    void assign_tensor(std::vector<void *> tensors, EinsumNode *node);

    // Helpers
    std::vector<TensorConfig::dim_t> infer_dim_types(const std::vector<int64_t> &dims);
    std::vector<int64_t> parse_dim_list(size_t &pos, const std::string &str);
    std::vector<int64_t> compute_strides(const std::vector<int64_t> &dims);

    // Cleanup
    void delete_tree(EinsumNode *node);

  public:
    EinsumTree(const std::string &tree_str, const std::vector<int> &sorted_dim_sizes);
    ~EinsumTree();

    ErrorParse parse_tree();
    EinsumNode *get_root() const;
    ErrorExecute execute(std::vector<void *> tensors);
  };
};  // namespace mini_jit

#endif