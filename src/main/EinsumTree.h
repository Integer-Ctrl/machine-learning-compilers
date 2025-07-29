#ifndef MINI_JIT_EINSUM_TREE_H
#define MINI_JIT_EINSUM_TREE_H

#include "TensorConfig.h"
#include "TensorOperation.h"
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
      UndefinedNode = 7,
      InvalidRoot = 8,

      err_wrong_dtype = 101,
      err_wrong_dimension = 102,
      err_wrong_primitive = 103,
      err_wrong_first_touch_primitive = 104,
      err_wrong_main_primitive = 105,
      err_wrong_last_touch_primitive = 106,
      err_execution_type_not_supported = 107,
      err_invalid_primitive_configuration = 108,
      err_invalid_first_touch_configuration = 109,
      err_invalid_main_configuration = 110,
      err_invalid_last_touch_configuration = 111,
      err_invalid_execution_order = 112,
      err_invalid_strides = 113,
      err_k_dimension_must_not_be_shared = 114,
      err_shared_required_for_parallel_execution = 115,
    };

    enum class ErrorExecute
    {
      None = 0,
      InvalidRoot = 1,
      NotEnoughInputTensors = 2,
      TooManyInputTensors = 3,
      NullPtrAsInputTensor = 5,
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
      int32_t input_tensor_index = -1;
      float *tensor = nullptr;
      mini_jit::TensorOperation tensor_op;

      // Always filled — dims of the output tensor
      std::vector<int64_t> output_dim_ids;

      // Pointers to children
      EinsumNode *left = nullptr;
      EinsumNode *right = nullptr;

      /**
       * Gets a string representation of the einsum tree.
       */
      std::string to_string() const;

      /**
       * @brief Gets the string representation of the dim ids of the node.
       */
      std::string name() const;

      /**
       * Get the size of the tensor represented by this node.
       *
       * @param dim_sizes A vector of dimension sizes corresponding to the output dimensions.
       */
      int64_t get_size(const std::vector<int64_t> dim_sizes) const;

    private:
      /**
       * This method recursively formats the node and its children into a string.
       *
       * @param depth The current depth in the tree, used for indentation.
       * @param connection A string representing the connection type.
       * @param depthString A string representation of the current depth.
       * @return A formatted string representing the einsum tree.
       */
      std::string _to_string(uint depth, std::string connection, std::string depthString) const;
    };

  private:
    uint32_t tensorIndex = 0;
    EinsumNode *root = nullptr;
    const std::string tree_str;
    ErrorParse error_parse = ErrorParse::None;
    std::vector<int64_t> dim_sizes;

    // Parser
    /**
     * Parses a node from the string starting at the given position in the einsum tree.
     * The node can be a leaf, contraction, or transposition node.
     *
     * @param pos The position in the string to start parsing from.
     * @param str The string containing the einsum tree representation.
     * @return A pointer to the parsed EinsumNode.
     */
    EinsumNode *parse_node(size_t &pos, const std::string &str);

    // Lowering
    /**
     * Retrieves the dimension types and sizes for the given EinsumNode.
     *
     * @param node The EinsumNode for which to retrieve the dimension types and sizes.
     * @param id_map A map that associates dimension IDs with a fixed index.
     * @param dim_types A vector to store the dimension types.
     * @param dim_sizes A vector to store the dimension sizes.
     * @param number_of_k A reference to store the number of 'k' dimensions.
     */
    void get_config_dim_types_and_sizes(const mini_jit::EinsumTree::EinsumNode *node, std::map<int64_t, size_t> &id_map,
                                        std::vector<mini_jit::TensorConfig::dim_t> &dim_types, std::vector<int64_t> &dim_sizes,
                                        uint32_t &number_of_k);

    /**
     * Retrieves the strides for the given EinsumNode based on the provided dimension ID map.
     *
     * @param node The EinsumNode for which to retrieve the strides.
     * @param id_map A map that associates dimension IDs with a fixed index.
     * @return A vector of strides corresponding to the dimensions of the node.
     */
    std::vector<int64_t> get_config_strides(const EinsumNode *node, std::map<int64_t, size_t> &id_map);

    /**
     * Executes the tensor operation for the given EinsumNode.
     *
     * @param input_tensors The tensors provided by the user.
     * @param node The EinsumNode to execute.
     * @return An ErrorExecute enum indicating the result of the execution.
     */
    ErrorExecute execute_node(const std::vector<void *> &input_tensors, EinsumNode *node);

    /**
     * Assigns intermediate tensors to the given EinsumNode.
     *
     * @param node The EinsumNode to which the tensors will be assigned.
     */
    void assign_tensor_indices(EinsumNode *node);

    /**
     * Checks if the given EinsumNode has unit stride in the 'n' dimension.
     *
     * @param node The EinsumNode to check.
     * @return true if the node has unit stride in the 'n' dimension, false otherwise.
     */
    bool is_unit_stride_n(EinsumNode *node);

    // Helpers
    /**
     * Parses a dimension list from the string starting at the given position.
     * The dimension list is expected to be a comma-separated list of integers.
     *
     * @param pos The position in the string to start parsing from.
     * @param str The string containing the dimension list.
     * @return A vector of integers representing the parsed dimensions.
     */
    std::vector<int64_t> parse_dim_list(size_t &pos, const std::string &str);

    /**
     * Computes the strides for the given dimension IDs based on the sorted dimension sizes.
     *
     * @param dim_ids A vector of dimension IDs for which to compute the strides.
     * @return A vector of computed strides corresponding to the dimension IDs.
     */
    std::vector<int64_t> compute_strides(const std::vector<int64_t> &dim_ids);

    /**
     * Retrieves the output dimensions for the given dimension IDs based on the sorted dimension sizes.
     *
     * @param dim_ids A vector of dimension IDs for which to retrieve the output dimensions.
     * @return A vector of output dimensions corresponding to the provided dimension IDs.
     */
    std::vector<int64_t> get_output_dims(const std::vector<int64_t> &dim_ids);

    /**
     * Parses the setup error from a TensorOperation error code to an ErrorExecute enum.
     *
     * @param error The error code from TensorOperation.
     * @return An ErrorExecute enum representing the parsed error.
     */
    ErrorParse parse_setup_error(TensorOperation::error_t error);

    // Cleanup
    /**
     * Recursively deletes the EinsumNode tree starting from the given node.
     */
    void delete_tree(EinsumNode *node);

    /**
     * Finds the k-dimension of the left or right child of the given node.
     *
     * @param Node The node to check.
     * @param getLeftIndex If true, finds the k-dimension in the left child; otherwise, in the right child.
     * @return int k-dim index if found, otherwise -1.
     */
    int32_t findKDim(EinsumNode *Node, bool getLeftIndex);

    /**
     * Finds the n-dimension of the right child of the given node.
     *
     * @param Node The node to check.
     * @return int n-dim index if found, otherwise -1.
     */
    int32_t findNDim(EinsumNode *Node);

    /**
     * Finds the n-dimension of the left child of the given node.
     *
     * @param Node The node to check.
     * @return int m-dim index if found, otherwise -1.
     */
    int32_t findMDim(EinsumNode *Node);

    /**
     * @brief Generates the operator to the parsed einsum tree.
     *
     * @return ErrorParse indicating the result of the parsing operation.
     */
    ErrorParse generate_operators();

    /**
     * @brief Swap the strides so that the strides position match the out Ids with the current stride location based on the inIds.
     *
     * @param strides The strides to adjust.
     * @param inIds The ids the strides got calculated with.
     * @param outIds The order of the strides that is expected for the strides.
     */
    std::vector<int64_t> swap_strides_id_based(const std::vector<int64_t> &strides, const std::vector<int64_t> &inIds,
                                               const std::vector<int64_t> &outIds);

  public:
    EinsumTree(const std::string &tree_str);
    EinsumTree(const std::string &tree_str, const std::vector<int64_t> &sorted_dim_sizes);
    ~EinsumTree();

    /**
     * @brief Set the sorted dime sizes of the input tensors.
     *
     * @param sorted_dim_sizes The sorted dim sizes
     */
    void set_sorted_dim_sizes(const std::vector<int64_t> &sorted_dim_sizes);

    const std::vector<int64_t> &get_sorted_dim_sizes();

    /**
     * Parses the einsum tree string and builds the tree structure.
     *
     * @param build_operators indicates if the operators should be generate with the parse.
     * @return ErrorParse indicating the result of the parsing operation.
     */
    ErrorParse parse_tree_no_optimization(bool build_operators = true);

    /**
     * Parses the einsum tree string, builds the tree structure and optimizes the tree.
     *
     * @return ErrorParse indicating the result of the parsing operation.
     */
    ErrorParse parse_tree();

    /**
     * Returns the root node of the EinsumTree.
     *
     * @return Pointer to the root EinsumNode.
     */
    EinsumNode *get_root() const;

    /**
     * @brief Optimizes the einsum tree structure.
     *
     * @param The node and its children to optimize.
     */
    void optimize(EinsumNode *node);

    /**
     * @brief Ensures that the 'm' dimension is unit stride, by swapping if 'n' dimension is unit stride.
     *
     * @param node The node of type tensor contraction.
     */
    void conditional_swap(mini_jit::EinsumTree::EinsumNode *node);

    /**
     * Reorders left node of a contraction to ensure the 'km' dimensions are at the right.
     * The 'm' dimension has unit-stride.
     *
     * @param node The EinsumNode representing the parent child of the contraction.
     */
    void reorder_left_node(EinsumNode *node);

    /**
     * Reorders right node of a contraction to ensure the 'nk' dimensions are at the right.
     * The 'k' dimension has unit-stride.
     *
     * @param node The EinsumNode representing the parent of the contraction.
     */
    void reorder_right_node(EinsumNode *node);

    /**
     * Executes the einsum operation defined by the tree.
     *
     * @param tensors A vector of pointers to the input tensors of the leafs.
     * @return ErrorExecute indicating the result of the execution operation.
     */
    ErrorExecute execute(const std::vector<void *> &tensors);

    /**
     * Lowers the given EinsumNode to a TensorConfig.
     *
     * @param node The EinsumNode to lower.
     * @return A TensorConfig representing the lowered node.
     */
    TensorConfig lower_node(const EinsumNode *node);
  };
};  // namespace mini_jit

#endif