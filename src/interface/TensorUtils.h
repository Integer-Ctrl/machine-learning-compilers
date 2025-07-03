#ifndef MLC_TENSORUTILS_H
#define MLC_TENSORUTILS_H
#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include "../main/release_assert.h"
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
     * @return mlc::Tensor nullptr as it should not be possible to get here.
     */
    template <typename T> constexpr const mlc::Tensor *getTensor(const T &)
    {
      static_assert(false, "No generic conversion of tensor possible.");
      release_assert(false, "No generic conversion of tensor possible.");
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
      return tensor;
    }

    /**
     * @brief Gets the pointer to the mlc::Tensor.
     *
     * @param tensor The tensor to get the pointer from.
     * @return Pointer to the mlc::Tensor.
     */
    template <> constexpr const mlc::Tensor *getTensor<const mlc::Tensor *>(const mlc::Tensor *const &tensor)
    {
      return tensor;
    }

    /**
     * @brief Gets the pointer to the mlc::Tensor.
     *
     * @param tensor The tensor to get the pointer from.
     * @return Pointer to the mlc::Tensor.
     */
    template <>
    constexpr const mlc::Tensor *
    getTensor<std::reference_wrapper<const mlc::Tensor>>(const std::reference_wrapper<const mlc::Tensor> &tensor)
    {
      return &(tensor.get());
    }

    /**
     * @brief Get the dim sizes of the input tensors in increased order of their dimension ids.
     *
     * @tparam T The type of the input tensors, either mlc::Tensor* or std::reference_wrapper<const mlc::Tensor>.
     * @param root The root of the EinsumNode tree.
     * @param inputs The input tensors.
     * @param sorted_dim_sizes The vector to store the sorted dimension sizes.
     */
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

    /**
     * @brief Get the dim sizes of the input tensors in increased order of their dimension ids.
     *
     * @param root The root of the EinsumNode tree.
     * @param inputs The input tensors.
     * @param sorted_dim_sizes The vector to store the sorted dimension sizes.
     */
    constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root,
                                               const std::vector<std::reference_wrapper<const mlc::Tensor>> &inputs,
                                               std::vector<int64_t> &sorted_dim_sizes)
    {
      get_sorted_dimensions_sizes<std::reference_wrapper<const mlc::Tensor>>(root, inputs, sorted_dim_sizes);
    }

    /**
     * @brief Get the dim sizes of the input tensors in increased order of their dimension ids.
     *
     * @param root The root of the EinsumNode tree.
     * @param inputs The input tensors.
     * @param sorted_dim_sizes The vector to store the sorted dimension sizes.
     */
    constexpr void get_sorted_dimensions_sizes(const mini_jit::EinsumTree::EinsumNode *root, const std::vector<mlc::Tensor *> &inputs,
                                               std::vector<int64_t> &sorted_dim_sizes)
    {
      get_sorted_dimensions_sizes<mlc::Tensor *>(root, inputs, sorted_dim_sizes);
    }

    /**
     * @brief Helper function to convert the parse error of the EinsumTree to the corresponding mlc::ErrorType.
     *
     * @param error The parse error of type mini_jit::EinsumTree::ErrorParse.
     * @return constexpr mlc::ErrorType The error code or ErrorType::None on success.
     */
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

    /**
     * @brief Converts the error of the EinsumTree execution to the corresponding mlc::ErrorType.
     *
     * @param error The error of type mini_jit::EinsumTree::ErrorExecute.
     * @return constexpr mlc::ErrorType The error code or ErrorType::None on success.
     */
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

    /**
     * @brief Converts the error of the TensorOperation to the corresponding mlc::ErrorType.
     *
     * @param error The error of type mini_jit::TensorOperation::error_t.
     * @return constexpr mlc::ErrorType The converted error of the interface.
     */
    constexpr mlc::ErrorType convertTensorOperationError(mini_jit::TensorOperation::error_t error)
    {
      switch (error)
      {
      case mini_jit::TensorOperation::error_t::success:
        return mlc::ErrorType::None;
      case mini_jit::TensorOperation::error_t::err_wrong_dtype:
        return mlc::ErrorType::ExecuteWrongDType;
      case mini_jit::TensorOperation::error_t::err_wrong_dimension:
        return mlc::ErrorType::ExecuteWrongDimension;
      case mini_jit::TensorOperation::error_t::err_wrong_primitive:
        return mlc::ErrorType::ExecuteWrongPrimitive;
      case mini_jit::TensorOperation::error_t::err_wrong_first_touch_primitive:
        return mlc::ErrorType::ExecuteFirstTouchPrimitive;
      case mini_jit::TensorOperation::error_t::err_wrong_main_primitive:
        return mlc::ErrorType::ExecuteWrongMainPrimitive;
      case mini_jit::TensorOperation::error_t::err_wrong_last_touch_primitive:
        return mlc::ErrorType::ExecuteWrongLastTouchPrimitive;
      case mini_jit::TensorOperation::error_t::err_execution_type_not_supported:
        return mlc::ErrorType::ExecuteTypeNotSupported;
      case mini_jit::TensorOperation::error_t::err_invalid_primitive_configuration:
        return mlc::ErrorType::ExecuteInvalidPrimitiveConfiguration;
      case mini_jit::TensorOperation::error_t::err_invalid_first_touch_configuration:
        return mlc::ErrorType::ExecuteInvalidFirstTouchConfiguration;
      case mini_jit::TensorOperation::error_t::err_invalid_main_configuration:
        return mlc::ErrorType::ExecuteInvalidMainConfiguration;
      case mini_jit::TensorOperation::error_t::err_invalid_last_touch_configuration:
        return mlc::ErrorType::ExecuteInvalidLastTouchConfiguration;
      case mini_jit::TensorOperation::error_t::err_invalid_execution_order:
        return mlc::ErrorType::ExecuteInvalidExecutionOrder;
      case mini_jit::TensorOperation::error_t::err_invalid_strides:
        return mlc::ErrorType::ExecuteInvalidStrides;
      case mini_jit::TensorOperation::error_t::err_k_dimension_must_not_be_shared:
        return mlc::ErrorType::ExecuteKDimensionMustNotBeShared;
      case mini_jit::TensorOperation::error_t::err_shared_required_for_parallel_execution:
        return mlc::ErrorType::ExecuteSharedRequiredForParallelExecution;
      default:
        return mlc::ErrorType::Undefined;
      }
    }

    /**
     * @brief Get the size of the given tensor.
     *
     * @param tensor The tensor to calculate the size from.
     * @return constexpr uint64_t The size of the tensor.
     */
    constexpr uint64_t getTensorSize(const mlc::Tensor *tensor)
    {
      uint64_t size = 1;
      for (auto dim : tensor->dim_sizes)
      {
        size *= dim;
      }
      return size;
    }

    /**
     * @brief Converts a primitive type from the interface unary to a corresponding primitive of the tensor config.
     *
     * @param type The unary type to convert.
     * @return constexpr mini_jit::TensorConfig::prim_t The converted primitive.
     */
    constexpr mini_jit::TensorConfig::prim_t convertPrimitiveType(mlc::UnaryType type)
    {
      switch (type)
      {
      case mlc::UnaryType::None:
        return mini_jit::TensorConfig::prim_t::none;
      case mlc::UnaryType::Identity:
        return mini_jit::TensorConfig::prim_t::copy;
      case mlc::UnaryType::Zero:
        return mini_jit::TensorConfig::prim_t::zero;
      case mlc::UnaryType::ReLU:
        return mini_jit::TensorConfig::prim_t::relu;
      default:
        return mini_jit::TensorConfig::prim_t::none;
      }
    }

    /**
     * @brief Recursively converts the given tensor into a string format.
     *
     * @param tensor The tensor to convert.
     * @param str The string to write to.
     * @param dim The current processed dimension.
     * @param offset The offset from the data to be processed.
     * @param indent The indentation of the current dimension.
     */
    void tensor_dim_to_string(mlc::Tensor *tensor, std::string &str, size_t dim, size_t offset, std::string indent);
  }  // namespace internal
}  // namespace mlc
#endif  // MLC_TENSORUTILS_H
