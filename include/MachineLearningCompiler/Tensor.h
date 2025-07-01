#ifndef MLC_TENSOR_H
#define MLC_TENSOR_H
#include "Error.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlc
{
  struct Tensor
  {
    bool ownsData = false;
    float *data = nullptr;
    std::vector<uint64_t> dim_sizes;

    // deletes the default constructor
    Tensor() = delete;

    // deletes the copy constructor
    Tensor(const Tensor &) = delete;

    /**
     * @brief Construct a new Tensor with with a pointer to memory and the dimension sizes sorted in by stride in descending order.
     *
     * @param data The pointer to the data array.
     * @param dim_sizes The dimension sizes sorted by stride in descending order.
     */
    Tensor(float *data, const std::vector<uint64_t> &dim_sizes) : data(data), dim_sizes(dim_sizes) {};

    /**
     * @brief Construct a new Tensor with the dimension sizes sorted by stride in descending order.
     *
     * @param dim_sizes The dimension sizes sorted by stride in descending order.
     */
    Tensor(const std::vector<uint64_t> &dim_sizes) : dim_sizes(dim_sizes)
    {
      uint64_t size = 1;
      for (auto dim : dim_sizes)
      {
        size *= dim;
      }
      data = new float[size];
      ownsData = true;
    };

    /**
     * @brief Destroys the tensor.
     */
    ~Tensor()
    {
      if (ownsData && data != nullptr)
      {
        delete[] data;
        data = nullptr;
      }
    }
  };

  /**
   * @brief Fills the tensor with random float data.
   *
   * @param tensor The tensor to fill.
   */
  void fill_random(Tensor &tensor);

  /**
   * @brief Fills the tensor with the given number.
   *
   * @param tensor The tensor to fill.
   * @param number The number used to fill the tensor.
   */
  void fill_number(Tensor &tensor, float number);

  /**
   * @brief Fills the tensor based on the given function.
   *
   * @param tensor The tensor to fill.
   * @param function The function that gets the current tensor and the current index of the tensor as input.
   *            index = index0 * stride0 + index1 * stride1 + ... + indexN * strideN.
   */
  void fill_lambda(Tensor &tensor, std::function<float(const Tensor &, size_t)> function);

  /**
   * @brief
   *
   * @param inputs The input tensors.
   * @param output The output tensor.
   * @param tree The (nested) einsum tree to contract in the format [in0],[in1]->[out].
   * @return Error The error code or ErrorType::None on success.
   */
  Error einsum(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output, const std::string &tree);

  /**
   * @brief Executes contractions based on the given tree.
   *
   * @param inputs The input tensors.
   * @param output The output tensor.
   * @param tree The (nested) einsum tree to contract in the format [in0],[in1]->[out].
   * @return Error The error code or ErrorType::None on success.
   */
  Error einsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree);

  /**
   * @brief Perform a binary contraction and adds it to the output.
   *
   * @param input0 The first input tensor.
   * @param input1 The second input tensor.
   * @param output The output to add the result to.
   * @param contraction The string to show the dimension to be contracted in the format [in0],[in1]->[out].
   * @return Error The error code or ErrorType::None on success.
   */
  Error contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction);

  /**
   * @brief Perform a general matrix-matrix multiplication and adds it to the output.
   *
   * @param input0 The first input tensor.
   * @param input1 The second input tensor.
   * @param output The output to add the result to.
   * @return Error The error code or ErrorType::None on success.
   */
  Error gemm(const Tensor &input0, const Tensor &input1, Tensor output);

  /**
   * @brief Performs a zero unary that sets the output tensor to zero.
   *
   * @param input The input tensor.
   * @param output The output tensor.
   * @return Error The error code or ErrorType::None on success.
   */
  Error unary_zero(const Tensor &input, Tensor &output);

  /**
   * @brief Performs a relu unary that applies Rectified Linear Unit on the tensor input.
   *
   * @param input The input tensor.
   * @param output The ouput tensor.
   * @return Error The error code or ErrorType::None on success.
   */
  Error unary_relu(const Tensor &input, Tensor &output);

  /**
   * @brief Performs a identity unary that copies the input tensor to the output tensor
   *
   * @param input The input tensor.
   * @param output The output tensor.
   * @return Error The error code or ErrorType::None on success.
   */
  Error unary_identity(const Tensor &input, Tensor output);
}  // namespace mlc

#endif  // MLC_TENSOR