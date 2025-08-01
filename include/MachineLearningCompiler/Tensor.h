#ifndef MLC_TENSOR_H
#define MLC_TENSOR_H
#include "Error.h"
#include "UnaryType.h"
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
    std::vector<uint64_t> strides;

    // deletes the default constructor
    Tensor() = delete;

    /**
     * @brief Construct a new Tensor with with a pointer to memory and the dimension sizes sorted in by stride in descending order.
     *
     * @param data The pointer to the data array.
     * @param dim_sizes The dimension sizes sorted by stride in descending order.
     */
    Tensor(float *data, const std::vector<uint64_t> &dim_sizes) : data(data), dim_sizes(dim_sizes)
    {
      strides.resize(dim_sizes.size());
      if (!dim_sizes.empty())
      {
        strides[dim_sizes.size() - 1] = 1;
        for (size_t i = dim_sizes.size() - 1; i > 0; --i)
        {
          strides[i - 1] = strides[i] * dim_sizes[i];
        }
      }
    };

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
      data = new float[size]{0};
      ownsData = true;

      strides.resize(dim_sizes.size());
      if (!dim_sizes.empty())
      {
        strides[dim_sizes.size() - 1] = 1;
        for (size_t i = dim_sizes.size() - 1; i > 0; --i)
        {
          strides[i - 1] = strides[i] * dim_sizes[i];
        }
      }
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

    /**
     * @brief Converts the tensor into its string representation.
     *
     * @param name Name of the tensor that is printed
     * @return std::string The string representation of the tensor.
     */
    std::string to_string(std::string name = "tensor");

    /**
     * @brief Returns the number of elements the tensor has.
     *
     * @return uint64_t The number of elements in the tensor.
     */
    uint64_t size();
  };

  class TensorOperation
  {
  public:
    virtual ~TensorOperation()
    {
    }

    /**
     * @brief Executes the setup einsum expression with input tensor of the same size.
     *
     * @param inputs The inputs to be einsum calculation.
     * @param output The output of the einsum calculation.
     * @return Error The error code or ErrorType::None on success.
     */
    virtual Error execute(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output) = 0;

    /**
     * @brief Executes the setup einsum expression with input tensor of the same size.
     *
     * @param inputs The inputs to be einsum calculation.
     * @param output The output of the einsum calculation.
     * @return Error The error code or ErrorType::None on success.
     */
    virtual Error execute(const std::vector<const Tensor *> &inputs, Tensor &output) = 0;

    /**
     * @brief Gets the error that was produces during the setup of the tree.
     *
     * @return Error The error code or ErrorType::None on success.
     */
    virtual Error getSetupError() const = 0;
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
   * @brief Fills the tensor with counting upwards numbers.
   *
   * @param tensor The tensor to fill.
   * @param start The number to start counting from.
   * @param step The amount to increase everytime.
   */
  void fill_counting_up(Tensor &tensor, float start, float step);

  /**
   * @brief Fills the tensor with counting downwards numbers.
   *
   * @param tensor The tensor to fill.
   * @param start The number to start counting from.
   * @param step The amount to decrease everytime.
   */
  void fill_counting_down(Tensor &tensor, float start, float step);

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
   * @brief Sets up the einsum tree for contraction based on the given tensor dimensions and tree.
   *
   * @param inputs The input tensors shapes.
   * @param output The output tensor shape.
   * @param tree The einsum tree to contract in the format [in0],[in1]->[out].
   */
  TensorOperation *einsum_operation(const std::vector<std::vector<uint64_t>> &inputs, const std::vector<uint64_t> &output,
                                    const std::string &tree);

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
   * @brief Performs a contraction on two input tensor and one output tensor. Before and after the contraction, a first touch unary and a
   * last touch unary are applied to the output tensor.
   *
   * @param input0 The first input tensor.
   * @param input1 The second input tensor.
   * @param output The output to add the result to.
   * @param contraction The string to show the dimension to be contracted in the format [in0],[in1]->[out].
   * @param firstTouch The unary that should be execute before the contraction.
   * @param lastTouch The unary that should be executed after the contraction.
   * @return Error The error code or ErrorType::None on success.
   */
  Error contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction, const UnaryType firstTouch,
                    const UnaryType lastTouch);

  /**
   * @brief Perform a general matrix-matrix multiplication and adds it to the output.
   *
   * @param input0 The first input tensor in the form MxK where M is the leading dimension.
   * @param input1 The second input tensor in the form KxN where K is the leading dimension.
   * @param output The output to add the result to in the form MxN where M is the leading dimension.
   * @return Error The error code or ErrorType::None on success.
   */
  Error gemm(const Tensor &input0, const Tensor &input1, Tensor &output);

  /**
   * @brief Performs a zero unary that sets the output tensor to zero.
   *
   * @param input The input tensor.
   * @return Error The error code or ErrorType::None on success.
   */
  Error unary_zero(Tensor &input);

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
  Error unary_identity(const Tensor &input, Tensor &output);
}  // namespace mlc

#endif  // MLC_TENSOR