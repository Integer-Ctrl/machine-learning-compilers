#ifndef MLC_TENSOR_H
#define MLC_TENSOR_H

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
   * @brief Executes contractions based on the given tree.
   *
   * @param inputs The input tensors.
   * @param output The output tensor.
   * @param tree The einsum tree to contract in the format [in0],[in1]->[out].
   */
  void einsum(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree);

}  // namespace mlc

#endif  // MLC_TENSOR