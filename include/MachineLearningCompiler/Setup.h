#ifndef MLC_SETUP_H
#define MLC_SETUP_H
#include "Error.h"
#include "Tensor.h"
#include <cstdint>
#include <vector>

namespace mlc
{
  class Setup
  {
  public:
    /**
     * @brief Executes the setup einsum expression with input tensor of the same size.
     *
     * @param inputs The inputs to be einsum calculation.
     * @param output The output of the einsum calculation.
     * @return Error The error during the
     */
    virtual Error execute(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output) = 0;

    /**
     * @brief Executes the setup einsum expression with input tensor of the same size.
     *
     * @param inputs The inputs to be einsum calculation.
     * @param output The output of the einsum calculation.
     * @return Error The error during the
     */
    virtual Error execute(const std::vector<const Tensor *> &inputs, Tensor &output) = 0;

    /**
     * @brief Gets the error that was produces during the setup of the tree.
     *
     * @return Error The error that was produces during the setup.
     */
    virtual Error getSetupError() const = 0;
  };

  /**
   * @brief Sets up the einsum tree for contraction based on the given tree.
   *
   * @param inputs The input tensors.
   * @param output The output tensor.
   * @param tree The einsum tree to contract in the format [in0],[in1]->[out].
   */
  Setup &einsum_setup(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output, const std::string &tree);

  /**
   * @brief Executes contractions based on the given tree.
   *
   * @param inputs The input tensors.
   * @param output The output tensor.
   * @param tree The einsum tree to contract in the format [in0],[in1]->[out].
   */
  Setup &einsum_setup(const std::vector<const Tensor *> &inputs, Tensor &output, const std::string &tree);
}  // namespace mlc

#endif  // MLC_SETUP_H