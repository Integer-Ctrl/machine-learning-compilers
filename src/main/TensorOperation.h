#ifndef MINI_JIT_TENSOR_OPERATION_H
#define MINI_JIT_TENSOR_OPERATION_H

#include "Brgemm.h"
#include "TensorConfig.h"
#include "Unary.h"
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace mini_jit
{
  class TensorOperation
  {

  public:
    /// execution type

    /// error codes
    enum class error_t : int32_t
    {
      success = 0,
      err_wrong_dtype = 1,
      err_wrong_dimension = 2,
      err_wrong_primitive = 3,
      err_wrong_first_touch_primitive = 4,
      err_wrong_main_primitive = 5,
      err_wrong_last_touch_primitive = 6,
      err_execution_type_not_supported = 7,
      err_invalid_primitive_configuration = 8,
      err_invalid_first_touch_configuration = 9,
      err_invalid_main_configuration = 10,
      err_invalid_last_touch_configuration = 11,
      err_invalid_execution_order = 12,
      err_invalid_strides = 13,
      err_k_dimension_must_not_be_shared = 14,
      err_shared_required_for_parallel_execution = 15,
    };

    // stride codes
    enum class stride_t : int32_t
    {
      in0 = 0,
      in1 = 1,
      out = 2,
    };

  private:
    // Keep track over configuration parameters
    TensorConfig config;
    TensorConfig::dtype_t dtype;
    TensorConfig::prim_t prim_first = TensorConfig::prim_t::none;
    TensorConfig::prim_t prim_main = TensorConfig::prim_t::none;
    TensorConfig::prim_t prim_last = TensorConfig::prim_t::none;
    std::span<const TensorConfig::dim_t> dim_types;
    std::span<const TensorConfig::exec_t> exec_types;
    std::span<const int64_t> dim_sizes;
    std::span<const int64_t> strides_in0;
    std::span<const int64_t> strides_in1;
    std::span<const int64_t> strides_out;

    int32_t indexPrimM = -1;
    int32_t indexPrimN = -1;
    int32_t indexPrimK = -1;
    int32_t indexPrimBatch = -1;

    std::variant<Brgemm, Unary> first_touch;
    std::variant<Brgemm, Unary> main_kernel;
    std::variant<Brgemm, Unary> last_touch;

    bool isParallel = false;  // default is sequential execution

    bool hasSetupError = false;

    /**
     * @brief Validates that exactly one m primitive dimension and one n primitive dimension exists.
     *
     * @param dim The dimension types to search through.
     * @param exec The execution types to search through.
     * @return true The configuration is a valid primitive setup.
     * @return false The configuration is NOT a valid primitive setup.
     */
    bool isValidPrimConfig(const std::span<const TensorConfig::dim_t> &dim, const std::span<const TensorConfig::exec_t> &exec,
                           const std::span<const int64_t> &strides_in0, const std::span<const int64_t> &strides_out);

    /**
     * @brief Checks if the K dimension is valid for the given primitive.
     *
     * @param dim The dimension types to search through.
     * @param exec The execution types to search through.
     * @param
     * @param prim The primitive i.e. Gemm or Brgemm to be executed.
     * @return true The configuration is a valid setup.
     * @return false The configuration is NOT a valid setup.
     */
    bool isValidKDim(const std::span<const TensorConfig::dim_t> &dim, const std::span<const TensorConfig::exec_t> &exec,
                     const std::span<const int64_t> &strides_in1, TensorConfig::prim_t prim);

    /**
     * @brief Checks if the configuration is sorted such that the primitives are last.
     *
     * @param exec The execution types of the configuration.
     * @return true The configuration align with the requirement.
     * @return false The configuration NOT algin with the requirement.
     */
    bool isSortedConfiguration(const std::span<const TensorConfig::exec_t> &exec);

    /**
     * @brief Generates the unary kernel.
     *
     * @param unary The unary used for generation.
     * @param prim The primitive that is generated.
     * @param dim_sizes The sizes of each dimension.
     * @return Unary::error_t
     */
    Unary::error_t generateUnary(Unary &unary, TensorConfig::prim_t prim, const std::span<const int64_t> &dim_sizes);

  public:
    /**
     * @brief Checks if the stride matches the given stride.
     *
     * @param expected The stride that is expected.
     * @param index The index of the stride.
     * @param strides The strides of the configuration.
     * @return true The stride matches the expected.
     * @return false The stride NOT matches the expected.
     */
    static bool isExpectedStride(int64_t expected, int index, const std::span<const int64_t> &strides);

    /**
     * @brief Checks if the strides are valid for the given dimension.
     *
     * @param dim The dimension types of the configuration.
     * @param strides The strides of the configuration.
     * @return true The strides are valid.
     * @return false The strides are NOT valid.
     */
    static bool isValidStride(const std::span<const TensorConfig::dim_t> &dim, const std::span<const int64_t> &strides,
                              const stride_t strideType);

    /**
     * @brief Indicates if a primitive fits the Unary generator.
     *
     * @param prim The primitive to check.
     * @return true The primitive is a unary.
     * @return false The primitive is NOT a unary.
     */
    static bool isUnary(TensorConfig::prim_t prim);

    /**
     * @brief Indicates if a primitive fits the Brgemm generator.
     *
     * @param prim The primitive to check.
     * @return true The primitive is a brgemm.
     * @return false The primitive is NOT a brgemm.
     */
    static bool isBrgemm(TensorConfig::prim_t prim);

    /**
     * @brief Finds the matching index of the given pair of dim and exec types.
     *
     * @param dim The dimension types to search through.
     * @param exec The execution types to search through.
     * @param searchDim The acceptable dimension type.
     * @param searchExec The acceptable execution type.
     * @param startIndex The optional start index for the search.
     * @return uint32_t The index of the found match. -1 if not match was found.
     */
    static int32_t findMatch(const std::span<const TensorConfig::dim_t> &dim, const std::span<const TensorConfig::exec_t> &exec,
                             TensorConfig::dim_t searchDim, TensorConfig::exec_t searchExec, uint32_t startIndex = 0);
    /**
     * @brief Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param config The configuration of the tensor dimension and primitives.
     * @return error_t error_t::success on success, other error values otherwise.
     */
    error_t setup(const TensorConfig &config);

    /**
     * Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param dtype             Datatype of all tensor elements.
     * @param prim_first_touch  Type of the first touch primitive.
     * @param prim_main         Type of the main primitive.
     * @param prim_last_touch   Type of the last touch primitive.
     * @param dim_types         Dimension type of the loops (c, m, n, or k).
     * @param exec_types        Execution type of the loops (seq, shared, or prim).
     * @param dim_sizes         Sizes of the dimensions.
     * @param strides_in0       Strides of the first input tensor.
     * @param strides_in1       Strides of the second input tensor (ignored if unary).
     * @param strides_out       Strides of the output tensor.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t setup_no_optimization(TensorConfig::dtype_t dtype, TensorConfig::prim_t prim_first_touch, TensorConfig::prim_t prim_main,
                                  TensorConfig::prim_t prim_last_touch, std::span<const TensorConfig::dim_t> dim_types,
                                  std::span<const TensorConfig::exec_t> exec_types, std::span<const int64_t> dim_sizes,
                                  std::span<const int64_t> strides_in0, std::span<const int64_t> strides_in1,
                                  std::span<const int64_t> strides_out);

    /**
     * Execute the tensor operation.
     *
     * @param tensor_in0 First input tensor.
     * @param tensor_in1 Second input tensor (use nullptr if unary).
     * @param tensor_out Output tensor.
     **/
    void execute(void const *tensor_in0, void const *tensor_in1, void *tensor_out);

    /**
     * General-purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param index_dimension      Dimension index of the loop which is executed.
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_dimension(int64_t index_dimension, char const *ptr_in0, char const *ptr_in1, char *ptr_out, bool first_access,
                           bool last_access);

    
    /**
     * @brief Get the current configuration object.
     * 
     * @return TensorConfig used by the Tensor operation. 
     */
    TensorConfig get_config();
  };
};  // namespace mini_jit

#endif