#ifndef EINSUM_BACKEND_TENSOR_OPERATION_H
#define EINSUM_BACKEND_TENSOR_OPERATION_H

#include "Brgemm.h"
#include "Unary.h"
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace mini_jit
{
  class TensorOperation;

}  // namespace mini_jit

class mini_jit::TensorOperation
{

public:
  /// execution type
  enum class exec_t : uint32_t
  {
    seq = 0,
    prim = 1,
    shared = 2,
  };

  /// primitive type
  enum class prim_t : uint32_t
  {
    none = 0,
    zero = 1,
    copy = 2,
    relu = 3,
    gemm = 4,
    brgemm = 5,
  };

  /// dimension type
  enum class dim_t : uint32_t
  {
    undefined = 0,
    c = 1,
    m = 2,
    n = 3,
    k = 4,
  };

  /// data type
  enum class dtype_t : uint32_t
  {
    fp32 = 0,
    fp64 = 1
  };

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
  mini_jit::TensorOperation::dtype_t dtype;
  mini_jit::TensorOperation::prim_t prim_first = prim_t::none;
  mini_jit::TensorOperation::prim_t prim_main = prim_t::none;
  mini_jit::TensorOperation::prim_t prim_last = prim_t::none;
  std::span<const mini_jit::TensorOperation::dim_t> dim_types;
  std::span<const mini_jit::TensorOperation::exec_t> exec_types;
  std::span<const int64_t> dim_sizes;
  std::span<const int64_t> strides_in0;
  std::span<const int64_t> strides_in1;
  std::span<const int64_t> strides_out;

  int32_t indexPrimM = -1;
  int32_t indexPrimN = -1;
  int32_t indexPrimK = -1;
  int32_t indexPrimBatch = -1;

  std::variant<mini_jit::Brgemm, mini_jit::Unary> first_touch;
  std::variant<mini_jit::Brgemm, mini_jit::Unary> main_kernel;
  std::variant<mini_jit::Brgemm, mini_jit::Unary> last_touch;

  bool hasSetupError = false;

  /**
   * @brief Indicates if a primitive fits the Unary generator.
   *
   * @param prim The primitive to check.
   * @return true The primitive is a unary.
   * @return false The primitive is NOT a unary.
   */
  static bool isUnary(prim_t prim);

  /**
   * @brief Indicates if a primitive fits the Brgemm generator.
   *
   * @param prim The primitive to check.
   * @return true The primitive is a brgemm.
   * @return false The primitive is NOT a brgemm.
   */
  static bool isBrgemm(prim_t prim);

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
  static int32_t findMatch(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec, dim_t searchDim, exec_t searchExec,
                           uint32_t startIndex = 0);

  /**
   * @brief Validates that exactly one m primitive dimension and one n primitive dimension exists.
   *
   * @param dim The dimension types to search through.
   * @param exec The execution types to search through.
   * @return true The configuration is a valid primitive setup.
   * @return false The configuration is NOT a valid primitive setup.
   */
  bool isValidPrimConfig(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec,
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
  bool isValidKDim(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec, const std::span<const int64_t> &strides_in1,
                   prim_t prim);

  /**
   * @brief Checks if the configuration is sorted such that the primitives are last.
   *
   * @param exec The execution types of the configuration.
   * @return true The configuration align with the requirement.
   * @return false The configuration NOT algin with the requirement.
   */
  bool isSortedConfiguration(const std::span<const exec_t> &exec);

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
  static bool isValidStride(const std::span<const dim_t> &dim, const std::span<const int64_t> &strides, const stride_t strideType);

  Unary::error_t generateUnary(Unary &unary, prim_t prim, const std::span<const int64_t> &dim_sizes);

public:
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
  error_t setup(dtype_t dtype, prim_t prim_first_touch, prim_t prim_main, prim_t prim_last_touch, std::span<const dim_t> dim_types,
                std::span<const exec_t> exec_types, std::span<const int64_t> dim_sizes, std::span<const int64_t> strides_in0,
                std::span<const int64_t> strides_in1, std::span<const int64_t> strides_out);

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
};

#endif