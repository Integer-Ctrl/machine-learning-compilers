#ifndef MINI_JIT_TENSORCONFIG_H
#define MINI_JIT_TENSORCONFIG_H

#include "TensorOperation.h"

namespace mini_jit
{
  struct TensorConfig
  {
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

    /// @brief The first touch primitive to be executed.
    prim_t first_touch;

    /// @brief The main primitive to be executed.
    prim_t main;

    /// @brief The last touch primitive to be executed.
    prim_t last_touch;

    /// @brief The dimensions types of each dimension.
    std::vector<dim_t> dim_types;

    /// @brief The execution types of each dimension.
    std::vector<exec_t> exec_types;

    /// @brief The dim_sizes that are supported.
    std::vector<int64_t> dim_sizes;

    /// @brief The strides of the first input of each dimension.
    std::vector<int64_t> strides_in0;

    /// @brief The strides of the second input of each dimension.
    std::vector<int64_t> strides_in1;

    /// @brief The strides of the output of each dimension.
    std::vector<int64_t> strides_out;

    /// @brief The data type to be used in the tensor operation.
    dtype_t dtype;

    /**
     * @brief Compares the two configuration and check if all values are equal.
     *
     * @param config1 The first configuration.
     * @param config2 The second configuration.
     * @return true Both configuration are equal.
     * @return false Both configuration are NOT equal.
     */
    static bool equals(const TensorConfig &config1, const TensorConfig config2);
  };
}  // namespace mini_jit

#endif  // MINI_JIT_TENSORCONFIG_H