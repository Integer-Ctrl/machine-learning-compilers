#ifndef MINI_JIT_TENSOROPTIMIZATION_H
#define MINI_JIT_TENSOROPTIMIZATION_H

#include "TensorConfig.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif  // USE_OPENMP

namespace mini_jit
{
  class TensorOptimization
  {
  private:
#ifdef USE_OPENMP
    /// @brief The number of processors to use for parallel work
    const int thread_count = omp_get_max_threads();

#else
    /// @brief The number of processors to use for parallel work
    const int thread_count = 1;
#endif  // USE_OPENMP

    /// @brief The inbalanced percentage of parallelism that can be achieved.
    const double maximum_inbalanced_parallel_precentage = 1.0 / 100;  // 1%

    /// @brief The dimension count when fusing or splitting is applied.
    const uint32_t fuse_split_dimension_size = 256;

    /**
     * @brief Runs the optimization primitive identification.
     *
     * @param config The configuration object to use.
     */
    void _primitive_identification(TensorConfig &config);

    /**
     * @brief Runs the optimization shared identification.
     *
     * @param config The configuration object to use.
     */
    void _shared_identification(TensorConfig &config);

    /**
     * @brief Runs the optimization dimension reordering favoring the shared optimization.
     *
     * @param config The configuration object to use.
     */
    void _dimension_reordering_shared(TensorConfig &config);

    /**
     * @brief Runs the optimization dimension reordering favoring the dimension fusing optimization.
     *
     * @param config The configuration object to use.
     */
    void _dimension_reordering_fusing(TensorConfig &config);

    /**
     * @brief Swaps two elements in the vectors of the config.
     *
     * @param config The configuration object to use.
     * @param index1 The index of element 1 to be set a position of index2.
     * @param index2 The index of element 2 ot be set a position of index1.
     */
    void _swap_elements(TensorConfig &config, size_t index1, size_t index2);

    /**
     * @brief Moves an element from the old index to the new index position.
     * 
     * @param config The configuration object to use.
     * @param old_index The index on the current postion.
     * @param new_index The index that should be the new position.
     */
    void _move_elements(TensorConfig &config, size_t old_index, size_t new_index);

    /**
     * @brief Runs the optimization dimension splitting.
     *
     * @param config The configuration object to use.
     */
    void _dimension_splitting(TensorConfig &config);

    /**
     * @brief Runs the optimization dimension fusing.
     *
     * @param config The configuration object to use.
     */
    void _dimension_fusing(TensorConfig &config);

  public:
    /**
     * @brief Optimize the given configuration.
     *
     * @param config The configuration to be optimized.
     * @return TensorConfig The optimized configuration.
     */
    TensorConfig optimize(TensorConfig config);

    /**
     * @brief Optimizes the config by identifying the primitive dimension.
     *
     * @param config The configuration to be optimized.
     * @return TensorConfig The optimized configuration.
     */
    TensorConfig optimize_primitive_identification(TensorConfig config);

    /**
     * @brief Optimizes the config by identifying the shared dimension.
     *
     * @param config The configuration to be optimized.
     * @return TensorConfig The optimized configuration.
     */
    TensorConfig optimize_shared_identification(TensorConfig config);

    /**
     * @brief Optimizes the config by dimension reordering favoring the shared optimization.
     *
     * @param config The configuration to be optimized.
     * @return TensorConfig The optimized configuration.
     */
    TensorConfig optimize_dimension_reordering_shared(TensorConfig config);

    /**
     * @brief Optimizes the config by dimension reordering favoring the dimension fusing optimization.
     *
     * @param config The configuration to be optimized.
     * @return TensorConfig The optimized configuration.
     */
    TensorConfig optimize_dimension_reordering_fusing(TensorConfig config);

    /**
     * @brief Optimizes the config by splitting the dimensions.
     *
     * @param config The configuration to be optimized.
     * @return TensorConfig The optimized configuration.
     */
    TensorConfig optimize_dimension_splitting(TensorConfig config);

    /**
     * @brief Optimizes the config by fusing the dimensions.
     *
     * @param config The configuration to be optimized.
     * @return TensorConfig The optimized configuration.
     */
    TensorConfig optimize_dimension_fusing(TensorConfig config);
  };
}  // namespace mini_jit

#endif  // MINI_JIT_TENSOROPTIMIZATION_H