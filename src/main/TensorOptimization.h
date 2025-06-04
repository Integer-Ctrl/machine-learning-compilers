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
    const int thread_count = omp_get_num_threads();
#else
    /// @brief The number of processors to use for parallel work
    const int thread_count = 1;
#endif  // USE_OPENMP

    /// @brief The inbalanced percentage of parallelism that can be achieved.
    const double maximum_inbalanced_parallel_precentage = 1.0 / 100;  // 1%

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
  };
}  // namespace mini_jit

#endif  // MINI_JIT_TENSOROPTIMIZATION_H