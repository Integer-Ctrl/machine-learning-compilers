#include "TensorOptimization.h"
#include "TensorOperation.h"
#include "release_assert.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <omp.h>

void mini_jit::TensorOptimization::_primitive_identification(TensorConfig &config)
{
  release_assert(config.dim_types.size() == config.strides_in0.size(), "Expected the dimension types size to match the strides_in0 size.");
  release_assert(config.dim_types.size() == config.strides_in1.size(), "Expected the dimension types size to match the strides_in1 size.");
  release_assert(config.dim_types.size() == config.strides_out.size(), "Expected the dimension types size to match the strides_out size.");

  int32_t primitive_m =
    TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::exec_t::prim);
  int32_t primitive_n =
    TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::exec_t::prim);
  int32_t primitive_k1 =
    TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::exec_t::prim);
  int32_t primitive_k2 = -1;
  if (primitive_k1 != -1)
  {
    primitive_k2 = primitive_k1;
    primitive_k1 = TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::k,
                                              mini_jit::TensorConfig::exec_t::prim, primitive_k2 + 1);
    if (primitive_k1 == -1)
    {
      primitive_k1 = primitive_k2;
      primitive_k2 = -1;
    }
  }

  bool fixed_n = primitive_n != -1;
  bool fixed_k2 = primitive_k2 != -1;

  for (auto [iDim, iStrideIn0, iStrideIn1, iStrideOut] =
         std::tuple{config.dim_types.begin(), config.strides_in0.begin(), config.strides_in1.begin(), config.strides_out.begin()};
       iDim != config.dim_types.end(); ++iDim, ++iStrideIn0, ++iStrideIn1, ++iStrideOut)
  {
    if (*iDim == TensorConfig::dim_t::k)
    {
      if (*iStrideIn1 == 1 && primitive_k1 == -1)
      {
        primitive_k1 = std::distance(config.dim_types.begin(), iDim);
        continue;
      }

      int64_t primitive_k2_stride = std::numeric_limits<int64_t>::max();
      if (primitive_k2 != -1)
      {
        primitive_k2_stride = std::min(config.strides_in0[primitive_k2], config.strides_in1[primitive_k2]);
      }

      int64_t primitive_stride = std::min(*iStrideIn0, *iStrideIn1);

      if (fixed_k2 == false && (primitive_k2 == -1 || primitive_stride < primitive_k2_stride))
      {
        primitive_k2 = std::distance(config.dim_types.begin(), iDim);
      }
    }
    else if (*iDim == TensorConfig::dim_t::m)
    {
      if (*iStrideIn0 == 1 && primitive_m == -1)
      {
        primitive_m = std::distance(config.dim_types.begin(), iDim);
      }
    }
    else if (*iDim == TensorConfig::dim_t::n)
    {
      int64_t primitive_n_stride = std::numeric_limits<int64_t>::max();
      if (primitive_n != -1)
      {
        primitive_n_stride = std::min(config.strides_out[primitive_n], config.strides_in1[primitive_n]);
      }

      int64_t primitive_stride = std::min(*iStrideOut, *iStrideIn1);
      if (TensorOperation::isUnary(config.main))  // ignore second input
      {
        primitive_n_stride = config.strides_out[primitive_n];
        primitive_stride = *iStrideOut;
      }

      if (fixed_n == false && (primitive_n == -1 || primitive_stride < primitive_n_stride))
      {
        primitive_n = std::distance(config.dim_types.begin(), iDim);
      }
    }
  }

  // m and n are always needed because the output has MxN
  if (primitive_m != -1)
  {
    config.exec_types[primitive_m] = TensorConfig::exec_t::prim;
  }

  if (primitive_n != -1)
  {
    config.exec_types[primitive_n] = TensorConfig::exec_t::prim;
  }

  if (TensorOperation::isBrgemm(config.main))
  {
    if (config.main == TensorConfig::prim_t::gemm)
    {
      // one additional k dim
      if (primitive_k1 != -1)
      {
        config.exec_types[primitive_k1] = TensorConfig::exec_t::prim;
      }
    }
    else if (config.main == TensorConfig::prim_t::brgemm)
    {
      // two additional k dims
      if (primitive_k1 != -1)
      {
        config.exec_types[primitive_k1] = TensorConfig::exec_t::prim;
      }
      if (primitive_k2 != -1)
      {
        config.exec_types[primitive_k2] = TensorConfig::exec_t::prim;
      }
    }
    else
    {
      release_assert(false, "Found unhandled brgemm class primitive.");
    }
  }
}

void mini_jit::TensorOptimization::_shared_identification(TensorConfig &config)
{
#ifdef USE_OPENMP
  release_assert(config.dim_types.size() == config.dim_sizes.size(),
                 "Expected the dimension types size to match the dimension sizes size.");

  int32_t end_parallel_index =
    TensorOperation::findMatch(config.dim_types, config.exec_types, TensorConfig::dim_t::k, TensorConfig::exec_t::seq);
  auto iter_first_prim_index = std::find(config.exec_types.begin(), config.exec_types.end(), TensorConfig::exec_t::prim);
  int32_t first_prim_index =
    iter_first_prim_index == config.exec_types.end() ? -1 : std::distance(config.exec_types.begin(), iter_first_prim_index);

  int32_t offset_end_prim = first_prim_index == -1 ? 0 : (config.strides_in0.size() - first_prim_index);
  int32_t offset_end_parallel = end_parallel_index == -1 ? 0 : (config.dim_types.size() - end_parallel_index);
  int32_t offset_end = std::max(offset_end_parallel, offset_end_prim);

  uint64_t parallel_size = 1;

  for (auto [iDim, iDimSize] = std::tuple{config.dim_types.begin(), config.dim_sizes.begin()};
       iDim != (config.dim_types.end() - offset_end); ++iDim, ++iDimSize)
  {
    parallel_size *= *iDimSize;
    config.exec_types[std::distance(config.dim_types.begin(), iDim)] = TensorConfig::exec_t::shared;

    if (parallel_size % thread_count == 0 ||
        (static_cast<double>(parallel_size % thread_count) / parallel_size) < maximum_inbalanced_parallel_precentage)
    {
      // perfect match for the processor count or under threshold, we don't parallelize any further as we don't benefit from it.
      return;
    }
  }
#else
  (void)config;
#endif  // USE_OPENMP
}

void mini_jit::TensorOptimization::_dimension_reordering(TensorConfig &config)
{
  release_assert(config.dim_types.size() == config.dim_sizes.size(),
                 "Expected the dimension types size to match the dimension sizes size.");
  release_assert(config.dim_types.size() == config.strides_in0.size(), "Expected the dimension types size to match the strides_in0 size.");
  release_assert(config.dim_types.size() == config.strides_in1.size(), "Expected the dimension types size to match the strides_in1 size.");
  release_assert(config.dim_types.size() == config.strides_out.size(), "Expected the dimension types size to match the strides_out size.");

  int32_t primitive_m =
    TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::exec_t::prim);
  int32_t primitive_n =
    TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::exec_t::prim);
  int32_t primitive_k1 =
    TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::exec_t::prim);
  int32_t primitive_k2 = -1;
  if (primitive_k1 != -1)
  {
    primitive_k2 = primitive_k1;
    primitive_k1 = TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::k,
                                              mini_jit::TensorConfig::exec_t::prim, primitive_k2 + 1);
    if (primitive_k1 == -1)
    {
      primitive_k1 = primitive_k2;
      primitive_k2 = -1;
    }
  }

  // Put the primitive dimension at the back
  int32_t first_prim_index = config.dim_types.size();
  if (primitive_k1 != -1)
  {
    _swap_elements(config, primitive_k1, config.dim_types.size() - 1);
    first_prim_index = std::min(primitive_k1, first_prim_index);
  }
  if (primitive_n != -1)
  {
    _swap_elements(config, primitive_n, config.dim_types.size() - 2);
    first_prim_index = std::min(primitive_n, first_prim_index);
  }
  if (primitive_m != -1)
  {
    _swap_elements(config, primitive_m, config.dim_types.size() - 3);
    first_prim_index = std::min(primitive_m, first_prim_index);
  }
  if (primitive_k2 != -1)
  {
    _swap_elements(config, primitive_k2, config.dim_types.size() - 4);
    first_prim_index = std::min(primitive_k2, first_prim_index);
  }

  int32_t offset_end = first_prim_index == -1 ? 0 : (config.dim_types.size() - first_prim_index);
  TensorConfig::dim_t previous_dim = TensorConfig::dim_t::undefined;

  // Order by the largest strides, but stop at first execution type primitive
  for (auto [iDim, iStrideIn0, iStrideIn1, iStrideOut, iExec] =
         std::tuple{config.dim_types.begin(), config.strides_in0.begin(), config.strides_in1.begin(), config.strides_out.begin(),
                    config.exec_types.begin()};
       iDim != (config.dim_types.end() - offset_end); ++iDim, ++iStrideIn0, ++iStrideIn1, ++iStrideOut, ++iExec)
  {
    uint64_t max_value = 0;
    int32_t max_index = std::distance(config.dim_types.begin(), iDim);

    for (auto [jDim, jStrideIn0, jStrideIn1, jStrideOut, jExec] = std::tuple{iDim, iStrideIn0, iStrideIn1, iStrideOut, iExec};
         jDim != (config.dim_types.end() - offset_end); ++jDim, ++jStrideIn0, ++jStrideIn1, ++jStrideOut, ++jExec)
    {
      if ((*iExec != TensorConfig::exec_t::shared && *jExec == TensorConfig::exec_t::shared) ||
          (*iExec == TensorConfig::exec_t::shared && *jExec != TensorConfig::exec_t::shared))
      {
        // Do not reorder shared with none shared dimension
        continue;
      }

      uint64_t value = (*jStrideIn0 * *jStrideIn0) + (*jStrideIn1 * *jStrideIn1) + (*jStrideOut * *jStrideOut);

      // value/8 if we have a k-dimension
      value >>= (*jDim == TensorConfig::dim_t::k) * 3;

      // value/2 if we have the same dimension type as the last dimension, but not for c dimension
      value >>= (*jDim == previous_dim && *jDim != TensorConfig::dim_t::c) * 1;

      if (value > max_value)
      {
        max_value = value;
        max_index = std::distance(config.dim_types.begin(), jDim);
      }
    }

    int32_t current_index = std::distance(config.dim_types.begin(), iDim);
    _swap_elements(config, max_index, current_index);
  }
}

void mini_jit::TensorOptimization::_swap_elements(TensorConfig &config, size_t index1, size_t index2)
{
  release_assert(config.dim_types.size() == config.dim_sizes.size(),
                 "Expected the dimension types size to match the dimension sizes size.");
  release_assert(config.dim_types.size() == config.exec_types.size(),
                 "Expected the dimension types size to match the execution types size.");
  release_assert(config.dim_types.size() == config.strides_in0.size(), "Expected the dimension types size to match the strides_in0 size.");
  release_assert(config.dim_types.size() == config.strides_in1.size(), "Expected the dimension types size to match the strides_in1 size.");
  release_assert(config.dim_types.size() == config.strides_out.size(), "Expected the dimension types size to match the strides_out size.");
  release_assert(index1 <= config.dim_types.size(), "Expected the index1 to be less than the dimension types size.");
  release_assert(index2 <= config.dim_types.size(), "Expected the index2 to be less than the dimension types size.");

  std::iter_swap(config.dim_types.begin() + index1, config.dim_types.begin() + index2);
  std::iter_swap(config.dim_sizes.begin() + index1, config.dim_sizes.begin() + index2);
  std::iter_swap(config.exec_types.begin() + index1, config.exec_types.begin() + index2);
  std::iter_swap(config.strides_in0.begin() + index1, config.strides_in0.begin() + index2);
  std::iter_swap(config.strides_in1.begin() + index1, config.strides_in1.begin() + index2);
  std::iter_swap(config.strides_out.begin() + index1, config.strides_out.begin() + index2);
}

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize(TensorConfig config)
{
  _primitive_identification(config);

  _dimension_reordering(config);

  // Only call shared after reordering it only parallelize the first loops until the first seq k-loops at maximum
  _shared_identification(config);
  return config;
}

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize_primitive_identification(TensorConfig config)
{
  _primitive_identification(config);
  return config;
}

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize_shared_identification(TensorConfig config)
{
  _shared_identification(config);
  return config;
}

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize_dimension_reordering(TensorConfig config)
{
  _dimension_reordering(config);
  return config;
}
