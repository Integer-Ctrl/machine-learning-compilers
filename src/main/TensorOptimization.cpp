#include "TensorOptimization.h"
#include "TensorOperation.h"
#include "release_assert.h"
#include <cmath>
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
    primitive_k2 = TensorOperation::findMatch(config.dim_types, config.exec_types, mini_jit::TensorConfig::dim_t::k,
                                              mini_jit::TensorConfig::exec_t::prim, primitive_k1);
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

  int end_parallel_index =
    TensorOperation::findMatch(config.dim_types, config.exec_types, TensorConfig::dim_t::k, TensorConfig::exec_t::seq);

  int offset_end = end_parallel_index == -1 ? 0 : (config.dim_types.size() - end_parallel_index);

  uint64_t parallel_size = 1;

  for (auto [iDim, iDimSize] = std::tuple{config.dim_types.begin(), config.dim_sizes.begin()};
       iDim != (config.dim_types.end() - offset_end); ++iDim, ++iDimSize)
  {
    if (*iDim != TensorConfig::dim_t::k)
    {
      parallel_size = *iDimSize;
      config.exec_types[std::distance(config.dim_types.begin(), iDim)] = TensorConfig::exec_t::shared;
      if (parallel_size % thread_count == 0 ||
          static_cast<double>(parallel_size % thread_count) / parallel_size < maximum_inbalanced_parallel_precentage)
      {
        // perfect match for the processor count or under threshold, we don't parallelize any further as we don't benefit from it.
        return;
      }
    }
    else
    {
      // Can not parallelize after k-dim
      return;
    }
  }

#else
  (void)config;
#endif  // USE_OPENMP
}

void mini_jit::TensorOptimization::_dimension_splitting(TensorConfig &config)
{
  for (size_t i = 0; i < config.dim_sizes.size(); ++i)
  {
    int64_t size = config.dim_sizes[i];
    if (size >= 256)
    {
      int64_t best_dominator = -1;
      for (int64_t d = std::floor(std::sqrt(size)); d > 1; --d)
      {
        if (size % d == 0)
        {
          best_dominator = d;
          break;
        }
      }
      if (best_dominator != -1)
      {
        int64_t new_size1 = best_dominator;
        int64_t new_size2 = size / best_dominator;

        // Insert new dimension after i
        config.dim_types.insert(config.dim_types.begin() + i, config.dim_types[i]);
        config.dim_sizes.insert(config.dim_sizes.begin() + i, new_size2);
        config.strides_in0.insert(config.strides_in0.begin() + i, config.strides_in0[i] * new_size1);
        config.strides_in1.insert(config.strides_in1.begin() + i, config.strides_in1[i] * new_size1);
        config.strides_out.insert(config.strides_out.begin() + i, config.strides_out[i] * new_size1);
        config.exec_types.insert(config.exec_types.begin() + i, config.exec_types[i]);

        // Update the original dimension
        config.dim_sizes[i + 1] = new_size1;

        // Skip the next dimension since it's the one we just inserted
        ++i;
      }
    }
  }
}

void mini_jit::TensorOptimization::_dimension_fusing(TensorConfig &config)
{
  for (size_t i = 0; i + 1 < config.dim_sizes.size(); ++i)
  {
    // Check if adjacent dims have the same type and their product is less equal than 256
    if (config.dim_types[i] == config.dim_types[i + 1])
    {
      int64_t fused_size = config.dim_sizes[i] * config.dim_sizes[i + 1];
      if (fused_size <= 256)
      {
        // Fuse dimension i and i+1
        config.dim_sizes[i] = fused_size;
        config.dim_types.erase(config.dim_types.begin() + i + 1);
        config.dim_sizes.erase(config.dim_sizes.begin() + i + 1);
        config.strides_in0.erase(config.strides_in0.begin() + i + 1);
        config.strides_in1.erase(config.strides_in1.begin() + i + 1);
        config.strides_out.erase(config.strides_out.begin() + i + 1);
        config.exec_types.erase(config.exec_types.begin() + i + 1);
        // Stay at the same index to check for further fusing
        --i;
      }
    }
  }
}

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize(TensorConfig config)
{
  _primitive_identification(config);

  // TODO: reordering for better shared performance

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

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize_dimension_splitting(TensorConfig config)
{
  _dimension_splitting(config);
  return config;
}

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize_dimension_fusing(TensorConfig config)
{
  _dimension_fusing(config);
  return config;
}
