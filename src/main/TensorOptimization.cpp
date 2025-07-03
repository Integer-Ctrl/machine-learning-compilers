#include "TensorOptimization.h"
#include "TensorOperation.h"
#include "release_assert.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>

void mini_jit::TensorOptimization::_reorder_helper_adjust_index(int32_t index, int32_t adjust_index, int32_t &primitive_m,
                                                                int32_t &primitive_n, int32_t &primitive_k1, int32_t &primitive_k2)
{
  release_assert(primitive_m != primitive_n, "Expected primitive index m and n to be unequal.");
  release_assert(primitive_m != primitive_k1, "Expected primitive index m and k1 to be unequal.");
  release_assert(primitive_m != primitive_k2, "Expected primitive index m and k2 to be unequal.");
  release_assert(primitive_n != primitive_k1, "Expected primitive index n and k1 to be unequal.");
  release_assert(primitive_n != primitive_k2, "Expected primitive index n and k2 to be unequal.");
  release_assert(primitive_k1 == -1 || primitive_k2 == -1 || primitive_k1 != primitive_k2,
                 "Expected primitive index k1 and k2 to be unequal.");

  if (index == primitive_n)
  {
    primitive_n = adjust_index;
    return;
  }

  if (index == primitive_m)
  {
    primitive_m = adjust_index;
    return;
  }

  if (index == primitive_k1)
  {
    primitive_k1 = adjust_index;
    return;
  }

  if (index == primitive_k2)
  {
    primitive_k2 = adjust_index;
  }
}

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
        int32_t index = std::distance(config.dim_types.begin(), iDim);
        if (index != primitive_k1)
        {
          primitive_k2 = index;
        }
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
    else if (TensorOperation::isUnary(config.main) && *iDim == TensorConfig::dim_t::c)
    {
      int32_t index = std::distance(config.dim_types.begin(), iDim);
      // m-dim = unit stride of in0
      if (*iStrideIn0 == 1)
      {
        primitive_m = index;
      }

      // n-dim = unit stride of out or next largest if m-dim == n-dim
      if (index != primitive_m && (primitive_n == -1 || (config.strides_out[primitive_n] > *iStrideOut)))
      {
        primitive_n = index;
      }
    }
  }

  // m and n are always needed because the output has MxN
  if (primitive_m != -1)
  {
    config.exec_types[primitive_m] = TensorConfig::exec_t::prim;

    if (config.dim_types[primitive_m] == TensorConfig::dim_t::c)
    {
      config.dim_types[primitive_m] = TensorConfig::dim_t::m;
    }
  }

  if (primitive_n != -1)
  {
    config.exec_types[primitive_n] = TensorConfig::exec_t::prim;

    if (config.dim_types[primitive_n] == TensorConfig::dim_t::c)
    {
      config.dim_types[primitive_n] = TensorConfig::dim_t::n;
    }
  }

  if (TensorOperation::isBrgemm(config.main))
  {
    if (primitive_k1 != -1)
    {
      config.exec_types[primitive_k1] = TensorConfig::exec_t::prim;
      config.main = TensorConfig::prim_t::gemm;
    }

    if (primitive_k2 != -1 && primitive_k1 != -1)
    {
      config.exec_types[primitive_k2] = TensorConfig::exec_t::prim;
      config.main = TensorConfig::prim_t::brgemm;
    }
  }
}

void mini_jit::TensorOptimization::_shared_identification(TensorConfig &config)
{
#ifdef MLC_USE_OPENMP
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
#endif  // MLC_USE_OPENMP
}

void mini_jit::TensorOptimization::_dimension_reordering_shared(TensorConfig &config)
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
    int32_t new_index = config.dim_types.size() - 1;
    _swap_elements(config, primitive_k1, new_index);
    _reorder_helper_adjust_index(new_index, primitive_k1, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_k1 = new_index;
    first_prim_index = std::min(primitive_k1, first_prim_index);
  }
  if (primitive_n != -1)
  {
    int32_t new_index = config.dim_types.size() - 1 - (primitive_k1 != -1);
    _swap_elements(config, primitive_n, new_index);
    _reorder_helper_adjust_index(new_index, primitive_n, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_n = new_index;
    first_prim_index = std::min(primitive_n, first_prim_index);
  }
  if (primitive_m != -1)
  {
    int32_t new_index = config.dim_types.size() - 1 - (primitive_n != -1) - (primitive_k1 != -1);
    _swap_elements(config, primitive_m, new_index);
    _reorder_helper_adjust_index(new_index, primitive_m, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_m = new_index;
    first_prim_index = std::min(primitive_m, first_prim_index);
  }
  if (primitive_k2 != -1)
  {
    int32_t new_index = config.dim_types.size() - 4;
    _swap_elements(config, primitive_k2, new_index);
    _reorder_helper_adjust_index(new_index, primitive_k2, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_k2 = new_index;
    first_prim_index = std::min(primitive_k2, first_prim_index);
  }

  TensorConfig::dim_t previous_dim = TensorConfig::dim_t::undefined;

  // Order by the largest strides, but stop at first execution type primitive
  for (int32_t i = 0; i < first_prim_index; ++i)
  {
    uint64_t max_value = 0;
    int32_t max_index = i;

    for (int32_t j = i; j < first_prim_index; ++j)
    {
      if ((config.exec_types[i] != TensorConfig::exec_t::shared && config.exec_types[j] == TensorConfig::exec_t::shared) ||
          (config.exec_types[i] == TensorConfig::exec_t::shared && config.exec_types[j] != TensorConfig::exec_t::shared))
      {
        // Do not reorder shared with none shared dimension
        continue;
      }

      uint64_t value = (config.strides_in0[j] * config.strides_in0[j]) + (config.strides_in1[j] * config.strides_in1[j]) +
                       (config.strides_out[j] * config.strides_out[j]);

      // value/8 if we have a k-dimension
      value >>= (config.dim_types[j] == TensorConfig::dim_t::k) * 3;

      // value/2 if we have the same dimension type as the last dimension, but not for c dimension
      value >>= (config.dim_types[j] == previous_dim && config.dim_types[j] != TensorConfig::dim_t::c) * 1;

      if (value > max_value)
      {
        max_value = value;
        max_index = j;
      }
    }

    _swap_elements(config, max_index, i);
  }
}

void mini_jit::TensorOptimization::_dimension_reordering_fusing(TensorConfig &config)
{
  release_assert(config.dim_types.size() == config.dim_sizes.size(),
                 "Expected the dimension types size to match the dimension sizes size.");
  release_assert(config.dim_types.size() == config.strides_in0.size(), "Expected the dimension types size to match the strides_in0 size.");
  release_assert(config.dim_types.size() == config.strides_in1.size(), "Expected the dimension types size to match the strides_in1 size.");
  release_assert(config.dim_types.size() == config.strides_out.size(), "Expected the dimension types size to match the strides_out size.");
  release_assert(config.dim_types.size() == config.dim_sizes.size(), "Expected the dimension types size to match the dim_sizes size.");

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
    int32_t new_index = config.dim_types.size() - 1;
    _swap_elements(config, primitive_k1, new_index);
    _reorder_helper_adjust_index(new_index, primitive_k1, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_k1 = new_index;
    first_prim_index = std::min(primitive_k1, first_prim_index);
  }
  if (primitive_n != -1)
  {
    int32_t new_index = config.dim_types.size() - 1 - (primitive_k1 != -1);
    _swap_elements(config, primitive_n, new_index);
    _reorder_helper_adjust_index(new_index, primitive_n, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_n = new_index;
    first_prim_index = std::min(primitive_n, first_prim_index);
  }
  if (primitive_m != -1)
  {
    int32_t new_index = config.dim_types.size() - 2 - (primitive_k1 != -1);
    _swap_elements(config, primitive_m, new_index);
    _reorder_helper_adjust_index(new_index, primitive_m, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_m = new_index;
    first_prim_index = std::min(primitive_m, first_prim_index);
  }
  if (primitive_k2 != -1)
  {
    int32_t new_index = config.dim_types.size() - 4;
    _swap_elements(config, primitive_k2, new_index);
    _reorder_helper_adjust_index(new_index, primitive_k2, primitive_m, primitive_n, primitive_k1, primitive_k2);
    primitive_k2 = new_index;
    first_prim_index = std::min(primitive_k2, first_prim_index);
  }

  // Order by the fusion capabilities, but stop at first execution type primitive
  for (int32_t i = 0; i < first_prim_index; ++i)
  {
    for (int32_t j = i; j < first_prim_index; ++j)
    {
      if ((config.exec_types[i] != TensorConfig::exec_t::shared && config.exec_types[j] == TensorConfig::exec_t::shared) ||
          (config.exec_types[i] == TensorConfig::exec_t::shared && config.exec_types[j] != TensorConfig::exec_t::shared))
      {
        // Do not reorder shared with none shared dimension
        continue;
      }

      if (config.dim_types[i] != config.dim_types[j])
      {
        continue;
      }

      if (config.strides_in0[i] == (config.dim_sizes[j] * config.strides_in0[j]) &&
          config.strides_in1[i] == (config.dim_sizes[j] * config.strides_in1[j]) &&
          config.strides_out[i] == (config.dim_sizes[j] * config.strides_out[j]))
      {
        _move_elements(config, i, j);
        break;
      }
      else if (config.strides_in0[j] == (config.dim_sizes[i] * config.strides_in0[i]) &&
               config.strides_in1[j] == (config.dim_sizes[i] * config.strides_in1[i]) &&
               config.strides_out[j] == (config.dim_sizes[i] * config.strides_out[i]))
      {
        _move_elements(config, j, i);
        break;
      }
    }
  }
}

void mini_jit::TensorOptimization::_swap_elements(TensorConfig &config, int64_t index1, int64_t index2)
{
  if (index1 == index2)
  {
    return;
  }

  release_assert(config.dim_types.size() == config.dim_sizes.size(),
                 "Expected the dimension types size to match the dimension sizes size.");
  release_assert(config.dim_types.size() == config.exec_types.size(),
                 "Expected the dimension types size to match the execution types size.");
  release_assert(config.dim_types.size() == config.strides_in0.size(), "Expected the dimension types size to match the strides_in0 size.");
  release_assert(config.dim_types.size() == config.strides_in1.size(), "Expected the dimension types size to match the strides_in1 size.");
  release_assert(config.dim_types.size() == config.strides_out.size(), "Expected the dimension types size to match the strides_out size.");
  release_assert(index1 < static_cast<int64_t>(config.dim_types.size()), "Expected the index1 to be less than the dimension types size.");
  release_assert(index2 < static_cast<int64_t>(config.dim_types.size()), "Expected the index2 to be less than the dimension types size.");
  release_assert(index1 >= 0, "Expected the index1 to be larger equal than 0.");
  release_assert(index2 >= 0, "Expected the index2 to be larger equal than 0.");

  std::iter_swap(config.dim_types.begin() + index1, config.dim_types.begin() + index2);
  std::iter_swap(config.dim_sizes.begin() + index1, config.dim_sizes.begin() + index2);
  std::iter_swap(config.exec_types.begin() + index1, config.exec_types.begin() + index2);
  std::iter_swap(config.strides_in0.begin() + index1, config.strides_in0.begin() + index2);
  std::iter_swap(config.strides_in1.begin() + index1, config.strides_in1.begin() + index2);
  std::iter_swap(config.strides_out.begin() + index1, config.strides_out.begin() + index2);
}

void mini_jit::TensorOptimization::_move_elements(TensorConfig &config, size_t old_index, size_t new_index)
{
  release_assert(config.dim_types.size() == config.dim_sizes.size(),
                 "Expected the dimension types size to match the dimension sizes size.");
  release_assert(config.dim_types.size() == config.exec_types.size(),
                 "Expected the dimension types size to match the execution types size.");
  release_assert(config.dim_types.size() == config.strides_in0.size(), "Expected the dimension types size to match the strides_in0 size.");
  release_assert(config.dim_types.size() == config.strides_in1.size(), "Expected the dimension types size to match the strides_in1 size.");
  release_assert(config.dim_types.size() == config.strides_out.size(), "Expected the dimension types size to match the strides_out size.");
  release_assert(old_index < config.dim_types.size(), "Expected the index1 to be less than the dimension types size.");
  release_assert(new_index < config.dim_types.size(), "Expected the index2 to be less than the dimension types size.");

  std::rotate(config.dim_types.begin() + new_index, config.dim_types.begin() + old_index, config.dim_types.begin() + old_index + 1);
  std::rotate(config.dim_sizes.begin() + new_index, config.dim_sizes.begin() + old_index, config.dim_sizes.begin() + old_index + 1);
  std::rotate(config.exec_types.begin() + new_index, config.exec_types.begin() + old_index, config.exec_types.begin() + old_index + 1);
  std::rotate(config.strides_in0.begin() + new_index, config.strides_in0.begin() + old_index, config.strides_in0.begin() + old_index + 1);
  std::rotate(config.strides_in1.begin() + new_index, config.strides_in1.begin() + old_index, config.strides_in1.begin() + old_index + 1);
  std::rotate(config.strides_out.begin() + new_index, config.strides_out.begin() + old_index, config.strides_out.begin() + old_index + 1);
}

void mini_jit::TensorOptimization::_dimension_splitting(TensorConfig &config)
{
  for (size_t i = 0; i < config.dim_sizes.size(); ++i)
  {
    int64_t size = config.dim_sizes[i];
    if (size >= fuse_split_dimension_size)
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
        int64_t new_size2 = best_dominator;
        int64_t new_size1 = size / best_dominator;

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
    if (config.dim_sizes.size() <= 2)
    {
      return;
    }

    // Check if adjacent dims have the same type and their product is less equal than 256
    // stride(X) = |Y| * stride(Y)
    if (config.dim_types[i] == config.dim_types[i + 1] && config.strides_in0[i] == (config.dim_sizes[i + 1] * config.strides_in0[i + 1]) &&
        config.strides_in1[i] == (config.dim_sizes[i + 1] * config.strides_in1[i + 1]) &&
        config.strides_out[i] == (config.dim_sizes[i + 1] * config.strides_out[i + 1]))
    {
      int64_t fused_size = config.dim_sizes[i] * config.dim_sizes[i + 1];
      if (fused_size <= fuse_split_dimension_size)
      {
        // Fuse dimension i and i+1
        config.dim_sizes[i + 1] = fused_size;
        config.dim_types.erase(config.dim_types.begin() + i);
        config.dim_sizes.erase(config.dim_sizes.begin() + i);
        config.strides_in0.erase(config.strides_in0.begin() + i);
        config.strides_in1.erase(config.strides_in1.begin() + i);
        config.strides_out.erase(config.strides_out.begin() + i);
        config.exec_types.erase(config.exec_types.begin() + i);
        // Stay at the same index to check for further fusing
        --i;
      }
    }
    else if (config.dim_types[i] == config.dim_types[i + 1] && config.strides_in0[i + 1] == (config.dim_sizes[i] * config.strides_in0[i]) &&
             config.strides_in1[i + 1] == (config.dim_sizes[i] * config.strides_in1[i]) &&
             config.strides_out[i + 1] == (config.dim_sizes[i] * config.strides_out[i]))
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
  _dimension_reordering_fusing(config);

  _dimension_splitting(config);

  _dimension_fusing(config);

  _primitive_identification(config);

  _dimension_reordering_shared(config);

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

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize_dimension_reordering_shared(TensorConfig config)
{
  _dimension_reordering_shared(config);
  return config;
}

mini_jit::TensorConfig mini_jit::TensorOptimization::optimize_dimension_reordering_fusing(TensorConfig config)
{
  _dimension_reordering_fusing(config);
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
