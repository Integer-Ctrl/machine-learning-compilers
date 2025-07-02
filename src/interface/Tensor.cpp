#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/TensorOperation.h"
#include "TensorUtils.h"
#include <iostream>

void mlc::fill_random(Tensor &tensor)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  uint64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; ++i)
  {
    float denominator = 1;
    denominator = static_cast<float>(std::rand());
    if (denominator == 0)
    {
      denominator = 1;
    }

    float numerator = 1;
    numerator = static_cast<float>(std::rand());

    float random = numerator / denominator;

    tensor.data[i] = random;
  }
}

void mlc::fill_number(Tensor &tensor, float number)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  uint64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++)
  {
    tensor.data[i] = number;
  }
}

void mlc::fill_lambda(Tensor &tensor, std::function<float(const Tensor &, size_t)> function)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  uint64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++)
  {
    tensor.data[i] = function(tensor, i);
  }
}
