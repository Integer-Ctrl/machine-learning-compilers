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

void mlc::fill_counting_up(Tensor &tensor, float start, float step)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  int64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < size; i++)
  {
    tensor.data[i] = start + i * step;
  }
}

void mlc::fill_counting_down(Tensor &tensor, float start, float step)
{
  if (tensor.dim_sizes.size() == 0)
  {
    return;
  }

  int64_t size = internal::getTensorSize(&tensor);

#ifdef MLC_USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < size; i++)
  {
    tensor.data[i] = start - i * step;
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

void mlc::internal::tensor_dim_to_string(mlc::Tensor *tensor, std::string &str, size_t dim, size_t offset, std::string indent)
{
  if (dim == tensor->dim_sizes.size() - 1)
  {
    str += "[";
    for (size_t i = 0; i < tensor->dim_sizes[dim]; ++i)
    {
      if (i > 0)
      {
        str += ", ";
      }
      if (tensor->data == nullptr)
      {
        str += "-";
      }
      else
      {
        str += std::to_string(tensor->data[offset + i]);
      }
    }
    str += "]";
  }
  else
  {
    str += "[";
    indent += " ";

    for (size_t i = 0; i < tensor->dim_sizes[dim]; ++i)
    {
      if (i > 0)
      {
        str += ",\n" + indent;
      }

      tensor_dim_to_string(tensor, str, dim + 1, offset + i * tensor->strides[dim], indent);
    }
    str += "]";
  }
}

std::string mlc::Tensor::to_string(std::string name)
{
  std::string str;
  str += name + "(\n";
  if (dim_sizes.empty())
  {
    str += "[]";
  }
  else
  {
    internal::tensor_dim_to_string(this, str, 0, 0, "");
  }
  str += ")";
  return str;
}

uint64_t mlc::Tensor::size()
{
  return internal::getTensorSize(this);
}
