#include "../../../include/MachineLearningCompiler/Tensor.h"
#include "../../interface/TensorUtils.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>
#include <vector>

TEST_CASE("Test tensor fill_random", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};

  size_t total_size1 = shape1[0] * shape1[1];
  float *data1 = new float[total_size1];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = std::nanf("1");
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::fill_random(tensor1);

  for (size_t i = 0; i < total_size1; i++)
  {
    CAPTURE(i);
    REQUIRE(!std::isnan(tensor1.data[i]));
  }
}

TEST_CASE("Test tensor einsum", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};
  std::vector<uint64_t> shape2 = {4, 5};
  std::vector<uint64_t> shape3 = {3, 5};

  size_t total_size1 = shape1[0] * shape1[1];
  size_t total_size2 = shape2[0] * shape2[1];
  size_t total_size3 = shape3[0] * shape3[1];

  float *data1 = new float[total_size1];
  float *data2 = new float[total_size2];
  float *data3 = new float[total_size3];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < total_size2; ++i)
  {
    data2[i] = static_cast<float>(2 * i);
  }
  for (size_t i = 0; i < total_size3; ++i)
  {
    data3[i] = static_cast<float>(3 * i);
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::Tensor tensor2(data2, shape2);
  mlc::Tensor tensor3(data2, shape3);

  mlc::einsum({tensor1, tensor2}, tensor3, "[0,1],[1,2]->[0,2]");
}