#include "../../../include/MachineLearningCompiler/Tensor.h"
#include "../../interface/TensorUtils.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>
#include <vector>

TEST_CASE("Test interface tensor fill_random", "[tensor][correctness]")
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

TEST_CASE("Test interface tensor fill_number", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};

  size_t total_size1 = shape1[0] * shape1[1];
  float *data1 = new float[total_size1];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = std::nanf("1");
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::fill_number(tensor1, 1);

  for (size_t i = 0; i < total_size1; i++)
  {
    CAPTURE(i);
    REQUIRE(tensor1.data[i] == 1);
  }
}

TEST_CASE("Test interface tensor fill_counting_up", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};

  size_t total_size1 = shape1[0] * shape1[1];
  float *data1 = new float[total_size1];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = std::nanf("1");
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::fill_counting_up(tensor1, 5, 0.5);

  for (size_t i = 0; i < total_size1; i++)
  {
    CAPTURE(i);
    REQUIRE(tensor1.data[i] == (0.5f * i + 5));
  }
}

TEST_CASE("Test interface tensor fill_counting_down", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};

  size_t total_size1 = shape1[0] * shape1[1];
  float *data1 = new float[total_size1];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = std::nanf("1");
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::fill_counting_down(tensor1, 5, 1.0);

  for (int64_t i = 0; i < static_cast<int64_t>(total_size1); i++)
  {
    CAPTURE(i);
    REQUIRE(tensor1.data[i] == (-i + 5));
  }
}

TEST_CASE("Test interface tensor fill_lambda", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};

  size_t total_size1 = shape1[0] * shape1[1];
  float *data1 = new float[total_size1];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = std::nanf("1");
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::fill_lambda(tensor1, [](const mlc::Tensor &, size_t index) { return index; });

  for (size_t i = 0; i < total_size1; i++)
  {
    CAPTURE(i);
    REQUIRE(tensor1.data[i] == i);
  }
}

TEST_CASE("Test interface tensor einsum reference", "[tensor][correctness]")
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

  REQUIRE(tensor1.strides.size() == 2);
  REQUIRE(tensor1.strides[0] == 4);
  REQUIRE(tensor1.strides[1] == 1);
  REQUIRE(tensor2.strides.size() == 2);
  REQUIRE(tensor2.strides[0] == 5);
  REQUIRE(tensor2.strides[1] == 1);
  REQUIRE(tensor3.strides.size() == 2);
  REQUIRE(tensor3.strides[0] == 5);
  REQUIRE(tensor3.strides[1] == 1);

  mlc::Error err = mlc::einsum({tensor1, tensor2}, tensor3, "[0,1],[1,2]->[0,2]");
  REQUIRE(err.type == mlc::ErrorType::None);

  delete[] data1;
  delete[] data2;
  delete[] data3;
}

TEST_CASE("Test interface tensor einsum pointer", "[tensor][correctness]")
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
  std::vector<mlc::Tensor *> inputs{&tensor1, &tensor2};

  REQUIRE(tensor1.strides.size() == 2);
  REQUIRE(tensor1.strides[0] == 4);
  REQUIRE(tensor1.strides[1] == 1);
  REQUIRE(tensor2.strides.size() == 2);
  REQUIRE(tensor2.strides[0] == 5);
  REQUIRE(tensor2.strides[1] == 1);
  REQUIRE(tensor3.strides.size() == 2);
  REQUIRE(tensor3.strides[0] == 5);
  REQUIRE(tensor3.strides[1] == 1);

  CAPTURE(inputs);
  mlc::Error err = mlc::einsum(inputs, tensor3, "[0,1],[1,2]->[0,2]");
  REQUIRE(err.type == mlc::ErrorType::None);

  delete[] data1;
  delete[] data2;
  delete[] data3;
}

TEST_CASE("Test interface tensor contraction", "[tensor][correctness]")
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

  REQUIRE(tensor1.strides.size() == 2);
  REQUIRE(tensor1.strides[0] == 4);
  REQUIRE(tensor1.strides[1] == 1);
  REQUIRE(tensor2.strides.size() == 2);
  REQUIRE(tensor2.strides[0] == 5);
  REQUIRE(tensor2.strides[1] == 1);
  REQUIRE(tensor3.strides.size() == 2);
  REQUIRE(tensor3.strides[0] == 5);
  REQUIRE(tensor3.strides[1] == 1);

  mlc::Error err = mlc::contraction(tensor1, tensor2, tensor3, "[0,1],[1,2]->[0,2]");
  REQUIRE(err.type == mlc::ErrorType::None);

  delete[] data1;
  delete[] data2;
  delete[] data3;
}

TEST_CASE("Test interface tensor gemm", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {4, 3};  // k, m
  std::vector<uint64_t> shape2 = {5, 4};  // n, k
  std::vector<uint64_t> shape3 = {5, 3};  // n, m

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

  REQUIRE(tensor1.strides.size() == 2);
  REQUIRE(tensor1.strides[0] == 3);
  REQUIRE(tensor1.strides[1] == 1);
  REQUIRE(tensor2.strides.size() == 2);
  REQUIRE(tensor2.strides[0] == 4);
  REQUIRE(tensor2.strides[1] == 1);
  REQUIRE(tensor3.strides.size() == 2);
  REQUIRE(tensor3.strides[0] == 3);
  REQUIRE(tensor3.strides[1] == 1);

  mlc::Error err = mlc::gemm(tensor1, tensor2, tensor3);
  REQUIRE(err.type == mlc::ErrorType::None);

  delete[] data1;
  delete[] data2;
  delete[] data3;
}

TEST_CASE("Test interface tensor gemm failure", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4, 5};  // Invalid shape for GEMM, should be 2D
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

  REQUIRE(tensor1.strides.size() == 3);
  REQUIRE(tensor1.strides[0] == 20);
  REQUIRE(tensor1.strides[1] == 5);
  REQUIRE(tensor1.strides[2] == 1);
  REQUIRE(tensor2.strides.size() == 2);
  REQUIRE(tensor2.strides[0] == 5);
  REQUIRE(tensor2.strides[1] == 1);
  REQUIRE(tensor3.strides.size() == 2);
  REQUIRE(tensor3.strides[0] == 5);
  REQUIRE(tensor3.strides[1] == 1);

  mlc::Error err = mlc::gemm(tensor1, tensor2, tensor3);
  REQUIRE(err.type == mlc::ErrorType::TensorExpected2DTensor);

  delete[] data1;
  delete[] data2;
  delete[] data3;
}

TEST_CASE("Test interface tensor unary zero", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4, 5};

  size_t total_size1 = shape1[0] * shape1[1] * shape1[2];

  float *data1 = new float[total_size1];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = static_cast<float>(i);
  }

  mlc::Tensor tensor1(data1, shape1);

  REQUIRE(tensor1.strides.size() == 3);
  REQUIRE(tensor1.strides[0] == 20);
  REQUIRE(tensor1.strides[1] == 5);
  REQUIRE(tensor1.strides[2] == 1);

  mlc::Error err = mlc::unary_zero(tensor1);
  REQUIRE(err.type == mlc::ErrorType::None);

  for (size_t i = 0; i < total_size1; i++)
  {
    CAPTURE(i);
    REQUIRE(tensor1.data[i] == 0);
  }

  delete[] data1;
}

TEST_CASE("Test interface tensor unary relu", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4, 5};
  std::vector<uint64_t> shape2 = {3, 4, 5};

  size_t total_size1 = shape1[0] * shape1[1] * shape1[2];
  size_t total_size2 = shape2[0] * shape2[1] * shape2[2];

  float *data1 = new float[total_size1];
  float *data2 = new float[total_size2];

  for (int64_t i = 0; i < static_cast<int64_t>(total_size1); ++i)
  {
    data1[i] = static_cast<float>(i * (2 * (i % 2) - 1));
  }
  for (size_t i = 0; i < total_size2; ++i)
  {
    data2[i] = 0;
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::Tensor tensor2(data2, shape2);

  REQUIRE(tensor1.strides.size() == 3);
  REQUIRE(tensor1.strides[0] == 20);
  REQUIRE(tensor1.strides[1] == 5);
  REQUIRE(tensor1.strides[2] == 1);
  REQUIRE(tensor2.strides.size() == 3);
  REQUIRE(tensor2.strides[0] == 20);
  REQUIRE(tensor2.strides[1] == 5);
  REQUIRE(tensor2.strides[2] == 1);

  mlc::Error err = mlc::unary_relu(tensor1, tensor2);
  REQUIRE(err.type == mlc::ErrorType::None);

  for (size_t i = 0; i < total_size1; i++)
  {
    CAPTURE(i);
    REQUIRE(tensor2.data[i] == std::max(0.0f, tensor1.data[i]));
  }

  delete[] data1;
  delete[] data2;
}

TEST_CASE("Test interface tensor unary identity", "[tensor][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4, 5};
  std::vector<uint64_t> shape2 = {3, 4, 5};

  size_t total_size1 = shape1[0] * shape1[1] * shape1[2];
  size_t total_size2 = shape2[0] * shape2[1] * shape2[2];

  float *data1 = new float[total_size1];
  float *data2 = new float[total_size2];

  for (size_t i = 0; i < total_size1; ++i)
  {
    data1[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < total_size2; ++i)
  {
    data2[i] = 0;
  }

  mlc::Tensor tensor1(data1, shape1);
  mlc::Tensor tensor2(data2, shape2);

  REQUIRE(tensor1.strides.size() == 3);
  REQUIRE(tensor1.strides[0] == 20);
  REQUIRE(tensor1.strides[1] == 5);
  REQUIRE(tensor1.strides[2] == 1);
  REQUIRE(tensor2.strides.size() == 3);
  REQUIRE(tensor2.strides[0] == 20);
  REQUIRE(tensor2.strides[1] == 5);
  REQUIRE(tensor2.strides[2] == 1);

  mlc::Error err = mlc::unary_identity(tensor1, tensor2);
  REQUIRE(err.type == mlc::ErrorType::None);

  for (size_t i = 0; i < total_size1; i++)
  {
    CAPTURE(i);
    REQUIRE(tensor1.data[i] == tensor2.data[i]);
  }

  delete[] data1;
  delete[] data2;
}

TEST_CASE("Test interface tensor contraction first+last", "[tensor][correctness]")
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

  REQUIRE(tensor1.strides.size() == 2);
  REQUIRE(tensor1.strides[0] == 4);
  REQUIRE(tensor1.strides[1] == 1);
  REQUIRE(tensor2.strides.size() == 2);
  REQUIRE(tensor2.strides[0] == 5);
  REQUIRE(tensor2.strides[1] == 1);
  REQUIRE(tensor3.strides.size() == 2);
  REQUIRE(tensor3.strides[0] == 5);
  REQUIRE(tensor3.strides[1] == 1);

  mlc::Error err = mlc::contraction(tensor1, tensor2, tensor3, "[0,1],[1,2]->[0,2]", mlc::UnaryType::None, mlc::UnaryType::None);
  REQUIRE(err.type == mlc::ErrorType::None);

  delete[] data1;
  delete[] data2;
  delete[] data3;
}

TEST_CASE("Test interface tensor einsum operation", "[setup][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};
  std::vector<uint64_t> shape2 = {4, 5};
  std::vector<uint64_t> shape3 = {3, 5};

  mlc::Tensor tensor1(shape1);
  mlc::Tensor tensor2(shape2);
  mlc::Tensor tensor3(shape3);

  mlc::TensorOperation *setup = mlc::einsum_operation({shape1, shape2}, shape3, "[0,1],[1,2]->[0,2]");

  mlc::Error error = setup->execute({tensor1, tensor2}, tensor3);
  INFO(error.message);
  REQUIRE(error.type == mlc::ErrorType::None);

  error = setup->execute({tensor1, tensor2}, tensor3);
  INFO(error.message);
  REQUIRE(error.type == mlc::ErrorType::None);

  error = setup->execute({tensor1, tensor2}, tensor3);
  INFO(error.message);
  REQUIRE(error.type == mlc::ErrorType::None);
  delete setup;
}