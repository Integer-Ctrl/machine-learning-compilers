#include "../../../include/MachineLearningCompiler/Setup.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>
#include <vector>

TEST_CASE("Test tensor einsum setup", "[setup][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};
  std::vector<uint64_t> shape2 = {4, 5};
  std::vector<uint64_t> shape3 = {3, 5};

  size_t total_size1 = shape1[0] * shape1[1];
  size_t total_size2 = shape2[0] * shape2[1];
  size_t total_size3 = shape3[0] * shape3[1];

  mlc::Tensor tensor1(shape1);
  mlc::Tensor tensor2(shape2);
  mlc::Tensor tensor3(shape3);

  mlc::Setup &setup = mlc::einsum_setup({tensor1, tensor2}, tensor3, "[0,1],[1,2]->[0,2]");
  setup.execute({tensor1, tensor2}, tensor3);
}