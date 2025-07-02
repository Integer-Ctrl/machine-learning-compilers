#include "../../../include/MachineLearningCompiler/Tensor.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>
#include <vector>

TEST_CASE("Test interface tensor einsum setup", "[setup][correctness]")
{
  std::vector<uint64_t> shape1 = {3, 4};
  std::vector<uint64_t> shape2 = {4, 5};
  std::vector<uint64_t> shape3 = {3, 5};

  mlc::Tensor tensor1(shape1);
  mlc::Tensor tensor2(shape2);
  mlc::Tensor tensor3(shape3);

  mlc::TensorOperation *setup = mlc::einsum_operation({shape1, shape2}, shape3, "[0,1],[1,2]->[0,2]");
  setup->execute({tensor1, tensor2}, tensor3);
  setup->execute({tensor1, tensor2}, tensor3);
  setup->execute({tensor1, tensor2}, tensor3);
  delete setup;
}