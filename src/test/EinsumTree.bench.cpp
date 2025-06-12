#include "../main/EinsumTree.h"
#include "../main/release_assert.h"
#include <benchmark/benchmark.h>
#include <ctime>
#include <iostream>
#include <string>
#include <utility>

// using namespace mini_jit;

//   std::string tree_str =
//   "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]"; std::vector<int64_t>
//   dim_sizes{
//     60,  // 0
//     60,  // 1
//     20,  // 2
//     20,  // 3
//     8,   // 4
//     8,   // 5
//     8,   // 6
//     8,   // 7
//     8,   // 8
//     8    // 9
//   };

//   CAPTURE(tree_str);
//   CAPTURE(dim_sizes);

//   EinsumTree tree(tree_str, dim_sizes);

//   mini_jit::EinsumTree::ErrorParse err_parse = tree.parse_tree();
//   REQUIRE(err_parse == mini_jit::EinsumTree::ErrorParse::None);
//   REQUIRE(tree.get_root() != nullptr);
//   INFO(tree.get_root()->to_string());

//   std::vector<void *> tensors{
//     new float[dim_sizes[3] * dim_sizes[6] * dim_sizes[8] * dim_sizes[9]],  // [3,6,8,9]
//     new float[dim_sizes[2] * dim_sizes[5] * dim_sizes[7] * dim_sizes[8]],  // [2,5,7,8]
//     new float[dim_sizes[0] * dim_sizes[4] * dim_sizes[5] * dim_sizes[6]],  // [0,4,5,6]
//     new float[dim_sizes[1] * dim_sizes[4] * dim_sizes[7] * dim_sizes[8]],  // [1,4,7,8]
//     new float[dim_sizes[0] * dim_sizes[1] * dim_sizes[2] * dim_sizes[3]],  // [0,1,2,3]
//   };

//   EinsumTree::ErrorExecute err_execute = tree.execute(tensors);
//   REQUIRE(err_execute == EinsumTree::ErrorExecute::None);

//   for (auto ptr : tensors)
//   {
//     delete static_cast<float *>(ptr);
//   }

class EinsumFixture : public benchmark::Fixture
{
public:
  std::vector<void *> tensors;
  double flops;

  double tensor_flops;
  std::string einsum_tree;
  std::vector<int64_t> dim_sizes;

  std::vector<std::string> config_einsum_trees{
    "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]",                           // First Example
    "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]",  // Second Example
  };

  std::vector<std::vector<int64_t>> config_dim_sizes{
    {100, 72, 128, 128, 3, 71, 305, 32, 3},  // First Example
    {60, 60, 20, 20, 8, 8, 8, 8, 8, 8},      // Second Example
  };

  std::vector<std::vector<std::vector<size_t>>> config_tensors{
    {
      {8, 4},
      {7, 3, 8},
      {2, 6, 7},
      {1, 5, 6},
      {0, 5},
      {0, 1, 2, 3, 4},
    },
    {
      {3, 6, 8, 9},
      {2, 5, 7, 8},
      {0, 4, 5, 6},
      {1, 4, 7, 8},
      {0, 1, 2, 3},
    },
  };

  // pair to differentiate between (br)gemm and unary in regard to the number of flops
  // pair of (multiplier, operation size)
  std::vector<std::vector<std::pair<double, std::vector<int64_t>>>> config_num_flops_iteration{
    {
      {2.0, {8, 4, 7, 3}},        // [8,4],[7,3,8]->[7,3,4]
      {2.0, {2, 6, 7, 1, 5}},     // [2,6,7],[1,5,6]->[1,2,5,7]
      {2.0, {1, 2, 5, 7, 0}},     // [1,2,5,7],[0,5]->[0,1,2,7]
      {2.0, {7, 3, 4, 0, 1, 2}},  // [7,3,4],[0,1,2,7]->[0,1,2,3,4]
    },
    {
      {0.0, {3, 6, 8, 9}},              // [3,6,8,9]->[8,6,9,3]
      {0.0, {2, 5, 7, 9}},              // [2,5,7,9]->[7,5,2,9]
      {2.0, {8, 6, 9, 3, 7, 5, 2}},     // [8,6,9,3],[7,5,2,9]->[7,8,5,6,2,3]
      {2.0, {7, 8, 5, 6, 2, 3, 0, 4}},  // [7,8,5,6,2,3],[0,4,5,6]->[0,4,7,8,2,3]
      {2.0, {0, 4, 7, 8, 2, 3, 1}},     // [0,4,7,8,2,3],[1,4,7,8]->[0,1,2,3]
    },
  };

  static void fill_random_matrix(float *matrix, uint32_t size)
  {
    std::srand(std::time(0));
    for (size_t i = 0; i < size; i++)
    {
      float denominator = 1;
      do
      {
        denominator = static_cast<float>(std::rand());
      } while (denominator == 0);

      float numerator = 1;
      do
      {
        numerator = static_cast<float>(std::rand());
      } while (numerator == 0);

      matrix[i] = numerator / denominator;
    }
  }

  void SetUp(::benchmark::State &state) override
  {
    tensors.clear();
    int64_t config_index = state.range(0);
    einsum_tree = config_einsum_trees[config_index];
    dim_sizes = config_dim_sizes[config_index];
    auto tensors_flops_ids = config_num_flops_iteration[config_index];
    auto tensors_ids = config_tensors[config_index];

    tensor_flops = 1;
    for (auto tensor_op_ids : tensors_flops_ids)
    {
      int64_t size = 1;
      for (auto id : tensor_op_ids.second)
      {
        size *= dim_sizes[id];
      }
      tensor_flops += size * tensor_op_ids.first;
    }

    for (const std::vector<size_t> &tensor_ids : tensors_ids)
    {
      int64_t size = 1;
      for (const size_t id : tensor_ids)
      {
        size *= dim_sizes[id];
      }

      float *tensor = new float[size];
      fill_random_matrix(tensor, size);
      tensors.push_back(tensor);
    }
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["FLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);

    for (void *tensor : tensors)
    {
      delete static_cast<float *>(tensor);
    }
  }
};

BENCHMARK_DEFINE_F(EinsumFixture, BM_tensor_optimization)(benchmark::State &state)
{
  mini_jit::EinsumTree tree(einsum_tree, dim_sizes);
  mini_jit::EinsumTree::ErrorParse err_parse = tree.parse_tree();

  release_assert(err_parse == mini_jit::EinsumTree::ErrorParse::None, "Failed to generate the setup");

  for (auto _ : state)
  {
    mini_jit::EinsumTree::ErrorExecute err_execute = tree.execute(tensors);
    release_assert(err_execute == mini_jit::EinsumTree::ErrorExecute::None, "Failed to execute einsum");
  }

  flops = tensor_flops * state.iterations();
}

BENCHMARK_REGISTER_F(EinsumFixture, BM_tensor_optimization)
  ->ArgNames({"einsum_tree"})
  ->Args({
    0,  // Selected einsum config
  })
  ->Name("BM_einsum_tree_first_example")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(EinsumFixture, BM_tensor_optimization)
  ->ArgNames({"einsum_tree"})
  ->Args({
    1,  // Selected einsum Config
  })
  ->Name("BM_einsum_tree_second_example")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds