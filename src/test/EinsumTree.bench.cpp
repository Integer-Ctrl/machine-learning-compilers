#include "../main/EinsumTree.h"
#include "../main/release_assert.h"
#include <benchmark/benchmark.h>
#include <ctime>
#include <iostream>
#include <string>
#include <utility>

class EinsumFixture : public benchmark::Fixture
{
public:
  std::vector<void *> tensors;
  double flops;
  bool optimize_tree;

  double tensor_flops;
  std::string einsum_tree;
  std::vector<int64_t> dim_sizes;

  std::vector<std::string> config_einsum_trees{
    "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]",                           // First Example
    "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]",  // Second Example

    // Optimize
    "[[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]",          // First Example
    "[1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]",           // Second Example
    "[[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]",  // Third Example
  };

  std::vector<std::vector<int64_t>> config_dim_sizes{
    {100, 72, 128, 128, 3, 71, 305, 32, 3},  // First Example
    {60, 60, 20, 20, 8, 8, 8, 8, 8, 8},      // Second Example

    // Optimize
    {100, 72, 128, 128, 3, 71, 305, 32, 3},    // First Example
    {60, 60, 20, 20, 8, 8, 8, 8, 8, 8},        // Second Example
    {40, 40, 40, 40, 40, 25, 25, 25, 25, 25},  // Third Example
  };

  std::vector<std::vector<std::vector<size_t>>> config_tensors{
    {
      // First Example
      {8, 4},
      {7, 3, 8},
      {2, 6, 7},
      {1, 5, 6},
      {0, 5},
      {0, 1, 2, 3, 4},
    },
    {
      // Second Example
      {3, 6, 8, 9},
      {2, 5, 7, 8},
      {0, 4, 5, 6},
      {1, 4, 7, 8},
      {0, 1, 2, 3},
    },

    // Optimize
    {
      // First Example
      {7, 3, 8},
      {8, 4},
      {0, 5},
      {5, 1, 6},
      {6, 2, 7},
      {0, 1, 2, 3, 4},
    },
    {
      // Second Example
      {1, 4, 7, 8},
      {0, 4, 5, 6},
      {2, 5, 7, 9},
      {3, 6, 8, 9},
      {0, 1, 2, 3},
    },
    {
      // Third Example
      {2, 7, 3},
      {3, 8, 4},
      {4, 9, 0},
      {0, 5, 1},
      {1, 6, 2},
      {5, 6, 7, 8, 9},
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

  double calculate_flops_node(mini_jit::EinsumTree::EinsumNode *node)
  {
    if (node->type == mini_jit::EinsumTree::NodeType::Leaf)
    {
      return 0;
    }
    else if (node->type == mini_jit::EinsumTree::NodeType::Transposition)
    {
      return calculate_flops_node(node->left);
    }
    else if (node->type == mini_jit::EinsumTree::NodeType::Contraction)
    {
      std::vector<int64_t> distinct_dims = node->left->output_dim_ids;

      for (auto dim_id : node->right->output_dim_ids)
      {
        if (std::find(distinct_dims.begin(), distinct_dims.end(), dim_id) == distinct_dims.end())
        {
          distinct_dims.push_back(dim_id);
        }
      }

      int64_t flops = 1;
      for (auto dim_id : distinct_dims)
      {
        flops *= dim_sizes[dim_id];
      }
      flops *= 2;  // 2 operations multiply + add

      flops += calculate_flops_node(node->left);
      flops += calculate_flops_node(node->right);

      return flops;
    }

    release_assert(false, "Found unhandled node type");
    return 0;
  }

  void SetUp(::benchmark::State &state) override
  {
    release_assert(config_einsum_trees.size() == config_dim_sizes.size(), "Must be equal size.");
    release_assert(config_einsum_trees.size() == config_tensors.size(), "Must be equal size.");

    tensors.clear();
    int64_t config_index = state.range(0);
    release_assert(config_index >= 0, "Expected value larger equal than 0.");
    release_assert(config_index < static_cast<int64_t>(config_einsum_trees.size()), "Expected config_index value is out of range.");

    einsum_tree = config_einsum_trees[config_index];
    dim_sizes = config_dim_sizes[config_index];
    auto tensors_ids = config_tensors[config_index];
    optimize_tree = state.range(1);

    mini_jit::EinsumTree tree(einsum_tree, dim_sizes);
    optimize_tree ? tree.parse_tree() : tree.parse_tree_no_optimization();
    tensor_flops = calculate_flops_node(tree.get_root());

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
      delete[] static_cast<float *>(tensor);
    }
  }
};

BENCHMARK_DEFINE_F(EinsumFixture, BM_tensor_optimization)(benchmark::State &state)
{
  mini_jit::EinsumTree tree(einsum_tree, dim_sizes);
  mini_jit::EinsumTree::ErrorParse err_parse = optimize_tree ? tree.parse_tree() : tree.parse_tree_no_optimization();

  release_assert(err_parse == mini_jit::EinsumTree::ErrorParse::None, "Failed to generate the setup");

  for (auto _ : state)
  {
    mini_jit::EinsumTree::ErrorExecute err_execute = tree.execute(tensors);
    release_assert(err_execute == mini_jit::EinsumTree::ErrorExecute::None, "Failed to execute einsum");
  }

  flops = tensor_flops * state.iterations();
}

BENCHMARK_REGISTER_F(EinsumFixture, BM_tensor_optimization)
  ->ArgNames({"config", "optimize"})
  ->Args({
    0,      // Selected einsum config
    false,  // Optimize
  })
  ->Name("BM_einsum_tree_first_example")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(EinsumFixture, BM_tensor_optimization)
  ->ArgNames({"config", "optimize"})
  ->Args({
    1,      // Selected einsum Config
    false,  // Optimize
  })
  ->Name("BM_einsum_tree_second_example")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(EinsumFixture, BM_tensor_optimization)
  ->ArgNames({"config", "optimize"})
  ->Args({
    2,     // Selected einsum Config
    true,  // Optimize
  })
  ->Name("BM_einsum_tree_optimize_first_example")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(EinsumFixture, BM_tensor_optimization)
  ->ArgNames({"config", "optimize"})
  ->Args({
    3,     // Selected einsum Config
    true,  // Optimize
  })
  ->Name("BM_einsum_tree_optimize_second_example")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds

BENCHMARK_REGISTER_F(EinsumFixture, BM_tensor_optimization)
  ->ArgNames({"config", "optimize"})
  ->Args({
    4,     // Selected einsum Config
    true,  // Optimize
  })
  ->Name("BM_einsum_tree_optimize_third_example")
  ->DisplayAggregatesOnly(true)
  ->MinWarmUpTime(0.3);  // WarmUp in seconds