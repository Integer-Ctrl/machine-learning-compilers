#include "../main/EinsumTree.h"
#include "BaseGeneration.test.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <iostream>

// ==========================
// Parse
// ==========================

TEST_CASE("Test einsum tree parser simple example", "[einsumtree][parse][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[0,1],[2,3,0]->[2,3,1]";
  std::vector<int64_t> sorted_dim_sizes{32, 128, 305, 128};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization(false);
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->left->output_dim_ids.size() == 2);
  REQUIRE(tree.get_root()->right->output_dim_ids.size() == 3);
  REQUIRE(tree.get_root()->output_dim_ids.size() == 3);
}

TEST_CASE("Test einsum tree parser first example", "[einsumtree][parse][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
  std::vector<int64_t> sorted_dim_sizes{100, 72, 128, 128, 3, 71, 305, 32, 3};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization(false);
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->left->output_dim_ids.size() == 3);
  REQUIRE(tree.get_root()->right->output_dim_ids.size() == 4);
  REQUIRE(tree.get_root()->output_dim_ids.size() == 5);
}

TEST_CASE("Test einsum tree parser second example", "[einsumtree][parse][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
  std::vector<int64_t> sorted_dim_sizes{60, 60, 20, 20, 8, 8, 8, 8, 8, 8};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err_parse = tree.parse_tree_no_optimization(false);
  REQUIRE(err_parse == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->left->output_dim_ids.size() == 6);
  REQUIRE(tree.get_root()->right->output_dim_ids.size() == 4);
  REQUIRE(tree.get_root()->output_dim_ids.size() == 4);
}

// ==========================
// Execute
// ==========================

TEST_CASE("Test einsum tree execute simple example", "[einsumtree][execute][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[0,1],[2,3,0]->[2,3,1]";
  std::vector<int64_t> dim_sizes{32, 128, 305, 128};

  CAPTURE(tree_str);
  CAPTURE(dim_sizes);

  EinsumTree tree(tree_str, dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization();
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  GenerationTest test(dim_sizes[1], dim_sizes[3], dim_sizes[0], 1,
                      dim_sizes[0] * dim_sizes[1],                  // [0,1]
                      dim_sizes[2] * dim_sizes[3] * dim_sizes[0],   // [2,3,0]
                      dim_sizes[2] * dim_sizes[3] * dim_sizes[1]);  // [2,3,1]

  test.SetUp(TestInfill::Random);

  std::vector<void *> tensors{
    test.matrix_a.data(),  // [0,1]
    test.matrix_b.data(),  // [2,3,0]
    test.matrix_c.data(),  // [2,3,1]
  };

  mini_jit::EinsumTree::ErrorExecute err_exe = tree.execute(tensors);
  REQUIRE(err_exe == mini_jit::EinsumTree::ErrorExecute::None);

  for (int64_t i0 = 0; i0 < dim_sizes[2]; ++i0)
  {
    uint64_t offset_a = i0 * 0;
    uint64_t offset_b = i0 * (dim_sizes[0] * dim_sizes[3]);
    uint64_t offset_c = i0 * (dim_sizes[1] * dim_sizes[3]);
    test.naive_matmul_M_N_K_Batch(test.matrix_a.data() + offset_a, test.matrix_b.data() + offset_b, test.matrix_c_verify.data() + offset_c,
                                  dim_sizes[1], dim_sizes[0], dim_sizes[1], dim_sizes[1] * dim_sizes[3], dim_sizes[0] * dim_sizes[3]);
  }

  test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
}

TEST_CASE("Test einsum tree execute first example", "[einsumtree][execute][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
  std::vector<int64_t> dim_sizes /*sorted*/ {
    100,  // 0
    72,   // 1
    128,  // 2
    128,  // 3
    3,    // 4
    71,   // 5
    305,  // 6
    32,   // 7
    3     // 8
  };

  CAPTURE(tree_str);
  CAPTURE(dim_sizes);

  EinsumTree tree(tree_str, dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization();
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  std::vector<void *> tensors{
    new float[dim_sizes[8] * dim_sizes[4]],                                              // [8,4]
    new float[dim_sizes[7] * dim_sizes[3] * dim_sizes[8]],                               // [7,3,8]
    new float[dim_sizes[2] * dim_sizes[6] * dim_sizes[7]],                               // [2,6,7]
    new float[dim_sizes[1] * dim_sizes[5] * dim_sizes[6]],                               // [1,5,6]
    new float[dim_sizes[0] * dim_sizes[5]],                                              // [0,5]
    new float[dim_sizes[0] * dim_sizes[1] * dim_sizes[2] * dim_sizes[3] * dim_sizes[4]]  // [0,1,2,3,4]
  };

  mini_jit::EinsumTree::ErrorExecute err_exe = tree.execute(tensors);
  REQUIRE(err_exe == mini_jit::EinsumTree::ErrorExecute::None);

  for (auto ptr : tensors)
  {
    delete[] static_cast<float *>(ptr);
  }
}

TEST_CASE("Test einsum tree execute second example", "[einsumtree][execute][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
  std::vector<int64_t> dim_sizes{
    60,  // 0
    60,  // 1
    20,  // 2
    20,  // 3
    8,   // 4
    8,   // 5
    8,   // 6
    8,   // 7
    8,   // 8
    8    // 9
  };

  CAPTURE(tree_str);
  CAPTURE(dim_sizes);

  EinsumTree tree(tree_str, dim_sizes);

  mini_jit::EinsumTree::ErrorParse err_parse = tree.parse_tree_no_optimization();
  REQUIRE(err_parse == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  std::vector<void *> tensors{
    new float[dim_sizes[3] * dim_sizes[6] * dim_sizes[8] * dim_sizes[9]],  // [3,6,8,9]
    new float[dim_sizes[2] * dim_sizes[5] * dim_sizes[7] * dim_sizes[8]],  // [2,5,7,8]
    new float[dim_sizes[0] * dim_sizes[4] * dim_sizes[5] * dim_sizes[6]],  // [0,4,5,6]
    new float[dim_sizes[1] * dim_sizes[4] * dim_sizes[7] * dim_sizes[8]],  // [1,4,7,8]
    new float[dim_sizes[0] * dim_sizes[1] * dim_sizes[2] * dim_sizes[3]],  // [0,1,2,3]
  };

  EinsumTree::ErrorExecute err_execute = tree.execute(tensors);
  REQUIRE(err_execute == EinsumTree::ErrorExecute::None);

  for (auto ptr : tensors)
  {
    delete[] static_cast<float *>(ptr);
  }
}

// ==========================
// Optimize
// ==========================

TEST_CASE("Test einsum tree optimize swap", "[einsumtree][optimize][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[2,3,0],[0,1]->[2,3,1]";
  std::vector<int64_t> sorted_dim_sizes{32, 128, 305, 128};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization(false);
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  tree.conditional_swap(tree.get_root());

  INFO("Optimized");
  INFO(tree.get_root()->to_string());

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->left->output_dim_ids.size() == 2);
  REQUIRE(tree.get_root()->right->output_dim_ids.size() == 3);
  REQUIRE(tree.get_root()->output_dim_ids.size() == 3);
}

TEST_CASE("Test einsum tree optimize reorder left", "[einsumtree][optimize][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[1,0],[2,3,0]->[2,3,1]";
  std::vector<int64_t> sorted_dim_sizes{32, 128, 305, 128};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization(false);
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  tree.reorder_left_node(tree.get_root());

  INFO("Optimized");
  INFO(tree.get_root()->to_string());

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->left->output_dim_ids.size() == 2);
  REQUIRE(tree.get_root()->right->output_dim_ids.size() == 3);
  REQUIRE(tree.get_root()->output_dim_ids.size() == 3);
  // Inserted node due to reorder
  REQUIRE(tree.get_root()->left->output_dim_ids[0] == 0);
  REQUIRE(tree.get_root()->left->output_dim_ids[1] == 1);
  REQUIRE(tree.get_root()->left->type == mini_jit::EinsumTree::NodeType::Transposition);
  // Original leaf node
  REQUIRE(tree.get_root()->left->left != nullptr);
  REQUIRE(tree.get_root()->left->left->output_dim_ids[0] == 1);
  REQUIRE(tree.get_root()->left->left->output_dim_ids[1] == 0);
}

TEST_CASE("Test einsum tree optimize reorder right", "[einsumtree][optimize][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[0,1],[0,3,2]->[2,3,1]";
  std::vector<int64_t> sorted_dim_sizes{32, 128, 305, 128};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization(false);
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  tree.reorder_right_node(tree.get_root());

  INFO("Optimized");
  INFO(tree.get_root()->to_string());

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->left->output_dim_ids.size() == 2);
  REQUIRE(tree.get_root()->right->output_dim_ids.size() == 3);
  REQUIRE(tree.get_root()->output_dim_ids.size() == 3);
  // Inserted node due to reorder
  REQUIRE(tree.get_root()->right->output_dim_ids[0] == 2);
  REQUIRE(tree.get_root()->right->output_dim_ids[1] == 3);
  REQUIRE(tree.get_root()->right->output_dim_ids[2] == 0);
  REQUIRE(tree.get_root()->right->type == mini_jit::EinsumTree::NodeType::Transposition);
  // Original leaf node
  REQUIRE(tree.get_root()->right->left != nullptr);
  REQUIRE(tree.get_root()->right->left->output_dim_ids[0] == 0);
  REQUIRE(tree.get_root()->right->left->output_dim_ids[1] == 3);
  REQUIRE(tree.get_root()->right->left->output_dim_ids[2] == 2);
}

TEST_CASE("Test einsum tree optimize", "[einsumtree][optimize][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[0,3,2],[1,0]->[2,3,1]";
  std::vector<int64_t> sorted_dim_sizes{32, 128, 305, 128};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree_no_optimization(false);
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  tree.optimize(tree.get_root());

  INFO("Optimized");
  INFO(tree.get_root()->to_string());

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->left->output_dim_ids.size() == 2);
  REQUIRE(tree.get_root()->right->output_dim_ids.size() == 3);
  REQUIRE(tree.get_root()->output_dim_ids.size() == 3);

  // Inserted node due to reorder
  REQUIRE(tree.get_root()->left->output_dim_ids[0] == 0);
  REQUIRE(tree.get_root()->left->output_dim_ids[1] == 1);
  REQUIRE(tree.get_root()->left->type == mini_jit::EinsumTree::NodeType::Transposition);
  // Original leaf node
  REQUIRE(tree.get_root()->left->left != nullptr);
  REQUIRE(tree.get_root()->left->left->output_dim_ids[0] == 1);
  REQUIRE(tree.get_root()->left->left->output_dim_ids[1] == 0);

  // Inserted right node due to reorder
  REQUIRE(tree.get_root()->right->output_dim_ids[0] == 2);
  REQUIRE(tree.get_root()->right->output_dim_ids[1] == 3);
  REQUIRE(tree.get_root()->right->output_dim_ids[2] == 0);
  REQUIRE(tree.get_root()->right->type == mini_jit::EinsumTree::NodeType::Transposition);
  // Original leaf node
  REQUIRE(tree.get_root()->right->left != nullptr);
  REQUIRE(tree.get_root()->right->left->output_dim_ids[0] == 0);
  REQUIRE(tree.get_root()->right->left->output_dim_ids[1] == 3);
  REQUIRE(tree.get_root()->right->left->output_dim_ids[2] == 2);
}

// ==========================
// Optimize & Execute
// ==========================

TEST_CASE("Test einsum tree optimize and execute first example", "[einsumtree][execute][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]";
  std::vector<int64_t> dim_sizes{
    100,  // 0
    72,   // 1
    128,  // 2
    128,  // 3
    3,    // 4
    71,   // 5
    305,  // 6
    32,   // 7
    3,    // 8
  };

  CAPTURE(tree_str);
  CAPTURE(dim_sizes);

  EinsumTree tree(tree_str, dim_sizes);
  EinsumTree tree_no_optimization(tree_str, dim_sizes);

  mini_jit::EinsumTree::ErrorParse err_parse = tree.parse_tree();
  REQUIRE(err_parse == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  std::string expected_optimization = "0,1,2,3,4\n"
                                      "├─ 3,7,4\n"
                                      "|  ├─ 8,4\n"
                                      "|  └─ 3,7,8\n"
                                      "|     └─ 7,3,8\n"
                                      "└─ 0,1,2,7\n"
                                      "   ├─ 2,1,5,7\n"
                                      "   |  ├─ 2,6,7\n"
                                      "   |  |  └─ 6,2,7\n"
                                      "   |  └─ 1,5,6\n"
                                      "   |     └─ 5,1,6\n"
                                      "   └─ 0,5\n";
  REQUIRE_THAT(expected_optimization, Catch::Matchers::Equals(tree.get_root()->to_string(), Catch::CaseSensitive::Yes));

  tree_no_optimization.parse_tree_no_optimization(false);
  INFO("No Optimization");
  INFO(tree_no_optimization.get_root()->to_string());

  std::vector<void *> tensors{
    new float[dim_sizes[7] * dim_sizes[3] * dim_sizes[8]],                                // [7,3,8]
    new float[dim_sizes[8] * dim_sizes[4]],                                               // [8,4]
    new float[dim_sizes[0] * dim_sizes[5]],                                               // [0,5]
    new float[dim_sizes[5] * dim_sizes[1] * dim_sizes[6]],                                // [5,1,6]
    new float[dim_sizes[6] * dim_sizes[2] * dim_sizes[7]],                                // [6,2,7]
    new float[dim_sizes[0] * dim_sizes[1] * dim_sizes[2] * dim_sizes[3] * dim_sizes[4]],  // [0,1,2,3,4]
  };

  EinsumTree::ErrorExecute err_execute = tree.execute(tensors);
  REQUIRE(err_execute == EinsumTree::ErrorExecute::None);

  for (auto ptr : tensors)
  {
    delete[] static_cast<float *>(ptr);
  }
}

TEST_CASE("Test einsum tree optimize and execute second example", "[einsumtree][execute][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]";
  std::vector<int64_t> dim_sizes{
    60,  // 0
    60,  // 1
    20,  // 2
    20,  // 3
    8,   // 4
    8,   // 5
    8,   // 6
    8,   // 7
    8,   // 8
    8,   // 9
  };

  CAPTURE(tree_str);
  CAPTURE(dim_sizes);

  EinsumTree tree(tree_str, dim_sizes);
  EinsumTree tree_no_optimization(tree_str, dim_sizes);

  mini_jit::EinsumTree::ErrorParse err_parse = tree.parse_tree();
  REQUIRE(err_parse == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  std::string expected_optimization = "0,1,2,3\n"
                                      "├─ 0,4,2,7,8,3\n"
                                      "|  ├─ 2,5,7,8,6,3\n"
                                      "|  |  ├─ 8,6,9,3\n"
                                      "|  |  |  └─ 3,6,8,9\n"
                                      "|  |  └─ 2,5,7,9\n"
                                      "|  └─ 0,5,4,6\n"
                                      "|     └─ 0,4,5,6\n"
                                      "└─ 7,4,1,8\n"
                                      "   └─ 1,4,7,8\n";
  REQUIRE_THAT(expected_optimization, Catch::Matchers::Equals(tree.get_root()->to_string(), Catch::CaseSensitive::Yes));

  tree_no_optimization.parse_tree_no_optimization(false);
  INFO("No Optimization");
  INFO(tree_no_optimization.get_root()->to_string());

  std::vector<void *> tensors{
    new float[dim_sizes[1] * dim_sizes[4] * dim_sizes[7] * dim_sizes[8]],  // [1,4,7,8]
    new float[dim_sizes[0] * dim_sizes[4] * dim_sizes[5] * dim_sizes[6]],  // [0,4,5,6]
    new float[dim_sizes[2] * dim_sizes[5] * dim_sizes[7] * dim_sizes[9]],  // [2,5,7,9]
    new float[dim_sizes[3] * dim_sizes[6] * dim_sizes[8] * dim_sizes[9]],  // [3,6,8,9]
    new float[dim_sizes[0] * dim_sizes[1] * dim_sizes[2] * dim_sizes[3]],  // [0,1,2,3]
  };

  EinsumTree::ErrorExecute err_execute = tree.execute(tensors);
  REQUIRE(err_execute == EinsumTree::ErrorExecute::None);

  for (auto ptr : tensors)
  {
    delete[] static_cast<float *>(ptr);
  }
}

TEST_CASE("Test einsum tree optimize and execute third example", "[einsumtree][execute][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]";
  std::vector<int64_t> dim_sizes{
    40,  // 0
    40,  // 1
    40,  // 2
    40,  // 3
    40,  // 4
    25,  // 5
    25,  // 6
    25,  // 7
    25,  // 8
    25,  // 9
  };

  CAPTURE(tree_str);
  CAPTURE(dim_sizes);

  EinsumTree tree(tree_str, dim_sizes);
  EinsumTree tree_no_optimization(tree_str, dim_sizes);

  mini_jit::EinsumTree::ErrorParse err_parse = tree.parse_tree();
  REQUIRE(err_parse == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  INFO(tree.get_root()->to_string());

  std::string expected_optimization = "5,6,7,8,9\n"
                                      "├─ 4,6,5,2,9\n"
                                      "|  ├─ 4,0,9\n"
                                      "|  |  └─ 4,9,0\n"
                                      "|  └─ 6,5,2,0\n"
                                      "|     ├─ 5,1,0\n"
                                      "|     |  └─ 0,5,1\n"
                                      "|     └─ 6,2,1\n"
                                      "|        └─ 1,6,2\n"
                                      "└─ 4,7,8,2\n"
                                      "   ├─ 7,3,2\n"
                                      "   |  └─ 2,7,3\n"
                                      "   └─ 4,8,3\n"
                                      "      └─ 3,8,4\n";

  REQUIRE_THAT(expected_optimization, Catch::Matchers::Equals(tree.get_root()->to_string(), Catch::CaseSensitive::Yes));

  tree_no_optimization.parse_tree_no_optimization(false);
  INFO("No Optimization");
  INFO(tree_no_optimization.get_root()->to_string());

  std::vector<void *> tensors{
    new float[dim_sizes[2] * dim_sizes[7] * dim_sizes[3]],                                // [2,7,3]
    new float[dim_sizes[3] * dim_sizes[8] * dim_sizes[4]],                                // [3,8,4]
    new float[dim_sizes[4] * dim_sizes[9] * dim_sizes[0]],                                // [4,9,0]
    new float[dim_sizes[0] * dim_sizes[5] * dim_sizes[1]],                                // [0,5,1]
    new float[dim_sizes[1] * dim_sizes[6] * dim_sizes[2]],                                // [1,6,2]
    new float[dim_sizes[5] * dim_sizes[6] * dim_sizes[7] * dim_sizes[8] * dim_sizes[9]],  // [5,6,7,8,9]
  };

  EinsumTree::ErrorExecute err_execute = tree.execute(tensors);
  REQUIRE(err_execute == EinsumTree::ErrorExecute::None);

  for (auto ptr : tensors)
  {
    delete[] static_cast<float *>(ptr);
  }
}