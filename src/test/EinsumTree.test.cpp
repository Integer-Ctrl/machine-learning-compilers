#include "../main/EinsumTree.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>

TEST_CASE("Test einsum tree parser simple example", "[einsumtree][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[1,2],[3,4,1]->[3,4,2]";
  std::vector<int> sorted_dim_sizes{32, 128, 305, 128};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree();
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  std::cout << "EinsumTree structure:\n" << tree.get_root()->to_string() << std::endl;

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->input_dims1.size() == 2);
  REQUIRE(tree.get_root()->input_dims2.size() == 3);
  REQUIRE(tree.get_root()->output_dims.size() == 3);
}

TEST_CASE("Test einsum tree parser first example", "[einsumtree][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
  std::vector<int> sorted_dim_sizes{100, 72, 128, 128, 3, 71, 305, 32, 3};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree();
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  std::cout << "EinsumTree structure:\n" << tree.get_root()->to_string() << std::endl;

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->input_dims1.size() == 3);
  REQUIRE(tree.get_root()->input_dims2.size() == 4);
  REQUIRE(tree.get_root()->output_dims.size() == 5);
}

TEST_CASE("Test einsum tree parser second example", "[einsumtree][correctness]")
{
  using namespace mini_jit;

  std::string tree_str = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
  std::vector<int> sorted_dim_sizes{60, 60, 20, 20, 8, 8, 8, 8, 8, 8};

  CAPTURE(tree_str);
  CAPTURE(sorted_dim_sizes);

  EinsumTree tree(tree_str, sorted_dim_sizes);

  mini_jit::EinsumTree::ErrorParse err = tree.parse_tree();
  REQUIRE(err == mini_jit::EinsumTree::ErrorParse::None);
  REQUIRE(tree.get_root() != nullptr);
  std::cout << "EinsumTree structure:\n" << tree.get_root()->to_string() << std::endl;

  REQUIRE(tree.get_root()->type == mini_jit::EinsumTree::NodeType::Contraction);
  REQUIRE(tree.get_root()->input_dims1.size() == 6);
  REQUIRE(tree.get_root()->input_dims2.size() == 4);
  REQUIRE(tree.get_root()->output_dims.size() == 4);
}