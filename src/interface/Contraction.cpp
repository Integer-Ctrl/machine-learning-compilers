#include "../../include/MachineLearningCompiler/Tensor.h"
#include "../main/EinsumTree.h"
#include "../main/TensorOperation.h"
#include "Einsum.h"
#include "TensorUtils.h"

mlc::Error mlc::contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction)
{
  return internal::einsum<std::reference_wrapper<const Tensor>>({input0, input1}, output, contraction);
}

mlc::Error mlc::contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction,
                            const UnaryType firstTouch, const UnaryType lastTouch)
{
  mini_jit::EinsumTree einsumTree(contraction);
  mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree(false);
  if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
  {
    mlc::ErrorType type = internal::convertParseError(errorParse);
    return {type, "Failed during parsing the given einsum tree."};
  }
  if (einsumTree.get_root()->left->type != mini_jit::EinsumTree::NodeType::Leaf ||
      einsumTree.get_root()->right->type != mini_jit::EinsumTree::NodeType::Leaf)
  {
    return {mlc::ErrorType::ExpectedSingleContraction, "Expected the given einsum string to be a single string."};
  }

  std::vector<int64_t> sorted_dim_sizes;
  internal::get_sorted_dimensions_sizes(einsumTree.get_root(), {input0, input1}, sorted_dim_sizes);
  einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);
  errorParse = einsumTree.generate_operators();
  if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
  {
    mlc::ErrorType type = internal::convertParseError(errorParse);
    return {type, "Failed during operator generation for the given einsum tree."};
  }

  mini_jit::TensorOperation op;
  mini_jit::TensorConfig config = einsumTree.lower_node(einsumTree.get_root());
  config.first_touch = internal::convertPrimitiveType(firstTouch);
  config.last_touch = internal::convertPrimitiveType(lastTouch);

  mini_jit::TensorOperation::error_t error = op.setup(config);
  mlc::ErrorType errorType = internal::convertTensorOperationError(error);
  if (errorType != mlc::ErrorType::None)
  {
    return {errorType, "Could not generate the kernels for the gemm operation."};
  }

  op.execute(input0.data, input1.data, output.data);
  return {ErrorType::None, "Success"};
}