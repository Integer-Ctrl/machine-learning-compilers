#include <MachineLearningCompiler/Tensor.h>
#include <iostream>

/**
 * Tensor object examples.
 */
void example_tensor()
{
  // Define tensors with different dimensions. The memory is allocated automatically based on the given dimensions and filled with zeros.
  mlc::Tensor tensor1D({5});        // 1D tensor with 5 elements
  mlc::Tensor tensor2D({3, 4});     // 2D tensor with 3 rows and 4 columns
  mlc::Tensor tensor3D({2, 3, 4});  // 3D tensor with 2 layers, 3 rows and 4 columns

  // Define a tensor with data
  float data1[] = {1, 2, 3, 4, 5};
  float data2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  mlc::Tensor tensorWithData1(data1, {2, 2});     // 2x2 tensor with specific data
  mlc::Tensor tensorWIthData2(data2, {3, 2, 2});  // 3D tensor with specific data

  // Print dimensions and sizes of the tensors
  std::cout << "Tensor 1D dim sizes: ";
  for (const auto &dim : tensor1D.dim_sizes)
  {
    std::cout << dim << " ";
  }
  std::cout << std::endl;
  std::cout << "Tensor 2D dim sizes: ";
  for (const auto &dim : tensor2D.dim_sizes)
  {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Print the sizes of the tensors
  std::cout << "Tensor 1D Size: " << tensor1D.size() << std::endl;
  std::cout << "Tensor 2D Size: " << tensor2D.size() << std::endl;
  std::cout << "Tensor 3D Size: " << tensor3D.size() << std::endl;
  std::cout << "Tensor with Data 1 Size: " << tensorWithData1.size() << std::endl;
  std::cout << "Tensor with Data 2 Size: " << tensorWIthData2.size() << std::endl;

  // Print the strides of the tensors
  std::cout << "Tensor 1D Strides: ";
  for (const auto &stride : tensor1D.strides)
  {
    std::cout << stride << " ";
  }
  std::cout << std::endl;
  std::cout << "Tensor 2D Strides: ";
  for (const auto &stride : tensor2D.strides)
  {
    std::cout << stride << " ";
  }
  std::cout << std::endl;

  // Print the tensors to the console
  std::cout << tensor1D.to_string("Tensor 1D") << std::endl;
  std::cout << tensor2D.to_string("Tensor 2D") << std::endl;
  std::cout << tensor3D.to_string("Tensor 3D") << std::endl;
  std::cout << tensorWithData1.to_string("Tensor with Data 1") << std::endl;
  std::cout << tensorWIthData2.to_string("Tensor with Data 2") << std::endl;
}

/**
 * Methods that can be used to fill a tensor.
 */
void example_fill()
{
  // Fill the memory of the tensors with random values
  mlc::Tensor tensorRandom({3, 3});
  mlc::fill_random(tensorRandom);
  std::cout << tensorRandom.to_string("Random") << std::endl;

  // Fill the memory of the tensors with all 1s.
  mlc::Tensor tensorSingleNumber({3, 3});
  mlc::fill_number(tensorSingleNumber, 1.43);
  std::cout << tensorSingleNumber.to_string("Ones") << std::endl;

  // Fill the memory of the tensors with counting upwards data starting from 0.
  mlc::Tensor tensorCountingUp({3, 3});
  mlc::fill_counting_up(tensorCountingUp, 0, 1.0);
  std::cout << tensorCountingUp.to_string("Counting Up") << std::endl;

  // Fill the memory of the tensors with counting downwards data starting from 5.
  mlc::Tensor tensorCountingDown({3, 3});
  mlc::fill_counting_down(tensorCountingDown, 5, 0.1);
  std::cout << tensorCountingDown.to_string("Counting Down") << std::endl;

  // Fill the memory of the tensor based on a user defined expression. The tensor itself and current index of the data that is currently
  // filled are given as additional parameter.
  // Here the tensor is filled with 1 2 3, 1 2 3, 1 2 3
  mlc::Tensor tensorLambda({3, 3});
  mlc::fill_lambda(tensorLambda, [](const mlc::Tensor &self, size_t index) { return index % self.strides[self.strides.size() - 1]; });
  std::cout << tensorLambda.to_string("Lambda 1 2 3") << std::endl;

  // We can also fill the tensor using outside defined variable.
  size_t size = tensorLambda.size();
  mlc::fill_lambda(tensorLambda, [&size](const mlc::Tensor &self, size_t index) { return size; });
  std::cout << tensorLambda.to_string("Lambda Outside") << std::endl;
}

/**
 * A GEneral Matrix Matrix multiplication requires the tensors to be in a matrix shape i.e. exactly 2 dimensions.
 */
void example_gemm()
{
  mlc::Tensor in0({5, 3});  // IDs: 0,1
  mlc::Tensor in1({2, 5});  // IDs: 2,0
  mlc::Tensor out({2, 3});  // IDs: 2,1

  // Fill the memory of the tensors with random values
  mlc::fill_counting_up(in0, 0, 1);
  mlc::fill_counting_up(in1, 0, 1);

  mlc::Error error = mlc::gemm(in0, in1, out);
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }

  std::cout << in0.to_string("in0") << std::endl;
  std::cout << in1.to_string("in1") << std::endl;
  std::cout << out.to_string("out") << std::endl;
}

/**
 * A unary operation zero, identity and ReLU can be performed on a Tensor.
 */
void example_unary()
{
  // Performs a zero unary
  mlc::Tensor tensorZero({3, 3});
  mlc::fill_random(tensorZero);
  mlc::Error error = mlc::unary_zero(tensorZero);
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }
  std::cout << tensorZero.to_string("Unary Zero") << std::endl;

  // Performs a identity unary
  mlc::Tensor tensorIdentityIn({3, 3});
  mlc::Tensor tensorIdentityOut({3, 3});
  mlc::fill_random(tensorIdentityIn);
  mlc::fill_number(tensorIdentityIn, 0);
  error = mlc::unary_identity(tensorIdentityIn, tensorIdentityOut);  // identity = copy from input to output
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }
  std::cout << tensorIdentityOut.to_string("Unary Identity Input") << std::endl;
  std::cout << tensorIdentityOut.to_string("Unary Identity Output") << std::endl;

  // Performs a ReLU unary
  mlc::Tensor tensorReluIn({3, 3});
  mlc::Tensor tensorReluOut({3, 3});
  // Fills even indices with positive and odd indices with negative numbers
  mlc::fill_lambda(tensorReluIn, [](const mlc::Tensor &, size_t index) { return index * (2 * (index % 2) - 1); });
  mlc::fill_number(tensorReluOut, 0);
  error = mlc::unary_relu(tensorReluIn, tensorReluOut);  // ReLU = max(x, 0)
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }
  std::cout << tensorReluIn.to_string("Unary ReLU Input") << std::endl;
  std::cout << tensorReluOut.to_string("Unary ReLU Output") << std::endl;
}

/**
 * A contraction of two tensors and add the result to the output.
 */
void example_contraction()
{
  mlc::Tensor in0({5, 4, 3});     // IDs: 0,1,2
  mlc::Tensor in1({5, 2, 4});     // IDs: 3,4,1
  mlc::Tensor out({5, 5, 2, 3});  // IDs: 0,3,4,2

  mlc::fill_counting_up(in0, 0, 1);
  mlc::fill_counting_down(in1, 0, 1);
  mlc::fill_number(in0, 1'000'000);

  mlc::Error error = mlc::contraction(in0, in1, out, "[0,1,2],[3,4,1]->[0,3,4,2]");
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }

  std::cout << in0.to_string("in0") << std::endl;
  std::cout << in1.to_string("in1") << std::endl;
  std::cout << out.to_string("out") << std::endl;
}

/**
 * A contraction of two tensors with unarys that are executed before (first touch) or after (last touch) the contraction on the output
 * tensor.
 */
void example_contraction_first_last_touch()
{
  mlc::Tensor in0({5, 4, 3});     // IDs: 0,1,2
  mlc::Tensor in1({5, 2, 4});     // IDs: 3,4,1
  mlc::Tensor out({5, 5, 2, 3});  // IDs: 0,3,4,2

  mlc::fill_counting_up(in0, 0, 1);
  mlc::fill_counting_down(in1, 0, 1);
  // The out is default initialized with zeros.

  mlc::Error error = mlc::contraction(in0, in1, out, "[0,1,2],[3,4,1]->[0,3,4,2]", mlc::UnaryType::None, mlc::UnaryType::ReLU);
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }

  std::cout << in0.to_string("in0") << std::endl;
  std::cout << in1.to_string("in1") << std::endl;
  std::cout << out.to_string("out") << std::endl;
}

/**
 * A simple einsum operation on three input tensors. The result is added to the output.
 */
void example_einsum()
{
  mlc::Tensor in0({5, 3});  // IDs: 0,1
  mlc::Tensor in1({2, 5});  // IDs: 2,0
  mlc::Tensor in2({3, 7});  // IDs: 1,3
  mlc::Tensor out({2, 7});  // IDs: 2,3

  mlc::fill_counting_up(in0, 0, 1);
  mlc::fill_number(in1, 1);
  mlc::fill_counting_down(in2, 0, 1);
  mlc::fill_number(out, 1'000);

  // Execute the defined einsum tree on the tensors.
  mlc::Error error = mlc::einsum({in0, in1, in2}, out, "[[0,1],[2,0]->[2,1]],[1,3]->[2,3]");
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }

  std::cout << in0.to_string("in0") << std::endl;
  std::cout << in1.to_string("in1") << std::endl;
  std::cout << in2.to_string("in2") << std::endl;
  std::cout << out.to_string("out") << std::endl;
}

/**
 * A einsum expression that is first defined by the shapes of the input and ouput tensors and can be multiple time called on any input and
 * output tensors that matches the same shape. This can be used to save the costs to setup and optimize the given einsum tree. The result is
 * added to the output.
 */
void example_einsum_operation()
{
  mlc::Tensor in0({5, 3});  // IDs: 0,1
  mlc::Tensor in1({2, 5});  // IDs: 2,0
  mlc::Tensor in2({3, 7});  // IDs: 1,3
  mlc::Tensor out({2, 7});  // IDs: 2,3

  mlc::fill_counting_down(in0, 0, 1);
  mlc::fill_number(in1, 1);
  mlc::fill_counting_down(in2, 0, 0.5);
  mlc::fill_number(out, 1'000);

  // Generates a tensor operation with fixed input and ouput tensor shapes.
  mlc::TensorOperation *op =
    mlc::einsum_operation({in0.dim_sizes, in1.dim_sizes, in2.dim_sizes}, out.dim_sizes, "[[0,1],[2,0]->[2,1]],[1,3]->[2,3]");

  // Process any error that may occurs during the setup of the operation.
  mlc::Error error = op->getSetupError();
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    delete op;
    return;
  }

  // Execute the operation and check for any error that can happen during execution.
  error = op->execute({in0, in1, in2}, out);
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    delete op;
    return;
  }

  std::cout << in0.to_string("in0") << std::endl;
  std::cout << in1.to_string("in1") << std::endl;
  std::cout << in2.to_string("in2") << std::endl;
  std::cout << out.to_string("out") << std::endl;

  // Create new tensors of the same shape.
  mlc::Tensor in0_2(in0.dim_sizes);  // IDs: 0,1
  mlc::Tensor in1_2(in1.dim_sizes);  // IDs: 2,0
  mlc::Tensor in2_2(in2.dim_sizes);  // IDs: 1,3
  mlc::Tensor out_2(out.dim_sizes);  // IDs: 2,3

  mlc::fill_random(in0_2);
  mlc::fill_random(in1_2);
  mlc::fill_random(in2_2);
  mlc::fill_random(out_2);

  // Execute the operation again but on different tensors of the same size.
  error = op->execute({in0_2, in1_2, in2_2}, out_2);
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    delete op;
    return;
  }

  delete op;
}

int main(int argc, const char **argv)
{
  example_tensor();
  example_fill();
  example_gemm();
  example_unary();
  example_contraction();
  example_contraction_first_last_touch();
  example_einsum();
  example_einsum_operation();

  return 0;
}