# CMake Library

In this user guide, we will cover our CMake library we made from the machine learning compiler project. This library was designed to simplify the usage of our machine learning compiler and to provide an easy to use interface for users.

## Overview

We will guide you through the process of integrating our CMake library into your project, highlight its features, and provide an example project to demonstrate its usage.

- [Library Usage](#library-usage)
  - [Integration into CMakeLists](#integration-into-cmakelists)
  - [Installing the Library](#installing-the-library)
- [Library Features](#library-features)
  - [Tensor Object](#tensor-object)
  - [Tensor Expressions](#tensor-expressions)
    - [GEMM](#gemm)
    - [Unary Operations](#unary-operations)
    - [Contraction](#contraction)
    - [Einsum](#einsum)
- [Example Project](#example-project)

# Library Usage

### Integration into CMakeLists

To integrate our CMake library into your project you can choose between two methods:

1. Directly fetch the content of this library from github and build it with your cmake:

    ```cmake
    # Optional: Toggles if included libraries is build as shared or static libraries. Default is ON.
    set(BUILD_SHARED_LIBS OFF)

    # Optional: Toggles if OpenMP should be used by the library. Default is ON.
    set(MLC_USE_OPENMP ON)

    Include(FetchContent)
    FetchContent_Declare(
        MachineLearningCompiler
        GIT_REPOSITORY https://github.com/Integer-Ctrl/machine-learning-compilers
        GIT_TAG        individual-phase
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(MachineLearningCompiler)
    ```

    If needed, you can specify two CMake options:

    1. `BUILD_SHARED_LIBS`: This option toggles if the included libraries are built as shared or static libraries. The default is `ON`, meaning shared libraries will be built.
    2. `MLC_USE_OPENMP`: This option toggles if OpenMP should be used by the library. The default is `ON`, meaning OpenMP will be used for parallelization if available.

2. Include it from the the current machine if installed on the system:

    ```cmake
    find_library(mlc::MachineLearningCompiler)
    ```

    If you want to install the library on your system, you can do this by following [Installing the Library](#installing-the-library).

### Installing the Library

  1. Clone the repository `git clone https://github.com/Integer-Ctrl/machine-learning-compilers.git`
  2. Navigate to the directory `cd machine-learning-compilers`
  3. Create a build directory `mkdir build && cd build`
  4. Run CMake to configure the build `cmake ..` \
    Optionally, you can specify the install directory with `cmake .. -DCMAKE_INSTALL_PREFIX=<installation_path>` (see [CMAKE_INSTALL_PREFIX](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html))
  5. Install the library `cmake --build . --target install`

  Now you can use the library in your CMake project by using the `find_library` command as shown in [Integration into CMakeLists](#integration-into-cmakelists).

## Library Features

In this section, we will cover the features of our CMake library. The library provides a simple interface to work with tensors and tensor expressions. It supports various tensor operations such as GEMM, unary operations, contraction, and einsum.

### Tensor Object

The library provides a `Tensor` class that represents a multi-dimensional array of data. This class is used as the input type for all tensor operations. Since the tensor compiler only supports unit-stride tensors, meaning elements must be stored contiguously in memory without gaps, strides can not be explicitly defined. Instead, they are automatically computed based on the tensor’s dimensions.

There are two ways to create a tensor. The first is to create a tensor with data and the suitable dimension sizes. The second is to create a tensor only by specifying the dimension sizes, which will allocate the data internally and fill it with zeros.

```cpp
#include <MachineLearningCompiler/Tensor.h>

float data[] = {1, 2, 3, 4};

mlc::Tensor tensor({2, 3, 4}); // 3D tensor with 2 layers, 3 rows and 4 columns initialized with zeros
mlc::Tensor tensorWithData1(data, {2, 2}); // 2D tensor with specific data

std::cout << "Tensor dimensions: " << tensor.dim_sizes << std::endl; // Dimensions of the tensor
std::cout << "Tensor strides: " << tensor.strides << std::endl; // Strides of the tensor
std::cout << tensor.to_string("Tensor") << std::endl; // String representation of the tensor
```

To fill a tensor with data a variety of functions are provided. Below are all available functions to fill a tensor with data:

```cpp
#include <MachineLearningCompiler/Tensor.h>

mlc::Tensor tensor({2, 3, 4}); // 3D tensor with 2 layers, 3 rows and 4 columns initialized with zeros
size_t size = tensor.size();

mlc::fill_random(tensor); // Fill the tensor with random values
mlc::fill_number(tensor, 3.2); // Fill the tensor with a single number, in this case 3.2
mlc::fill_counting_up(tensor, 0.1, 0.1); // Fill the tensor with counting up values starting from 4 and increasing by 0.1
mlc::fill_counting_down(tensor, 5, 1); // Fill the tensor with counting down values starting from 5 reducing by 1
mlc::fill_lambda(tensor, [&size](const mlc::Tensor &self, size_t index) { return size; }); // Fill the tensor with a user defined function, in this case the size of the tensor
```

### Tensor Expressions

Next we will cover the tensor expressions which the library provides. All tensor expressions return an `mlc::Error` object which contains the result of the operation. If the operation was successful, the `type` field of the `Error` object will be set to `mlc::ErrorType::None`. If there was an error, the `type` field will contain the type of error that occurred.

#### GEMM

To perform a general matrix-matrix multiplication (GEMM), three tensors are required: two input tensors and one output tensor. The input tensors must have compatible dimensions for matrix multiplication, and the output tensor must have the correct dimensions to store the result.

```cpp
#include <MachineLearningCompiler/Tensor.h>

mlc::Tensor in0({5, 3});  // IDs: 0,1
mlc::Tensor in1({2, 5});  // IDs: 2,0
mlc::Tensor out({2, 3});  // IDs: 2,1

mlc::Error error = mlc::gemm(in0, in1, out);
```

#### Unary Operations

Our library supports three unary operations: **zero**, **identity** and **ReLU** (Rectified Linear Unit). **zero** receive one input tensor and produce one output tensor, while **ReLU** and **identity** receive one input tensor and and one output tensor which will be filled with the same data as the input tensor but with the ReLU or identity operation applied.

```cpp
#include <MachineLearningCompiler/Tensor.h>

mlc::Tensor in({2, 2});
mlc::Tensor out({2, 2});

mlc::Error error = mlc::unary_zero(in);
mlc::Error error = mlc::unary_identity(in, out);
mlc::Error error = mlc::unary_relu(in, out);
```

#### Contraction

To get more advanced, lets look at the contraction operation. This operation allows you to perform a contraction of two tensors based on a user defined expression. The expression defines which dimensions of the input tensors are contracted (reduce dimensions) and which dimensions are retained (output dimensions) in the output tensor. 

```cpp
#include <MachineLearningCompiler/Tensor.h>

mlc::Tensor in0({5, 4, 3});     // IDs: 0,1,2
mlc::Tensor in1({5, 2, 4});     // IDs: 3,4,1
mlc::Tensor out({5, 5, 2, 3});  // IDs: 0,3,4,

mlc::Error error = mlc::contraction(in0, in1, out, "[0,1,2],[3,4,1]->[0,3,4,2]");
```

In the example above, the contraction operation takes two input tensors `in0` and `in1`, and produces an output tensor `out`. The expression `"[0,1,2],[3,4,1]->[0,3,4,2]"` defines that the dimensions with IDs `0`, `2`, `3`and `4` are retained in the output tensor, while the dimensions with IDs `1` is contracted. The output tensor will have the dimensions `[5, 5, 2, 3]`.

To further advance the contraction operation, a first touch primitive and a last touch primitive can be specified. The first touch primitive is applied to the output tensor before the contraction operation, while the last touch primitive is applied to the output tensor after the contraction operation. The supported primitives are `mlc::UnaryType::None`, `mlc::UnaryType::Zero`, `mlc::UnaryType::Identity` and `mlc::UnaryType::ReLU`.

```cpp
#include <MachineLearningCompiler/Tensor.h>

mlc::Tensor in0({5, 4, 3});     // IDs: 0,1,2
mlc::Tensor in1({5, 2, 4});     // IDs: 3,4,1
mlc::Tensor out({5, 5, 2, 3});  // IDs: 0,3,4,2

mlc::Error error = mlc::contraction(in0, in1, out, "[0,1,2],[3,4,1]->[0,3,4,2]", mlc::UnaryType::None, mlc::UnaryType::ReLU);
```

In the example above, the first touch primitive is set to `mlc::UnaryType::None`, meaning that no operation is applied to the output tensor before the contraction operation. The last touch primitive is set to `mlc::UnaryType::ReLU`, meaning that the **ReLU** operation is applied to the output tensor after the contraction operation.

#### Einsum

The last operation we will cover is the einsum operation, better known as **Einsum Tree**. This operation allows you to perform a contraction of multiple tensors based on a user defined expression. The expression defines which dimensions of the input tensors are contracted (reduce dimensions) and which dimensions are retained (output dimensions) in the output tensor. The expression is similar to the one used in the contraction operation, but it can handle multiple input tensors and a single output tensor. This allows you to perform multiple contractions in a single operation.

```cpp
#include <MachineLearningCompiler/Tensor.h>

mlc::Tensor in0({5, 3});  // IDs: 0,1
mlc::Tensor in1({2, 5});  // IDs: 2,0
mlc::Tensor in2({3, 7});  // IDs: 1,3
mlc::Tensor out({2, 7});  // IDs: 2,3

mlc::Error error = mlc::einsum({in0, in1, in2}, out, "[[0,1],[2,0]->[2,1]],[1,3]->[2,3]");
```

The example above shows a einsum tree with three input tensors (leafs), one output tensor (root) and two contraction operations. The first contraction operation is defined by the expression `[[0,1],[2,0]->[2,1]]`, using the first two input tensors `in0`and `in1`. The second contraction operation uses the intermediate output of the first contraction and the third input tensor `in2`, defined by the expression `[2,1]],[1,3]->[2,3]`.

Einsum trees can be increase in complexity very quickly, so jitting the expression every time can create an overhead. To avoid this it is possible to create a einsum tree once and reuse it. Therefore, the library provides the function `mlc::einsum_operation` which receives the shapes of the input tensors and the output tensor, as well as the expression. This function returns an `mlc::TensorOperation` object which can be used to execute the einsum tree multiple times with different input tensors.

```cpp
#include <MachineLearningCompiler/Tensor.h>

mlc::Tensor in0({5, 3});  // IDs: 0,1
mlc::Tensor in1({2, 5});  // IDs: 2,0
mlc::Tensor in2({3, 7});  // IDs: 1,3
mlc::Tensor out({2, 7});  // IDs: 2,3

mlc::Tensor in0_2(in0.dim_sizes);  // IDs: 0,1
mlc::Tensor in1_2(in1.dim_sizes);  // IDs: 2,0
mlc::Tensor in2_2(in2.dim_sizes);  // IDs: 1,3
mlc::Tensor out_2(out.dim_sizes);  // IDs: 2,3

// Generates a tensor operation with fixed input and ouput tensor shapes.
mlc::TensorOperation *op = mlc::einsum_operation({in0.dim_sizes, in1.dim_sizes, in2.dim_sizes}, out.dim_sizes, "[[0,1],[2,0]->[2,1]],[1,3]->[2,3]");

// Process any error that may occurs during the setup of the operation.
mlc::Error error = op->getSetupError();

// Execute the operation.
error = op->execute({in0, in1, in2}, out);

// Execute the operation again but on different tensors of the same size.
error = op->execute({in0_2, in1_2, in2_2}, out_2);

delete op; // Don't forget to delete the operation object after you are done with it.
```

**Important**: Don't forget to delete the `mlc::TensorOperation` object after you are done with it to avoid memory leaks.

## Example Project

To demonstrate the usage of our CMake library, we have created an example project. This project showcases the features which we introduced in the previous section. You can find the example project in the `cmake-library/example-project` directory. There you can have a look at the `CMakeLists.txt` file and the `Example.cpp` file which contains the example code.
