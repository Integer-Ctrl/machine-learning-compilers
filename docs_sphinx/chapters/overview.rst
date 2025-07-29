Overview
========

In this project, we will develop a domain-specific compiler for tensor expressions from scratch. Tensor compilers are used to translate
high-level tensor operations into efficient, low-level code that can be executed on various hardware platforms.

Tensor expressions are mathematical representations of operations on multi-dimensional arrays (tensors). They are widely used in machine
learning, scientific computing, and data analysis. The goal of this project is to create a compiler that efficiently translates these
expressions into machine code, enabling high-performance execution on modern hardware.

The compiler developed in this project is designed with the following goals in mind:

- **High Throughput**: Efficient execution of repeated evaluations of the same expression.
- **Low Latency**: Fast response time for single expression evaluations.
- **Short Compile Times**: Rapid compilation from expression to executable code.
- **Flexibility**: Support for a wide range of tensor expressions.

To meet these goals, the compiler will be primitive-based, meaning that complex tensor operations are built from a set of manually tuned
low-level primitives. These primitives are handcrafted during development and enable just-in-time (*JIT*) compilation of tensor
expressions to machine code. In this project, the primitives are optimized for the **ARM64** architecture.

Documentation
-------------

This overview chapter will guide you through the key components of the project. The documentation is structured into several chapters,
each focusing on a specific aspect of the project.

Assembly
--------

In the first chapter, :doc:`assembly`, we revisit the basics of assembly language as an essential requirement before diving into implementing
the compiler. This chapter is intended as a refresher and is not directly part of the project's goal.

The next chapter is as well not directly part of the project's goal, but it is used to get familiar with some ARM64 assembly instructions
and on how to benchmark the performance of these instructions.

Base
----

In the :doc:`base` chapter, we will implement the functionality of several given C functions using only base instructions. After that, we
will benchmark the execution throughput and latency of selected ARM64 instructions.

With that, the next chapters will focus on the implementation of the compiler itself.

Neon
----

This chapter focuses on implementing the first kernels using ARM64 `Neon <https://developer.arm.com/Architectures/Neon>`_ instructions.
We begin with a small microkernel for matrix multiplication on fixed-size matrices. Next, we scale the microkernel to handle larger matrices
by introducing loops over the *K*, *M*, and *N* dimensions over our microkernel.

We then address edge cases where the *M* dimension of the matrix is not a multiple of 4, a prerequirement assumed up to this point.
After that, we extend the microkernel to support batch-reduced matrix multiplication, a widely used operation in machine learning workloads.
Finally, we explore how to transpose a matrix in memory using Neon instructions on a fixed-sized :math:`8 \times 8` matrix.

Code Generation
---------------

In this chapter, we took a look on how to JIT (Just-In-Time) generate the kernel we written in the :doc:`neon` chapter.
Furthermore, we dynamically adjust the generated machine code to perform arbitrary-sized matrix multiplications using fast kernels. 

To begin, we wrap the necessary machine code instructions in an assembly-like manner using C++ functions to make kernel generation easier.
We then implement kernel generation for Batch-Reduce General Matrix-Matrix Multiplication (BRGEMM) and Unary Operations for zero, identity and ReLU.
Finally, we measure the performance of our generated kernels across different size configurations.

Tensor Operation
----------------

This chapter introduces an additional layer of abstraction to :doc:`code_generation` by describing higher-level tensor operations.
We therefore examine how to generate the correct kernel based on a provided tensor configuration object, i.e. the abstraction.
This object describes which operations on parameters, such as the size and type of dimensions, the execution type and the strides of the involved tensors, are required to generate and execute a kernel.
Furthermore, we also perform optimization passes such as primitive and shared identification, dimension splitting, dimension fusion and dimension reordering.
These optimizations help to boost the performance of the generated kernel for a given tensor operation.

Einsum Tree
-----------

In this chapter, we introduce an additional layer of abstraction by defining a tree representation of multiple chained contractions on a set of two or more input tensors.
We therefore process a string representation of nested tensor operations alongside a list of the dimension sizes of the tensors used.
We then generate a tree representation from these input values, where each non-leaf node represents a single tensor operation. These operations are lowered to kernels, as described in the :doc:`tensor_operations` chapter.
Furthermore, we optimize this tree representation by performing optimization passes: Swap, Reorder and Permutation Insert on a node of the tree.

Individual Phase
----------------

In the final chapter, :doc:`report_individual`, we developed a plan on how to further develop the project.
We created a draft to convert the project into a CMake library with a convenient tensor interface.
We then provide a step-by-step description of how we converted our project into a CMake library.
We also present our library interface, which defines a high-level tensor structure and operations such as unary, GEMM, contraction and Einsum expressions.
Finally, to help users work with our library, we provide an example project that uses all the tensor operations, as well as extensive documentation with examples.