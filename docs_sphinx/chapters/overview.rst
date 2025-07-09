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
