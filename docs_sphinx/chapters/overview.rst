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
