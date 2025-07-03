# Machine Learning Compilers

This repository was created as part of the **Machine Learning Compilers** lecture and lab at Friedrich Schiller University Jena during the summer term 2025. While the lecture focused on theoretical concepts, the lab had a practical orientation, with the goal of implementing a domain-specific compiler for tensor expressions.

The main objective of the lab was to build a Just-In-Time (JIT) compiler from scratch that supports a variety of tensor operations. Tensor compilers automate the transformation of tensor expressions into executable code, aiming for high throughput, low latency, short compile times, flexibility and portability.

The lab involved weekly tasks that guided the development of this compiler. The corresponding code and implementations are part of this repository.

## Overview

This repository includes:

- Implementations of all lab tasks
- Source code of a functional JIT compiler for tensor operations
- Modular code structured for reuse and extensibility

The weekly tasks from the lab can be found here: [scalable-analyses](https://github.com/scalable-analyses/pbtc/tree/main/lab)

## CMake Library

To make the compiler easy to integrate into other projects, we structured it as a CMake library. This allows users to include and build upon our functionality directly in their own CMake-based projects. More details about the library and how to use it can be found in the [user-guide](https://github.com/Integer-Ctrl/machine-learning-compilers/blob/main/cmake-library/README.md).

## Technical Documentation

A detailed technical documentation of our implementation including the design decisions and solutions to the lab tasks, and explanations of the source code is available on our [project website](https://integer-ctrl.github.io/machine-learning-compilers/).
