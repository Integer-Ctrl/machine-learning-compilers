Individual Phase
================

Draft Ideas
-----------

1. C++ Library
^^^^^^^^^^^^^^

Currently, our project is primarily usable by the developers, us, themselves. The goal for this
phase is to transition the project from its developmental stage into a production-ready C++
library that can be easily integrated into other C++ projects.

To achieve this, we plan to package the existing functionality into a distributable library.
In addition, we aim to improve usability by designing a clean tensor interface
that allows users to easily create, manipulate, and convert tensors to and from our internal
format.

Of course, documentation will also be a key part of this phase. We will provide clear
instructions on how to install and use the library, along with detailed guidance on its
features and functionality.

2. Python Library
^^^^^^^^^^^^^^^^^

To make our C++ tensor operations even more accessible to a wider audience, we plan to
publish a Python package that bridges the ease of Python with the performance of JIT-compiled
C++ code. This will involve creating Python bindings for our C++ project and implementing
a Python interface that fully exposes its functionality. The final package is intended to
be published on `PyPi <https://pypi.org>`_, allowing users to easily install and use it
with minimal setup.

Suggestions
-----------

1. Implementation of the C++ library
2. Implementation fo the Python library
3. Implementation of the C++ library and creating the Python library on top of the C++ library