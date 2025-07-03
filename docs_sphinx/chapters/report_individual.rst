Individual Phase
================

Draft Ideas
-----------

C++ Library
^^^^^^^^^^^

Currently, our project is primarily usable by the developers, us, themselves. The goal for this
phase is to transition the project from its developmental stage into a production-ready C++
library that can be easily integrated into other C++ projects.

To achieve this, we plan to package the existing functionality into a distributable library. In order
to simplify usage and improve consistency across our tensor operations, we are developing a dedicated
tensor structure in C++. This structure will serve as a unified interface for creating and managing
tensors, making it easier to use them for our tensor operations.

Documentation will also be a key part of this phase. We will provide clear instructions on how to
install and use the library, along with detailed guidance on its features and functionality.

.. 2. Python Library
.. ^^^^^^^^^^^^^^^^^

.. To make our C++ tensor operations even more accessible to a wider audience, we plan to
.. publish a Python package that bridges the ease of Python with the performance of JIT-compiled
.. C++ code. This will involve creating Python bindings for our C++ project and implementing
.. a Python interface that fully exposes its functionality. The final package is intended to
.. be published on `PyPi <https://pypi.org>`_, allowing users to easily install and use it
.. with minimal setup.

.. Suggestions
.. -----------

.. 1. Implementation of the C++ library
.. 2. Implementation fo the Python library
.. 3. Implementation of the C++ library and creating the Python library on top of the C++ library

Execution
---------

This section is divided into two parts. The first part describes how we created a CMake library for our project. The second part
describes how we extended the project to make our tensor operations easier to use by creating a dedicated interface for them.

CMake Library
^^^^^^^^^^^^^

Let's start with the first part, creating a CMake library for our project. The goal is to create a library that can be easily integrated into other C++ projects.
Because our project is already based on CMake, we can use it to create a library that can be integrated into other CMake projects. To gather information on how to
create a CMake library, we researched the web and found two sources from which we collected all information we needed. The first one is a
`blog post by decovar <https://decovar.dev/blog/2021/03/08/cmake-cpp-library/>`_ that explains step by step how to make a C++ library with CMake.
The second source is the `official CMake documentation <https://cmake.org/>`_, which provides detailed information on the different configuration
options and commands available in CMake.

With that information we can start creating our CMake library.

Project Structure
"""""""""""""""""

We start with the project structure, which is important because it defines how files are organized and how CMake accesses them.
We differentiate between public and private headers. Public headers are visible to users and specify the functions available in
our library, including their signatures. Private headers, on the other hand, are intended for internal use and are not visible
by default so they cannot be used by the user.

.. code-block:: bash

  ├── CMakeLists.txt
  ├── include
  │   └── MachineLearningCompiler
  │       ├── Error.h
  │       ├── Tensor.h
  │       └── Unary.h
  └── src
      ├── interface/
      ├── main/
      └── test/


Public headers are located in the ``include/`` directory and provide all functions meant
for users of the library. All files under ``src/`` are considered private.

CMakeLists
""""""""""

Next, we adjust the CMakeLists of our project so it full fills all required settings to be included as an library in another CMake project.

Top-level-project vs. Sub-level-project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A CMake project can be either a top-level project, meaning it is the main project and may include other projects, or a sub-project
(an included library) within another project. This distinction is important because the build targets can vary depending on the
project's level. To determine this, CMake provides a boolean variable
`PROJECT_IS_TOP_LEVEL <https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html>`_, which returns true if the project
is the top-level project, or false if it was included by another top-level ``CMakeLists.txt`` file.

In our project, the main build difference is that tests and benchmarks are not built when the project is used as a library within
another project.

Library Target
~~~~~~~~~~~~~~

Next, we dynamically determine whether our library should be built as a static or shared library. CMake provides a global flag
for this called `BUILD_SHARED_LIBS <https://cmake.org/cmake/help/latest/variable/BUILD_SHARED_LIBS.html>_` which controls the
default library type for calls to ``add_library()`` without an explicit type.  If ``BUILD_SHARED_LIBS`` is set to true, the default library
type is ``SHARED``. Otherwise, it defaults to ``STATIC``.

In our project, the build targets for the library include all files under ``src/*``.

.. code-block:: cmake

  add_library(${PROJECT_NAME})

  target_sources(${PROJECT_NAME}
      PRIVATE
          "${SOURCE_FILEPATHS}"
  )

Target Include Directories
~~~~~~~~~~~~~~~~~~~~~~~~~~

Target include directories specify where the compiler should look for header files when compiling a target. In CMake, this
is done using the `target_include_directories <https://cmake.org/cmake/help/latest/command/target_include_directories.html>`_
command. 

.. code-block:: cmake

  target_include_directories(${PROJECT_NAME}
      PRIVATE   
          # where the library itself will look for its internal headers
          ${CMAKE_CURRENT_SOURCE_DIR}/src/interface
          ${CMAKE_CURRENT_SOURCE_DIR}/src/main
          ${CMAKE_CURRENT_SOURCE_DIR}/src/main/kernels
          ${CMAKE_CURRENT_SOURCE_DIR}/src/main/kernels/unary
          ${CMAKE_CURRENT_SOURCE_DIR}/src/main/arm_instructions
          ${CMAKE_CURRENT_SOURCE_DIR}/src/main/arm_instructions/base
          ${CMAKE_CURRENT_SOURCE_DIR}/src/main/arm_instructions/register
          ${CMAKE_CURRENT_SOURCE_DIR}/src/main/arm_instructions/simd_fp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/test
          ${CMAKE_CURRENT_SOURCE_DIR}/src/test/kernels
          ${CMAKE_CURRENT_SOURCE_DIR}/src/test/kernels/unary
          ${CMAKE_CURRENT_SOURCE_DIR}/src/test/arm_instructions
          ${CMAKE_CURRENT_SOURCE_DIR}/src/test/arm_instructions/base
          ${CMAKE_CURRENT_SOURCE_DIR}/src/test/arm_instructions/simd_fp
      PUBLIC
          # using the project name as additional directory to include <project_name>/header.h instead of header.h if it is included as internal library
          # where top-level project will look for the library's public headers
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
          # where external projects will look for the library's public headers
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

Paths marked as private are used internally by the library to find headers that are not visible to users. Paths marked as
public are used both by the top-level project during the build (BUILD_INTERFACE) and by external projects consuming the
installed library (INSTALL_INTERFACE) to locate the public headers.

Include Path
~~~~~~~~~~~~

When including headers within an internal project (means as project library), the include looks like this: ``#include <tensor.h>``.
However, when the library is installed and used by an external project, the include has a prefix and looks like this:
``#include <MachineLearningCompiler/tensor.h>``. To unify these include paths, we can place the public headers
inside an sub-directory named same as the project. This ensures that the include path can remain consistent across internal and external usage.
Therefore, our public headers are located in ``include/${PROJECT_NAME}/*``. This allows both internal and external projects to use the same
include style ``#include <MachineLearningCompiler/tensor.h>``.

OpenMP
~~~~~~

Our project uses OpenMP for parallelization. Therefore, we need to check if OpenMP is available on the system and if OpenMP should be used.
Therefore we created the CMake option ``MLC_USE_OPENMP`` to enable or disable OpenMP usage for our library. If it is enabled, we check
if OpenMP is available on the system and if so, link it to the library.

.. code-block:: cmake

  if(MLC_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
  endif()

Installation Path
~~~~~~~~~~~~~~~~~

If the library is intended to be installed globally on the system, we need to ensure the correct installation path is used. If no path is set,
CMake uses the default system installation path, whatever that may be. However, we want to specify a custom installation path explicitly.

CMake provides the variable `CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT <https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT.html>`_
to check whether the installation directory variable `CMAKE_INSTALL_PREFIX <https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html#variable:CMAKE_INSTALL_PREFIX>`_
has been set by the user or is still at its default. If ``CMAKE_INSTALL_PREFIX`` is still set to the default, we override it with a custom
path to direct where the library artifacts will be installed. In our case, we set it to the directory ``install`` within the source directory.

.. code-block:: cmake

  if(PROJECT_IS_TOP_LEVEL)
    if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(
            STATUS
            "CMAKE_INSTALL_PREFIX is not set\n"
            "   ├ Default value: ${CMAKE_INSTALL_PREFIX}\n"
            "   └ Will set it to ${CMAKE_SOURCE_DIR}/install"
        )
        set(CMAKE_INSTALL_PREFIX
            "${CMAKE_SOURCE_DIR}/install"
            CACHE PATH "Where the library will be installed to" FORCE
        )
    else()
        message(
            STATUS
            "CMAKE_INSTALL_PREFIX was already set\n"
            "   └ Current value: ${CMAKE_INSTALL_PREFIX}"
        )
    endif()
  endif()

Public Headers
~~~~~~~~~~~~~~

Public headers are the headers visible to users of the library. To specify them in CMake, we use the option  `PUBLIC_HEADER <https://cmake.org/cmake/help/latest/prop_tgt/PUBLIC_HEADER.html>`.
``PUBLIC_HEADER`` is used to specify which headers are considered public and should be installed when the library is installed.

.. code-block:: cmake

  # without it public headers won't get installed
  set(public_headers
      include/${PROJECT_NAME}/tensor.h
  )
  set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${public_headers})

  set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

Additionally, we use `set_target_properties` to define a debug postfix for the library name. This helps differentiate between release
and debug builds. The debug postfix is set using the `DEBUG_POSTFIX <https://cmake.org/cmake/help/latest/prop_tgt/DEBUG_POSTFIX.html>`_.

Install Destinations
~~~~~~~~~~~~~~~~~~~~

To this point we only configured the installation but we did not execute it. To do so, we first set the installation destinations for the
library, headers, and other files. To do this we use the `install <https://cmake.org/cmake/help/latest/command/install.html>`_ command and
`GNUInstallDirs <https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html>`_ module, which provides standard installation directories.

.. code-block:: cmake

  # definitions of CMAKE_INSTALL_LIBDIR, CMAKE_INSTALL_INCLUDEDIR and others
  include(GNUInstallDirs)

  # install the target and create export-set
  install(TARGETS ${PROJECT_NAME}
      EXPORT "${PROJECT_NAME}Targets"
      # these get default values from GNUInstallDirs, no need to set them
      #RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # bin
      #LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
      #ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
      # except for public headers, as we want them to be inside a library folder
      PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} # include/SomeLibrary
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # include
  )

Installation Config
~~~~~~~~~~~~~~~~~~~

During the installation, CMake creates ``*.cmake`` files that help other projects configure and use the library. To guide CMake on where
to find these files, we create a configuration file named ``Config.cmake.in``.

.. code-block:: none

  @PACKAGE_INIT@

  include("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@Targets.cmake")

  check_required_components(@PROJECT_NAME@)

This configuration instructs CMake to include the exported target file created during installation and checks that all required components
are available. We use the ``PROJECT_NAME`` variable in the template by enclosing it in ``@`` signs, which will be replaced with the actual
project name during installation.

Then, we define a namespace for the exported targets, ``mlc::``. Next, we export the targets, installing a CMake export file containing
the build targets under this namespace into the ``cmake/`` directory.  The function ``write_basic_package_version_file`` generates a version
file to handle compatibility checks when importing the package. This will create ``SomeLibraryConfigVersion.cmake`` file in the install folder.
Using ``configure_package_config_file``, CMake generates the final configuration files based on our ``Config.cmake.in`` template.
Finally, these generated configuration files are installed into the ``cmake/`` directory alongside the exported targets.

.. code-block:: cmake

  set(namespace mlc)

  # generate and install export file
  install(EXPORT "${PROJECT_NAME}Targets"
      FILE "${PROJECT_NAME}Targets.cmake"
      NAMESPACE ${namespace}::
      DESTINATION cmake
  )

  include(CMakePackageConfigHelpers)

  # generate the version file for the config file
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
      VERSION "${version}"
      COMPATIBILITY AnyNewerVersion
  )
  # create config file
  configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
      INSTALL_DESTINATION cmake
  )
  # install config files
  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
      DESTINATION cmake
  )

.. _building_and_installing:

Building and Installing
"""""""""""""""""""""""

To build the library and install it system wide, the following commands can be used:

.. code-block:: bash

  mkdir build
  cd build
  cmake ..
  cmake --build . --target install

Linking to the Library
"""""""""""""""""""""""

Linking depends on how the library is integrated into the ``CMakeLists.txt``. In general, two methods can be chosen:

1. Directly fetch the content of this library from `GitHub <https://github.com/Integer-Ctrl/machine-learning-compilers>`_ and include it in ``CMakeLists.txt``:

    .. code-block:: cmake

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


    If needed, you can specify two CMake options:

    1. ``BUILD_SHARED_LIBS``: This option toggles if the included libraries are built as shared or static libraries. The default is ``ON``, meaning shared libraries will be built.
    2. ``MLC_USE_OPENMP``: This option toggles if OpenMP should be used by the library. The default is ``ON``, meaning OpenMP will be used for parallelization if available.

2. Include it from the the current machine if installed on the system:

    .. code-block:: cmake

      find_library(mlc::MachineLearningCompiler)

    Checkout :ref:`building_and_installing` for more information on how to build and install the library.
      
Library Interface
^^^^^^^^^^^^^^^^^

As the second part of our project, we developed a library interface to simplify the usage of our tensor operations. This interface is
designed to be user-friendly, providing a simple tensor object that serves as data object for all tensor operations.

Tensor Object
"""""""""""""

The tensor object represents a multidimensional array and is implemented as a struct to allow easy creation and direct access to its members.
Since all fields and functions are public, the tensor can be used flexibly without additional effort.

A tensor can be initialized either with a pointer to existing data and the corresponding dimension sizes, or with just the
dimension sizes. In the second case the tensor allocates its own memory and initializes it with zeros.

.. code-block:: cpp

  struct Tensor
  {
    bool ownsData = false;
    float *data = nullptr;
    std::vector<uint64_t> dim_sizes;
    std::vector<uint64_t> strides;

    Tensor() = delete;
    Tensor(float *data, const std::vector<uint64_t> &dim_sizes);
    Tensor(const std::vector<uint64_t> &dim_sizes); 
    ~Tensor();

    std::string to_string(std::string name = "tensor");
    uint64_t size();
  };

These definitions allow users to easily generate tensors of arbitrary dimensions:

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  float data[] = {1, 2, 3, 4};

  mlc::Tensor tensor({2, 3, 4}); // 3D tensor with 2 layers, 3 rows and 4 columns initialized with zeros
  mlc::Tensor tensorWithData1(data, {2, 2}); // 2D tensor with specific data

To simplify usage, we provide various functions for filling tensors with data.

.. note::

  All of the listed functions below are implemented as regular C++ functions and do not generate *jitted* kernels.

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Tensor tensor({2, 3, 4}); // 3D tensor with 2 layers, 3 rows and 4 columns initialized with zeros
  size_t size = tensor.size();

  mlc::fill_random(tensor); // Fill the tensor with random values
  mlc::fill_number(tensor, 3.2); // Fill the tensor with a single number, in this case 3.2
  mlc::fill_counting_up(tensor, 0.1, 0.1); // Fill the tensor with counting up values starting from 4 and increasing by 0.1
  mlc::fill_counting_down(tensor, 5, 1); // Fill the tensor with counting down values starting from 5 reducing by 1
  mlc::fill_lambda(tensor, [&size](const mlc::Tensor &self, size_t index) { return size; }); // Fill the tensor with a user defined function, in this case the size of the tensor

- ``fill_random`` fills the given tensor with random floating-point numbers greater than or equal to zero.
- ``fill_number`` sets every element of the tensor to a specified value. For example ``fill_number(tensor, 0)`` fills the tensor with zeros,
  or ``fill_number(tensor, 1)`` fills the tensor with ones which is commonly used in frameworks like `Pytroch <https://docs.pytorch.org/docs/stable/generated/torch.ones.html>`_.
- ``fill_counting_up`` fills the tensor with a sequence starting at a given value and increasing by a specified step size ``data[index] = start + index * step``.
- ``fill_counting_down`` fills the tensor with a sequence starting at a given value and decreasing by a specified step size ``data[index] = start - index * step``.
- ``fill_lambda`` allows the tensor to be filled using a user-defined lambda function. One can also use outside defined variables in the lambda expression,
  see ``size`` in the example code above.


An more complex example using ``fill_lambda`` is shown below. In this case, the tensor is filled with the values ``[[1,2,3], [1,2,3], [1,2,3]]``:

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Tensor tensor({3, 3});
  mlc::fill_lambda(tensor, 
    [](const mlc::Tensor &self, size_t index) { return index % self.strides[0] + 1; });


Tensor Expressions
""""""""""""""""""

The tensor expression include all *jitted* operations developed during this project. Each expression is implemented as an independent
function within the ``mlc`` namespace and returns an ``mlc::Error`` object. This object contains a ``type`` and a ``message`` field
which should be used to catch any errors that may occur during the execution of the expression. To check if an expression
executed successfully, check whether the ``type`` is equal to ``mlc::ErrorType::None``. See the example below with the ``mlc::gemm`` expression.

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Error error = mlc::gemm(in0, in1, out);
  if (error.type != mlc::ErrorType::None)
  {
    std::cout << error.message << std::endl;
    return;
  }

Unary
~~~~~

During the project, we implemented the unary operations ``zero``, ``identity`` and ``ReLU``.  These operations can also be accessed through
the interface like shown below:

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Tensor in({2, 2});
  mlc::Tensor out({2, 2});

  mlc::Error error = mlc::unary_zero(in);
  mlc::Error error = mlc::unary_identity(in, out);
  mlc::Error error = mlc::unary_relu(in, out);

These functions internally generate the appropriate *jitted* code and execute it directly on the memory space of the tensors.
This is achieved using our ``mini_jit::TensorOperation`` class, which operates based on a configuration object.
For example, the implementation of the ``unary_zero`` operation is shown below. The other two unary operations are implemented
similarly.

.. code-block:: cpp

  mlc::Error mlc::unary_zero(Tensor &input)
  {
    int64_t stride = 1;
    std::vector<int64_t> dimSizes(input.dim_sizes.size());
    std::vector<int64_t> strides(input.dim_sizes.size());

    for (int64_t i = input.dim_sizes.size() - 1; i >= 0; i--)
    {
      strides[i] = stride;
      dimSizes[i] = static_cast<int64_t>(input.dim_sizes[i]);
      stride *= input.dim_sizes[i];
    }

    mini_jit::TensorOperation op;
    mini_jit::TensorConfig config{
      mini_jit::TensorConfig::prim_t::none,                                      // first_touch
      mini_jit::TensorConfig::prim_t::zero,                                      // main
      mini_jit::TensorConfig::prim_t::none,                                      // last touch
      std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::dim_t::c),     // dim_types
      std::vector(input.dim_sizes.size(), mini_jit::TensorConfig::exec_t::seq),  // exec_types
      dimSizes,                                                                  // dim_sizes
      strides,                                                                   // strides_in0
      std::vector<int64_t>(input.dim_sizes.size(), 0),                           // strides_in1
      strides,                                                                   // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                     // dtype_t
    };

    mini_jit::TensorOperation::error_t error = op.setup(config);
    mlc::ErrorType errorType = internal::convertTensorOperationError(error);
    if (errorType != mlc::ErrorType::None)
    {
      return {errorType, "Could not generate the kernels for the gemm operation."};
    }

    op.execute(input.data, nullptr, input.data);
    return {ErrorType::None, "Success"};
  }

GEMM
~~~~

We implemented a general matrix-matrix multiplication operation that can be called on tensors representing matrices of any
shape. All tensors have rank 2 with dimensions in the format: **in0**: (k, m), **in1**: (n, k), and **out**: (n, m).
As long as the input tensors are rank 2, matrix multiplication is applicable.

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Tensor in0({5, 3});  // IDs: 0,1
  mlc::Tensor in1({2, 5});  // IDs: 2,0
  mlc::Tensor out({2, 3});  // IDs: 2,1

  mlc::Error error = mlc::gemm(in0, in1, out);

Similar to the unary operations, the ``mini_jit::TensorOperation`` class is used with a configuration object specifically designed for GEMM
operations. This config enables *jit* compilation of the appropriate GEMM kernel.

.. code-block:: cpp

  mlc::Error mlc::gemm(const Tensor &input0, const Tensor &input1, Tensor &output)
  {
    if (input0.dim_sizes.size() != 2 || input1.dim_sizes.size() != 2 || output.dim_sizes.size() != 2)
    {
      return {ErrorType::TensorExpected2DTensor, "GEMM requires input0 and input1 to be 2D tensors and output to be a 2D tensor."};
    }

    int64_t mSize = static_cast<int64_t>(input0.dim_sizes[1]);
    int64_t nSize = static_cast<int64_t>(input1.dim_sizes[0]);
    int64_t kSize = static_cast<int64_t>(input0.dim_sizes[0]);

    if (static_cast<int64_t>(output.dim_sizes[1]) != mSize)
    {
      return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same m dimension size as the input0."};
    }

    if (static_cast<int64_t>(output.dim_sizes[0]) != nSize)
    {
      return {ErrorType::ExecuteWrongDimension, "Expected the output tensor to have the same n dimension size as the input1."};
    }

    if (static_cast<int64_t>(input1.dim_sizes[1]) != kSize)
    {
      return {ErrorType::ExecuteWrongDimension, "Expected the input1 tensor to have the same k dimension size as the input0."};
    }

    mini_jit::TensorOperation op;
    mini_jit::TensorConfig config{
      mini_jit::TensorConfig::prim_t::none,                                                                                // first_touch
      mini_jit::TensorConfig::prim_t::gemm,                                                                                // main
      mini_jit::TensorConfig::prim_t::none,                                                                                // last touch
      {mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},              // dim_types
      {mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
      {mSize, nSize, kSize},                                                                                               // dim_sizes
      {1, 0, mSize},                                                                                                       // strides_in0
      {0, kSize, 1},                                                                                                       // strides_in1
      {1, mSize, 0},                                                                                                       // strides_out
      mini_jit::TensorConfig::dtype_t::fp32,                                                                               // dtype_t
    };

    mini_jit::TensorOperation::error_t error = op.setup(config);
    mlc::ErrorType errorType = internal::convertTensorOperationError(error);
    if (errorType != mlc::ErrorType::None)
    {
      return {errorType, "Could not generate the kernels for the gemm operation."};
    }

    op.execute(input0.data, input1.data, output.data);
    return {ErrorType::None, "Success"};
  }


Contraction
~~~~~~~~~~~

The next tensor operation we implemented is a contraction operation, which generalizes matrix multiplication. It allows
the user to specify how to contract two input tensors and define the shape of the output tensor. To do this, the user
must provide a contraction string that describes the contraction pattern between the two input tensors and the desired
output tensor layout. This contraction string is similar to an einsum expression but is restricted to exactly two input
tensors and one output tensor.

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Tensor in0({5, 4, 3});     // IDs: 0,1,2
  mlc::Tensor in1({5, 2, 4});     // IDs: 3,4,1
  mlc::Tensor out({5, 5, 2, 3});  // IDs: 0,3,4,

  mlc::Error error = mlc::contraction(in0, in1, out, "[0,1,2],[3,4,1]->[0,3,4,2]");

Since a single contraction is essentially an einsum expression, we internally delegate its implementation to the  :ref:`einsum`.
function that we also implemented and introduce in the section after this one.

.. code-block:: cpp

  mlc::Error mlc::contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction)
  {
    return internal::einsum<std::reference_wrapper<const Tensor>>({input0, input1}, output, contraction);
  }

In addition, we also implemented an **advanced contraction** operation that allows the user to specify additional **first-touch** and **last-touch**
primitives. These primitives can be any of our defined unary operations, including ``none`` (``mlc::UnaryType::None``), ``zero``
(``mlc::UnaryType::Zero``), ``identity`` (``mlc::UnaryType::Identity``) and ``ReLU`` (``mlc::UnaryType::ReLU``). The unary types are defined
in ``<MachineLearningCompiler/UnaryType.h>``.

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Tensor in0({5, 4, 3});     // IDs: 0,1,2
  mlc::Tensor in1({5, 2, 4});     // IDs: 3,4,1
  mlc::Tensor out({5, 5, 2, 3});  // IDs: 0,3,4,2

  mlc::Error error = mlc::contraction(in0, in1, out, "[0,1,2],[3,4,1]->[0,3,4,2]", mlc::UnaryType::None, mlc::UnaryType::ReLU);

In this case, we cannot use the einsum expression as a substitute for the contraction because einsum does not support the first-touch and
last-touch primitives. Therefore, we use the ``mini_jit::TensorOperation`` class again, together with a specialized configuration object.

.. code-block:: cpp

  mlc::Error mlc::contraction(const Tensor &input0, const Tensor &input1, Tensor &output, const std::string &contraction,
                              const UnaryType firstTouch, const UnaryType lastTouch)
  {
    mini_jit::EinsumTree einsumTree(contraction);
    mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
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

.. _einsum:

Einsum
~~~~~~

Lastly, we also support the einsum expression, which accepts multiple input tensors, an output tensor, and a contraction tree that defines
how the inputs are combined.

.. code-block:: cpp

  #include <MachineLearningCompiler/Tensor.h>

  mlc::Tensor in0({5, 3});  // IDs: 0,1
  mlc::Tensor in1({2, 5});  // IDs: 2,0
  mlc::Tensor in2({3, 7});  // IDs: 1,3
  mlc::Tensor out({2, 7});  // IDs: 2,3

  mlc::Error error = mlc::einsum({in0, in1, in2}, out, "[[0,1],[2,0]->[2,1]],[1,3]->[2,3]");

Here we support two types on how the inputs can be passed into einsum function.
  1. The Tensors are passed as an vector of references i.e. ``std::vector<std::reference_wrapper<const Tensor>>``
  2. The Tensors are passed as an vector of pointers i.e. ``std::vector<Tensor *>``

This approach provides users with flexibility and simplicity in writing einsum expressions in C++ code. To achieve this, we implemented a
generic einsum function that internally uses a helper function ``Tensor *getTensor(<type> tensor)``, which converts any tensor representation
into a tensor pointer i.e. ``Tensor *``.

Our implementation leverages the ``mini_jit::EinsumTree`` class to parse and optimize the user-provided contraction tree before executing
the complete einsum operation.

.. code-block:: cpp

  template <typename T> mlc::Error einsum(const std::vector<T> &inputs, mlc::Tensor &output, const std::string &tree)
  {
    mini_jit::EinsumTree einsumTree(tree);
    mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
    if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
    {
      mlc::ErrorType type = convertParseError(errorParse);
      return {type, "Failed during parsing the given einsum tree."};
    }

    std::vector<int64_t> sorted_dim_sizes;
    get_sorted_dimensions_sizes(einsumTree.get_root(), inputs, sorted_dim_sizes);
    einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

    std::vector<void *> tensors(inputs.size() + 1);
    for (size_t i = 0; i < inputs.size(); i++)
    {
      tensors[i] = getTensor<T>(inputs[i])->data;
      assert(tensors[i] != nullptr);
    }
    tensors[inputs.size()] = output.data;

    mini_jit::EinsumTree::ErrorExecute errorExecute = einsumTree.execute(tensors);
    if (errorExecute != mini_jit::EinsumTree::ErrorExecute::None)
    {
      mlc::ErrorType type = convertErrorExecute(errorExecute);
      return {type, "Failed during calculation of the einsum tree."};
    }

    return {mlc::ErrorType::None, "Success"};
  }

Einsum trees can quickly become very complex, so *jitting* the expression every time can introduce significant overhead. To mitigate this,
it is possible to create an einsum tree once and reuse it multiple times.

For this purpose, we implemented the einsum operation as ``mlc::einsum_operation``, which takes the shapes of the input tensors and output
tensor, along with the contraction expression to parse and optimize. This function returns an ``mlc::TensorOperation`` object that can be
executed repeatedly with different input tensors of the same shape, avoiding redundant *jit* compilation.

.. code-block:: cpp

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

.. note::
 
  The ``mlc::einsum_operation`` function returns a pointer to the created operation, and it is the user's responsibility to delete it
  after use. This design is partly due to a technical limitation, as ``mlc::TensorOperation`` is an abstract class and cannot be
  instantiated directly. However, this also provides the advantage that the user can decide how long to keep the preprocessed einsum
  operation in memory, allowing flexible control over its lifetime and reuse.

Internally, the ``einsum_operation`` creates an instance of ``mlc::EinsumOperation``, which implements the functionality of
``mlc::TensorOperation``. However, ``mlc::EinsumOperation`` is not exposed to the user. The ``mlc::EinsumOperation`` essentially
separates the implementation of ``mlc::einsum`` into two main phases, the setup and execution.

.. code-block:: cpp

  mlc::TensorOperation *mlc::einsum_operation(const std::vector<std::vector<uint64_t>> &inputs, const std::vector<uint64_t> &output,
                                              const std::string &tree)
  {
    // ...

    EinsumOperation *operation = new EinsumOperation(inputTensors, outputTensor, tree);
    return operation;
  }

  // ...

  mlc::EinsumOperation::EinsumOperation(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &, const std::string &tree)
      : einsumTree(tree)
  {
    mini_jit::EinsumTree::ErrorParse errorParse = einsumTree.parse_tree();
    if (errorParse != mini_jit::EinsumTree::ErrorParse::None)
    {
      mlc::ErrorType type = internal::convertParseError(errorParse);
      error = {type, "Failed to parse the tree."};
    }

    std::vector<int64_t> sorted_dim_sizes;
    internal::get_sorted_dimensions_sizes<std::reference_wrapper<const Tensor>>(einsumTree.get_root(), inputs, sorted_dim_sizes);
    einsumTree.set_sorted_dim_sizes(sorted_dim_sizes);

    error = {mlc::ErrorType::None, "Success"};
  }

And the execution part:

.. code-block:: cpp

  mlc::Error mlc::EinsumOperation::execute(const std::vector<const Tensor *> &inputs, Tensor &output)
  {
    if (error.type != ErrorType::None)
    {
      return error;
    }

    Error checkError = hasSameDimensions<const Tensor *>(inputs, output);
    if (checkError.type != ErrorType::None)
    {
      return checkError;
    }

    return execute<const Tensor *>(inputs, output);
  }

  mlc::Error mlc::EinsumOperation::execute(const std::vector<std::reference_wrapper<const Tensor>> &inputs, Tensor &output)
  {
    // similar to execute with 'const Tensor *', see above
  }

  // ...

  template <typename T> inline Error EinsumOperation::execute(const std::vector<T> &inputs, Tensor &output)
  {
    std::vector<void *> tensors(inputs.size() + 1);
    for (size_t i = 0; i < inputs.size(); i++)
    {
      tensors[i] = internal::getTensor<T>(inputs[i])->data;
    }
    tensors[inputs.size()] = output.data;

    mini_jit::EinsumTree::ErrorExecute errorExecute = einsumTree.execute(tensors);
    if (errorExecute != mini_jit::EinsumTree::ErrorExecute::None)
    {
      mlc::ErrorType type = internal::convertErrorExecute(errorExecute);
      return {type, "Failed to execute the einsum operation."};
    }

    return {mlc::ErrorType::None, "Success"};
  }

.. note::

  The execute function performs checks to ensure that the input and output tensors match the sizes defined during the operation's
  creation phase, helping to catch mismatches at runtime.

Documentation
~~~~~~~~~~~~~

A library is nothing without good documentation. Therefore, we created a
`user documentation <https://github.com/Integer-Ctrl/machine-learning-compilers/blob/main/cmake-library/README.md>`_ file that explains the
concept of the tensor object and the defined operations, together with examples of how to use them. In addition to the user guide, we provide
an `example project <https://github.com/Integer-Ctrl/machine-learning-compilers/tree/main/cmake-library/example-project>`_  demonstrating
how to integrate our library into a CMake project and how to use the library interface.
