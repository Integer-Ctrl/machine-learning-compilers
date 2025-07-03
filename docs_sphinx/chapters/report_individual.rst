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

Creating a C++ library with CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We started with creating library of our project with CMake. Therefor we used two tutorial, found be research on the web.
We followed both tutorials and collected all information on how to create a library with CMake.

https://www.ics.com/blog/creating-reusable-libraries-cmake
https://decovar.dev/blog/2021/03/08/cmake-cpp-library/

1. Project Structure

We differentiate between public and private headers. Public headers are the header files that will be visible to the users,
showing what functions available in our library, and what signature they have. Private headers on the other hand are discouraged,
not visible and by default not usable to the user.

```bash
├── CMakeLists.txt
├── include
│   └── tensor.h
└── src
    ├── interface/
    ├── main/
    └── test/
```

Public headers are in the directory ``include/``, ``include/tensor.h`` providing all functions that are for usage in out library.
All files under ``src/`` are private.

CMakeLists
^^^^^^^^^^

Next, we adjust the CMakeLists of our project so it full fills all required settings to be included as an library.

**Top-level-project vs. Sub-level-project**

A Cmake project can either be a Top-level-project, meaning that it is the main project, possibly including other project,
or it is a sub-project (included library) of another project. This differentiation is important because depending on which
level the project is build, the build targets may vary. To check this, CMake has a boolean variable `PROJECT_IS_TOP_LEVEL <https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html>`,
returning if the project is the top score, or was was called from another top level ``CMakeLists.txt`` file.

In our case we do not build tests nor benchmarks if the project is used as library for another project.

**Library Target**

Next we dynamically check whether our library should be a static or a shared library. Again, CMake provided a global flag for this.
`BUILD_SHARED_LIBS <https://cmake.org/cmake/help/latest/variable/BUILD_SHARED_LIBS.html>` which means calls to ``add_library()``
without any explicit library type check the current ``BUILD_SHARED_LIBS``. If it is true, then the default library type is
``SHARED``. Otherwise, the default is ``STATIC``.

The targets themselves are all files under ``src/*``.

.. code-block:: cmake

  add_library(${PROJECT_NAME})

  target_sources(${PROJECT_NAME}
      PRIVATE
          "${SOURCE_FILEPATHS}"
  )

**Include Directories**

Next we add include directories to a target with `target_include_directories <https://cmake.org/cmake/help/latest/command/target_include_directories.html>`.
Does directories are used when compiling a given target. 

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

The pats in private are used by the library to find internal headers, that are not visible to the user. The paths in public are used by the
top-level project (BUILD_INTERFACE) and by external projects (INSTALL_INTERFACE) to find the public headers of the library

**Include Path**

When including the header in an internal project (project library) the include looks something like this: ``#include <tensor.h>``. However,
when including the header in an external project (library installed on the system), the include looks like this: ``#include <MachineLearningCompiler/tensor.h>``.
To unify the include path, there is a fix for this in CMake. By using a intermediate directory with the same name as the project,
the include path is also for external projects ``#include <tensor.h>``. Therefore our public headers are located in ``include/${PROJECT_NAME}/tensor.h``.

**OpenMP**

Our project uses OpenMP for parallelization. Therefore, we need to check if OpenMP is available on the system and if OpenMP should be used.
We created the CMake option ``MLC_USE_OPENMP`` to enable or disable OpenMP usage for the library. If it is enabled, we check if OpenMP
is available and if so, link it to the library.

.. code-block:: cmake

  if(MLC_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
  endif()

**Installation**

If the library is intended to be installed global on the system, we need to ensure that the correct installation path is used.
If not set, CMake uses the default system installation path, what ever that is. However, we want to ensure a specific installation path is used.
With CMake, the variable `CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT <https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT.html>`_
can be used to check if the `CMAKE_INSTALL_PREFIX <https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html#variable:CMAKE_INSTALL_PREFIX>` which holds the install directory,
is set. If not set, is has the default installation path. If so, we overwrite the default installation path with a custom one, installing the artifact
into ``install``.

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

**Public Headers**

The public headers are the headers that are visible to the user of the library. To set them, we use the option  `PUBLIC_HEADER <https://cmake.org/cmake/help/latest/prop_tgt/PUBLIC_HEADER.html>`.
This property is used to specify which headers are considered public and should be installed when the library is installed.

.. code-block:: cmake

  # without it public headers won't get installed
  set(public_headers
      include/${PROJECT_NAME}/tensor.h
  )
  set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${public_headers})

  set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

The second `set_target_properties` is used to set the debug postfix for the library. This is useful to differentiate between the release and debug versions of the library.
The debug option is to be set with `DEBUG_POSTFIX <https://cmake.org/cmake/help/latest/prop_tgt/DEBUG_POSTFIX.html>`_.

**Destinations**

To this point we only configured the installation but we did not execute it. To do so, we use the `install <https://cmake.org/cmake/help/latest/command/install.html>`_ command.
`GNUInstallDirs <https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html>`_ is a CMake module that provides standard installation directories.

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

**Config**

During the installation, CMake createds ``*.cmake`` files that are used to configure the library for other projects.
To tell CMake where to find these files, we need to create a small configuration file ``Config.cmake.in``.

.. code-block:: cmake

  @PACKAGE_INIT@

  include("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@Targets.cmake")

  check_required_components(@PROJECT_NAME@)

This tells CMake to include the targets file that was created during the installation and to check if the required components are available.
We can use the ``PROJECT_NAME`` there as well by wrapping it in `@` signs, which will be replaced by the actual project name during the installation.

Then, first we define a namespace for the exported targets ``mlc::``.
Next, we export the targets with installs a CMake export file containing build targets under the specified namespace into the ``cmake/`` directory.
``write_basic_package_version_file`` generates a version file to handle compatibility checks when importing the package. It will
create ``SomeLibraryConfigVersion.cmake`` file in the install folder.
With ``configure_package_config_file`` CMake generated the configuration files from our defined configuration template ``Config.cmake.in``.
Lastly, the generated configuration files are installed into the same ``cmake/`` directory.

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

**Building and Installing**

To build the library, we can use the following commands:

```bash
mkdir build
cd build
cmake ..
cmake --build . --target install
```

**Linking to the Library**

???


