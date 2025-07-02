cmake_minimum_required(VERSION 3.28.0)
project(ExampleProject VERSION 0.1.0 LANGUAGES C CXX ASM)

# The MachineLearningCompiler library is only supported on Linux on arm.
if(NOT (UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm)"))
   message(FATAL_ERROR "Only arm on Linux is supported.") 
endif()


# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IS_MULTI_CONFIG)
    message(NOTICE "Using multi-config generator. Compile with: cmake --build . --config [Debug|Release] --target <target>")
else()
    message(NOTICE "Using single-config generator. Generate with: cmake .. -DCMAKE_BUILD_TYPE=[Debug|Release]")
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release")
        message(WARNING "No Build type is set. Using Release!")
    endif()
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")


# ===========================================
# Include the MachineLearningCompiler Library
# ===========================================

# Option 1: Including the MachineLearningCompiler Library

# Optional: Toggles if included libraries is build as shared or static libraries. Default is ON.
set(BUILD_SHARED_LIBS ON)

# Optional: Toggles if OpenMP should be used by the library. Default is ON.
set(MLC_USE_OPENMP ON)

Include(FetchContent)
FetchContent_Declare(
    MachineLearningCompiler
    GIT_REPOSITORY https://github.com/Integer-Ctrl/machine-learning-compilers
    GIT_TAG        individual-phase #TODO change
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(MachineLearningCompiler)

# Option 2: Include it from the the current machine if installed.
# find_library(mlc::MachineLearningCompiler)

# ===========================================

add_executable(example 
    Example.cpp
)
target_link_libraries(example mlc::MachineLearningCompiler)