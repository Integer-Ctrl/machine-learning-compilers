cmake_minimum_required(VERSION 3.10.0)
project(MachineLearningCompiler VERSION 0.1.0 LANGUAGES C CXX ASM)

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

# Setup compile Flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Og")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Og")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_VERBOSE_MAKEFILE off)

# Fetch Catch2
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)

# set(BENCHMARK_DOWNLOAD_DEPENDENCIES on)
set(BENCHMARK_ENABLE_GTEST_TESTS off)
set(BENCHMARK_ENABLE_TESTING off)

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG        v1.9.2
)
FetchContent_MakeAvailable(google_benchmark)

# Add source files
set(SRC_FILES
    neon_1_1.s
    neon_1_2.s
    neon_2_1_simple.s
    neon_2_1_unrolled.s
)

set(TEST_FILES
    neon_2_1.test.cpp
)

set(BENCH_FILES
    neon_2_1.bench.cpp
)

add_executable(neon_1_1 "${SRC_FILES}"
    neon_1_1_driver.cpp
)

add_executable(neon_1_2 "${SRC_FILES}"
    neon_1_2_driver.cpp
)

add_executable(neon_2_1 "${SRC_FILES}"
    neon_2_1_driver.cpp
)

# add_executable(loops "${SRC_FILES}")
# target_sources(loops loops_main.cpp)

add_executable(tests "${SRC_FILES}" "${TEST_FILES}")
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

add_executable(benchmarks "${SRC_FILES}" "${BENCH_FILES}")
target_link_libraries(benchmarks benchmark::benchmark_main)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)