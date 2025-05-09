
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

# ==============================================================
# Setup compile Flags
# ==============================================================
add_compile_options(-Werror -Wall -Wextra -Wpedantic)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Og")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_VERBOSE_MAKEFILE off)

# ==============================================================
# Fetch external libs
# ==============================================================
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

# ==============================================================
# Source files
# ==============================================================
set(SRC_MAIN_FILES
    Kernel.cpp
    Kernel.h
    Brgemm.cpp
    Brgemm.h
    release_assert.h
)

set(KERNEL_FILES
    matmul_16_6_1.cpp
    matmul_16_6_1.h
    matmul_16_6_k.cpp
    matmul_16_6_k.h
    matmul_16m_4n_k.h
    matmul_16m_4n_k.cpp
)

set(ARM_INSTRUCTION_FILES
    register.h

    register/general_purpose.h
    register/vector.h

    base/base_all.h
    base/ldr.h
    base/lsl.h
    base/add.h
    base/cbnz.h
    base/ldp.h
    base/stp.h
    base/ret.h
    base/sub.h
    base/mov.h
    base/orr.h
    base/madd.h
    
    simd_fp/ld1.h
    simd_fp/st1.h
    simd_fp/fmla.h
    simd_fp/ldp.h
    simd_fp/stp.h
)

set(TEST_FILES
    BaseGeneration.test.h
    BaseGeneration.test.cpp
    Brgemm.test.cpp
)

set(TEST_KERNELS
    matmul.test.h
    matmul_16_6_1.test.cpp
    matmul_16_6_k.test.cpp
    matmul_16m_4n_k.test.cpp
)

set(TEST_ARM_INSTRUCTION_FILES
    base/ldr.test.cpp
    base/lsl.test.cpp
    base/ret.test.cpp
    base/add.test.cpp
    base/ldp.test.cpp
    base/stp.test.cpp
    base/sub.test.cpp
    base/orr.test.cpp
    base/mov.test.cpp
    base/movz.test.cpp
    base/madd.test.cpp

    simd_fp/fmla.test.cpp
    simd_fp/ld1.test.cpp
    simd_fp/st1.test.cpp
    simd_fp/ldp.test.cpp
    simd_fp/stp.test.cpp
    simd_fp/ldr.test.cpp
)

set(BENCH_FILES
)

foreach(file ${SRC_MAIN_FILES})
    list(APPEND SOURCE_FILEPATHS src/main/${file})
endforeach()

foreach(file ${KERNEL_FILES})
    list(APPEND SOURCE_FILEPATHS src/main/kernels/${file})
endforeach()

foreach(file ${ARM_INSTRUCTION_FILES})
    list(APPEND SOURCE_FILEPATHS src/main/arm_instructions/${file})
endforeach()

foreach(file ${TEST_FILES})
    list(APPEND TEST_FILEPATHS src/test/${file})
endforeach()

foreach(file ${TEST_KERNELS})
    list(APPEND TEST_FILEPATHS src/test/kernels/${file})
endforeach()

foreach(file ${TEST_ARM_INSTRUCTION_FILES})
    list(APPEND TEST_FILEPATHS src/test/arm_instructions/${file})
endforeach()
    
foreach(file ${BENCH_FILES})
    list(APPEND BENCH_FILEPATHS src/bench/${file})
endforeach()

# =============================================================
# TARGETS
# =============================================================
add_executable(tests "${SOURCE_FILEPATHS}" "${TEST_FILEPATHS}")
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

add_executable(benchmarks "${SOURCE_FILEPATHS}" "${BENCH_FILEPATHS}")
target_link_libraries(benchmarks benchmark::benchmark_main)

# ==============================================================
# Ctest setup
# ==============================================================
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)