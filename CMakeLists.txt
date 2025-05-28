
cmake_minimum_required(VERSION 3.10.0)
project(MachineLearningCompiler VERSION 0.1.0 LANGUAGES C CXX ASM)

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

# =============================================================
# Extra build options
# =============================================================
option(SAVE_JITS_TO_FILE "Saves the jitted kernels into a file if activated." OFF)


if(SAVE_JITS_TO_FILE)
    message(NOTICE "The saved kernels can be disassembled with: 'objdump -D -b binary -m aarch64 <inputFile> > <outputFile>'")
    # set per target
endif()

# ==============================================================
# Setup compile Flags
# ==============================================================
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Werror -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_VERBOSE_MAKEFILE off)

# ==============================================================
# Fetch external libs
# ==============================================================
Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
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
    Unary.h
    Unary.cpp
    TensorOperation.h
    TensorOperation.cpp
)

set(KERNEL_FILES
    matmul_16_6_1.cpp
    matmul_16_6_1.h
    matmul_16_6_k.cpp
    matmul_16_6_k.h
    matmul_16m_4n_k.h
    matmul_16m_4n_k.cpp
    matmul_16m_lt4nRest_k.h
    matmul_16m_lt4nRest_k.cpp
    matmul_16mRest_4n_k.h
    matmul_16mRest_4n_k.cpp
    matmul_16mRest_lt4nRest_k.h
    matmul_16mRest_lt4nRest_k.cpp
    matmul_lt16_4n_k.h
    matmul_lt16_4n_k.cpp
    matmul_lt16_lt4nRest_k.h
    matmul_lt16_lt4nRest_k.cpp

    br_matmul_16m_4n_k.h
    br_matmul_16m_4n_k.cpp
    br_matmul_16m_lt4nRest_k.h
    br_matmul_16m_lt4nRest_k.cpp
    br_matmul_16mRest_4n_k.h
    br_matmul_16mRest_4n_k.cpp
    br_matmul_16mRest_lt4nRest_k.h
    br_matmul_16mRest_lt4nRest_k.cpp
    br_matmul_lt16_4n_k.h
    br_matmul_lt16_4n_k.cpp
    br_matmul_lt16_lt4nRest_k.h
    br_matmul_lt16_lt4nRest_k.cpp

    unary/unary_all.h
    unary/unary_zero_16m_n.h
    unary/unary_zero_16m_n.cpp
    unary/unary_identity.h
    unary/unary_identity.cpp
    unary/unary_identity_transpose.h
    unary/unary_identity_transpose.cpp
    unary/unary_zero.h
    unary/unary_zero.cpp
    unary/unary_relu.h
    unary/unary_relu.cpp
    unary/unary_relu_transpose.h
    unary/unary_relu_transpose.cpp
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
    base/movn.h
    base/movz.h
    
    simd_fp/ld1.h
    simd_fp/st1.h
    simd_fp/fmla.h
    simd_fp/ldp.h
    simd_fp/stp.h
    simd_fp/str.h
    simd_fp/fmax.h
    simd_fp/trn1.h
    simd_fp/trn2.h
    simd_fp/zip1.h
    simd_fp/zip2.h
    simd_fp/eor.h
)

set(TEST_FILES
    BaseGeneration.test.h
    BaseGeneration.test.cpp
    Brgemm.test.cpp
    TensorOperation.test.cpp
)

set(TEST_KERNELS
    matmul.test.h
    matmul.test.cpp
    matmul_16_6_1.test.cpp
    matmul_16_6_k.test.cpp
    matmul_16m_4n_k.test.cpp
    matmul_16mRest_4n_k.test.cpp
    matmul_16m_lt4nRest_k.test.cpp
    matmul_16mRest_lt4nRest_k.test.cpp
    matmul_lt16_4n_k.test.cpp
    matmul_lt16_lt4nRest_k.test.cpp

    br_matmul_16m_4n_k.test.cpp
    br_matmul_16mRest_4n_k.test.cpp
    br_matmul_16m_lt4nRest_k.test.cpp
    br_matmul_16mRest_lt4nRest_k.test.cpp
    br_matmul_lt16_4n_k.test.cpp
    br_matmul_lt16_lt4nRest_k.test.cpp

    unary/unary.test.h
    unary/unary.test.cpp
    unary/unary_zero_16m_n.test.cpp
    unary/unary_identity.test.cpp
    unary/unary_identity_transpose.test.cpp
    unary/unary_zero.test.cpp
    unary/unary_relu.test.cpp
    unary/unary_relu_transpose.test.cpp
)

set(TEST_ARM_INSTRUCTION_FILES
    base/ldr.test.cpp
    base/lsl.test.cpp
    base/ret.test.cpp
    base/add.test.cpp
    base/cbnz.test.cpp
    base/ldp.test.cpp
    base/stp.test.cpp
    base/sub.test.cpp
    base/orr.test.cpp
    base/mov.test.cpp
    base/movz.test.cpp
    base/madd.test.cpp
    base/movn.test.cpp

    simd_fp/fmla.test.cpp
    simd_fp/ld1.test.cpp
    simd_fp/st1.test.cpp
    simd_fp/ldp.test.cpp
    simd_fp/stp.test.cpp
    simd_fp/ldr.test.cpp
    simd_fp/str.test.cpp
    simd_fp/fmax.test.cpp
    simd_fp/trn1.test.cpp
    simd_fp/trn2.test.cpp
    simd_fp/zip1.test.cpp
    simd_fp/zip2.test.cpp
    simd_fp/eor.test.cpp
)

set(BENCH_FILES
    matmul_16_6_1.bench.cpp
    matmul_16_6_k.bench.cpp
    matmul.bench.cpp

    unary/unary_zero.bench.cpp
    unary/unary_identity.bench.cpp
    unary/unary_identity_transpose.bench.cpp
    unary/unary_relu.bench.cpp
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
    list(APPEND BENCH_FILEPATHS src/test/kernels/${file})
endforeach()

# =============================================================
# TARGETS
# =============================================================
add_executable(tests "${SOURCE_FILEPATHS}" "${TEST_FILEPATHS}")
if(SAVE_JITS_TO_FILE)
target_compile_definitions(tests PUBLIC SAVE_JITS_TO_FILE)
endif(SAVE_JITS_TO_FILE)
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