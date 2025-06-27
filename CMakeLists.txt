cmake_minimum_required(VERSION 3.28.0)
project(MachineLearningCompiler VERSION 0.1.0 LANGUAGES C CXX ASM)

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

# =============================================================
# Extra build options
# =============================================================
option(SAVE_JITS_TO_FILE "Saves the jitted kernels into a file if activated." OFF)


if(SAVE_JITS_TO_FILE)
    message(NOTICE "The saved kernels can be disassembled with: 'objdump -D -b binary -m aarch64 <inputFile> > <outputFile>'")
    # set per target
endif()

option(MLC_USE_OPENMP "Use OpenMP for parallelization of the Tensor Operation" ON)

if(MLC_USE_OPENMP)
    find_package(OpenMP REQUIRED)

    if(OpenMP_CXX_FOUND)
        add_compile_options(-fopenmp)
        add_compile_definitions(MLC_USE_OPENMP)
    endif()
endif()

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# ==============================================================
# Setup compile Flags
# ==============================================================
add_compile_options(-Wall -Wextra -Wpedantic -Werror)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_VERBOSE_MAKEFILE off)

# ==============================================================
# Fetch external libs
# ==============================================================
if(PROJECT_IS_TOP_LEVEL)
    Include(FetchContent)

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2
        GIT_TAG        v3.8.1
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(Catch2)

    # set(BENCHMARK_DOWNLOAD_DEPENDENCIES on)
    set(BENCHMARK_ENABLE_GTEST_TESTS off)
    set(BENCHMARK_ENABLE_TESTING off)

    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark
        GIT_TAG        v1.9.2
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(google_benchmark)
endif()

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
    TensorConfig.h
    TensorConfig.cpp
    TensorOperation.h
    TensorOperation.cpp
    TensorOptimization.h
    TensorOptimization.cpp
    EinsumTree.h
    EinsumTree.cpp
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
    TensorOptimization.test.cpp
    EinsumTree.test.cpp
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
    TensorOperation.bench.cpp
    TensorOptimization.bench.cpp
    EinsumTree.bench.cpp
)

set(BENCH_KERNLES_FILES
    matmul_16_6_1.bench.cpp
    matmul_16_6_k.bench.cpp
    matmul.bench.cpp

    unary/unary_zero.bench.cpp
    unary/unary_identity.bench.cpp
    unary/unary_identity_transpose.bench.cpp
    unary/unary_relu.bench.cpp
)

set(SRC_INTERFACE_FILES
    TensorUtils.h
    Tensor.cpp
)

set(TEST_INTERFACE_FILES
    TensorUtils.test.cpp
    Tensor.test.cpp
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

foreach(file ${TEST_INTERFACE_FILES})
    list(APPEND TEST_FILEPATHS src/test/interface/${file})
endforeach()

foreach(file ${TEST_ARM_INSTRUCTION_FILES})
    list(APPEND TEST_FILEPATHS src/test/arm_instructions/${file})
endforeach()

foreach(file ${BENCH_FILES})
    list(APPEND BENCH_FILEPATHS src/test/${file})
endforeach()
    
foreach(file ${BENCH_KERNELS_FILES})
    list(APPEND BENCH_FILEPATHS src/test/kernels/${file})
endforeach()

foreach(file ${SRC_INTERFACE_FILES})
    list(APPEND INTERFACE_FILEPATHS src/interface/${file})
endforeach()

# ==== Public headers of the installed library ====
set(public_headers
    include/${PROJECT_NAME}/Tensor.h
)

list(APPEND TEST_FILEPATHS "${INTERFACE_FILEPATHS}" "${public_headers}")

# =============================================================
# TARGETS
# =============================================================
if(PROJECT_IS_TOP_LEVEL)
    # tests
    add_executable(tests "${SOURCE_FILEPATHS}" "${TEST_FILEPATHS}")
    if(SAVE_JITS_TO_FILE)
        target_compile_definitions(tests PUBLIC SAVE_JITS_TO_FILE)
    endif(SAVE_JITS_TO_FILE)

    target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # tests sanitized
    add_executable(tests_sanitized "${SOURCE_FILEPATHS}" "${TEST_FILEPATHS}")
    if(SAVE_JITS_TO_FILE)
        target_compile_definitions(tests_sanitized PUBLIC SAVE_JITS_TO_FILE)
    endif(SAVE_JITS_TO_FILE)

    target_compile_options(tests_sanitized PRIVATE -g -fsanitize=float-divide-by-zero -fsanitize=bounds -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(tests_sanitized PRIVATE -g -fsanitize=address -fsanitize=undefined)
    target_link_libraries(tests_sanitized PRIVATE Catch2::Catch2WithMain)

    if(MLC_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(tests_sanitized PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # benchmarks
    add_executable(benchmarks "${SOURCE_FILEPATHS}" "${BENCH_FILEPATHS}")

    target_link_libraries(benchmarks PRIVATE benchmark::benchmark_main)
    if(MLC_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(benchmarks PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# ==============================================================
# Ctest setup
# ==============================================================
if(PROJECT_IS_TOP_LEVEL)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    catch_discover_tests(tests)
endif()

# ==============================================================
# Library Setup
# ==============================================================

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

add_library(${PROJECT_NAME})

# without it public headers won't get installed
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${public_headers})

target_sources(${PROJECT_NAME}
    PRIVATE
        "${SOURCE_FILEPATHS}"
        "${INTERFACE_FILEPATHS}"
        "${public_headers}"
)

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MLC_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

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

set(namespace mlc)

# generate and install export file
install(EXPORT "${PROJECT_NAME}Targets"
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${namespace}::
    DESTINATION cmake
)
add_library(mlc::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

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
