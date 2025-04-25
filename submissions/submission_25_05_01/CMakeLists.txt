cmake_minimum_required(VERSION 3.10.0)
project(TsunamiSimulation VERSION 0.1.0 LANGUAGES C CXX)

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
if(MSVC)
    message(FATAL_ERROR "This project cannot be compiled with MSVC due to https://developercommunity.visualstudio.com/t/using-c17-new-stdalign-val-tn-syntax-results-in-er/528320")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_compile_options(-Wall -Wextra -pthread -qopenmp)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "" FORCE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -pthread -fopenmp)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_VERBOSE_MAKEFILE on)

# Fetch Catch2
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)

FetchContent_MakeAvailable(Catch2)

# Add source files
set(SRC_FILES
    neon_throughput_latency_driver.cpp
    neon_throughput_latency.s
)

set(TEST_FILES

)

add_executable(benchmark "${SRC_FILES}")
target_sources(benchmark PRIVATE neon_throughput_latency_driver.cpp)

# add_executable(mircokernel "${SRC_FILES}")
# target_sources(mircokernel mircokernel_main.cpp)

# add_executable(loops "${SRC_FILES}")
# target_sources(loops loops_main.cpp)

add_executable(tests "${SRC_FILES}" "${TEST_FILES}")
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)