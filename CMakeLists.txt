cmake_minimum_required(VERSION 3.20)

project(cuda_algorithms LANGUAGES CXX CUDA)

# C++/CUDA Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# If not specified, set default CUDA architecture (can be adjusted or overridden in command line)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Override common architectures: Pascal(60), Volta(70), Turing(75), Ampere(80/86), Ada(89), Hopper(90)
  set(CMAKE_CUDA_ARCHITECTURES 60;70;75;80;86;89;90)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(CUDA_TEST_TIME "Test time consumption of CUDA kernels" OFF)
if(CUDA_TEST_TIME)
    add_compile_definitions(CUDA_TEST_TIME)
    message(STATUS "CUDA_TEST_TIME enabled: CUDA_TEST_TIME macro defined")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)