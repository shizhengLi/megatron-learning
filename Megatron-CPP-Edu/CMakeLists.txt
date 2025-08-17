cmake_minimum_required(VERSION 3.16)
project(MegatronCPP VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, using custom implementation")
    add_definitions(-DUSE_CUSTOM_EIGEN)
endif()

find_package(MPI QUIET)
if(NOT MPI_FOUND)
    message(STATUS "MPI not found, using custom implementation")
    add_definitions(-DUSE_CUSTOM_MPI)
endif()

# Compile options
option(USE_CUDA "Use CUDA support" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Add subdirectories
add_subdirectory(core)
add_subdirectory(models)
add_subdirectory(distributed)
# add_subdirectory(training)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation configuration
# install(DIRECTORY include/ DESTINATION include)
# install(DIRECTORY models/ DESTINATION include/models)
# install(DIRECTORY training/ DESTINATION include/training)
# install(DIRECTORY distributed/ DESTINATION include/distributed)