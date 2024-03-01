cmake_minimum_required(VERSION 3.10)

project(PathTracing-PBR LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Add src
include_directories(src)
add_subdirectory(src)