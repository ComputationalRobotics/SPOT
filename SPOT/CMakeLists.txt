cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 11)  # Or 14, 17, or 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cstss VERSION 0.1.0 LANGUAGES C CXX)

# Set include directories
include_directories(include)

# Specify source files
set(SOURCES
    src/main.cpp
    src/gen_test.cpp
    src/Polys.cpp
)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Create executable and link source files
add_executable(cstss ${SOURCES})
target_link_libraries(cstss Eigen3::Eigen)