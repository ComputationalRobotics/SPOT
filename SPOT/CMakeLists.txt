cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 11)  # Or 14, 17, or 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cstss VERSION 0.1.0 LANGUAGES C CXX)

# 设置头文件目录
include_directories(include)

# 指定源文件
set(SOURCES
    src/main.cpp
    src/gen_test.cpp
    src/Polys.cpp
)

# 查找 Eigen
find_package(Eigen3 REQUIRED)

# 创建可执行文件并链接源文件
add_executable(cstss ${SOURCES})
target_link_libraries(cstss Eigen3::Eigen)