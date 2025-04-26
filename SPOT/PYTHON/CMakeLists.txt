cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(CSTSS_Python_wrapper VERSION 0.1.0 LANGUAGES C CXX)

# Force compilation for x86_64
# set(CMAKE_OSX_ARCHITECTURES "x86_64")

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

set(TARGET_NAME CSTSS_Python_pybind)
set(TOP_SOURCE_NAME CSTSS_Python_pybind.cpp)

file(
    GLOB_RECURSE PYBIND_SOURCES
    ${TOP_SOURCE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gen_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Polys.cpp
)

set(
    INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_DIRS})

add_library(${TARGET_NAME} SHARED ${PYBIND_SOURCES})

target_link_libraries(
    ${TARGET_NAME} PRIVATE 
    pybind11::module 
    Eigen3::Eigen
)

set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "" SUFFIX ".so")

