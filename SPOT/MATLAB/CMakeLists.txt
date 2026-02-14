cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(CSTSS_MATLAB_wrapper VERSION 0.1.0 LANGUAGES C CXX)

# Build with O2 optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# Force compilation for x86_64
set(CMAKE_OSX_ARCHITECTURES "x86_64")

find_package(Matlab REQUIRED)
find_package(Eigen3 REQUIRED)

set(TARGET_NAME CSTSS_MATLAB_mex)
set(TOP_SOURCE_NAME CSTSS_MATLAB_mex.cpp)

file(
    GLOB_RECURSE MEX_SOURCES
    ${TOP_SOURCE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gen_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Polys.cpp
)

set(
    INCLUDE_DIRS
    ${Matlab_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_DIRS})

matlab_add_mex(
    NAME ${TARGET_NAME}
    SRC ${MEX_SOURCES}
    OUTPUT_NAME ${TARGET_NAME}
    LINK_TO 
    ${Matlab_LIBRARIES} 
    Eigen3::Eigen
)

