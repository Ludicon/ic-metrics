cmake_minimum_required(VERSION 3.20)
project(ic_metrics CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()


################################
# Metrics library

add_library(ic_metrics STATIC
    src/ic_ssimulacra2.cc
    src/ic_ssimulacra2.h
    src/ic_vars.cpp
    src/ic_vars.h
    src/ic_shared.h
)
target_include_directories(ic_metrics PUBLIC src)


################################
# stb (header-only, shared include path)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE extern/stb)


################################
# Executables

add_executable(ssimudiff src/ssimudiff.cpp)
target_link_libraries(ssimudiff PRIVATE ic_metrics stb)

add_executable(compare src/compare.cpp)
target_link_libraries(compare PRIVATE ic_metrics stb)

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE ic_metrics stb)
