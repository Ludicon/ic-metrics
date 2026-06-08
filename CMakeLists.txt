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
# Third-party impls (compared against ours)

set(IMPL_SOURCES "")
set(IMPL_LIBS "")
set(IMPL_DEFINES "")
set(IMPL_RPATH "")

# --- fssimu2: Zig project, already exposes a C ABI -------------------------
find_program(ZIG_EXECUTABLE zig)
if(ZIG_EXECUTABLE AND EXISTS ${CMAKE_SOURCE_DIR}/extern/fssimu2/build.zig)
    set(FSSIMU2_DIR ${CMAKE_SOURCE_DIR}/extern/fssimu2)
    set(FSSIMU2_LIB ${FSSIMU2_DIR}/zig-out/lib/libssimu2.dylib)
    set(FSSIMU2_INCLUDE ${FSSIMU2_DIR}/zig-out/include)

    add_custom_command(
        OUTPUT ${FSSIMU2_LIB} ${FSSIMU2_INCLUDE}/ssimu2.h
        COMMAND ${ZIG_EXECUTABLE} build --release=fast
        WORKING_DIRECTORY ${FSSIMU2_DIR}
        COMMENT "Building fssimu2 (zig)"
        VERBATIM
    )
    add_custom_target(fssimu2_build DEPENDS ${FSSIMU2_LIB})

    add_library(fssimu2 SHARED IMPORTED)
    set_target_properties(fssimu2 PROPERTIES
        IMPORTED_LOCATION ${FSSIMU2_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${FSSIMU2_INCLUDE}
    )
    add_dependencies(fssimu2 fssimu2_build)

    list(APPEND IMPL_SOURCES src/impl_fssimu2.cpp)
    list(APPEND IMPL_LIBS fssimu2)
    list(APPEND IMPL_DEFINES HAVE_FSSIMU2=1)
    list(APPEND IMPL_RPATH ${FSSIMU2_DIR}/zig-out/lib)

    message(STATUS "fssimu2: enabled (${FSSIMU2_LIB})")
else()
    message(STATUS "fssimu2: disabled (zig not found or submodule missing)")
endif()


################################
# Executables

add_executable(ssimudiff src/ssimudiff.cpp)
target_link_libraries(ssimudiff PRIVATE ic_metrics stb)

add_executable(compare src/compare.cpp ${IMPL_SOURCES})
target_link_libraries(compare PRIVATE ic_metrics stb ${IMPL_LIBS})
target_compile_definitions(compare PRIVATE ${IMPL_DEFINES})
if(IMPL_RPATH)
    set_target_properties(compare PROPERTIES BUILD_RPATH "${IMPL_RPATH}")
endif()

add_executable(bench src/bench.cpp ${IMPL_SOURCES})
target_link_libraries(bench PRIVATE ic_metrics stb ${IMPL_LIBS})
target_compile_definitions(bench PRIVATE ${IMPL_DEFINES})
if(IMPL_RPATH)
    set_target_properties(bench PROPERTIES BUILD_RPATH "${IMPL_RPATH}")
endif()
