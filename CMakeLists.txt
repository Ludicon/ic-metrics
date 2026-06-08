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

# Enable AVX2 + FMA on x86_64 builds for the f32x8 SIMD path in ic_ssimulacra2.
# On arm64 the helpers use NEON (always available on aarch64).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_compile_options(-mavx2 -mfma)
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

# --- cloudinary: reference C++ impl; bundled libjxl pieces -----------------
if(EXISTS ${CMAKE_SOURCE_DIR}/extern/cloudinary_ssimulacra2/src/CMakeLists.txt)
    # Their CMakeLists declares project(LIBJXL) and builds the bundled libjxl
    # bits plus an ssimulacra2 CLI. We bring in the lib targets (jxl,
    # jxl_extras-static, hwy) but don't build their executable.
    add_subdirectory(extern/cloudinary_ssimulacra2/src EXCLUDE_FROM_ALL)

    add_library(cloudinary_impl STATIC
        src/impl_cloudinary.cpp
        extern/cloudinary_ssimulacra2/src/ssimulacra2.cc
    )
    target_include_directories(cloudinary_impl PRIVATE
        extern/cloudinary_ssimulacra2/src
        extern/stb)
    target_link_libraries(cloudinary_impl PUBLIC jxl jxl_extras-static hwy)

    list(APPEND IMPL_SOURCES "")  # impl_cloudinary.cpp is inside cloudinary_impl
    list(APPEND IMPL_LIBS cloudinary_impl)
    list(APPEND IMPL_DEFINES HAVE_CLOUDINARY=1)

    message(STATUS "cloudinary: enabled")
else()
    message(STATUS "cloudinary: disabled (submodule missing)")
endif()


# --- rust-av: Rust crate, via a small companion C-ABI bridge crate ---------
find_program(CARGO_EXECUTABLE cargo)
if(CARGO_EXECUTABLE AND EXISTS ${CMAKE_SOURCE_DIR}/extern/rust_av_bridge/Cargo.toml)
    set(RUST_AV_DIR ${CMAKE_SOURCE_DIR}/extern/rust_av_bridge)
    set(RUST_AV_LIB ${RUST_AV_DIR}/target/release/librust_av_bridge.a)

    add_custom_command(
        OUTPUT ${RUST_AV_LIB}
        COMMAND ${CARGO_EXECUTABLE} build --release --manifest-path ${RUST_AV_DIR}/Cargo.toml
        DEPENDS ${RUST_AV_DIR}/Cargo.toml ${RUST_AV_DIR}/src/lib.rs
        COMMENT "Building rust_av_bridge (cargo)"
        VERBATIM
    )
    add_custom_target(rust_av_build DEPENDS ${RUST_AV_LIB})

    add_library(rust_av STATIC IMPORTED)
    set_target_properties(rust_av PROPERTIES IMPORTED_LOCATION ${RUST_AV_LIB})
    add_dependencies(rust_av rust_av_build)
    # macOS system frameworks required by Rust std for things like rayon's thread pool.
    if(APPLE)
        set_property(TARGET rust_av APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES "-framework CoreFoundation" "-framework Security" "-liconv")
    endif()

    list(APPEND IMPL_SOURCES src/impl_rust_av.cpp)
    list(APPEND IMPL_LIBS rust_av)
    list(APPEND IMPL_DEFINES HAVE_RUST_AV=1)

    message(STATUS "rust-av: enabled (${RUST_AV_LIB})")
else()
    message(STATUS "rust-av: disabled (cargo not found or bridge missing)")
endif()


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

add_executable(compare src/compare.cpp src/harness.cpp src/harness.h ${IMPL_SOURCES})
target_link_libraries(compare PRIVATE ic_metrics stb ${IMPL_LIBS})
target_compile_definitions(compare PRIVATE ${IMPL_DEFINES})
if(IMPL_RPATH)
    set_target_properties(compare PROPERTIES BUILD_RPATH "${IMPL_RPATH}")
endif()
if(APPLE)
    # Works around an ld64 fixup-chains crash when linking libjxl.a's __dso_handle
    # references on arm64 (https://github.com/llvm/llvm-project/issues/86599).
    target_link_options(compare PRIVATE -Wl,-ld_classic)
endif()

add_executable(bench src/bench.cpp src/harness.cpp src/harness.h ${IMPL_SOURCES})
target_link_libraries(bench PRIVATE ic_metrics stb ${IMPL_LIBS})
target_compile_definitions(bench PRIVATE ${IMPL_DEFINES})
if(IMPL_RPATH)
    set_target_properties(bench PROPERTIES BUILD_RPATH "${IMPL_RPATH}")
endif()
if(APPLE)
    target_link_options(bench PRIVATE -Wl,-ld_classic)
endif()
