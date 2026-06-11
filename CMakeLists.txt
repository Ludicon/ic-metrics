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

# Enable AVX2 + FMA on x86_64 builds for the f32x8 SIMD path in ic_metrics.
# On arm64 the helpers use NEON (always available on aarch64).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2 -mfma)
    endif()
endif()


################################
# Metrics library

add_library(ic_metrics STATIC
    src/ic_metrics.cpp
    src/ic_metrics.h
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
#
# Toggleable so CI / minimal builds don't need to drag in libjxl + Highway
# (cloudinary), the rust toolchain (rust-av), or zig (fssimu2). When OFF,
# the `compare` executable is also skipped — without other implementations
# it has nothing useful to do.

option(IC_METRICS_BUILD_COMPARE "Build the cross-implementation compare tool and pull in third-party ssimulacra2 impls" ON)

set(IMPL_SOURCES "")
set(IMPL_LIBS "")
set(IMPL_DEFINES "")
set(IMPL_RPATH "")

if(IC_METRICS_BUILD_COMPARE)

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
# The bridge has a path dependency on extern/rust_av_ssimulacra2/ssimulacra2,
# so require that submodule too — otherwise the cargo build fails outright on
# a normal clone that didn't pull the nested submodule.
find_program(CARGO_EXECUTABLE cargo)
if(CARGO_EXECUTABLE
   AND EXISTS ${CMAKE_SOURCE_DIR}/extern/rust_av_bridge/Cargo.toml
   AND EXISTS ${CMAKE_SOURCE_DIR}/extern/rust_av_ssimulacra2/ssimulacra2/Cargo.toml)
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
    message(STATUS "rust-av: disabled (cargo not found, or bridge/ssimulacra2 submodule missing)")
endif()


# --- fast-ssim2: Rust crate (imazen), via a small companion C-ABI bridge ----
# Built as a cdylib (see the bridge's Cargo.toml) so its bundled Rust std stays
# private and doesn't clash with rust_av_bridge's static std. Linked like the
# fssimu2 dylib: SHARED IMPORTED + a build rpath. The bridge has a path
# dependency on the fast-ssim2 submodule's inner crate, so require it too.
if(CARGO_EXECUTABLE
   AND EXISTS ${CMAKE_SOURCE_DIR}/extern/fast_ssim2_bridge/Cargo.toml
   AND EXISTS ${CMAKE_SOURCE_DIR}/extern/fast-ssim2/fast-ssim2/Cargo.toml)
    set(FAST_SSIM2_DIR ${CMAKE_SOURCE_DIR}/extern/fast_ssim2_bridge)
    set(FAST_SSIM2_LIB ${FAST_SSIM2_DIR}/target/release/${CMAKE_SHARED_LIBRARY_PREFIX}fast_ssim2_bridge${CMAKE_SHARED_LIBRARY_SUFFIX})

    add_custom_command(
        OUTPUT ${FAST_SSIM2_LIB}
        COMMAND ${CARGO_EXECUTABLE} build --release --manifest-path ${FAST_SSIM2_DIR}/Cargo.toml
        DEPENDS ${FAST_SSIM2_DIR}/Cargo.toml ${FAST_SSIM2_DIR}/src/lib.rs
        COMMENT "Building fast_ssim2_bridge (cargo)"
        VERBATIM
    )
    add_custom_target(fast_ssim2_build DEPENDS ${FAST_SSIM2_LIB})

    add_library(fast_ssim2 SHARED IMPORTED)
    set_target_properties(fast_ssim2 PROPERTIES IMPORTED_LOCATION ${FAST_SSIM2_LIB})
    add_dependencies(fast_ssim2 fast_ssim2_build)

    list(APPEND IMPL_SOURCES src/impl_fast_ssim2.cpp)
    list(APPEND IMPL_LIBS fast_ssim2)
    list(APPEND IMPL_DEFINES HAVE_FAST_SSIM2=1)
    list(APPEND IMPL_RPATH ${FAST_SSIM2_DIR}/target/release)

    message(STATUS "fast-ssim2: enabled (${FAST_SSIM2_LIB})")
else()
    message(STATUS "fast-ssim2: disabled (cargo not found, or bridge/submodule missing)")
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


# --- vszip: SSIMULACRA2 filter from dnjulek/vapoursynth-zip ----------------
# That repo is a VapourSynth plugin, but its src/filters/ssimulacra2.zig is a
# standalone (std-only) algorithm. Our companion bridge (extern/vszip_bridge)
# pulls in just that file and wraps it in a small C ABI dylib.
if(ZIG_EXECUTABLE
   AND EXISTS ${CMAKE_SOURCE_DIR}/extern/vszip_bridge/build.zig
   AND EXISTS ${CMAKE_SOURCE_DIR}/extern/vapoursynth-zip/src/filters/ssimulacra2.zig)
    set(VSZIP_DIR ${CMAKE_SOURCE_DIR}/extern/vszip_bridge)
    set(VSZIP_LIB ${VSZIP_DIR}/zig-out/lib/${CMAKE_SHARED_LIBRARY_PREFIX}vszip_ssimu2${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(VSZIP_INCLUDE ${VSZIP_DIR}/zig-out/include)

    add_custom_command(
        OUTPUT ${VSZIP_LIB} ${VSZIP_INCLUDE}/vszip_ssimu2.h
        COMMAND ${ZIG_EXECUTABLE} build --release=fast
        WORKING_DIRECTORY ${VSZIP_DIR}
        DEPENDS ${VSZIP_DIR}/src/wrapper.zig ${VSZIP_DIR}/build.zig
                ${CMAKE_SOURCE_DIR}/extern/vapoursynth-zip/src/filters/ssimulacra2.zig
        COMMENT "Building vszip_bridge (zig)"
        VERBATIM
    )
    add_custom_target(vszip_build DEPENDS ${VSZIP_LIB})

    add_library(vszip SHARED IMPORTED)
    set_target_properties(vszip PROPERTIES
        IMPORTED_LOCATION ${VSZIP_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${VSZIP_INCLUDE}
    )
    add_dependencies(vszip vszip_build)

    list(APPEND IMPL_SOURCES src/impl_vszip.cpp)
    list(APPEND IMPL_LIBS vszip)
    list(APPEND IMPL_DEFINES HAVE_VSZIP=1)
    list(APPEND IMPL_RPATH ${VSZIP_DIR}/zig-out/lib)

    message(STATUS "vszip: enabled (${VSZIP_LIB})")
else()
    message(STATUS "vszip: disabled (zig not found or bridge/submodule missing)")
endif()


# --- vship: GPU compute SSIMULACRA2 (Line-fr/Vship) ------------------------
# Vship runs SSIMULACRA2 on the GPU via HIP/CUDA or Vulkan. We build its Vulkan
# backend (host C++ + precompiled SPIR-V, compiled with the C++ compiler), which
# on macOS runs through MoltenVK. Gated on a Vulkan loader being present, so
# machines without one (incl. typical CI) just skip it. Note: Vship's Vulkan
# path is upstream-flagged experimental, and via MoltenVK scores differ slightly
# from a native run (e.g. identical images land near 99.8, not 100).
find_program(MAKE_EXECUTABLE NAMES make gmake)
find_library(VULKAN_LIBRARY NAMES vulkan vulkan-1 PATHS /usr/local/lib)
if(MAKE_EXECUTABLE AND VULKAN_LIBRARY AND EXISTS ${CMAKE_SOURCE_DIR}/extern/Vship/Makefile)
    set(VSHIP_DIR ${CMAKE_SOURCE_DIR}/extern/Vship)
    set(VSHIP_LIB ${VSHIP_DIR}/libvship.so)  # Vship's Makefile names it .so even on macOS

    add_custom_command(
        OUTPUT ${VSHIP_LIB}
        COMMAND ${MAKE_EXECUTABLE} build BACKEND=Vulkan
        WORKING_DIRECTORY ${VSHIP_DIR}
        DEPENDS ${VSHIP_DIR}/src/VshipLib.cpp ${VSHIP_DIR}/src/VshipAPI.h
        COMMENT "Building libvship (make BACKEND=Vulkan)"
        VERBATIM
    )
    add_custom_target(vship_build DEPENDS ${VSHIP_LIB})

    add_library(vship SHARED IMPORTED)
    set_target_properties(vship PROPERTIES
        IMPORTED_LOCATION ${VSHIP_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${VSHIP_DIR}/src
    )
    add_dependencies(vship vship_build)

    list(APPEND IMPL_SOURCES src/impl_vship.cpp)
    list(APPEND IMPL_LIBS vship)
    list(APPEND IMPL_DEFINES HAVE_VSHIP=1)
    # libvship.so resolves by its absolute install name; libvulkan via the
    # SDK lib dir on the executable's rpath (libvship references @rpath/libvulkan).
    list(APPEND IMPL_RPATH ${VSHIP_DIR})
    get_filename_component(VULKAN_LIBDIR ${VULKAN_LIBRARY} DIRECTORY)
    list(APPEND IMPL_RPATH ${VULKAN_LIBDIR})

    message(STATUS "vship: enabled (${VSHIP_LIB}, Vulkan/MoltenVK)")
else()
    message(STATUS "vship: disabled (make/Vulkan loader/submodule missing)")
endif()

endif() # IC_METRICS_BUILD_COMPARE


################################
# Executables

add_executable(ssimudiff src/ssimudiff.cpp)
target_link_libraries(ssimudiff PRIVATE ic_metrics stb)

if(IC_METRICS_BUILD_COMPARE)
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
