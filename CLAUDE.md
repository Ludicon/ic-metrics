
# C/C++ Style Guide

This is a C-style C++ codebase. Write plain, direct code. Use C++ selectively where it earns its keep.

## Philosophy

- Prefer simple, readable code over clever abstractions.
- Write functions, not class hierarchies. Data is structs with public members.
- No speculative generality. Write what the code needs now.

## Naming

| Thing              | Convention                      | Example                                |
|--------------------|--------------------------------|----------------------------------------|
| Variables          | `snake_case`                   | `block_size`, `output_data`            |
| Functions          | `snake_case`                   | `spark_encode_image`, `read_block`     |
| Structs/Types      | `PascalCase`                   | `SparkCodecT`, `Vector3`, `Image`      |
| Enums              | `PascalCase`                   | `SparkFormat`, `SparkQuality`          |
| Enum values        | `EnumName_Value`               | `SparkFormat_BC1_RGB`, `SparkVendor_Intel` |
| Macros/constants   | `UPPER_SNAKE_CASE`             | `IC_FORCEINLINE`, `IC_OS_WINDOWS`      |
| Type aliases       | Lowercase short form           | `u8`, `u16`, `u32`, `u64`, `s32`, `f32` |
| File names         | `snake_case` with prefix       | `ic_math.h`, `spark_base.cpp`          |
| Globals            | `g_` prefix                    | `g_printf`, `g_malloc`                 |

- Use short variable names (`w`, `h`, `c`, `i`, `x`, `y`, `bx`, `by`) when scope is small and meaning is clear.

## Formatting

- **Indentation:** 4 spaces. No tabs.
- **Braces:** Same line (K&R / "Egyptian" style):
  ```cpp
  if (condition) {
      // ...
  }
  else {
      // ...
  }
  ```
- `else` on its own line after the closing brace.
- One-liners are fine when short: `float operator[](int idx) const { return c[idx]; }`
- Space after control keywords: `if (`, `for (`, `while (`.
- No space before function call parens: `func(arg)`.
- No strict line length limit, but keep lines reasonable (~100 chars typical).

## Sections

Organize code into sections using visual separators:
```cpp
////////////////////////////////
// Section Name
```

Typical section order in headers: types, declarations, inline implementations.
Typical section order in .cpp: includes, static helpers, public API.

## Headers

- Always use `#pragma once`.
- Copyright line first: `// Copyright 2026 Ludicon LLC. All Rights Reserved.`
- Include order: local headers, then system headers.
- Use forward declarations to avoid unnecessary includes.
- Platform-specific includes guarded with `#if IC_OS_WINDOWS`, `#if __ANDROID__`, etc.

## Types

- Use `int` for general integers (loop counters, sizes, counts, dimensions). Prefer `int` over `s32` unless the exact size matters.
- Use the sized integer types (`u8`, `u16`, `u32`, `u64`, `s8`, `s16`, `s32`, `s64`, `f32`) when the size is semantically important — binary file formats, bit-packed structures, hash values, wire protocols, SIMD lanes, GPU-side data, etc.
- Use `typedef` (not `using`) for type aliases and function pointer types.
- Enums: plain `enum` (not `enum class`). Prefix values with the enum name to avoid collisions.
- Structs: all public members, no inheritance. Use default member initializers when useful.
- Opaque handles: `typedef struct FooT* Foo;`

## C++ Features — Use

- **Templates:** for math utilities, containers, and `constexpr` functions.
- **Operator overloading:** for math types (`Vector2`, `Vector3`, `Vector4`).
- **Lambdas:** primarily with the `defer` macro for cleanup.
- **Namespaces:** for codec modules (`namespace ic::astc`, `namespace ic::eac`).
- **`constexpr`:** for compile-time constants and simple functions.
- **Unions:** for type punning in vector types (`struct { float x, y; }` / `float c[2]`).
- **Default constructors:** `= default` on math types.
- **Placement new:** when needed.

## C++ Features — Avoid

- **Classes with inheritance / virtual functions.** Use structs + function pointers.
- **Exceptions.** Never. Use return codes and asserts.
- **RTTI** (`dynamic_cast`, `typeid`). Never.
- **STL containers** (`std::vector`, `std::map`, `std::string`). Use C arrays, custom `DynamicArray<T>`, or `FixedArray<T, N>`.
- **Smart pointers** (`unique_ptr`, `shared_ptr`). Manual ownership.
- **`std::function`**, **`std::algorithm`**. Use function pointers and plain loops.
- **`auto`** for variable declarations (except in `defer` macro internals).
- **Range-based for loops** over STL containers (there are no STL containers).

## Memory Management

- Prefer fixed size allocations whenever possible.
- Use `malloc`/`free` (or the project's `g_malloc`/`g_free` callbacks) for allocations.
- Caller-allocates / caller-frees as the default ownership model.
- Use `calloc` for zero-initialized allocations.
- Use `defer { free(ptr); };` for scope-based cleanup instead of RAII wrappers.

## Error Handling

- Use `ic_assert(condition)` for internal invariants (compiled out in FINAL builds).
- Use `ic_check(statement)` when the statement must execute but the check is debug-only.
- No exceptions. No `std::optional`. Return `bool` or error codes.

## Comments

- Use `//` line comments. Avoid `/* */` block comments.
- Use `@@` for TODOs and open questions in documents or comments: `// @@ Should we use float or double?`
- No Doxygen or structured doc comments. Keep comments informal and explanatory.
- Don't state the obvious. Comment *why*, not *what*.

## Platform Abstraction

- Use `#if IC_OS_WINDOWS`, `#if IC_OS_APPLE`, `#if IC_OS_ANDROID`, `#if IC_OS_LINUX`.
- Use `#if IC_CC_MSVC`, `#if IC_CC_GCC`, `#if IC_CC_CLANG` for compiler-specific code.
- Use `#if IC_CPU_X86`, `#if IC_CPU_ARM`, `#if IC_CPU_ARM64` for architecture-specific code.
- Keep platform-specific code behind these guards, never raw `_WIN32` or `__APPLE__` in new code.

## Flags

When making changes to the core implementation of ic-metrics, protect the code changes using variable flags. You can use ic_vars for this:

```
IC_VAR_BOOL(new_implementation, true)

if (var::new_implementation) {
    // new implementation goes here
}
else {
    // old implementation goes here
}
```

This allows us testing both variants and ensure there are no regressions.

# Code Navigation

When tracing where a symbol is defined or finding all references to it, use LSP (goToDefinition, findReferences, hover) instead of Grep. LSP gives exact results; Grep gives text matches.

Use Grep/Glob for discovery (finding files, searching patterns). Use LSP for understanding (definitions, references, type info).

After locating a file with Grep/Glob, use LSP to navigate within it rather than reading the whole file.


# Building

## Tools and Libraries

Use cmake to build the project.

On macos use the build-release and build-debug directories. To build the resulting projects, use ninja.

On windows use the build-msvc directory.

For wasm, use the build-wasm directory.
