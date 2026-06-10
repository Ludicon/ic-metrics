// Copyright 2026 Ludicon LLC. All Rights Reserved.
#pragma once

////////////////////////////////
// Basic types

#include <stdint.h>

typedef char s8;
typedef unsigned char u8;
typedef short s16;
typedef unsigned short u16;
typedef int s32;
typedef unsigned int u32;
typedef int64_t s64;
typedef uint64_t u64;
typedef float f32;


////////////////////////////////
// OS

#if ((defined(_WIN32) || defined WIN32 || defined __NT__ || defined __WIN32__) && !defined __CYGWIN__)
#define IC_OS_WINDOWS 1
#endif

#if defined(__APPLE__) || defined (__MACH__)
#define IC_OS_APPLE 1
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define IC_OS_IOS 1
#else
#define IC_OS_MACOS 1
#endif
#endif

#if defined __ANDROID__
#define IC_OS_ANDROID 1
#endif

#if (defined linux || defined __linux__)
#define IC_OS_LINUX 1
#endif


////////////////////////////////
// CPU

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
    #define IC_CPU_X86 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define IC_CPU_X64 1
#endif

#if (defined(__arm__) || defined(__aarch64__) || defined(_M_ARM))
    #define IC_CPU_ARM 1
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
    #define IC_CPU_ARM64 1
#endif


////////////////////////////////
// Compiler

#if defined _MSC_VER
#define IC_CC_MSVC 1
#endif

#if defined __GNUC__
#define IC_CC_GCC 1
#endif

#if defined __clang__
#define IC_CC_CLANG 1
#endif

#if IC_CC_MSVC
#define __attribute__(...)
#endif

#if IC_CC_GCC
#define IC_FORCEINLINE inline __attribute__((always_inline))
#else
#define IC_FORCEINLINE __forceinline
#endif

#if IC_CC_MSVC
#define IC_PRINTF_FORMAT_STRING _Printf_format_string_
#else
#define IC_PRINTF_FORMAT_STRING
#endif


////////////////////////////////
// defer

#define IC_CONCAT_INTERNAL(x,y) x##y
#define IC_CONCAT(x,y) IC_CONCAT_INTERNAL(x,y)

template<typename T>
struct ExitScope {
    T lambda;
    ExitScope(T lambda) : lambda(lambda) {}
    ~ExitScope() { lambda(); }
private:
    ExitScope& operator=(const ExitScope&);
};

struct ExitScopeHelp {
    template<typename T>
    ExitScope<T> operator+(T t) { return t; }
};

#if IC_CC_MSVC
#define defer [[maybe_unused]] const auto& IC_CONCAT(defer__, __LINE__) = ExitScopeHelp() + [&]()
#else
#define defer const auto& __attribute__((unused)) IC_CONCAT(defer__, __LINE__) = ExitScopeHelp() + [&]()
#endif


////////////////////////////////
// countof

template <typename T, unsigned int N>
char(*__countof_helper(T(&_Array)[N]))[N];
#define countof(array) (sizeof(*__countof_helper(array)) + 0)


////////////////////////////////
// Assert

#if IC_CC_MSVC
    #define ic_debug_break() __debugbreak()
    #if __cplusplus >= 202002L
        #define ic_unlikely(x)   (x) [[unlikely]]
    #else
        #define ic_unlikely(x)   (x)
    #endif
#else
    // __builtin_debugtrap exists on clang/apple-clang but not on gcc.
    // gcc has __builtin_trap (SIGILL) — less debugger-friendly but works
    // everywhere we care about.
    #if defined(__has_builtin) && __has_builtin(__builtin_debugtrap)
        #define ic_debug_break() __builtin_debugtrap()
    #else
        #define ic_debug_break() __builtin_trap()
    #endif
    #define ic_unlikely(x)   (__builtin_expect((x), 0))
#endif

#ifndef ic_assert
    #if !FINAL
        #define ic_assert(x) do { if ic_unlikely(!(x)) ic_debug_break(); } while(false)
        #define ic_check(st) ic_assert(st)
    #else
        #define ic_assert(x) (void)sizeof(x)
        #define ic_check(st) (void)(st)
    #endif
#endif


////////////////////////////////
// Unused

#define IC_UNUSED(x) (void)x


////////////////////////////////
// Basic templates

template <typename T> inline T max(const T& a, const T& b) {
    return (b < a) ? a : b;
}

template <typename T> inline T min(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template <typename T> inline T clamp(const T& x, const T& a, const T& b) {
    return min(max(x, a), b);
}

IC_FORCEINLINE float saturate(float x) {
    return min(max(x, 0.0f), 1.0f);
}

template <typename T> IC_FORCEINLINE void swap(T& a, T& b) {
    T tmp(a);
    a = b;
    b = tmp;
}
