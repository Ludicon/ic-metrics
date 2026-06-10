// Copyright 2026 Ludicon LLC. All Rights Reserved.

#if __clang__
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    #pragma clang diagnostic ignored "-Wsign-conversion"
    #pragma clang diagnostic ignored "-Wimplicit-int-conversion"
    #pragma clang diagnostic ignored "-Wundef"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "harness.h"
#include "ic_shared.h"  // IC_OS_WINDOWS

#if IC_OS_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <glob.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


////////////////////////////////
// PNG load

bool harness_load_png(const char* path, PngImage* out) {
    int c;
    out->rgba = stbi_load(path, &out->w, &out->h, &c, 4);
    return out->rgba != nullptr;
}

void harness_free_png(PngImage* img) {
    if (img->rgba) {
        stbi_image_free(img->rgba);
        img->rgba = nullptr;
    }
}


////////////////////////////////
// In-memory JPEG distortion

struct ByteBuffer {
    u8* data;
    size_t size;
    size_t capacity;
};

static void bb_write(void* ctx, void* data, int size) {
    ByteBuffer* b = (ByteBuffer*)ctx;
    size_t needed = b->size + (size_t)size;
    if (needed > b->capacity) {
        size_t new_cap = b->capacity ? b->capacity * 2 : 64 * 1024;
        while (new_cap < needed) new_cap *= 2;
        b->data = (u8*)realloc(b->data, new_cap);
        b->capacity = new_cap;
    }
    memcpy(b->data + b->size, data, (size_t)size);
    b->size += (size_t)size;
}

u8* harness_jpeg_distort(const u8* rgba, int w, int h, int quality) {
    ByteBuffer jpg = {};
    defer { free(jpg.data); };

    if (!stbi_write_jpg_to_func(bb_write, &jpg, w, h, 4, rgba, quality)) {
        return nullptr;
    }

    int dw, dh, dc;
    u8* decoded = stbi_load_from_memory(jpg.data, (int)jpg.size, &dw, &dh, &dc, 4);
    if (!decoded || dw != w || dh != h) {
        if (decoded) stbi_image_free(decoded);
        return nullptr;
    }
    return decoded;
}

void harness_free_distortion(u8* p) {
    if (p) stbi_image_free(p);
}


////////////////////////////////
// Glob *.png

int harness_for_each_png(const char* data_dir, HarnessPathCallback cb, void* ctx) {
#if IC_OS_WINDOWS
    // Windows has no glob(3); use FindFirstFile/FindNextFile. Same shape:
    // build a "<dir>\*.png" wildcard, enumerate, build absolute paths on
    // the fly. Excludes "." and ".." (FindFirstFile doesn't return them
    // for a *.png wildcard).
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\*.png", data_dir);

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return -1;

    int count = 0;
    do {
        count++;
        if (cb) {
            char path[1024];
            snprintf(path, sizeof(path), "%s\\%s", data_dir, fd.cFileName);
            cb(path, ctx);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return count;
#else
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s/*.png", data_dir);

    glob_t g;
    if (glob(pattern, 0, nullptr, &g) != 0) {
        return -1;
    }
    defer { globfree(&g); };

    if (cb) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            cb(g.gl_pathv[i], ctx);
        }
    }
    return (int)g.gl_pathc;
#endif
}


////////////////////////////////
// Misc

const char* harness_basename(const char* path) {
    const char* slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}
