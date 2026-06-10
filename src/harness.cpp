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

#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <string>
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
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(data_dir, ec)) return -1;

    int count = 0;
    for (const auto& entry : fs::directory_iterator(data_dir, ec)) {
        if (ec) return -1;
        if (entry.path().extension() == ".png") {
            count++;
            if (cb) {
                std::string s = entry.path().string();
                cb(s.c_str(), ctx);
            }
        }
    }
    return count;
}


////////////////////////////////
// Misc

const char* harness_basename(const char* path) {
    const char* slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}
