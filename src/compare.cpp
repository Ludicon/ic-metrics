// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// compare: cross-implementation correctness harness.
//
// For every PNG in <data_dir> (default: "data"), generates a set of in-memory
// JPEG distortions and runs ComputeSSIMULACRA2Score on each (orig, dist) pair.
// Emits CSV to stdout:  image,impl,distortion,score
//
// Phase 1: ours only. Phase 2 will add the other implementations as
// additional `impl` rows.

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

#include "ic_ssimulacra2.h"
#include "ic_shared.h"
#include "impls.h"

#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


////////////////////////////////
// Path helpers

static const char* basename_ptr(const char* path) {
    const char* slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}


////////////////////////////////
// Growable byte buffer (for in-memory JPEG encode)

struct ByteBuffer {
    u8* data;
    size_t size;
    size_t capacity;
};

static void bb_init(ByteBuffer* b) {
    b->data = nullptr;
    b->size = 0;
    b->capacity = 0;
}

static void bb_free(ByteBuffer* b) {
    free(b->data);
    b->data = nullptr;
    b->size = 0;
    b->capacity = 0;
}

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


////////////////////////////////
// Distortion: RGBA -> JPEG@q -> RGBA  (in memory)
//
// Returns a freshly malloc'd RGBA buffer of size 4 * w * h.
// Caller frees with free().
// On failure returns nullptr.

static u8* jpeg_distort(const u8* rgba, int w, int h, int quality) {
    ByteBuffer jpg;
    bb_init(&jpg);
    defer { bb_free(&jpg); };

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


////////////////////////////////
// Impl table

typedef double (*ScoreFn)(int w, int h, const u8* orig, const u8* dist);

static double ours_compute_score(int w, int h, const u8* orig, const u8* dist) {
    return ComputeSSIMULACRA2Score(w, h, orig, dist);
}

struct Impl {
    const char* name;
    ScoreFn fn;
};

static const Impl IMPLS[] = {
    { "ours", ours_compute_score },
#if HAVE_FSSIMU2
    { "fssimu2", fssimu2_compute_score },
#endif
#if HAVE_RUST_AV
    { "rust-av", rust_av_compute_score },
#endif
#if HAVE_CLOUDINARY
    { "cloudinary", cloudinary_compute_score },
#endif
};
static const int NUM_IMPLS = (int)(sizeof(IMPLS) / sizeof(IMPLS[0]));


////////////////////////////////
// Per-image processing

static void emit_row(const char* image, const char* distortion, const u8* orig, const u8* dist, int w, int h) {
    for (int i = 0; i < NUM_IMPLS; i++) {
        double s = IMPLS[i].fn(w, h, orig, dist);
        printf("%s,%s,%s,%.6f\n", image, IMPLS[i].name, distortion, s);
        fflush(stdout);
    }
}

static void process_image(const char* path) {
    int w, h, c;
    u8* orig = stbi_load(path, &w, &h, &c, 4);
    if (!orig) {
        fprintf(stderr, "compare: failed to load %s\n", path);
        return;
    }
    defer { stbi_image_free(orig); };

    const char* name = basename_ptr(path);

    // self vs self — sanity check, should print ~100
    emit_row(name, "self", orig, orig, w, h);

    static const int qualities[] = { 40, 70, 90 };
    for (int i = 0; i < (int)(sizeof(qualities) / sizeof(qualities[0])); i++) {
        int q = qualities[i];
        u8* dist = jpeg_distort(orig, w, h, q);
        if (!dist) {
            fprintf(stderr, "compare: jpeg_distort q%d failed on %s\n", q, name);
            continue;
        }
        defer { stbi_image_free(dist); };

        char distortion[16];
        snprintf(distortion, sizeof(distortion), "jpeg-q%d", q);
        emit_row(name, distortion, orig, dist, w, h);
    }
}


////////////////////////////////
// Entry

int main(int argc, char** argv) {
    const char* data_dir = (argc > 1) ? argv[1] : "data";

    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s/*.png", data_dir);

    glob_t g;
    int rc = glob(pattern, 0, nullptr, &g);
    if (rc != 0) {
        fprintf(stderr, "compare: no PNGs found at %s\n", pattern);
        return 1;
    }
    defer { globfree(&g); };

    printf("image,impl,distortion,score\n");
    for (size_t i = 0; i < g.gl_pathc; i++) {
        process_image(g.gl_pathv[i]);
    }
    return 0;
}
