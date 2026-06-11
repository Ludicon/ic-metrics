// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// compare: cross-implementation correctness harness.
//
// For every PNG in <data_dir> (default: "data"), generates a set of in-memory
// JPEG distortions and runs each enabled ssimulacra2 implementation on each
// (orig, dist) pair. Emits CSV to stdout:  image,impl,distortion,score

#include "harness.h"
#include "ic_shared.h"
#include "ic_metrics.h"
#include "ic_vars.h"
#include "impls.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


////////////////////////////////
// Impl table

typedef double (*ScoreFn)(int w, int h, const u8* orig, const u8* dist);

static double ours_compute_score(int w, int h, const u8* orig, const u8* dist) {
    void* scratch = malloc(ic_ssimulacra2_score_scratch_size(w, h));
    defer { free(scratch); };
    return ic_ssimulacra2_score(w, h, orig, dist, scratch);
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
#if HAVE_FAST_SSIM2
    { "fast-ssim2", fast_ssim2_compute_score },
#endif
#if HAVE_VSZIP
    { "vszip", vszip_compute_score },
#endif
#if HAVE_VSHIP
    { "vship", vship_compute_score },
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

static void process_image(const char* path, void* /*ctx*/) {
    PngImage img;
    if (!harness_load_png(path, &img)) {
        fprintf(stderr, "compare: failed to load %s\n", path);
        return;
    }
    defer { harness_free_png(&img); };

    const char* name = harness_basename(path);

    // self vs self — sanity check, should print ~100
    emit_row(name, "self", img.rgba, img.rgba, img.w, img.h);

    static const int qualities[] = { 40, 70, 90 };
    for (int i = 0; i < (int)(sizeof(qualities) / sizeof(qualities[0])); i++) {
        int q = qualities[i];
        u8* dist = harness_jpeg_distort(img.rgba, img.w, img.h, q);
        if (!dist) {
            fprintf(stderr, "compare: jpeg_distort q%d failed on %s\n", q, name);
            continue;
        }
        defer { harness_free_distortion(dist); };

        char distortion[16];
        snprintf(distortion, sizeof(distortion), "jpeg-q%d", q);
        emit_row(name, distortion, img.rgba, dist, img.w, img.h);
    }
}


////////////////////////////////
// Entry

int main(int argc, char** argv) {
    const char* data_dir = "data";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            // Load var overrides (e.g. ssimu2_prune_threshold) from a file
            // without recompiling. Format: one "name value" per line.
            ic_vars_load_config(argv[++i], false);
        }
        else if (argv[i][0] == '-') {
            fprintf(stderr, "compare: unknown flag %s\n", argv[i]);
            return 1;
        }
        else {
            data_dir = argv[i];
        }
    }

    int n = harness_for_each_png(data_dir, nullptr, nullptr);
    if (n < 0) {
        fprintf(stderr, "compare: no PNGs found in %s\n", data_dir);
        return 1;
    }

    printf("image,impl,distortion,score\n");
    harness_for_each_png(data_dir, process_image, nullptr);
    return 0;
}
