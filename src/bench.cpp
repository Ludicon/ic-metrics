// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// bench: cross-implementation performance harness.
//
// For each (image, distortion, impl): one warm-up iteration, then N timed
// iterations of the score function. Reports CSV with median / p10 / p90 ms
// plus the final score (sanity check).
//
// CLI:  bench [--iters N] [--threads K] [--impl X] [data_dir]
//   --iters N      iterations per cell, default 5
//   --threads K    request K threads per impl. Sets RAYON_NUM_THREADS for
//                  rust-av. ours/fssimu2/cloudinary are single-threaded today
//                  (Phase 4 will add equivalent controls).
//   --impl X       run only impl X (ours|fssimu2|rust-av|cloudinary)
//   data_dir       defaults to "data"

#include "harness.h"
#include "ic_shared.h"
#include "ic_ssimulacra2.h"
#include "ic_vars.h"
#include "impls.h"

IC_VAR_DECL(bool, ssimu2_blur_clamp_to_border);

#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


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
// CLI state (file-scope so process_image can read it)

static int g_iters = 5;
static const char* g_impl_filter = nullptr;


////////////////////////////////
// Timing helpers

static double now_ms() {
    using namespace std::chrono;
    auto t = steady_clock::now().time_since_epoch();
    return duration<double, std::milli>(t).count();
}

static double percentile_sorted(const std::vector<double>& v, double p) {
    // v must be sorted ascending. p in [0,1].
    if (v.empty()) return 0.0;
    if (v.size() == 1) return v[0];
    double idx = p * (double)(v.size() - 1);
    int lo = (int)idx;
    int hi = lo + 1 < (int)v.size() ? lo + 1 : lo;
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}


////////////////////////////////
// Per-cell bench

static void bench_cell(const char* image, const Impl& impl, const char* distortion,
                       const u8* orig, const u8* dist, int w, int h) {
    // Warm-up (untimed) — populates caches and lazily-init'd pools (e.g. rayon).
    double last_score = impl.fn(w, h, orig, dist);

    std::vector<double> samples;
    samples.reserve((size_t)g_iters);
    for (int i = 0; i < g_iters; i++) {
        double t0 = now_ms();
        last_score = impl.fn(w, h, orig, dist);
        double t1 = now_ms();
        samples.push_back(t1 - t0);
    }
    std::sort(samples.begin(), samples.end());

    double median = percentile_sorted(samples, 0.5);
    double p10 = percentile_sorted(samples, 0.1);
    double p90 = percentile_sorted(samples, 0.9);

    printf("%s,%s,%s,%d,%.3f,%.3f,%.3f,%.6f\n",
           image, impl.name, distortion, g_iters, median, p10, p90, last_score);
    fflush(stdout);
}


////////////////////////////////
// Per-image driver

static void process_image(const char* path, void* /*ctx*/) {
    PngImage img;
    if (!harness_load_png(path, &img)) {
        fprintf(stderr, "bench: failed to load %s\n", path);
        return;
    }
    defer { harness_free_png(&img); };

    const char* name = harness_basename(path);

    // Pre-generate distortions once per image — outside the timing loop.
    struct Pair { const char* name; u8* buf; };
    static const int qualities[] = { 40, 70, 90 };
    Pair pairs[4];
    int n_pairs = 0;
    pairs[n_pairs++] = { "self", img.rgba };
    for (int i = 0; i < (int)(sizeof(qualities) / sizeof(qualities[0])); i++) {
        int q = qualities[i];
        u8* dist = harness_jpeg_distort(img.rgba, img.w, img.h, q);
        if (!dist) {
            fprintf(stderr, "bench: jpeg_distort q%d failed on %s\n", q, name);
            continue;
        }
        // Leak distortion buffers on early return; main exits soon enough.
        static char names[3][16];
        snprintf(names[i], sizeof(names[i]), "jpeg-q%d", q);
        pairs[n_pairs++] = { names[i], dist };
    }
    defer {
        for (int i = 1; i < n_pairs; i++) harness_free_distortion(pairs[i].buf);
    };

    for (int p = 0; p < n_pairs; p++) {
        for (int i = 0; i < NUM_IMPLS; i++) {
            if (g_impl_filter && strcmp(g_impl_filter, IMPLS[i].name) != 0) continue;
            bench_cell(name, IMPLS[i], pairs[p].name, img.rgba, pairs[p].buf, img.w, img.h);
        }
    }
}


////////////////////////////////
// Entry

int main(int argc, char** argv) {
    int threads = -1;
    const char* data_dir = "data";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            g_iters = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            g_impl_filter = argv[++i];
        }
        else if (strcmp(argv[i], "--wrap") == 0 && i + 1 < argc) {
            const char* w = argv[++i];
            if (strcmp(w, "edge") == 0) {
                var::ssimu2_blur_clamp_to_border = false;
            }
            else if (strcmp(w, "border") == 0) {
                var::ssimu2_blur_clamp_to_border = true;
            }
            else {
                fprintf(stderr, "bench: --wrap must be edge|border (got %s)\n", w);
                return 1;
            }
        }
        else if (argv[i][0] == '-') {
            fprintf(stderr, "bench: unknown flag %s\n", argv[i]);
            return 1;
        }
        else {
            data_dir = argv[i];
        }
    }

    if (g_iters < 1) g_iters = 1;

    // Set rayon thread count BEFORE the first rust-av call (lazy pool init).
    if (threads > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", threads);
        setenv("RAYON_NUM_THREADS", buf, 1);
    }

    if (harness_for_each_png(data_dir, nullptr, nullptr) < 0) {
        fprintf(stderr, "bench: no PNGs found in %s\n", data_dir);
        return 1;
    }

    printf("image,impl,distortion,n,median_ms,p10_ms,p90_ms,score\n");
    harness_for_each_png(data_dir, process_image, nullptr);
    return 0;
}
