// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// Wrapper around Vship's C API (extern/Vship/src/VshipAPI.h) — a GPU compute
// SSIMULACRA2 implementation (HIP/CUDA or Vulkan). On macOS we build its
// Vulkan backend and it runs through MoltenVK.
//
// Vship takes three planar uint8 planes per image in a described colorspace
// and does the sRGB->linear / XYB conversion on the GPU itself, so we just
// deinterleave our RGBA8 into R/G/B planes and hand it the matching colorspace.

#include "impls.h"

#include "VshipAPI.h"

#include <stdio.h>
#include <stdlib.h>

namespace {

// Lazy one-time device selection (GPU 0 = MoltenVK on this Mac).
bool g_device_ready = false;
bool g_device_ok = false;

// Handler + scratch are cached and only rebuilt when the image size changes,
// so repeated calls on the same image (bench warm-up + iterations) measure
// steady-state compute, not per-call setup.
bool g_have_handler = false;
Vship_SSIMU2Handler g_handler;
int g_hw = 0, g_hh = 0;
unsigned char* g_planes = nullptr;  // 6 planes: r1 g1 b1 r2 g2 b2
size_t g_planes_cap = 0;

Vship_Colorspace_t make_colorspace(int w, int h) {
    Vship_Colorspace_t cs;
    cs.width = w;
    cs.height = h;
    cs.target_width = -1;
    cs.target_height = -1;
    cs.sample = Vship_SampleUINT8;
    cs.range = Vship_RangeFull;
    cs.subsampling.subw = 0;
    cs.subsampling.subh = 0;
    cs.chromaLocation = Vship_ChromaLoc_Left;
    cs.colorFamily = Vship_ColorRGB;
    cs.YUVMatrix = Vship_MATRIX_RGB;       // identity: input is RGB
    cs.transferFunction = Vship_TRC_sRGB;
    cs.primaries = Vship_PRIMARIES_BT709;
    cs.crop.top = cs.crop.bottom = cs.crop.left = cs.crop.right = 0;
    return cs;
}

} // namespace

double vship_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist) {
    if (!g_device_ready) {
        g_device_ready = true;
        g_device_ok = (Vship_GPUFullCheck(0) == Vship_NoError) && (Vship_SetDevice(0) == Vship_NoError);
        if (!g_device_ok) fprintf(stderr, "vship: no usable GPU 0\n");
    }
    if (!g_device_ok) return 0.0;

    const size_t n = (size_t)w * (size_t)h;

    // (Re)create the size-specific handler.
    if (!g_have_handler || g_hw != w || g_hh != h) {
        if (g_have_handler) { Vship_SSIMU2Free(g_handler); g_have_handler = false; }
        Vship_Colorspace_t cs = make_colorspace(w, h);
        if (Vship_SSIMU2Init(&g_handler, cs, cs) != Vship_NoError) {
            fprintf(stderr, "vship: SSIMU2Init failed for %dx%d\n", w, h);
            return 0.0;
        }
        g_have_handler = true; g_hw = w; g_hh = h;
    }

    // Grow the planar scratch if needed (6 planes of w*h bytes).
    if (g_planes_cap < n * 6) {
        free(g_planes);
        g_planes = (unsigned char*)malloc(n * 6);
        g_planes_cap = n * 6;
    }
    unsigned char* r1 = g_planes;        unsigned char* g1 = r1 + n; unsigned char* b1 = g1 + n;
    unsigned char* r2 = b1 + n;          unsigned char* g2 = r2 + n; unsigned char* b2 = g2 + n;
    for (size_t i = 0; i < n; i++) {
        r1[i] = orig[4*i + 0]; g1[i] = orig[4*i + 1]; b1[i] = orig[4*i + 2];
        r2[i] = dist[4*i + 0]; g2[i] = dist[4*i + 1]; b2[i] = dist[4*i + 2];
    }

    const unsigned char* p1[3] = { r1, g1, b1 };
    const unsigned char* p2[3] = { r2, g2, b2 };
    const int64_t ls[3] = { w, w, w };

    double score = 0.0;
    if (Vship_ComputeSSIMU2(g_handler, &score, p1, p2, ls, ls) != Vship_NoError) {
        fprintf(stderr, "vship: ComputeSSIMU2 failed\n");
        return 0.0;
    }
    return score;
}
