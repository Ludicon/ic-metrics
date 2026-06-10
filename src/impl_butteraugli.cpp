// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// Wrapper around upstream butteraugli (google/butteraugli). Lets us call
// the reference implementation from compare/bench to validate our own
// butteraugli port.
//
// Upstream expects 3 float ImageF planes (R, G, B) with sRGB-encoded
// values in [0, 255] — we round-trip our RGBA8 buffer through them.
// Score is a distance: 0 = identical, larger = more different.

#include "impls.h"

#include "butteraugli/butteraugli.h"

#include <vector>

double butteraugli_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist) {
    using butteraugli::ImageF;

    std::vector<ImageF> rgb0;
    std::vector<ImageF> rgb1;
    rgb0.reserve(3);
    rgb1.reserve(3);
    for (int c = 0; c < 3; c++) {
        rgb0.emplace_back(w, h);
        rgb1.emplace_back(w, h);
    }

    // Unpack RGBA8 -> 3 planar float images, sRGB [0, 255] preserved.
    for (int y = 0; y < h; y++) {
        float* r0 = rgb0[0].Row(y);
        float* g0 = rgb0[1].Row(y);
        float* b0 = rgb0[2].Row(y);
        float* r1 = rgb1[0].Row(y);
        float* g1 = rgb1[1].Row(y);
        float* b1 = rgb1[2].Row(y);
        const unsigned char* p0 = orig + (size_t)y * w * 4;
        const unsigned char* p1 = dist + (size_t)y * w * 4;
        for (int x = 0; x < w; x++) {
            r0[x] = p0[4*x + 0];
            g0[x] = p0[4*x + 1];
            b0[x] = p0[4*x + 2];
            r1[x] = p1[4*x + 0];
            g1[x] = p1[4*x + 1];
            b1[x] = p1[4*x + 2];
        }
    }

    ImageF diffmap;
    double diffvalue = 0.0;
    // hf_asymmetry = 1.0 is the upstream-recommended default (treats
    // brighter-than-reference and darker-than-reference equally).
    butteraugli::ButteraugliInterface(rgb0, rgb1, /*hf_asymmetry=*/1.0f,
                                      diffmap, diffvalue);
    return diffvalue;
}
