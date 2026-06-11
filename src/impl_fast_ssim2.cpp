// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// Wrapper around the fast_ssim2_bridge C ABI (extern/fast_ssim2_bridge/src/lib.rs).

#include "impls.h"

extern "C" double fast_ssim2_score(
    const unsigned char* orig, const unsigned char* dist, unsigned int w, unsigned int h);

double fast_ssim2_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist) {
    return fast_ssim2_score(orig, dist, (unsigned)w, (unsigned)h);
}
