// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// Wrapper around the fssimu2 C ABI (extern/fssimu2/zig-out/include/ssimu2.h).

#include "impls.h"

extern "C" {
#include "ssimu2.h"
}

#include <stdio.h>

double fssimu2_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist) {
    double score = 0.0;
    int err = ssimulacra2_score(orig, dist, (unsigned)w, (unsigned)h, /*channels=*/4, &score);
    if (err != SSIMU2_OK) {
        fprintf(stderr, "fssimu2: error %d on %dx%d\n", err, w, h);
        return 0.0;
    }
    return score;
}
