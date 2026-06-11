// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// Wrapper around the vszip_bridge C ABI (extern/vszip_bridge/include/vszip_ssimu2.h),
// which wraps the SSIMULACRA2 filter from dnjulek/vapoursynth-zip.

#include "impls.h"

extern "C" {
#include "vszip_ssimu2.h"
}

double vszip_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist) {
    return vszip_ssimu2_score(orig, dist, (unsigned)w, (unsigned)h);
}
