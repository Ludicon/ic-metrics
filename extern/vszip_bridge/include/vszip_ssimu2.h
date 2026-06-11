// Copyright 2026 Ludicon LLC. All Rights Reserved.
#ifndef VSZIP_SSIMU2_H
#define VSZIP_SSIMU2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Compute a SSIMULACRA2 score from two RGBA8 buffers (sRGB), each at least
// width*height*4 bytes. Returns the score, or NaN on allocation failure.
double vszip_ssimu2_score(
    const uint8_t *orig,
    const uint8_t *dist,
    uint32_t width,
    uint32_t height);

#ifdef __cplusplus
}
#endif

#endif // VSZIP_SSIMU2_H
