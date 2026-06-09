// Copyright (c) Ignacio Castaño <castano@gmail.com>
#pragma once

#include <stddef.h> // size_t

// Computes the SSIMULACRA 2 score between reference image 'orig' and
// distorted image 'dist'. Both buffers are RGBA8; alpha is currently
// honored per the ssimu2_alpha_blend var.
//
// Caller-managed scratch buffer. Allocate ic_*_score_scratch_size(w, h) bytes
// (malloc is fine — any allocator returning a float-aligned pointer works)
// and pass via scratch_ptr. Reuse across calls with the same w/h to avoid
// per-call allocations.
size_t ic_ssimulacra2_score_scratch_size(int w, int h);
double ic_ssimulacra2_score(int w, int h, const unsigned char *orig,
                            const unsigned char *dist, void *scratch_ptr,
                            unsigned char *error_map = nullptr);

size_t ic_ssim_score_scratch_size(int w, int h);
double ic_ssim_score(int w, int h, const unsigned char *orig,
                     const unsigned char *dist, void *scratch_ptr,
                     unsigned char *error_map = nullptr);
