// Copyright 2026 Ludicon LLC. All Rights Reserved.
#pragma once

// Wrappers around third-party image-quality metric implementations.
// Each takes two RGBA8 buffers and returns a double score. SSIMULACRA 2
// impls return a quality score (higher = better). Butteraugli returns a
// distance (lower = better) — not directly comparable to the others.
//
// Declarations are unconditional. CMake decides at build time which impl_*.cpp
// files to compile and link against. compare.cpp gates calls with HAVE_*.

double fssimu2_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double rust_av_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double cloudinary_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double butteraugli_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
