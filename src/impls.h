// Copyright 2026 Ludicon LLC. All Rights Reserved.
#pragma once

// Wrappers around third-party ssimulacra2 implementations.
// Each takes two RGBA8 buffers and returns a double score, matching the
// signature of ic_ssimulacra2_score in ic_metrics.h.
//
// Declarations are unconditional. CMake decides at build time which impl_*.cpp
// files to compile and link against. compare.cpp gates calls with HAVE_*.

double fssimu2_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double rust_av_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double cloudinary_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double fast_ssim2_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double vszip_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
double vship_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist);
