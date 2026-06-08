// Copyright (c) Ignacio Castaño <castano@gmail.com>
// Copyright (c) Jon Sneyers, Cloudinary. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

// Computes the SSIMULACRA 2 score between reference image 'orig' and
// distorted image 'dist'. Both buffers are RGBA8; alpha is currently
// honored per the ssimu2_alpha_blend var.
double ComputeSSIMULACRA2Score(int w, int h, const unsigned char * orig, const unsigned char * dist, unsigned char* error_map = nullptr);

double ComputeSSIMScore(int w, int h, const unsigned char * orig, const unsigned char * dist, unsigned char* error_map = nullptr);
