// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// Wrapper around the cloudinary reference impl. Their public API takes
// jxl::ImageBundle, so we round-trip our RGBA8 buffer through an in-memory
// PNG (lossless) and feed it through libjxl's SetFromBytes loader — the
// same path their CLI driver uses.

#include "impls.h"

#include "stb_image_write.h"

#include "ssimulacra2.h"  // cloudinary's, in extern/cloudinary_ssimulacra2/src/

#include "lib/extras/codec.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"

#include <algorithm>
#include <stdio.h>
#include <vector>


static void png_mem_write(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<unsigned char>*>(ctx);
    auto* p = static_cast<unsigned char*>(data);
    v->insert(v->end(), p, p + size);
}

static bool encode_png_mem(int w, int h, const unsigned char* rgba, std::vector<unsigned char>* out) {
    return stbi_write_png_to_func(png_mem_write, out, w, h, 4, rgba, w * 4) != 0;
}

double cloudinary_compute_score(int w, int h, const unsigned char* orig, const unsigned char* dist) {
    std::vector<unsigned char> orig_png, dist_png;
    if (!encode_png_mem(w, h, orig, &orig_png) || !encode_png_mem(w, h, dist, &dist_png)) {
        fprintf(stderr, "cloudinary: in-memory PNG encode failed\n");
        return 0.0;
    }

    jxl::CodecInOut io1, io2;
    auto orig_span = jxl::Span<const uint8_t>(orig_png.data(), orig_png.size());
    auto dist_span = jxl::Span<const uint8_t>(dist_png.data(), dist_png.size());
    if (!jxl::SetFromBytes(orig_span, jxl::extras::ColorHints(), &io1)) {
        fprintf(stderr, "cloudinary: SetFromBytes failed (orig)\n");
        return 0.0;
    }
    if (!jxl::SetFromBytes(dist_span, jxl::extras::ColorHints(), &io2)) {
        fprintf(stderr, "cloudinary: SetFromBytes failed (dist)\n");
        return 0.0;
    }

    if (io1.Main().HasAlpha()) {
        // Their CLI takes the worst of two background blends — matches that.
        Msssim m_dark = ComputeSSIMULACRA2(io1.Main(), io2.Main(), 0.1f);
        Msssim m_bright = ComputeSSIMULACRA2(io1.Main(), io2.Main(), 0.9f);
        return std::min(m_dark.Score(), m_bright.Score());
    }
    return ComputeSSIMULACRA2(io1.Main(), io2.Main()).Score();
}
