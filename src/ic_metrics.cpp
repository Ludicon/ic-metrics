// Copyright 2026 Ludicon LLC. All Rights Reserved.
// Portions copyright (c) Jon Sneyers, Cloudinary. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
SSIMULACRA 2
Structural SIMilarity Unveiling Local And Compression Related Artifacts

Perceptual metric developed by Jon Sneyers (Cloudinary) in July 2022,
updated in April 2023.
Design:
- XYB color space (rescaled to a 0..1 range and with B-Y)
- SSIM map (with correction: no double gamma correction)
- 'blockiness/ringing' map (distorted has edges where original is smooth)
- 'smoothing' map (distorted is smooth where original has edges)
- error maps are computed at 6 scales (1:1 to 1:32) for each component (X,Y,B)
- downscaling is done in linear RGB
- for all 6*3*3=54 maps, two norms are computed: 1-norm (mean) and 4-norm
- a weighted sum of these 54*2=108 norms leads to the final score
- weights were tuned based on a large set of subjective scores
  (CID22, TID2013, Kadid10k, KonFiG-IQA).
*/

#include "ic_metrics.h"

#include <stddef.h> // size_t
#include <string.h> // memcpy
#include <math.h>

#if __has_include("ic_shared.h")
  #include "ic_shared.h"
  #define JXL_CHECK(st) ic_check(st)
#else
  #if _MSC_VER
    #define ic_debug_break() __debugbreak()
    #if __cplusplus >= 202002L
      #define ic_unlikely(x)   (x) [[unlikely]]
    #else
      #define ic_unlikely(x)   (x)
    #endif
    template <typename T> inline T max(const T & a, const T & b) {
      return (b < a) ? a : b;
    }

    template <typename T> inline T min(const T & a, const T & b) {
      return (a < b) ? a : b;
    }

    template <typename T> inline T clamp(const T & x, const T & lo, const T & hi) {
      return max(min(x, hi), lo);
    }
  #else
    #define ic_debug_break() __builtin_debugtrap()
    #define ic_unlikely(x)   (__builtin_expect((x), 0))
  #endif
  #define JXL_CHECK(st) assert(st)
  #define JXL_CHECK(x) do { if ic_unlikely(!(x)) ic_debug_break(); } while(false)
#endif

#if __has_include("ic_profiler.h")
  #include "ic_profiler.h"
  #define JXL_PROFILE_FUNC IC_PROFILE_FUNC
#else
  #define JXL_PROFILE_FUNC
#endif

#if __has_include("ic_vars.h")
  #include "ic_vars.h"
#else
  #define IC_VAR_BOOL(name, value) namespace var { const bool name = value; }
#endif

#if IC_CPU_ARM64
  #include <arm_neon.h>
  #define IC_HAS_F32X8 1
#elif IC_CPU_X64
  #include <immintrin.h>
  #define IC_HAS_F32X8 1
#endif

IC_VAR_BOOL(ssimu2_alpha_blend, false);

// Boundary policy for the FIR Gaussian.
//   ClampEdge   = off-image is the nearest edge pixel (default; more physically
//                 meaningful for typical texture content).
//   ClampBorder = off-image is 0. Matches cloudinary / rust-av.
//   Mirror      = RAD-style mirror with edge repeat: pixel[-1] = pixel[0],
//                 pixel[-2] = pixel[1], ... pixel[w] = pixel[w-1].
//                 (fssimu2 uses a *no-edge-repeat* mirror — pixel[-1] = pixel[1]
//                 — which we don't currently expose.)
enum BlurWrapMode {
    BlurWrapMode_ClampEdge   = 0,
    BlurWrapMode_ClampBorder = 1,
    BlurWrapMode_Mirror      = 2,
};
IC_VAR_INT(ssimu2_blur_wrap_mode, BlurWrapMode_ClampEdge);

// Use the f32x8 SIMD path in the blur interior (NEON on arm64, AVX2+FMA on
// x86_64). Falls back to scalar when off, or on platforms where neither
// backend is compiled in.
#if IC_HAS_F32X8
    IC_VAR_BOOL(ssimu2_blur_simd, true);
#else
    IC_VAR_BOOL(ssimu2_blur_simd, false);
#endif


#define JXL_RESTRICT __restrict


namespace {

// Pointer-sized signed/unsigned integers for indexing — short aliases used
// throughout this file. Local to ic_metrics.cpp on purpose; not exposed
// in any public header. If we ever target a non-64-bit platform, change
// these here and we're done.
typedef intptr_t  isize;
typedef uintptr_t usize;

// Internal aggregate; not part of the public API.
struct MsssimScale {
  double avg_ssim[3 * 2];
  double avg_edgediff[3 * 4];
};

static const int kNumScales = 6;
struct Msssim {
  MsssimScale scales[kNumScales];

  double Score() const;
};


template <typename T> inline T abs(T f) {
  return f < 0 ? -f : f;
}

template <typename T> inline T lerp(T a, T b, T t) {
    //return a + t * (b - a);
    return a * (1 - t) + b * t;
}


struct ScratchBuffer {
  void* data;
  usize size;
  usize offset;
  void* allocate(usize byte_count) {
      ic_assert(offset + byte_count <= size);
      void* ptr = (char*)data + offset;
      offset += byte_count;
      return ptr;
  }
};

// Non-owning view: data lives in the caller-provided scratch buffer. Trivial
// copy/move are correct semantics — they just copy the fat pointer.
struct ImageF {
  ImageF() = default;
  ImageF(size_t xs, size_t ys, ScratchBuffer& scratch) : xs(xs), ys(ys), orig_xs(xs), orig_ys(ys) {
    data = (float*)scratch.allocate(xs * ys * sizeof(float));
  }

  void ShrinkTo(size_t xsize, size_t ysize) {
    JXL_CHECK(xsize <= size_t(orig_xs));
    JXL_CHECK(ysize <= size_t(orig_ys));
    xs = xsize;
    ys = ysize;
  }
  void Clear() {
    memset(data, 0, sizeof(float)*xs*ys);
  }

  float* Row(size_t y) {
    return data + y * xs;
  }
  const float* Row(size_t y) const {
    return data + y * xs;
  }
  size_t xsize() const { return xs; }
  size_t ysize() const { return ys; }

  float texel(int x, int y) const {
    return data[y * xs + size_t(x)];
  }

  float sample(float x, float y) const {

    x *= float(xs);
    y *= float(ys);

    float fx = x - floorf(x);
    float fy = y - floorf(y);

    int ix0 = clamp(int(x), 0, int(xs)-1);
    int iy0 = clamp(int(y), 0, int(ys)-1);
    int ix1 = clamp(int(x)+1, 0, int(xs)-1);
    int iy1 = clamp(int(y)+1, 0, int(ys)-1);

    float f00 = texel(ix0, iy0);
    float f10 = texel(ix1, iy0);
    float f01 = texel(ix0, iy1);
    float f11 = texel(ix1, iy1);

    return lerp(lerp(f00, f10, fx), lerp(f01, f11, fx), fy);
  }

  size_t xs = 0, ys = 0;
  size_t orig_xs = 0, orig_ys = 0;
  float* data = nullptr;
};

// Non-owning view of three ImageF planes, all sharing the same ScratchBuffer.
struct Image3F {
  Image3F() = default;
  Image3F(size_t xs, size_t ys, ScratchBuffer& scratch) {
    for (int i = 0; i < 3; i++)
      planes[i] = ImageF(xs, ys, scratch);
  }

  ImageF& Plane(size_t p) { return planes[p]; }
  const ImageF& Plane(size_t p) const { return planes[p]; }
  size_t xsize() const { return planes[0].xs; }
  size_t ysize() const { return planes[0].ys; }

  void ShrinkTo(size_t xsize, size_t ysize) {
    for (int i = 0; i < 3; i++)
      planes[i].ShrinkTo(xsize, ysize);
  }
  float* PlaneRow(size_t p, size_t x) {
    return planes[p].Row(x);
  }
  const float* PlaneRow(size_t p, size_t x) const {
    return planes[p].Row(x);
  }
  const float* ConstPlaneRow(size_t p, size_t x) const {
    return planes[p].Row(x);
  }

  ImageF planes[3];
};

static bool SameSize(const Image3F& a, const Image3F& b) { return a.xsize() == b.xsize() && a.ysize() == b.ysize(); }


// Parameters for opsin absorbance.
static const float kM02 = 0.078f;
static const float kM00 = 0.30f;
static const float kM01 = 1.0f - kM02 - kM00;

static const float kM12 = 0.078f;
static const float kM10 = 0.23f;
static const float kM11 = 1.0f - kM12 - kM10;

static const float kM20 = 0.24342268924547819f;
static const float kM21 = 0.20476744424496821f;
static const float kM22 = 1.0f - kM20 - kM21;

// Opsin absorbance matrix is now frozen.
static const float kOpsinAbsorbanceMatrix[9] = {
    kM00, kM01, kM02, kM10, kM11, kM12, kM20, kM21, kM22,
};

static const float kOpsinAbsorbanceBias = 0.0037930732552754493f;

inline float ZeroIfNegative(float x) {
  return x < 0.0f ? 0.0f : x;
}

template <typename T, typename Q>
inline T BitCast(Q x) {
    static_assert(sizeof(T) == sizeof(Q), "");
    T tmp;
    memcpy(&tmp, &x, sizeof(tmp));
    return tmp;
}

// Returns cbrt(x) + add with 6 ulp max error.
// Modified from vectormath_exp.h, Apache 2 license.
// https://www.agner.org/optimize/vectorclass.zip
inline float CubeRootAndAdd(float x, float add) {
  //return cbrtf(x) + a;

  const int32_t kExpBias = 0x54800000;  // cast(1.) + cast(1.) / 3
  const int32_t kExpMul = 0x002AAAAA;   // shifted 1/3
  const float k1_3 = 1.0f / 3;
  const float k4_3 = 4.0f / 3;

  const float xa = x;  // assume inputs never negative
  const float xa_3 = k1_3 * xa;

  // Multiply exponent by -1/3
  const int32_t m1 = BitCast<int32_t>(xa);
  // Special case for 0. 0 is represented with an exponent of 0, so the
  // "kExpBias - 1/3 * exp" below gives the wrong result. The IfThenZeroElse()
  // sets those values as 0, which prevents having NaNs in the computations
  // below.
  // TODO(eustas): use fused op
  int32_t m2 = m1 == 0 ? 0 : (kExpBias - (m1 >> 23) * kExpMul);
  auto r = BitCast<float>(m2);

  // Newton-Raphson iterations
  for (int i = 0; i < 3; i++) {
      const float r2 = r * r;
      r = -xa_3 * (r2 * r2) + k4_3 * r;
  }
  // Final iteration
  float r2 = r * r;
  r = k1_3 * (-xa * r2 * r2 + r) + r;
  r2 = r * r;
  r = r2 * x + add;

  return r;
}

// Approximates smooth functions via rational polynomials (i.e. dividing two
// polynomials). Evaluates polynomials via Horner's scheme, which is faster than
// Clenshaw recurrence for Chebyshev polynomials. LoadDup128 allows us to
// specify constants (replicated 4x) independently of the lane count.
inline float EvalRationalPolynomial(float x, const float (&p)[5], const float (&q)[5]) {
    constexpr size_t kDegP = 4;
    constexpr size_t kDegQ = 4;
    float yp = p[kDegP];
    float yq = q[kDegQ];

    yp = yp * x + p[kDegP - 1];
    yq = yq * x + q[kDegQ - 1];

    yp = yp * x + p[kDegP - 2];
    yq = yq * x + q[kDegQ - 2];

    yp = yp * x + p[kDegP - 3];
    yq = yq * x + q[kDegQ - 3];

    yp = yp * x + p[kDegP - 4];
    yq = yq * x + q[kDegQ - 4];

    return yp / yq;
}

static float LinearFromSRGB(float x) {
    JXL_CHECK(x >= 0.0f);
    JXL_CHECK(x <= 1.0f);

    constexpr float kThreshSRGBToLinear = 0.04045f;
    //constexpr float kThreshLinearToSRGB = 0.0031308f;
    constexpr float kLowDiv = 12.92f;
    constexpr float kLowDivInv = 1.0f / kLowDiv;

    // Computed via af_cheb_rational (k=100); replicated 4x.
    constexpr float p[5] = { 2.200248328e-04f, 1.043637593e-02f, 1.624820318e-01f, 7.961564959e-01f, 8.210152774e-01f };
    constexpr float q[5] = { 2.631846970e-01f, 1.076976492e+00f, 4.987528350e-01f, -5.512498495e-02f, 6.521209011e-03f };

    const float linear = x * kLowDivInv;
    const float poly = EvalRationalPolynomial(x, p, q);
    const float magnitude = x > kThreshSRGBToLinear ? poly : linear;
    return magnitude;
}

static void ToXYB(Image3F& img) {
  JXL_PROFILE_FUNC

  // Pre-broadcasted constants
  float M[12];
  for (size_t i = 0; i < 9; ++i) {
    //const auto absorb = kOpsinAbsorbanceMatrix[i] * (in.metadata()->IntensityTarget() / 255.0f);
    const float absorb = kOpsinAbsorbanceMatrix[i];
    M[i] = absorb;
  }
  const float neg_bias_cbrt = -cbrtf(kOpsinAbsorbanceBias);
  for (size_t i = 0; i < 3; ++i) {
    M[9 + i] = neg_bias_cbrt;
  }

  const size_t xsize = img.xsize();
  const size_t ysize = img.ysize();

  for (size_t y = 0; y < ysize; y ++) {
    float* JXL_RESTRICT row0 = img.PlaneRow(0, y);
    float* JXL_RESTRICT row1 = img.PlaneRow(1, y);
    float* JXL_RESTRICT row2 = img.PlaneRow(2, y);

    for (size_t x = 0; x < xsize; x ++) {
      // Read all three inputs first. After this, the per-pixel kernel only
      // touches `r`, `g`, `b` locals, so writes to row_xyb*[x] below can't
      // disturb the data we still need.
      const float r = row0[x];
      const float g = row1[x];
      const float b = row2[x];

      // OpsinAbsorbance (inlined).
      float mixed0 = M[0] * r + M[1] * g + M[2] * b + kOpsinAbsorbanceBias;
      float mixed1 = M[3] * r + M[4] * g + M[5] * b + kOpsinAbsorbanceBias;
      float mixed2 = M[6] * r + M[7] * g + M[8] * b + kOpsinAbsorbanceBias;

      mixed0 = ZeroIfNegative(mixed0);
      mixed1 = ZeroIfNegative(mixed1);
      mixed2 = ZeroIfNegative(mixed2);

      mixed0 = CubeRootAndAdd(mixed0, M[9]);
      mixed1 = CubeRootAndAdd(mixed1, M[10]);
      mixed2 = CubeRootAndAdd(mixed2, M[11]);

      // StoreXYB (inlined): X = (R-G)/2, Y = (R+G)/2, B' = B.
      float X = (mixed0 - mixed1) * 0.5f;
      float Y = (mixed0 + mixed1) * 0.5f;
      float B = mixed2;

      // MakePositiveXYB
      /* Get all components in more or less 0..1 range
         Range of Rec2020 with these adjustments:
          X: 0.017223..0.998838
          Y: 0.010000..0.855303
          B: 0.048759..0.989551
         Range of sRGB:
          X: 0.204594..0.813402
          Y: 0.010000..0.855308
          B: 0.272295..0.938012
         The maximum pixel-wise difference has to be <= 1 for the ssim formula to make
         sense.
      */
      B = B - Y + 0.55f;
      X = X * 14.f + 0.42f;
      Y += 0.01f;

      row0[x] = X;
      row1[x] = Y;
      row2[x] = B;
    }
  }

}



// Blur kernel parameters.
static constexpr float kBlurSigma  = 1.5f;
static constexpr int   kBlurRadius = 4;
static constexpr int   kBlurSize   = 2 * kBlurRadius + 1;


inline void GaussianKernel(int radius, float sigma, float* kernel) {
  JXL_CHECK(sigma > 0.0);

  const int size = 2 * radius + 1;
  const float scaler = -1.0f / (2.0f * sigma * sigma);
  float sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const float val = expf(scaler * i * i);
    kernel[i + radius] = val;
    sum += val;
  }
  for (int i = 0; i < size; ++i) {
    kernel[i] /= sum;
  }
}



// Sample at index xi using the chosen wrap policy. The interior loop never
// calls this; only the border loops do.
static inline float sample_h(const float* JXL_RESTRICT row, isize xi, isize w, int mode) {
  if (xi >= 0 && xi < w) return row[xi];
  if (mode == BlurWrapMode_ClampBorder) return 0.0f;
  if (mode == BlurWrapMode_Mirror) {
    // RAD mirror with edge repeat: pixel[-1] = pixel[0], pixel[w] = pixel[w-1].
    isize mxi = (xi < 0) ? (-xi - 1) : (2 * w - xi - 1);
    return row[clamp(mxi, (isize)0, w - 1)]; // extra clamp protects tiny w
  }
  return row[clamp(xi, (isize)0, w - 1)]; // ClampEdge
}


////////////////////////////////
// Minimal 8-wide SIMD helpers. f32x8 = 2x NEON 128-bit regs on arm64, 1x
// AVX2 256-bit reg on x86_64. Use these in inner loops in place of raw
// intrinsics so the same code body works on both backends.
//
// Ops the blur needs today: load / store / add / mul-by-scalar /
// fma(acc, vec, scalar). Add more as other kernels need them.

#if IC_CPU_ARM64

struct f32x8 {
  float32x4_t lo;
  float32x4_t hi;
};

IC_FORCEINLINE f32x8 load(const float* p) {
  return { vld1q_f32(p), vld1q_f32(p + 4) };
}

IC_FORCEINLINE void store(float* p, f32x8 v) {
  vst1q_f32(p,     v.lo);
  vst1q_f32(p + 4, v.hi);
}

IC_FORCEINLINE f32x8 add(f32x8 a, f32x8 b) {
  return { vaddq_f32(a.lo, b.lo), vaddq_f32(a.hi, b.hi) };
}

IC_FORCEINLINE f32x8 mul(f32x8 a, float s) {
  return { vmulq_n_f32(a.lo, s), vmulq_n_f32(a.hi, s) };
}

// acc + vec * scalar
IC_FORCEINLINE f32x8 fma(f32x8 acc, f32x8 v, float s) {
  return { vfmaq_n_f32(acc.lo, v.lo, s), vfmaq_n_f32(acc.hi, v.hi, s) };
}

#elif IC_CPU_X64

// AVX2 + FMA backend. CMake enables -mavx2 -mfma on x86_64 builds.
struct f32x8 {
  __m256 v;
};

IC_FORCEINLINE f32x8 load(const float* p) {
  return { _mm256_loadu_ps(p) };
}

IC_FORCEINLINE void store(float* p, f32x8 v) {
  _mm256_storeu_ps(p, v.v);
}

IC_FORCEINLINE f32x8 add(f32x8 a, f32x8 b) {
  return { _mm256_add_ps(a.v, b.v) };
}

IC_FORCEINLINE f32x8 mul(f32x8 a, float s) {
  return { _mm256_mul_ps(a.v, _mm256_set1_ps(s)) };
}

// acc + vec * scalar — uses VFMADD231PS via _mm256_fmadd_ps(v, scalar, acc).
IC_FORCEINLINE f32x8 fma(f32x8 acc, f32x8 v, float s) {
  return { _mm256_fmadd_ps(v.v, _mm256_set1_ps(s), acc.v) };
}

#endif // IC_CPU_*


static void ConvolveHorizontal(const ImageF& in, ImageF* JXL_RESTRICT out, const float* JXL_RESTRICT kernel) {
  const isize w = in.xsize();
  const isize h = in.ysize();
  const int mode = var::ssimu2_blur_wrap_mode;

  // Snapshot the right half of the symmetric kernel into a stack array.
  // With just JXL_RESTRICT the compiler was reloading kernel[r+i] every
  // iteration — measured ~5% slower. R constexpr lets the i loops unroll.
  // Used by all three sections (border, NEON interior, scalar interior).
  constexpr int R = kBlurRadius;
  float kloc[R + 1];
  for (int i = 0; i <= R; ++i) kloc[i] = kernel[R + i];

  for (isize y = 0; y < h; ++y) {
    const float* JXL_RESTRICT rowp = in.Row(y);
    float* JXL_RESTRICT rowout = out->Row(y);
    isize x = 0;

    // Left border: [0, min(R, w)). Uses sample_h for off-image taps.
    const isize x_left_end = min((isize)R, w);
    for (; x < x_left_end; ++x) {
      float sum = kloc[0] * sample_h(rowp, x, w, mode);
      for (int i = 1; i <= R; ++i) {
        const float l = sample_h(rowp, x - i, w, mode);
        const float right = sample_h(rowp, x + i, w, mode);
        sum += kloc[i] * (l + right);
      }
      rowout[x] = sum;
    }

    // Interior: x ∈ [R, w-R), no bounds checks needed.
#if IC_HAS_F32X8
    if (var::ssimu2_blur_simd) {
      // 8-wide via the f32x8 helper layer. Up to 7 leftover pixels fall
      // through to the right-border loop (sample_h, ~0.15% overhead).
      for (; x + 7 < w - R; x += 8) {
        f32x8 sum = mul(load(rowp + x), kloc[0]);
        for (int i = 1; i <= R; ++i) {
          sum = fma(sum, add(load(rowp + x - i), load(rowp + x + i)), kloc[i]);
        }
        store(rowout + x, sum);
      }
    }
    else
#endif
    {
      for (; x < w - R; ++x) {
        float sum = kloc[0] * rowp[x];
        for (int i = 1; i <= R; ++i) {
          sum += kloc[i] * (rowp[x + i] + rowp[x - i]);
        }
        rowout[x] = sum;
      }
    }

    // Right border (and NEON interior remainder): from wherever we stopped to w.
    for (; x < w; ++x) {
      float sum = kloc[0] * sample_h(rowp, x, w, mode);
      for (int i = 1; i <= R; ++i) {
        const float l = sample_h(rowp, x - i, w, mode);
        const float right = sample_h(rowp, x + i, w, mode);
        sum += kloc[i] * (l + right);
      }
      rowout[x] = sum;
    }
  }
}

// Map a (possibly out-of-bounds) row index to an in-bounds one per wrap mode.
// Returns -1 for ClampBorder when out-of-bounds (caller treats that as 0.0f).
static inline isize sample_v_row(isize yi, isize h, int mode) {
  if (yi >= 0 && yi < h) return yi;
  if (mode == BlurWrapMode_ClampBorder) return -1;
  if (mode == BlurWrapMode_Mirror) {
    isize myi = (yi < 0) ? (-yi - 1) : (2 * h - yi - 1);
    return clamp(myi, (isize)0, h - 1);
  }
  return clamp(yi, (isize)0, h - 1); // ClampEdge
}

// Resolve a (possibly OOB) row index into either a row pointer or null
// (for ClampBorder out-of-bounds), per wrap mode.
static inline const float* v_row(const ImageF& in, isize yi, isize h, int mode) {
  isize row = sample_v_row(yi, h, mode);
  return (row < 0) ? nullptr : in.Row(row);
}

static void ConvolveVertical(const ImageF& in, ImageF* JXL_RESTRICT out, const float* JXL_RESTRICT kernel) {
  const isize w = in.xsize();
  const isize h = in.ysize();
  const int mode = var::ssimu2_blur_wrap_mode;

  // See ConvolveHorizontal for rationale.
  constexpr int R = kBlurRadius;
  float kloc[R + 1];
  for (int i = 0; i <= R; ++i) kloc[i] = kernel[R + i];

  // Strip width was tuned via sweep at both 1K and 4K images. No-striping
  // wins at both sizes: even at 4K (working set 9 × 4096 × 4 = 144 KB,
  // exceeding L1 on M1/M2), the strip-loop overhead costs more than the
  // L1-miss penalty (likely because the inner loop is FMA-throughput-bound).
  // Numbers (4K, --iters 3):
  //   strip=128: 812 ms   strip=1024: 753 ms   no-strip: 727 ms
  // If you ever bench on much larger images, re-run the sweep.
  {
    const isize x0 = 0;
    const isize x1 = w;

    // Top border.
    for (isize y = 0; y < min((isize)R, h); ++y) {
      float* JXL_RESTRICT rowout = out->Row(y);
      for (isize x = x0; x < x1; ++x) {
        float sum = kloc[0] * in.Row(y)[x];
        for (int i = 1; i <= R; ++i) {
          const float* r_up   = v_row(in, y - i, h, mode);
          const float* r_down = v_row(in, y + i, h, mode);
          float v_up   = r_up   ? r_up[x]   : 0.0f;
          float v_down = r_down ? r_down[x] : 0.0f;
          sum += kloc[i] * (v_up + v_down);
        }
        rowout[x] = sum;
      }
    }

    // Interior: no bounds checks.
    for (isize y = R; y < h - R; ++y) {
      float* JXL_RESTRICT rowout = out->Row(y);
      const float* JXL_RESTRICT row_c = in.Row(y);
      isize x = x0;
#if IC_HAS_F32X8
      if (var::ssimu2_blur_simd) {
        // Row pointers hoisted once per y; inner i loop unrolls with R constexpr.
        const float* row_up[R];
        const float* row_dn[R];
        for (int i = 0; i < R; ++i) {
          row_up[i] = in.Row(y - (i + 1));
          row_dn[i] = in.Row(y + (i + 1));
        }
        for (; x + 7 < x1; x += 8) {
          f32x8 sum = mul(load(row_c + x), kloc[0]);
          for (int i = 0; i < R; ++i) {
            sum = fma(sum, add(load(row_up[i] + x), load(row_dn[i] + x)), kloc[i + 1]);
          }
          store(rowout + x, sum);
        }
      }
#endif
      for (; x < x1; ++x) {
        float sum = kloc[0] * row_c[x];
        for (int i = 1; i <= R; ++i) {
          sum += kloc[i] * (in.Row(y + i)[x] + in.Row(y - i)[x]);
        }
        rowout[x] = sum;
      }
    }

    // Bottom border.
    for (isize y = max(h - R, (isize)R); y < h; ++y) {
      float* JXL_RESTRICT rowout = out->Row(y);
      for (isize x = x0; x < x1; ++x) {
        float sum = kloc[0] * in.Row(y)[x];
        for (int i = 1; i <= R; ++i) {
          const float* r_up   = v_row(in, y - i, h, mode);
          const float* r_down = v_row(in, y + i, h, mode);
          float v_up   = r_up   ? r_up[x]   : 0.0f;
          float v_down = r_down ? r_down[x] : 0.0f;
          sum += kloc[i] * (v_up + v_down);
        }
        rowout[x] = sum;
      }
    }
  }
}

// Caller pre-allocates `*out` at the desired downsampled size; we don't
// allocate here. Safe in-place when in and out share the same backing
// buffer (writes lag reads in row-major order).
static void Downsample(const Image3F &in, size_t fx, size_t fy, Image3F* out) {
  JXL_PROFILE_FUNC
  const size_t out_xsize = out->xsize();
  const size_t out_ysize = out->ysize();
  JXL_CHECK(out_xsize == (in.xsize() + fx - 1) / fx);
  JXL_CHECK(out_ysize == (in.ysize() + fy - 1) / fy);
  const float normalize = 1.0f / (fx * fy);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t oy = 0; oy < out_ysize; ++oy) {
      float *JXL_RESTRICT row_out = out->PlaneRow(c, oy);
      for (size_t ox = 0; ox < out_xsize; ++ox) {
        float sum = 0.0f;
        for (size_t iy = 0; iy < fy; ++iy) {
          for (size_t ix = 0; ix < fx; ++ix) {
            const size_t x = min(ox * fx + ix, in.xsize() - 1);
            const size_t y = min(oy * fy + iy, in.ysize() - 1);
            sum += in.PlaneRow(c, y)[x];
          }
        }
        row_out[ox] = sum * normalize;
      }
    }
  }
}

static void Multiply(const Image3F &a, const Image3F &b, Image3F *mul) {
  JXL_PROFILE_FUNC
  JXL_CHECK(SameSize(a, b));
  JXL_CHECK(SameSize(a, *mul));
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < a.ysize(); ++y) {
      const float *JXL_RESTRICT in1 = a.PlaneRow(c, y);
      const float *JXL_RESTRICT in2 = b.PlaneRow(c, y);
      float *JXL_RESTRICT out = mul->PlaneRow(c, y);
      for (size_t x = 0; x < a.xsize(); ++x) {
        out[x] = in1[x] * in2[x];
      }
    }
  }
}

static void UpscaleAndAccumulate(const ImageF &in, ImageF &out) {
  JXL_PROFILE_FUNC

  // For each pixel in out, bilinearly sample in?
  const size_t h = out.ysize();
  const size_t w = out.xsize();

  for (size_t y = 0; y < h; y++) {
    float fy = float(y) * (1.0f / h);
    float* row_out = out.Row(y);
    for (size_t x = 0; x < w; x++) {
      float fx = float(x) * (1.0f / w);
      float value = in.sample(fx, fy);
      JXL_CHECK(isfinite(value));
      row_out[x] += value;
    }
  }
}


// Separable FIR Gaussian blur. `temp` is reused across scales via ShrinkTo.
struct Blur {
  Blur(size_t xsize, size_t ysize, ScratchBuffer& scratch) : temp(xsize, ysize, scratch) {
    GaussianKernel(kBlurRadius, kBlurSigma, kernel);
  }

  void operator()(const ImageF& in, ImageF* JXL_RESTRICT out) {
    ConvolveHorizontal(in, &temp, kernel);
    ConvolveVertical(temp, out, kernel);
  }

  // Caller pre-allocates `out` at the desired size; we don't allocate here.
  void operator()(const Image3F& in, Image3F* out) {
    operator()(in.Plane(0), &out->Plane(0));
    operator()(in.Plane(1), &out->Plane(1));
    operator()(in.Plane(2), &out->Plane(2));
  }

  void ShrinkTo(const size_t xsize, const size_t ysize) {
    temp.ShrinkTo(xsize, ysize);
  }

  float kernel[kBlurSize];
  ImageF temp;
};

inline double tothe4th(double x) {
  x *= x;
  x *= x;
  return x;
}

inline float tothe4th(float x) {
  x *= x;
  x *= x;
  return x;
}

constexpr double weight[108] = {
  // ssim                                         artifact                                        detail_lost
  // X
  0.0,                    0.0007376606707406586,  0.0,                    0.0,                    0.0007793481682867309,  0.0,
  0.0,                    0.0004371155730107379,  0.0,                    1.1041726426657346,     0.00066284834129271,    0.00015231632783718752,
  0.0,                    0.0016406437456599754,  0.0,                    1.8422455520539298,     11.441172603757666,     0.0,
  0.0007989109436015163,  0.000176816438078653,   0.0,                    1.8787594979546387,     10.94906990605142,      0.0,
  0.0007289346991508072,  0.9677937080626833,     0.0,                    0.00014003424285435884, 0.9981766977854967,     0.00031949755934435053,
  0.0004550992113792063,  0.0,                    0.0,                    0.0013648766163243398,  0.0,                    0.0,

  // Y
  0.0,                    0.0,                    0.0,                    7.466890328078848,      0.0,                    17.445833984131262,
  0.0006235601634041466,  0.0,                    0.0,                    6.683678146179332,      0.00037724407979611296, 1.027889937768264,
  225.20515300849274,     0.0,                    0.0,                    19.213238186143016,     0.0011401524586618361,  0.001237755635509985,
  176.39317598450694,     0.0,                    0.0,                    24.43300999870476,      0.28520802612117757,    0.0004485436923833408,
  0.0,                    0.0,                    0.0,                    34.77906344483772,      44.835625328877896,     0.0,
  0.0,                    0.0,                    0.0,                    0.0,                    0.0,                    0.0,

  // B
  0.0,                    0.0008680556573291698,  0.0,                    0.0,                    0.0,                    0.0,
  0.0,                    0.0005313191874358747,  0.0,                    0.00016533814161379112, 0.0,                    0.0,
  0.0,                    0.0,                    0.0,                    0.0004179171803251336,  0.0017290828234722833,  0.0,
  0.0020827005846636437,  0.0,                    0.0,                    8.826982764996862,      23.19243343998926,      0.0,
  95.1080498811086,       0.9863978034400682,     0.9834382792465353,     0.0012286405048278493,  171.2667255897307,      0.9807858872435379,
  0.0,                    0.0,                    0.0,                    0.0005130064588990679,  0.0,                    0.00010854057858411537
};

inline double get_weight(int c, int scale, int map, int norm) {
  JXL_CHECK(c >= 0 && c < 3);
  JXL_CHECK(scale >= 0 && scale < 6);
  JXL_CHECK(map >= 0 && map < 3);
  JXL_CHECK(norm >= 0 && norm < 2);
  int idx = 36 * c + 6 * scale + 3 * norm + map;
  JXL_CHECK(idx >= 0 && idx < 108);
  return weight[idx];
}


static void SSIMMap(const Image3F &m1, const Image3F &m2, const Image3F &s11,
                    const Image3F &s22, const Image3F &s12, double* JXL_RESTRICT plane_averages, ImageF* error_map, int scale) {
  JXL_PROFILE_FUNC
  static const float kC2 = 0.0009f;
  const double onePerPixels = 1.0 / (m1.ysize() * m1.xsize());
  for (int c = 0; c < 3; ++c) {
    double sum1[2] = {};
    const float w_err = error_map ? float(get_weight(c, scale, 0, 0) + get_weight(c, scale, 0, 1)) : 0.0f;
    for (size_t y = 0; y < m1.ysize(); ++y) {
      const float *JXL_RESTRICT row_m1 = m1.PlaneRow(c, y);
      const float *JXL_RESTRICT row_m2 = m2.PlaneRow(c, y);
      const float *JXL_RESTRICT row_s11 = s11.PlaneRow(c, y);
      const float *JXL_RESTRICT row_s22 = s22.PlaneRow(c, y);
      const float *JXL_RESTRICT row_s12 = s12.PlaneRow(c, y);
      float *JXL_RESTRICT error_row = error_map ? error_map->Row(y) : nullptr;
      for (size_t x = 0; x < m1.xsize(); ++x) {
        float mu1 = row_m1[x];
        float mu2 = row_m2[x];
        float mu11 = mu1 * mu1;
        float mu22 = mu2 * mu2;
        float mu12 = mu1 * mu2;
        /* Correction applied compared to the original SSIM formula, which has:

             luma_err = 2 * mu1 * mu2 / (mu1^2 + mu2^2)
                      = 1 - (mu1 - mu2)^2 / (mu1^2 + mu2^2)

           The denominator causes error in the darks (low mu1 and mu2) to weigh
           more than error in the brights (high mu1 and mu2). This would make
           sense if values correspond to linear luma. However, the actual values
           are either gamma-compressed luma (which supposedly is already
           perceptually uniform) or chroma (where weighing green more than red
           or blue more than yellow does not make any sense at all). So it is
           better to simply drop this denominator.
        */
        float num_m = 1.0f - (mu1 - mu2) * (mu1 - mu2);
        float num_s = 2 * (row_s12[x] - mu12) + kC2;
        float denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + kC2;

        // Use 1 - SSIM' so it becomes an error score instead of a quality
        // index. This makes it make sense to compute an L_4 norm.
        double d = 1.0 - (num_m * num_s / denom_s);
        d = max(d, 0.0);
        double d4 = tothe4th(d);
        sum1[0] += d;
        sum1[1] += d4;

        if (error_row) {
          error_row[x] += w_err * float(d);
          JXL_CHECK(isfinite(error_row[x]));
        }
      }
    }
    plane_averages[c * 2] = onePerPixels * sum1[0];
    plane_averages[c * 2 + 1] = sqrt(sqrt(onePerPixels * sum1[1]));
  }
}

static void EdgeDiffMap(const Image3F &img1, const Image3F &mu1, const Image3F &img2,
                        const Image3F &mu2, double *plane_averages, ImageF* error_map, int scale) {
  JXL_PROFILE_FUNC
  const double onePerPixels = 1.0 / (img1.ysize() * img1.xsize());
  for (int c = 0; c < 3; ++c) {
    float sum1[4] = {0.0f};
    const float w_artif  = error_map ? float(get_weight(c, scale, 1, 0) + get_weight(c, scale, 1, 1)) : 0.0f;
    const float w_detail = error_map ? float(get_weight(c, scale, 2, 0) + get_weight(c, scale, 2, 1)) : 0.0f;
    for (size_t y = 0; y < img1.ysize(); ++y) {
      const float *JXL_RESTRICT row1 = img1.PlaneRow(c, y);
      const float *JXL_RESTRICT row2 = img2.PlaneRow(c, y);
      const float *JXL_RESTRICT rowm1 = mu1.PlaneRow(c, y);
      const float *JXL_RESTRICT rowm2 = mu2.PlaneRow(c, y);
      float *JXL_RESTRICT error_row = error_map ? error_map->Row(y) : nullptr;
      for (size_t x = 0; x < img1.xsize(); ++x) {
        float d1 = (1.0f + fabsf(row2[x] - rowm2[x])) / (1.0f + fabsf(row1[x] - rowm1[x])) - 1.0f;

        // d1 > 0: distorted has an edge where original is smooth
        //         (indicating ringing, color banding, blockiness, etc)
        float artifact = max(d1, 0.0f);
        float artifact4 = tothe4th(artifact);
        sum1[0] += artifact;
        sum1[1] += artifact4;

        // d1 < 0: original has an edge where distorted is smooth
        //         (indicating smoothing, blurring, smearing, etc)
        float detail_lost = max(-d1, 0.0f);
        float detail_lost4 = tothe4th(detail_lost);
        sum1[2] += detail_lost;
        sum1[3] += detail_lost4;

        if (error_row) {
          error_row[x] += w_artif  * fabsf(artifact)
                        + w_detail * fabsf(detail_lost);
          JXL_CHECK(isfinite(error_row[x]));
        }
      }
    }
    plane_averages[c * 4] = onePerPixels * sum1[0];
    plane_averages[c * 4 + 1] = sqrt(sqrt(onePerPixels * sum1[1]));
    plane_averages[c * 4 + 2] = onePerPixels * sum1[2];
    plane_averages[c * 4 + 3] = sqrt(sqrt(onePerPixels * sum1[3]));
  }
}


static const uint32_t MagmaMap[256] = {
    0xFF040000, 0xFF050001, 0xFF060101, 0xFF080101, 0xFF090102, 0xFF0B0202, 0xFF0D0202, 0xFF0F0303,
    0xFF120303, 0xFF140404, 0xFF160405, 0xFF180506, 0xFF1A0506, 0xFF1C0607, 0xFF1E0708, 0xFF200709,
    0xFF22080A, 0xFF24090B, 0xFF26090C, 0xFF290A0D, 0xFF2B0B0E, 0xFF2D0B10, 0xFF2F0C11, 0xFF310D12,
    0xFF340D13, 0xFF360E14, 0xFF380E15, 0xFF3B0F16, 0xFF3D0F18, 0xFF3F1019, 0xFF42101A, 0xFF44101C,
    0xFF47111D, 0xFF49111E, 0xFF4B1120, 0xFF4E1121, 0xFF501122, 0xFF531224, 0xFF551225, 0xFF581227,
    0xFF5A1129, 0xFF5C112A, 0xFF5F112C, 0xFF61112D, 0xFF63112F, 0xFF651131, 0xFF671033, 0xFF691034,
    0xFF6B1036, 0xFF6C1038, 0xFF6E0F39, 0xFF700F3B, 0xFF710F3D, 0xFF720F3F, 0xFF740F40, 0xFF750F42,
    0xFF760F44, 0xFF771045, 0xFF781047, 0xFF781049, 0xFF79104A, 0xFF7A114C, 0xFF7B114E, 0xFF7B124F,
    0xFF7C1251, 0xFF7C1352, 0xFF7D1354, 0xFF7D1456, 0xFF7E1557, 0xFF7E1559, 0xFF7E165A, 0xFF7F165C,
    0xFF7F175D, 0xFF7F185F, 0xFF801860, 0xFF801962, 0xFF801A64, 0xFF801A65, 0xFF801B67, 0xFF811C68,
    0xFF811C6A, 0xFF811D6B, 0xFF811D6D, 0xFF811E6E, 0xFF811F70, 0xFF811F72, 0xFF812073, 0xFF812175,
    0xFF812176, 0xFF812278, 0xFF822279, 0xFF82237B, 0xFF82237C, 0xFF82247E, 0xFF822580, 0xFF812581,
    0xFF812683, 0xFF812684, 0xFF812786, 0xFF812788, 0xFF812889, 0xFF81298B, 0xFF81298C, 0xFF812A8E,
    0xFF812A90, 0xFF812B91, 0xFF802B93, 0xFF802C94, 0xFF802C96, 0xFF802D98, 0xFF802D99, 0xFF7F2E9B,
    0xFF7F2E9C, 0xFF7F2F9E, 0xFF7F2FA0, 0xFF7E30A1, 0xFF7E30A3, 0xFF7E31A5, 0xFF7D31A6, 0xFF7D32A8,
    0xFF7D33AA, 0xFF7C33AB, 0xFF7C34AD, 0xFF7B34AE, 0xFF7B35B0, 0xFF7B35B2, 0xFF7A36B3, 0xFF7A36B5,
    0xFF7937B7, 0xFF7937B8, 0xFF7838BA, 0xFF7839BC, 0xFF7739BD, 0xFF773ABF, 0xFF763AC0, 0xFF753BC2,
    0xFF753CC4, 0xFF743CC5, 0xFF733DC7, 0xFF733EC8, 0xFF723ECA, 0xFF713FCC, 0xFF7140CD, 0xFF7040CF,
    0xFF6F41D0, 0xFF6F42D2, 0xFF6E43D3, 0xFF6D44D5, 0xFF6C45D6, 0xFF6C45D8, 0xFF6B46D9, 0xFF6A47DB,
    0xFF6948DC, 0xFF6849DE, 0xFF684ADF, 0xFF674CE0, 0xFF664DE2, 0xFF654EE3, 0xFF644FE4, 0xFF6450E5,
    0xFF6352E7, 0xFF6253E8, 0xFF6254E9, 0xFF6156EA, 0xFF6057EB, 0xFF6058EC, 0xFF5F5AED, 0xFF5E5BEE,
    0xFF5E5DEF, 0xFF5E5FF0, 0xFF5D60F1, 0xFF5D62F2, 0xFF5C64F2, 0xFF5C65F3, 0xFF5C67F4, 0xFF5C69F4,
    0xFF5C6BF5, 0xFF5C6CF6, 0xFF5C6EF6, 0xFF5C70F7, 0xFF5C72F7, 0xFF5C74F8, 0xFF5C76F8, 0xFF5D78F9,
    0xFF5D79F9, 0xFF5D7BF9, 0xFF5E7DFA, 0xFF5E7FFA, 0xFF5F81FA, 0xFF5F83FB, 0xFF6085FB, 0xFF6187FB,
    0xFF6189FC, 0xFF628AFC, 0xFF638CFC, 0xFF648EFC, 0xFF6590FC, 0xFF6692FD, 0xFF6794FD, 0xFF6896FD,
    0xFF6998FD, 0xFF6A9AFD, 0xFF6B9BFD, 0xFF6C9DFE, 0xFF6D9FFE, 0xFF6EA1FE, 0xFF6FA3FE, 0xFF71A5FE,
    0xFF72A7FE, 0xFF73A9FE, 0xFF74AAFE, 0xFF76ACFE, 0xFF77AEFE, 0xFF78B0FE, 0xFF7AB2FE, 0xFF7BB4FE,
    0xFF7CB6FE, 0xFF7EB7FE, 0xFF7FB9FE, 0xFF81BBFE, 0xFF82BDFE, 0xFF84BFFE, 0xFF85C1FE, 0xFF87C2FE,
    0xFF88C4FE, 0xFF8AC6FE, 0xFF8CC8FE, 0xFF8DCAFE, 0xFF8FCCFE, 0xFF90CDFE, 0xFF92CFFE, 0xFF94D1FE,
    0xFF95D3FE, 0xFF97D5FE, 0xFF99D7FE, 0xFF9AD8FE, 0xFF9CDAFD, 0xFF9EDCFD, 0xFFA0DEFD, 0xFFA1E0FD,
    0xFFA3E2FD, 0xFFA5E3FD, 0xFFA7E5FD, 0xFFA9E7FD, 0xFFAAE9FD, 0xFFACEBFD, 0xFFAEECFC, 0xFFB0EEFC,
    0xFFB2F0FC, 0xFFB4F2FC, 0xFFB6F4FC, 0xFFB8F6FC, 0xFFB9F7FC, 0xFFBBF9FC, 0xFFBDFBFC, 0xFFBFFDFC
};


static Msssim ComputeSSIMULACRA2(ScratchBuffer& scratch, Image3F& orig, Image3F& dist, unsigned char* error_map) {
  JXL_PROFILE_FUNC
  Msssim msssim = {};

  const size_t w = orig.xsize();
  const size_t h = orig.ysize();

  // Error-map workspace is only needed when the caller asked for one.
  // Saves two ImageF allocs and the per-pixel error_row writes inside
  // SSIMMap / EdgeDiffMap (gated by null pointer).
  const bool want_error_map = (error_map != nullptr);
  ImageF error_accum;
  ImageF error_scale;
  if (want_error_map) {
    error_accum = ImageF(w, h, scratch);
    error_scale = ImageF(w, h, scratch);
    error_scale.Clear();
  }

  // ds_orig/ds_dist hold the linear-RGB downsample at scale-1 size. After
  // pre-loop setup, orig/dist contain XYB at scale 0 (in-place converted)
  // and ds_orig/ds_dist contain linear RGB at scale 1.
  //
  // Inside the scale loop we ping-pong: at any scale s, *xyb_orig/dist
  // hold XYB at scale s (analyzed), and *linear_orig/dist hold linear RGB
  // at scale s+1 (ready to downsample). After analysis we downsample
  // linear_* into xyb_*'s freed buffer (linear at scale s+2), convert
  // linear_* in-place to XYB (so it now holds XYB at scale s+1), and swap
  // the pointers. ds_orig/ds_dist are sized for scale 1; orig/dist
  // (full-size buffers) hold subsequent shrunk-down scales.
  Image3F ds_orig(w / 2, h / 2, scratch);
  Image3F ds_dist(w / 2, h / 2, scratch);

  Downsample(orig, 2, 2, &ds_orig);
  Downsample(dist, 2, 2, &ds_dist);
  ToXYB(orig);
  ToXYB(dist);

  Image3F* xyb_orig    = &orig;     // XYB, scale 0
  Image3F* xyb_dist    = &dist;     // XYB, scale 0
  Image3F* linear_orig = &ds_orig;  // linear RGB, scale 1
  Image3F* linear_dist = &ds_dist;  // linear RGB, scale 1

  Image3F mul(w, h, scratch);
  Blur blur(w, h, scratch);
  Image3F sigma1_sq(w, h, scratch);
  Image3F sigma2_sq(w, h, scratch);
  Image3F sigma12  (w, h, scratch);
  Image3F mu1      (w, h, scratch);
  Image3F mu2      (w, h, scratch);

  for (int scale = 0; scale < kNumScales; scale++) {
    if (xyb_orig->xsize() < 8 || xyb_orig->ysize() < 8) {
      break;
    }

    const size_t sx = xyb_orig->xsize();
    const size_t sy = xyb_orig->ysize();
    mul      .ShrinkTo(sx, sy);
    blur     .ShrinkTo(sx, sy);
    sigma1_sq.ShrinkTo(sx, sy);
    sigma2_sq.ShrinkTo(sx, sy);
    sigma12  .ShrinkTo(sx, sy);
    mu1      .ShrinkTo(sx, sy);
    mu2      .ShrinkTo(sx, sy);
    if (want_error_map && scale > 0) {
      error_scale.ShrinkTo(sx, sy);
      error_scale.Clear();
    }

    Multiply(*xyb_orig, *xyb_orig, &mul);  blur(mul, &sigma1_sq);
    Multiply(*xyb_dist, *xyb_dist, &mul);  blur(mul, &sigma2_sq);
    Multiply(*xyb_orig, *xyb_dist, &mul);  blur(mul, &sigma12);
    blur(*xyb_orig, &mu1);
    blur(*xyb_dist, &mu2);

    ImageF* err = want_error_map ? &error_scale : nullptr;
    SSIMMap(mu1, mu2, sigma1_sq, sigma2_sq, sigma12, msssim.scales[scale].avg_ssim, err, scale);
    EdgeDiffMap(*xyb_orig, mu1, *xyb_dist, mu2, msssim.scales[scale].avg_edgediff, err, scale);

    if (want_error_map) {
      if (scale == 0) {
        memcpy(error_accum.data, error_scale.data, sizeof(float) * error_scale.xs * error_scale.ys);
      }
      else if (scale < 3) {
        UpscaleAndAccumulate(error_scale, error_accum);
      }
    }

    // Prepare for next scale: linear_* holds linear RGB at scale+1.
    // Downsample it into xyb_*'s now-free buffer (linear at scale+2),
    // then in-place ToXYB linear_* (so it holds XYB at scale+1), then
    // swap so the next iteration sees XYB in xyb_* and linear in linear_*.
    const size_t next_xs = linear_orig->xsize() / 2;
    const size_t next_ys = linear_orig->ysize() / 2;
    if (scale + 1 < kNumScales && next_xs >= 8 && next_ys >= 8) {
      xyb_orig->ShrinkTo(next_xs, next_ys);
      xyb_dist->ShrinkTo(next_xs, next_ys);
      Downsample(*linear_orig, 2, 2, xyb_orig);
      Downsample(*linear_dist, 2, 2, xyb_dist);
      ToXYB(*linear_orig);
      ToXYB(*linear_dist);
      Image3F* tmp;
      tmp = xyb_orig; xyb_orig = linear_orig; linear_orig = tmp;
      tmp = xyb_dist; xyb_dist = linear_dist; linear_dist = tmp;
    }
  }

  if (error_map != nullptr) {
    // Remap scalar error values to magma color palette.
    for (size_t y = 0; y < h; y++) {
      for (size_t x = 0; x < w; x++) {
        float ssim = error_accum.texel(x,y);

        ssim = ssim * 0.9562382616834844f;
        ssim = 2.326765642916932f * ssim - 0.020884521182843837f * ssim * ssim + 6.248496625763138e-05f * ssim * ssim * ssim;
        if (ssim > 0.0f) {
          ssim = 100.0f - 10.0f * powf(ssim, 0.6276336467831387f);
        } else {
          ssim = 100.0f;
        }

        ssim = 1.0f - ssim / 100.0f;

        int value = int(255 * clamp(ssim, 0.0f, 1.0f));
        ((uint32_t*)error_map)[y * w + x] = MagmaMap[value];
      }
    }
  }

  return msssim;
}


/*
The final score is based on a weighted sum of 108 sub-scores:
- for 6 scales (1:1 to 1:32, downsampled in linear RGB)
- for 3 components (X, Y, B-Y, rescaled to 0..1 range)
- using 2 norms (the 1-norm and the 4-norm)
- over 3 error maps:
    - SSIM' (SSIM without the spurious gamma correction term)
    - "ringing" (distorted edges where there are no orig edges)
    - "blurring" (orig edges where there are no distorted edges)

The weights were obtained by running Nelder-Mead simplex search,
optimizing to minimize MSE for the CID22 training set and to
maximize Kendall rank correlation (and with a lower weight,
also Pearson correlation) with the CID22 training set and the
TID2013, Kadid10k and KonFiG-IQA datasets.
Validation was done on the CID22 validation set.

Final results after tuning (Kendall | Spearman | Pearson):
   CID22:     0.6903 | 0.8805 | 0.8583
   TID2013:   0.6590 | 0.8445 | 0.8471
   KADID-10k: 0.6175 | 0.8133 | 0.8030
   KonFiG(F): 0.7668 | 0.9194 | 0.9136
*/
double Msssim::Score() const {
  JXL_PROFILE_FUNC
  double ssim = 0.0;

  size_t i = 0;
  for (size_t c = 0; c < 3; ++c) {
    for (size_t scale = 0; scale < kNumScales; ++scale) {
      for (size_t n = 0; n < 2; n++) {
        ssim += weight[i++] * abs(scales[scale].avg_ssim[c * 2 + n]);
        ssim += weight[i++] * abs(scales[scale].avg_edgediff[c * 4 + n]);
        ssim += weight[i++] * abs(scales[scale].avg_edgediff[c * 4 + n + 2]);
      }
    }
  }

  ssim = ssim * 0.9562382616834844;
  ssim = 2.326765642916932 * ssim - 0.020884521182843837 * ssim * ssim + 6.248496625763138e-05 * ssim * ssim * ssim;
  if (ssim > 0) {
    ssim = 100.0 - 10.0 * pow(ssim, 0.6276336467831387);
  } else {
    ssim = 100.0;
  }
  return ssim;
}


Msssim ComputeSSIMULACRA2(ScratchBuffer& scratch, int w, int h, const unsigned char* orig, const unsigned char* dist, unsigned char* error_map) {
  JXL_PROFILE_FUNC
  Image3F orig_img(w, h, scratch);
  Image3F dist_img(w, h, scratch);

  for (int y = 0; y < h; y++) {
    float* orig_r = orig_img.PlaneRow(0, y);
    float* orig_g = orig_img.PlaneRow(1, y);
    float* orig_b = orig_img.PlaneRow(2, y);
    for (int x = 0; x < w; x++) {
      float r = LinearFromSRGB(float(orig[(y * w + x) * 4 + 0]) / 255.0f);
      float g = LinearFromSRGB(float(orig[(y * w + x) * 4 + 1]) / 255.0f);
      float b = LinearFromSRGB(float(orig[(y * w + x) * 4 + 2]) / 255.0f);
      float a = float(orig[(y * w + x) * 4 + 3]) / 255.0f;

      if (var::ssimu2_alpha_blend) {
        orig_r[x] = lerp(0.5f, r, a);
        orig_g[x] = lerp(0.5f, g, a);
        orig_b[x] = lerp(0.5f, b, a);
      }
      else {
        orig_r[x] = r;
        orig_g[x] = g;
        orig_b[x] = b;
      }
    }
  }

  for (int y = 0; y < h; y++) {
    float* dist_r = dist_img.PlaneRow(0, y);
    float* dist_g = dist_img.PlaneRow(1, y);
    float* dist_b = dist_img.PlaneRow(2, y);
    for (int x = 0; x < w; x++) {
      float r = LinearFromSRGB(float(dist[(y * w + x) * 4 + 0]) / 255.0f);
      float g = LinearFromSRGB(float(dist[(y * w + x) * 4 + 1]) / 255.0f);
      float b = LinearFromSRGB(float(dist[(y * w + x) * 4 + 2]) / 255.0f);
      float a = float(dist[(y * w + x) * 4 + 3]) / 255.0f;
      if (var::ssimu2_alpha_blend) {
        dist_r[x] = lerp(0.5f, r, a);
        dist_g[x] = lerp(0.5f, g, a);
        dist_b[x] = lerp(0.5f, b, a);
      }
      else {
        dist_r[x] = r;
        dist_g[x] = g;
        dist_b[x] = b;
      }
    }
  }

  return ComputeSSIMULACRA2(scratch, orig_img, dist_img, error_map);
}

} // namespace


// Scratch budget, in floats, sized at the largest scale (w*h):
//   ic_ssimulacra2_score                without error map     with error map
//   ----------------------------------- --------------------- --------------------
//   orig_img, dist_img                     6                    6  (2 Image3F)
//   ds_orig, ds_dist (at w/2 × h/2)        1.5                  1.5
//   mul, sigma1_sq, sigma2_sq, sigma12     12                   12
//   mu1, mu2                               6                    6
//   blur.temp                              1                    1
//   error_accum, error_scale               0                    2  (2 ImageF)
//   ----------------------------------- --------------------- --------------------
//   Total                                  26.5                 28.5
//
// Rounded up to integer-multiple-of-wh = 29 (caller doesn't know in advance
// whether they'll pass an error_map). Down from 35 wh before the in-place
// ToXYB refactor that retired the separate img1/img2 buffers.
size_t ic_ssimulacra2_score_scratch_size(int w, int h) {
    return usize(w) * usize(h) * sizeof(float) * 29;
}

double ic_ssimulacra2_score(int w, int h, const unsigned char* orig, const unsigned char* dist, void* scratch_ptr, unsigned char* error_map) {
  ScratchBuffer scratch = { scratch_ptr, ic_ssimulacra2_score_scratch_size(w, h), 0 };
  return ComputeSSIMULACRA2(scratch, w, h, orig, dist, error_map).Score();
}


size_t ic_ssim_score_scratch_size(int w, int h) {
  return usize(w) * usize(h) * sizeof(float) * 9;
}

double ic_ssim_score(int w, int h, const unsigned char* orig, const unsigned char* dist, void* scratch_ptr, unsigned char* error_map) {
  JXL_PROFILE_FUNC

  ScratchBuffer scratch = { scratch_ptr, ic_ssim_score_scratch_size(w, h), 0 };

  // Standard SSIM constants for [0,1] range (Wang et al. 2004).
  // C1 = (K1*L)^2, C2 = (K2*L)^2, with K1=0.01, K2=0.03, L=1.
  static const float kC1 = 0.0001f;
  static const float kC2 = 0.0009f;

  ImageF img1(w, h, scratch);
  ImageF img2(w, h, scratch);

  // RGBA8 → Rec.601 luma in [0,1]. Standard SSIM operates on
  // gamma-corrected luma (i.e. the sRGB bytes directly, no
  // linearization).
  constexpr float kR = 0.299f / 255.0f;
  constexpr float kG = 0.587f / 255.0f;
  constexpr float kB = 0.114f / 255.0f;

  const size_t n = size_t(w) * size_t(h);

  {
    float* row1 = img1.data;
    float* row2 = img2.data;
    for (size_t i = 0; i < n; i++) {
      row1[i] = kR * orig[4*i + 0] + kG * orig[4*i + 1] + kB * orig[4*i + 2];
      row2[i] = kR * dist[4*i + 0] + kG * dist[4*i + 1] + kB * dist[4*i + 2];
    }
  }

  Blur blur(w, h, scratch);

  // mu1 = blur(img1), mu2 = blur(img2)
  ImageF mu1(w, h, scratch);
  ImageF mu2(w, h, scratch);
  blur(img1, &mu1);
  blur(img2, &mu2);

  // sigma1_sq = blur(img1*img1) - mu1*mu1
  // sigma2_sq = blur(img2*img2) - mu2*mu2
  // sigma12   = blur(img1*img2) - mu1*mu2
  ImageF tmp(w, h, scratch);
  ImageF sigma1_sq(w, h, scratch);
  ImageF sigma2_sq(w, h, scratch);
  ImageF sigma12(w, h, scratch);

  {
    const float* p1 = img1.data;
    const float* p2 = img2.data;
    float* pt = tmp.data;
    for (size_t i = 0; i < n; i++) pt[i] = p1[i] * p1[i];
    blur(tmp, &sigma1_sq);
    for (size_t i = 0; i < n; i++) pt[i] = p2[i] * p2[i];
    blur(tmp, &sigma2_sq);
    for (size_t i = 0; i < n; i++) pt[i] = p1[i] * p2[i];
    blur(tmp, &sigma12);
  }

  // Compute SSIM map and accumulate mean.
  double ssim_sum = 0.0;
  const double one_per_pixels = 1.0 / double(n);
  const float* pm1 = mu1.data;
  const float* pm2 = mu2.data;
  const float* ps11 = sigma1_sq.data;
  const float* ps22 = sigma2_sq.data;
  const float* ps12 = sigma12.data;
  uint32_t* err_out = (uint32_t*)error_map;

  for (size_t i = 0; i < n; i++) {
    float m1 = pm1[i];
    float m2 = pm2[i];
    float m1m2 = m1 * m2;
    float m1sq = m1 * m1;
    float m2sq = m2 * m2;
    float s1sq = ps11[i] - m1sq;
    float s2sq = ps22[i] - m2sq;
    float s12  = ps12[i] - m1m2;

    float num   = (2.0f * m1m2 + kC1) * (2.0f * s12 + kC2);
    float denom = (m1sq + m2sq + kC1) * (s1sq + s2sq + kC2);
    float ssim  = num / denom;

    ssim_sum += ssim;

    if (err_out) {
      // Map 1-SSIM error to magma palette, same as ssimulacra2.
      int value = int(255 * clamp(1.0f - ssim, 0.0f, 1.0f));
      err_out[i] = MagmaMap[value];
    }
  }

  return ssim_sum * one_per_pixels;
}


// MS-SSIM (Multi-Scale SSIM), Wang/Simoncelli/Bovik 2003. Same Gaussian,
// C1/C2 and Rec.601-luma input as ic_ssim_score — at scale 0, MS-SSIM and
// SSIM use the same per-pixel formula.
//
// At each scale j we compute pixel-wise SSIM = l·cs and CS, take their mean
// over the image, then downsample (2×2 box average) for the next scale.
// Combination follows the paper:
//   MS-SSIM = mean_SSIM_M ^ w_M  *  prod_{j<M} mean_CS_j ^ w_j
// with weights {0.0448, 0.2856, 0.3001, 0.2363, 0.1333} and M = 5. We
// compute in log space and divide by the sum of *used* weights so the
// score still makes sense when small images can't reach all 5 scales.

// 2×2 box-average downsample for a single plane. In-place safe in
// row-major order — output stride < input stride so writes always lag reads.
static void DownsampleAvg2(const ImageF& in, ImageF* out) {
  const size_t out_w = out->xsize();
  const size_t out_h = out->ysize();
  JXL_CHECK(out_w == (in.xsize() + 1) / 2);
  JXL_CHECK(out_h == (in.ysize() + 1) / 2);
  const float norm = 0.25f;
  const size_t in_w = in.xsize();
  const size_t in_h = in.ysize();
  for (size_t oy = 0; oy < out_h; ++oy) {
    const size_t iy0 = oy * 2;
    const size_t iy1 = (iy0 + 1 < in_h) ? iy0 + 1 : iy0;     // odd-height fold
    const float* row_in0 = in.Row(iy0);
    const float* row_in1 = in.Row(iy1);
    float* row_out = out->Row(oy);
    for (size_t ox = 0; ox < out_w; ++ox) {
      const size_t ix0 = ox * 2;
      const size_t ix1 = (ix0 + 1 < in_w) ? ix0 + 1 : ix0;   // odd-width fold
      row_out[ox] = (row_in0[ix0] + row_in0[ix1] + row_in1[ix0] + row_in1[ix1]) * norm;
    }
  }
}

size_t ic_msssim_score_scratch_size(int w, int h) {
  // Identical layout to ic_ssim_score: img1, img2, mu1, mu2, tmp,
  // sigma1_sq, sigma2_sq, sigma12, blur.temp — 9 ImageF at the largest
  // scale. Coarser scales reuse the same buffers via ShrinkTo.
  return usize(w) * usize(h) * sizeof(float) * 9;
}

double ic_msssim_score(int w, int h, const unsigned char* orig, const unsigned char* dist, void* scratch_ptr, unsigned char* error_map) {
  JXL_PROFILE_FUNC

  ScratchBuffer scratch = { scratch_ptr, ic_msssim_score_scratch_size(w, h), 0 };

  // Same constants as SSIM (Wang 2004) for the [0,1] luma range.
  static const float kC1 = 0.0001f;
  static const float kC2 = 0.0009f;

  // Wang/Simoncelli/Bovik 2003 default scale weights. Sum to 1.0.
  static const double kWeights[5] = { 0.0448, 0.2856, 0.3001, 0.2363, 0.1333 };
  static constexpr int kScales = 5;

  ImageF img1(w, h, scratch);
  ImageF img2(w, h, scratch);

  // RGBA8 → Rec.601 luma in [0,1]. Same path as ic_ssim_score.
  constexpr float kR = 0.299f / 255.0f;
  constexpr float kG = 0.587f / 255.0f;
  constexpr float kB = 0.114f / 255.0f;
  {
    const size_t n0 = size_t(w) * size_t(h);
    float* row1 = img1.data;
    float* row2 = img2.data;
    for (size_t i = 0; i < n0; i++) {
      row1[i] = kR * orig[4*i + 0] + kG * orig[4*i + 1] + kB * orig[4*i + 2];
      row2[i] = kR * dist[4*i + 0] + kG * dist[4*i + 1] + kB * dist[4*i + 2];
    }
  }

  Blur blur(w, h, scratch);
  ImageF mu1(w, h, scratch);
  ImageF mu2(w, h, scratch);
  ImageF tmp(w, h, scratch);
  ImageF sigma1_sq(w, h, scratch);
  ImageF sigma2_sq(w, h, scratch);
  ImageF sigma12(w, h, scratch);

  double log_msssim = 0.0;
  double weights_used = 0.0;

  for (int j = 0; j < kScales; j++) {
    const size_t sx = img1.xsize();
    const size_t sy = img1.ysize();
    if (sx < 8 || sy < 8) break;

    blur     .ShrinkTo(sx, sy);
    mu1      .ShrinkTo(sx, sy);
    mu2      .ShrinkTo(sx, sy);
    tmp      .ShrinkTo(sx, sy);
    sigma1_sq.ShrinkTo(sx, sy);
    sigma2_sq.ShrinkTo(sx, sy);
    sigma12  .ShrinkTo(sx, sy);

    blur(img1, &mu1);
    blur(img2, &mu2);

    // sigma1_sq = blur(img1²)   (the mu1² subtraction is folded into the per-pixel formula)
    const size_t n = sx * sy;
    {
      const float* p1 = img1.data;
      const float* p2 = img2.data;
      float* pt = tmp.data;
      for (size_t i = 0; i < n; i++) pt[i] = p1[i] * p1[i];
      blur(tmp, &sigma1_sq);
      for (size_t i = 0; i < n; i++) pt[i] = p2[i] * p2[i];
      blur(tmp, &sigma2_sq);
      for (size_t i = 0; i < n; i++) pt[i] = p1[i] * p2[i];
      blur(tmp, &sigma12);
    }

    // Mean SSIM (= l·cs) and mean CS over pixels. The cs/l split here uses
    // C3 = C2/2 so that l·cs simplifies to (2σxy+C2)/(σx²+σy²+C2) · l —
    // saves one division per pixel.
    double ssim_sum = 0.0;
    double cs_sum = 0.0;
    const double one_per_pixels = 1.0 / double(n);
    const bool fill_err = (error_map != nullptr) && (j == 0);
    const float* pm1  = mu1.data;
    const float* pm2  = mu2.data;
    const float* ps11 = sigma1_sq.data;
    const float* ps22 = sigma2_sq.data;
    const float* ps12 = sigma12.data;
    uint32_t* err_out = fill_err ? (uint32_t*)error_map : nullptr;

    for (size_t i = 0; i < n; i++) {
      float m1 = pm1[i];
      float m2 = pm2[i];
      float m1m2 = m1 * m2;
      float m1sq = m1 * m1;
      float m2sq = m2 * m2;
      float s1sq = ps11[i] - m1sq;
      float s2sq = ps22[i] - m2sq;
      float s12  = ps12[i] - m1m2;

      float l_num  = 2.0f * m1m2 + kC1;
      float l_den  = m1sq + m2sq + kC1;
      float cs_num = 2.0f * s12  + kC2;
      float cs_den = s1sq + s2sq + kC2;

      float cs   = cs_num / cs_den;
      float ssim = (l_num / l_den) * cs;

      ssim_sum += ssim;
      cs_sum   += cs;

      if (err_out) {
        int value = int(255 * clamp(1.0f - ssim, 0.0f, 1.0f));
        err_out[i] = MagmaMap[value];
      }
    }

    const double ssim_mean = ssim_sum * one_per_pixels;
    const double cs_mean   = cs_sum   * one_per_pixels;

    // Coarsest scale we'll actually run contributes l·cs; finer scales
    // contribute CS only. Detect "is this the last scale we'll run" by
    // checking whether the next scale would be too small.
    const bool is_last_scale = (j == kScales - 1) || (sx / 2 < 8 || sy / 2 < 8);
    double contribution = is_last_scale ? ssim_mean : cs_mean;
    if (contribution < 1e-10) contribution = 1e-10;  // guard the log
    log_msssim += kWeights[j] * log(contribution);
    weights_used += kWeights[j];

    if (is_last_scale) break;

    // Downsample img1, img2 for the next scale (in-place safe).
    const ImageF src1 = img1;
    const ImageF src2 = img2;
    const size_t next_sx = (sx + 1) / 2;
    const size_t next_sy = (sy + 1) / 2;
    img1.ShrinkTo(next_sx, next_sy);
    img2.ShrinkTo(next_sx, next_sy);
    DownsampleAvg2(src1, &img1);
    DownsampleAvg2(src2, &img2);
  }

  if (weights_used <= 0.0) return 0.0;
  return exp(log_msssim / weights_used);
}
