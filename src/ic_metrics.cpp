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

  #if _DEBUG
    #define ic_assert(x) do { if ic_unlikely(!(x)) ic_debug_break(); } while(false)
  #else
    #define ic_assert(x) (void)sizeof(x)
  #endif
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
  #define IC_VAR_INT(name, value) namespace var { const int name = value; }
  #define IC_VAR_FLOAT(name, value) namespace var { const float name = value; }
#endif

#if IC_CPU_ARM64
  #include <arm_neon.h>
  #define IC_HAS_F32X8 1
#elif IC_CPU_X64
  #include <immintrin.h>
  #define IC_HAS_F32X8 1
#endif


// Boundary policy for the FIR Gaussian.
//   ClampEdge   = off-image is the nearest edge pixel (default; more physically
//                 meaningful for typical texture content).
//   ClampBorder = off-image is 0. Matches cloudinary / rust-av.
//   Mirror      = Mirror with edge repeat: pixel[-1] = pixel[0],
//                 pixel[-2] = pixel[1], ... pixel[w] = pixel[w-1].
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

// Weight-based pruning. SSIMULACRA2's score is a weighted sum of 108 sub-scores
// (3 components x 6 scales x 3 maps x 2 norms), and many weights are exactly 0.
// A whole (component, scale, map) can contribute nothing. We skip the blurs,
// multiplies, and map kernel for any sub-score whose weights (both norms) are
// <= this threshold:
//   < 0   disables pruning (compute everything; the pre-pruning behavior).
//   0.0   prunes only exact-zero weights (lossless).
//   > 0   prunes small weights too for a tiny, bounded score change.
// Default 0.01 (matching vapoursynth-zip): ~25-30% faster than no pruning, with
// a measured score shift of at most +0.0006 across our test set — well under the
// ~0.01 agreement with the cloudinary reference. Set to 0.0 for bit-exact scores.
IC_VAR_FLOAT(ssimu2_prune_threshold, 0.01f);

// IW-SSIM: evaluate the per-pixel weight as one log2 of the product of the
// per-component terms instead of a sum of 10 log2f calls. The product is a
// fixed polynomial of the eigenvalues evaluated by Horner's rule in double
// (its terms reach ~1e8 each, so float would overflow).
IC_VAR_BOOL(iwssim_log_product, true);

// IW-SSIM: evaluate the GSM multiplier s^2 = v^T C_u^-1 v as |R v|^2 with R
// the Cholesky factor of C_u^-1 (upper triangular, 55 FMAs for n = 10)
// instead of projecting v onto all n eigenvectors (n^2 = 100 FMAs). Both are
// sums of squares, so neither cancels; the plain symmetric quadratic form
// would, since C_u^-1 holds inverse eigenvalues.
IC_VAR_BOOL(iwssim_cholesky_projection, true);

// IW-SSIM: build the neighborhood covariance from SIMD dot products of
// shifted rows (converted to double once per row) instead of gathering a
// 10-vector per pixel and accumulating its outer product in scalar double.
// Same double arithmetic, so results match to rounding.
IC_VAR_BOOL(iwssim_simd_covariance, true);

// IW-SSIM: run the per-pixel weight pass (s^2, block statistics, weight,
// CS) eight pixels at a time with f32x8. The scalar path handles row tails
// and serves as the reference variant. The vector path uses a polynomial
// log2 accurate to ~2e-7 absolute; score differences are at rounding level.
#if IC_HAS_F32X8
    IC_VAR_BOOL(iwssim_simd_weights, true);
#else
    IC_VAR_BOOL(iwssim_simd_weights, false);
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
    ic_assert(xsize <= size_t(orig_xs));
    ic_assert(ysize <= size_t(orig_ys));
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
    ic_assert(x >= 0.0f);
    ic_assert(x <= 1.0f);

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
  ic_assert(sigma > 0.0);

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
    // Mirror with edge repeat: pixel[-1] = pixel[0], pixel[w] = pixel[w-1].
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

IC_FORCEINLINE f32x8 set1(float s) {
  return { vdupq_n_f32(s), vdupq_n_f32(s) };
}

IC_FORCEINLINE f32x8 sub(f32x8 a, f32x8 b) {
  return { vsubq_f32(a.lo, b.lo), vsubq_f32(a.hi, b.hi) };
}

IC_FORCEINLINE f32x8 mul(f32x8 a, f32x8 b) {
  return { vmulq_f32(a.lo, b.lo), vmulq_f32(a.hi, b.hi) };
}

IC_FORCEINLINE f32x8 div(f32x8 a, f32x8 b) {
  return { vdivq_f32(a.lo, b.lo), vdivq_f32(a.hi, b.hi) };
}

// acc + a * b
IC_FORCEINLINE f32x8 fma(f32x8 acc, f32x8 a, f32x8 b) {
  return { vfmaq_f32(acc.lo, a.lo, b.lo), vfmaq_f32(acc.hi, a.hi, b.hi) };
}

// All-ones lanes where a < b, as a float-typed mask for select().
IC_FORCEINLINE f32x8 lt(f32x8 a, f32x8 b) {
  return { vreinterpretq_f32_u32(vcltq_f32(a.lo, b.lo)), vreinterpretq_f32_u32(vcltq_f32(a.hi, b.hi)) };
}

// mask ? a : b, per lane.
IC_FORCEINLINE f32x8 select(f32x8 mask, f32x8 a, f32x8 b) {
  return { vbslq_f32(vreinterpretq_u32_f32(mask.lo), a.lo, b.lo), vbslq_f32(vreinterpretq_u32_f32(mask.hi), a.hi, b.hi) };
}

// Unbiased exponent of positive normal floats, as floats.
IC_FORCEINLINE float32x4_t exponent4(float32x4_t x) {
  return vcvtq_f32_s32(vsubq_s32(vshrq_n_s32(vreinterpretq_s32_f32(x), 23), vdupq_n_s32(127)));
}
IC_FORCEINLINE f32x8 exponent(f32x8 x) {
  return { exponent4(x.lo), exponent4(x.hi) };
}

// Mantissa of positive normal floats, in [1, 2).
IC_FORCEINLINE float32x4_t mantissa4(float32x4_t x) {
  uint32x4_t bits = vreinterpretq_u32_f32(x);
  return vreinterpretq_f32_u32(vorrq_u32(vandq_u32(bits, vdupq_n_u32(0x007FFFFFu)), vdupq_n_u32(0x3F800000u)));
}
IC_FORCEINLINE f32x8 mantissa(f32x8 x) {
  return { mantissa4(x.lo), mantissa4(x.hi) };
}

IC_FORCEINLINE float hsum(f32x8 v) {
  return vaddvq_f32(vaddq_f32(v.lo, v.hi));
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

// acc + vec * scalar - uses VFMADD231PS via _mm256_fmadd_ps(v, scalar, acc).
IC_FORCEINLINE f32x8 fma(f32x8 acc, f32x8 v, float s) {
  return { _mm256_fmadd_ps(v.v, _mm256_set1_ps(s), acc.v) };
}

IC_FORCEINLINE f32x8 set1(float s) {
  return { _mm256_set1_ps(s) };
}

IC_FORCEINLINE f32x8 sub(f32x8 a, f32x8 b) {
  return { _mm256_sub_ps(a.v, b.v) };
}

IC_FORCEINLINE f32x8 mul(f32x8 a, f32x8 b) {
  return { _mm256_mul_ps(a.v, b.v) };
}

IC_FORCEINLINE f32x8 div(f32x8 a, f32x8 b) {
  return { _mm256_div_ps(a.v, b.v) };
}

// acc + a * b
IC_FORCEINLINE f32x8 fma(f32x8 acc, f32x8 a, f32x8 b) {
  return { _mm256_fmadd_ps(a.v, b.v, acc.v) };
}

// All-ones lanes where a < b, as a float-typed mask for select().
IC_FORCEINLINE f32x8 lt(f32x8 a, f32x8 b) {
  return { _mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ) };
}

// mask ? a : b, per lane.
IC_FORCEINLINE f32x8 select(f32x8 mask, f32x8 a, f32x8 b) {
  return { _mm256_blendv_ps(b.v, a.v, mask.v) };
}

// Unbiased exponent of positive normal floats, as floats.
IC_FORCEINLINE f32x8 exponent(f32x8 x) {
  __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(x.v), 23), _mm256_set1_epi32(127));
  return { _mm256_cvtepi32_ps(e) };
}

// Mantissa of positive normal floats, in [1, 2).
IC_FORCEINLINE f32x8 mantissa(f32x8 x) {
  __m256i bits = _mm256_castps_si256(x.v);
  bits = _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000));
  return { _mm256_castsi256_ps(bits) };
}

IC_FORCEINLINE float hsum(f32x8 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v.v), _mm256_extractf128_ps(v.v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

#endif // IC_CPU_*

#if IC_HAS_F32X8
// log2 of positive normal floats: exponent from the bits, then log2 of the
// mantissa m in [1, 2) from the atanh series in s = (m-1)/(m+1), |s| <= 1/3,
// truncated after s^11. Absolute error below 2e-7.
IC_FORCEINLINE f32x8 log2_approx(f32x8 x) {
  const f32x8 one = set1(1.0f);
  const f32x8 m = mantissa(x);
  const f32x8 s = div(sub(m, one), add(m, one));
  const f32x8 s2 = mul(s, s);
  // 2/ln2 * (s + s^3/3 + s^5/5 + s^7/7 + s^9/9 + s^11/11), Horner in s^2.
  f32x8 p = set1(2.8853900817779268f / 11.0f);
  p = fma(set1(2.8853900817779268f / 9.0f), p, s2);
  p = fma(set1(2.8853900817779268f / 7.0f), p, s2);
  p = fma(set1(2.8853900817779268f / 5.0f), p, s2);
  p = fma(set1(2.8853900817779268f / 3.0f), p, s2);
  p = fma(set1(2.8853900817779268f), p, s2);
  return fma(exponent(x), p, s);
}
#endif


////////////////////////////////
// 4-wide double helpers, same shape as f32x8: 2x NEON float64x2 on arm64,
// 1x AVX 256-bit reg on x86_64. Used by the IW-SSIM covariance dot products,
// which need double accumulation (see iwssim_simd_covariance).

#if IC_CPU_ARM64

#define IC_HAS_F64X4 1

struct f64x4 {
  float64x2_t lo;
  float64x2_t hi;
};

IC_FORCEINLINE f64x4 load(const double* p) {
  return { vld1q_f64(p), vld1q_f64(p + 2) };
}

IC_FORCEINLINE f64x4 zero_f64x4() {
  return { vdupq_n_f64(0.0), vdupq_n_f64(0.0) };
}

// acc + a * b
IC_FORCEINLINE f64x4 fma(f64x4 acc, f64x4 a, f64x4 b) {
  return { vfmaq_f64(acc.lo, a.lo, b.lo), vfmaq_f64(acc.hi, a.hi, b.hi) };
}

IC_FORCEINLINE double hsum(f64x4 v) {
  return vaddvq_f64(vaddq_f64(v.lo, v.hi));
}

#elif IC_CPU_X64

#define IC_HAS_F64X4 1

struct f64x4 {
  __m256d v;
};

IC_FORCEINLINE f64x4 load(const double* p) {
  return { _mm256_loadu_pd(p) };
}

IC_FORCEINLINE f64x4 zero_f64x4() {
  return { _mm256_setzero_pd() };
}

IC_FORCEINLINE f64x4 fma(f64x4 acc, f64x4 a, f64x4 b) {
  return { _mm256_fmadd_pd(a.v, b.v, acc.v) };
}

IC_FORCEINLINE double hsum(f64x4 v) {
  __m128d lo = _mm256_castpd256_pd128(v.v);
  __m128d hi = _mm256_extractf128_pd(v.v, 1);
  __m128d s = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

#endif

// Dot product of two double ranges. Two accumulators keep the FMA chain
// short; the tail is scalar.
static double dot_f64(const double* a, const double* b, isize count) {
  isize i = 0;
  double sum = 0.0;
#if IC_HAS_F64X4
  f64x4 acc0 = zero_f64x4();
  f64x4 acc1 = zero_f64x4();
  for (; i + 8 <= count; i += 8) {
    acc0 = fma(acc0, load(a + i), load(b + i));
    acc1 = fma(acc1, load(a + i + 4), load(b + i + 4));
  }
  sum = hsum(acc0) + hsum(acc1);
#endif
  for (; i < count; i++) sum += a[i] * b[i];
  return sum;
}


static void ConvolveHorizontal(const ImageF& in, ImageF* JXL_RESTRICT out, const float* JXL_RESTRICT kernel) {
  const isize w = in.xsize();
  const isize h = in.ysize();
  const int mode = var::ssimu2_blur_wrap_mode;

  // Snapshot the right half of the symmetric kernel into a stack array.
  // With just JXL_RESTRICT the compiler was reloading kernel[r+i] every
  // iteration, measured ~5% slower. R constexpr lets the i loops unroll.
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
  ic_assert(out_xsize == (in.xsize() + fx - 1) / fx);
  ic_assert(out_ysize == (in.ysize() + fy - 1) / fy);
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

static void MultiplyPlane(const ImageF &a, const ImageF &b, ImageF *mul) {
  JXL_PROFILE_FUNC
  ic_assert(a.xsize() == b.xsize() && a.ysize() == b.ysize());
  ic_assert(a.xsize() == mul->xsize() && a.ysize() == mul->ysize());
  for (size_t y = 0; y < a.ysize(); ++y) {
    const float *JXL_RESTRICT in1 = a.Row(y);
    const float *JXL_RESTRICT in2 = b.Row(y);
    float *JXL_RESTRICT out = mul->Row(y);
    for (size_t x = 0; x < a.xsize(); ++x) {
      out[x] = in1[x] * in2[x];
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
      ic_assert(isfinite(value));
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
  ic_assert(c >= 0 && c < 3);
  ic_assert(scale >= 0 && scale < 6);
  ic_assert(map >= 0 && map < 3);
  ic_assert(norm >= 0 && norm < 2);
  int idx = 36 * c + 6 * scale + 3 * norm + map;
  ic_assert(idx >= 0 && idx < 108);
  return weight[idx];
}


static void SSIMMap(const Image3F &m1, const Image3F &m2, const Image3F &s11,
                    const Image3F &s22, const Image3F &s12, double* JXL_RESTRICT plane_averages, ImageF* error_map, int scale,
                    const bool* JXL_RESTRICT active) {
  JXL_PROFILE_FUNC
  static const float kC2 = 0.0009f;
  const double onePerPixels = 1.0 / (m1.ysize() * m1.xsize());
  for (int c = 0; c < 3; ++c) {
    // Pruned: weights are below threshold, so s11/s22/s12/mu for this plane
    // were never computed. Averages stay 0 (Msssim is zero-initialized).
    if (!active[c]) continue;
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
          ic_assert(isfinite(error_row[x]));
        }
      }
    }
    plane_averages[c * 2] = onePerPixels * sum1[0];
    plane_averages[c * 2 + 1] = sqrt(sqrt(onePerPixels * sum1[1]));
  }
}

static void EdgeDiffMap(const Image3F &img1, const Image3F &mu1, const Image3F &img2,
                        const Image3F &mu2, double *plane_averages, ImageF* error_map, int scale,
                        const bool* JXL_RESTRICT active) {
  JXL_PROFILE_FUNC
  const double onePerPixels = 1.0 / (img1.ysize() * img1.xsize());
  for (int c = 0; c < 3; ++c) {
    // Pruned: mu for this plane was never computed. Averages stay 0.
    if (!active[c]) continue;
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
          ic_assert(isfinite(error_row[x]));
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
  // Ceil division so odd dimensions keep their last row/column, matching
  // Downsample()'s size invariant.
  Image3F ds_orig((w + 1) / 2, (h + 1) / 2, scratch);
  Image3F ds_dist((w + 1) / 2, (h + 1) / 2, scratch);

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

    // Per-component weight pruning for this scale. The SSIM map needs the
    // sigma blurs (+ multiplies); both maps need the mu blurs. A map whose
    // weights (both norms) are <= threshold contributes nothing, so we skip
    // its work for that component. See ssimu2_prune_threshold.
    const float prune_thr = var::ssimu2_prune_threshold;
    bool ssim_on[3], edge_on[3], mu_on[3];
    for (int c = 0; c < 3; ++c) {
      const float w_ssim   = max(fabsf((float)get_weight(c, scale, 0, 0)), fabsf((float)get_weight(c, scale, 0, 1)));
      const float w_artif  = max(fabsf((float)get_weight(c, scale, 1, 0)), fabsf((float)get_weight(c, scale, 1, 1)));
      const float w_detail = max(fabsf((float)get_weight(c, scale, 2, 0)), fabsf((float)get_weight(c, scale, 2, 1)));
      ssim_on[c] = w_ssim > prune_thr;
      edge_on[c] = (w_artif > prune_thr) || (w_detail > prune_thr);
      mu_on[c]   = ssim_on[c] || edge_on[c];
    }

    for (int c = 0; c < 3; ++c) {
      if (ssim_on[c]) {
        MultiplyPlane(xyb_orig->Plane(c), xyb_orig->Plane(c), &mul.Plane(c));  blur(mul.Plane(c), &sigma1_sq.Plane(c));
        MultiplyPlane(xyb_dist->Plane(c), xyb_dist->Plane(c), &mul.Plane(c));  blur(mul.Plane(c), &sigma2_sq.Plane(c));
        MultiplyPlane(xyb_orig->Plane(c), xyb_dist->Plane(c), &mul.Plane(c));  blur(mul.Plane(c), &sigma12.Plane(c));
      }
      if (mu_on[c]) {
        blur(xyb_orig->Plane(c), &mu1.Plane(c));
        blur(xyb_dist->Plane(c), &mu2.Plane(c));
      }
    }

    ImageF* err = want_error_map ? &error_scale : nullptr;
    SSIMMap(mu1, mu2, sigma1_sq, sigma2_sq, sigma12, msssim.scales[scale].avg_ssim, err, scale, ssim_on);
    EdgeDiffMap(*xyb_orig, mu1, *xyb_dist, mu2, msssim.scales[scale].avg_edgediff, err, scale, edge_on);

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
    const size_t next_xs = (linear_orig->xsize() + 1) / 2;
    const size_t next_ys = (linear_orig->ysize() + 1) / 2;
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


Msssim ComputeSSIMULACRA2(ScratchBuffer& scratch, int w, int h, const unsigned char* orig, const unsigned char* dist, unsigned char* error_map, bool alpha_blend, float background) {
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

      if (alpha_blend) {
        orig_r[x] = lerp(background, r, a);
        orig_g[x] = lerp(background, g, a);
        orig_b[x] = lerp(background, b, a);
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
      
      if (alpha_blend) {
        dist_r[x] = lerp(background, r, a);
        dist_g[x] = lerp(background, g, a);
        dist_b[x] = lerp(background, b, a);
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

double ic_ssimulacra2_score(int w, int h, const unsigned char* orig, const unsigned char* dist, void* scratch_ptr, unsigned char* error_map, bool alpha_blend, float background) {
  // SSIMULACRA2 is undefined below 8x8: no scale of the multi-scale loop runs,
  // so the all-zero Msssim would map to a bogus perfect score of 100. Reject.
  ic_assert(w >= 8 && h >= 8);
  if (w < 8 || h < 8) {
    return NAN;
  }
  ScratchBuffer scratch = { scratch_ptr, ic_ssimulacra2_score_scratch_size(w, h), 0 };
  return ComputeSSIMULACRA2(scratch, w, h, orig, dist, error_map, alpha_blend, background).Score();
}


size_t ic_ssim_score_scratch_size(int w, int h) {
  return usize(w) * usize(h) * sizeof(float) * 9;
}

// Shared core of ic_ssim_score / ic_ssim_score_channel: single-scale SSIM over
// two [0,1] planes the caller has already extracted into img1/img2. Consumes
// the remaining 7 planes of the caller's scratch (blur temp, mu, sigma).
static double ssim_plane_score(int w, int h, ImageF& img1, ImageF& img2, ScratchBuffer& scratch, unsigned char* error_map) {
  // Standard SSIM constants for [0,1] range (Wang et al. 2004).
  // C1 = (K1*L)^2, C2 = (K2*L)^2, with K1=0.01, K2=0.03, L=1.
  static const float kC1 = 0.0001f;
  static const float kC2 = 0.0009f;

  const size_t n = size_t(w) * size_t(h);

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

double ic_ssim_score(int w, int h, const unsigned char* orig, const unsigned char* dist, void* scratch_ptr, unsigned char* error_map) {
  JXL_PROFILE_FUNC

  ScratchBuffer scratch = { scratch_ptr, ic_ssim_score_scratch_size(w, h), 0 };

  ImageF img1(w, h, scratch);
  ImageF img2(w, h, scratch);

  // RGBA8 -> Rec.601 luma in [0,1]. Standard SSIM operates on
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

  return ssim_plane_score(w, h, img1, img2, scratch, error_map);
}

double ic_ssim_score_channel(int w, int h, const unsigned char* orig, const unsigned char* dist, void* scratch_ptr, int channel, unsigned char* error_map) {
  JXL_PROFILE_FUNC

  ic_assert(channel >= 0 && channel < 4);

  ScratchBuffer scratch = { scratch_ptr, ic_ssim_score_scratch_size(w, h), 0 };

  ImageF img1(w, h, scratch);
  ImageF img2(w, h, scratch);

  // One plane of the RGBA8 input, bytes mapped to [0,1]. No luma projection:
  // for non-color data (normal XY, roughness, alpha) mixing channels would let
  // cross-channel errors cancel.
  const size_t n = size_t(w) * size_t(h);

  {
    float* row1 = img1.data;
    float* row2 = img2.data;
    for (size_t i = 0; i < n; i++) {
      row1[i] = float(orig[4*i + channel]) / 255.0f;
      row2[i] = float(dist[4*i + channel]) / 255.0f;
    }
  }

  return ssim_plane_score(w, h, img1, img2, scratch, error_map);
}


// MS-SSIM (Multi-Scale SSIM), Wang/Simoncelli/Bovik 2003. Same Gaussian,
// C1/C2 and Rec.601-luma input as ic_ssim_score. At scale 0, MS-SSIM and
// SSIM use the same per-pixel formula.
//
// At each scale j we compute pixel-wise SSIM = l·cs and CS, take their mean
// over the image, then downsample (2×2 box average) for the next scale.
// Combination follows the paper:
//   MS-SSIM = mean_SSIM_M ^ w_M  *  prod_{j<M} mean_CS_j ^ w_j
// with weights {0.0448, 0.2856, 0.3001, 0.2363, 0.1333} and M = 5. We
// compute in log space and divide by the sum of *used* weights so the
// score still makes sense when small images can't reach all 5 scales.

// 2×2 box-average downsample for a single plane. Safe to use in place.
static void DownsampleAvg2(const ImageF& in, ImageF* out) {
  const size_t out_w = out->xsize();
  const size_t out_h = out->ysize();
  ic_assert(out_w == (in.xsize() + 1) / 2);
  ic_assert(out_h == (in.ysize() + 1) / 2);
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

  // Multi-scale like SSIMULACRA2: below 8x8 no scale runs (weights_used stays
  // 0) and the result collapses to a meaningless 0.0. Reject.
  ic_assert(w >= 8 && h >= 8);
  if (w < 8 || h < 8) {
    return NAN;
  }

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


////////////////////////////////
// IW-SSIM

// IW-SSIM (Information Content Weighted SSIM), Wang/Li 2011. Same luma
// input, Gaussian window and C1/C2 as ic_ssim_score, and the same scale
// weights and log-space combination as ic_msssim_score. What changes:
//
//  - The image is decomposed with a Laplacian pyramid (matlabPyrTools
//    conventions: 5-tap binomial filter, reflected edges). Every scale but
//    the coarsest computes CS on its bandpass band; the coarsest computes
//    full SSIM on the lowpass residual, exactly like the last MS-SSIM scale.
//
//  - At the bandpass scales the CS map is pooled with information-content
//    weights instead of a plain mean. The reference band is modeled as a
//    Gaussian scale mixture: each 3x3 neighborhood (plus the co-located
//    coefficient of the parent band when that band is itself bandpass) is
//    c = s*u with u ~ N(0, C_u). The distortion is d = g*c + v, and
//    perception adds white noise of variance sigma_nsq to both c and d. The
//    weight is the mutual information between (c, d) and their perceived
//    versions, summed over the eigen-components of C_u.
//
// Follows the authors' MATLAB code (via the piq port) for the pyramid, the
// GSM estimate and the weight formula. Deviations, in the spirit of our SSIM
// and MS-SSIM: the SSIM window is the shared 9-tap clamp-to-edge Gaussian
// producing full-size maps (the reference uses an 11x11 'valid' window), the
// parent coefficient is a pixel-centered bilinear upsample, and the weighted
// pooling runs over every pixel whose 3x3 neighborhood is in range.

// matlabPyrTools 'binom5': [1 4 6 4 1]/16 scaled by sqrt(2). Each lowpass
// level therefore carries a gain of 2 over the one above it. The reference
// keeps that gain and so do we: C1/C2 and sigma_nsq are absolute, so the
// published parameters assume it.
static const float kBinom5[5] = { 0.08838834764f, 0.35355339059f, 0.53033008589f, 0.35355339059f, 0.08838834764f };

// Reference parameters are on the 8-bit scale; we run on [0,1] luma, so the
// visual noise variance (0.4) and the zero-variance threshold (float epsilon)
// scale by 1/255^2. The weight formula is scale invariant otherwise.
static const double kIwSigmaNsq = 0.4 / (255.0 * 255.0);
static const double kIwVarEps   = 1.1920929e-7 / (255.0 * 255.0);
static const double kIwWeightEps = 1.1920929e-7;   // weights are in bits, unscaled

static const int kIwMaxN = 10;   // 3x3 neighborhood + parent

// Reflect without repeating the edge sample: [-1] -> [1], [n] -> [n-2].
static inline isize reflect_index(isize i, isize n) {
  if (i < 0) i = -i;
  if (i >= n) i = 2 * n - 2 - i;
  return i;
}

// lo = decimate(binom5 * in). out must be ceil(in/2) along each axis; tmp
// holds out.xsize() * in.ysize() floats.
static void PyrReduce(const ImageF& in, ImageF* out, float* tmp) {
  const isize w = in.xsize();
  const isize h = in.ysize();
  const isize ow = out->xsize();
  const isize oh = out->ysize();
  ic_assert(ow == (w + 1) / 2 && oh == (h + 1) / 2);

  for (isize y = 0; y < h; y++) {
    const float* row = in.Row(y);
    float* t = tmp + y * ow;
    for (isize ox = 0; ox < ow; ox++) {
      const isize ix = 2 * ox;
      float sum = 0.0f;
      for (int k = 0; k < 5; k++) sum += kBinom5[k] * row[reflect_index(ix + k - 2, w)];
      t[ox] = sum;
    }
  }
  for (isize oy = 0; oy < oh; oy++) {
    float* row_out = out->Row(oy);
    const isize iy = 2 * oy;
    for (int k = 0; k < 5; k++) {
      const float* t = tmp + reflect_index(iy + k - 2, h) * ow;
      const float kk = kBinom5[k];
      if (k == 0) for (isize ox = 0; ox < ow; ox++) row_out[ox]  = kk * t[ox];
      else        for (isize ox = 0; ox < ow; ox++) row_out[ox] += kk * t[ox];
    }
  }
}

// One axis of the pyramid expand: zero-stuff lo (n samples) to 2n, filter
// with binom5 using reflected edges on the stuffed signal, keep the first m
// outputs (m = 2n or 2n-1). Written per output parity so the zeros are never
// touched; reflection preserves parity so only even stuffed indices appear.
static inline void expand_line(const float* lo, isize n, float* out, isize m) {
  const isize n2 = 2 * n;
  for (isize i = 0; i < m; i++) {
    isize j = i - 2;
    if (j & 1) j++;
    float sum = 0.0f;
    for (; j <= i + 2; j += 2) {
      sum += kBinom5[j - i + 2] * lo[reflect_index(j, n2) / 2];
    }
    out[i] = sum;
  }
}

// out = expand(lo), with lo = ceil(out/2) along each axis. tmp holds
// out.xsize() * lo.ysize() floats.
static void PyrExpand(const ImageF& lo, ImageF* out, float* tmp) {
  const isize lw = lo.xsize();
  const isize lh = lo.ysize();
  const isize w = out->xsize();
  const isize h = out->ysize();
  ic_assert(lw == (w + 1) / 2 && lh == (h + 1) / 2);

  for (isize y = 0; y < lh; y++) {
    expand_line(lo.Row(y), lw, tmp + y * w, w);
  }
  const isize h2 = 2 * lh;
  for (isize i = 0; i < h; i++) {
    float* row_out = out->Row(i);
    isize j = i - 2;
    if (j & 1) j++;
    bool first = true;
    for (; j <= i + 2; j += 2) {
      const float kk = kBinom5[j - i + 2];
      const float* t = tmp + (reflect_index(j, h2) / 2) * w;
      if (first) for (isize x = 0; x < w; x++) row_out[x]  = kk * t[x];
      else       for (isize x = 0; x < w; x++) row_out[x] += kk * t[x];
      first = false;
    }
  }
}

// Reference parent coefficient for child pixel (x, y): pixel-centered
// bilinear sample of the parent band at (x/2 - 1/4, y/2 - 1/4), clamped.
static inline float parent_sample(const ImageF& p, isize x, isize y) {
  const isize pw = p.xsize();
  const isize ph = p.ysize();
  const isize x0 = (x - 1) >> 1;
  const isize y0 = (y - 1) >> 1;
  const float fx = (x & 1) ? 0.25f : 0.75f;
  const float fy = (y & 1) ? 0.25f : 0.75f;
  const isize xa = clamp(x0, (isize)0, pw - 1);
  const isize xb = clamp(x0 + 1, (isize)0, pw - 1);
  const isize ya = clamp(y0, (isize)0, ph - 1);
  const isize yb = clamp(y0 + 1, (isize)0, ph - 1);
  const float* ra = p.Row(ya);
  const float* rb = p.Row(yb);
  return lerp(lerp(ra[xa], ra[xb], fx), lerp(rb[xa], rb[xb], fx), fy);
}

// Neighborhood vector of the GSM model: the 3x3 reference neighborhood in
// row-major order, then the parent coefficient if there is one.
static inline void iw_gather(const ImageF& ref, const ImageF* parent, isize x, isize y, float* v) {
  int k = 0;
  for (isize dy = -1; dy <= 1; dy++) {
    const float* row = ref.Row(y + dy);
    for (isize dx = -1; dx <= 1; dx++) v[k++] = row[x + dx];
  }
  if (parent) v[k] = parent_sample(*parent, x, y);
}

// Cyclic Jacobi eigen-decomposition of a symmetric n x n matrix, n <= kIwMaxN.
// a is destroyed. On return eigvals[k] and column k of q are the k-th pair.
static void jacobi_eigen(double* a, int n, double* eigvals, double* q) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) q[i * n + j] = (i == j) ? 1.0 : 0.0;
  }
  for (int sweep = 0; sweep < 50; sweep++) {
    double off = 0.0, diag = 0.0;
    for (int i = 0; i < n; i++) {
      diag += a[i * n + i] * a[i * n + i];
      for (int j = i + 1; j < n; j++) off += a[i * n + j] * a[i * n + j];
    }
    if (off <= 1e-30 * diag) break;

    for (int p = 0; p < n; p++) {
      for (int r = p + 1; r < n; r++) {
        const double apr = a[p * n + r];
        if (apr == 0.0) continue;
        const double theta = (a[r * n + r] - a[p * n + p]) / (2.0 * apr);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
        const double c = 1.0 / sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < n; k++) {   // columns p, r
          const double akp = a[k * n + p], akr = a[k * n + r];
          a[k * n + p] = c * akp - s * akr;
          a[k * n + r] = s * akp + c * akr;
        }
        for (int k = 0; k < n; k++) {   // rows p, r
          const double apk = a[p * n + k], ark = a[r * n + k];
          a[p * n + k] = c * apk - s * ark;
          a[r * n + k] = s * apk + c * ark;
        }
        for (int k = 0; k < n; k++) {
          const double qkp = q[k * n + p], qkr = q[k * n + r];
          q[k * n + p] = c * qkp - s * qkr;
          q[k * n + r] = s * qkp + c * qkr;
        }
      }
    }
  }
  for (int k = 0; k < n; k++) eigvals[k] = a[k * n + k];
}

// CS term of SSIM from the blurred moments, with the luminance term as a
// side output. C3 = C2/2 as in ic_msssim_score.
static inline float ssim_cs_terms(float m1, float m2, float p11, float p22, float p12, float c1, float c2, float* l) {
  const float m1m2 = m1 * m2;
  const float m1sq = m1 * m1;
  const float m2sq = m2 * m2;
  const float s1sq = p11 - m1sq;
  const float s2sq = p22 - m2sq;
  const float s12  = p12 - m1m2;
  *l = (2.0f * m1m2 + c1) / (m1sq + m2sq + c1);
  return (2.0f * s12 + c2) / (s1sq + s2sq + c2);
}

// Information-content-weighted mean of the CS map at one bandpass scale.
// ref/dst are the reference/distorted bands, parent the reference band one
// scale up (or null). CS is evaluated on the fly from the SSIM moments.
//
// Per-pixel math is float: everything is scale invariant and built from
// positive terms, and the block variances use the two-pass form, so float
// loses nothing measurable (|delta| <= 1.1e-7 vs an all-double version over
// our test set, flat and near-flat crops included). The pooled totals are
// accumulated per row and flushed to double; the covariance and the small
// eigen-solve run in double.
// row_scratch holds 4 * ref.xsize() doubles for the SIMD covariance path.
static double iw_pooled_cs(const ImageF& ref, const ImageF& dst, const ImageF* parent,
                           const ImageF& mu1, const ImageF& mu2, const ImageF& p11, const ImageF& p22, const ImageF& p12,
                           float c1, float c2, double* row_scratch) {
  const isize w = ref.xsize();
  const isize h = ref.ysize();
  const int n = parent ? kIwMaxN : kIwMaxN - 1;
  ic_assert(w >= 3 && h >= 3);

  // Covariance of the reference neighborhood vectors. Always in double: the
  // off-diagonal entries cancel heavily, and the smallest eigenvalues (which
  // the s^2 estimate divides by) sit well below float rounding of the sums.
  double cov[kIwMaxN * kIwMaxN] = {};
  if (var::iwssim_simd_covariance) {
    // Entry (i, j) is the dot product of two shifted rows of the reference
    // band (or the upsampled parent row), summed over interior rows. Rows
    // y-1, y, y+1 live in a sliding window of double copies; one new row is
    // converted per y.
    double* drow[3] = { row_scratch, row_scratch + w, row_scratch + 2 * w };
    double* prow = row_scratch + 3 * w;
    for (isize y = 0; y < 2; y++) {
      const float* src = ref.Row(y);
      for (isize x = 0; x < w; x++) drow[y][x] = src[x];
    }
    const isize count = w - 2;
    for (isize y = 1; y < h - 1; y++) {
      {
        double* d = drow[(y + 1) % 3];
        const float* src = ref.Row(y + 1);
        for (isize x = 0; x < w; x++) d[x] = src[x];
      }
      if (parent) {
        for (isize x = 1; x < w - 1; x++) prow[x] = parent_sample(*parent, x, y);
      }
      // Neighbor k = (dy + 1) * 3 + (dx + 1), same order as iw_gather.
      const double* nb[kIwMaxN];
      for (int k = 0; k < 9; k++) nb[k] = drow[(y + k / 3 - 1 + 3) % 3] + 1 + (k % 3 - 1);
      if (parent) nb[9] = prow + 1;
      for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) cov[i * n + j] += dot_f64(nb[i], nb[j], count);
      }
    }
  }
  else {
    for (isize y = 1; y < h - 1; y++) {
      for (isize x = 1; x < w - 1; x++) {
        float v[kIwMaxN];
        iw_gather(ref, parent, x, y, v);
        for (int i = 0; i < n; i++) {
          for (int j = i; j < n; j++) cov[i * n + j] += double(v[i]) * double(v[j]);
        }
      }
    }
  }
  const double inv_count = 1.0 / double((w - 2) * (h - 2));
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      cov[i * n + j] *= inv_count;
      cov[j * n + i] = cov[i * n + j];
    }
  }

  double eig[kIwMaxN], q[kIwMaxN * kIwMaxN];
  jacobi_eigen(cov, n, eig, q);

  // Negative eigenvalues are rounding noise: zero them and rescale the rest
  // so the trace is preserved, as the reference does. Components with no
  // variance carry no signal, so they drop out of the s^2 estimate instead
  // of dividing by zero.
  double sum_all = 0.0, sum_pos = 0.0;
  for (int k = 0; k < n; k++) {
    sum_all += eig[k];
    if (eig[k] > 0.0) sum_pos += eig[k];
  }
  const double rescale = sum_pos > 0.0 ? sum_all / sum_pos : 0.0;
  float eig_f[kIwMaxN], inv_eig[kIwMaxN], q_f[kIwMaxN * kIwMaxN];
  double inv_eig_d[kIwMaxN];
  double coef[kIwMaxN + 1] = { 1.0 };   // prod_k (1 + t*eig_k) = sum_j coef[j] t^j
  for (int k = 0; k < n; k++) {
    const double e = eig[k] > 0.0 ? eig[k] * rescale : 0.0;
    eig_f[k] = float(e);
    inv_eig_d[k] = e > 1e-12 * sum_all ? 1.0 / e : 0.0;
    inv_eig[k] = float(inv_eig_d[k]);
    for (int i = 0; i < n; i++) q_f[i * n + k] = float(q[i * n + k]);
    for (int j = k + 1; j >= 1; j--) coef[j] += e * coef[j - 1];
  }

  // Upper-triangular R with R^T R = C_u^-1 = Q diag(inv_eig) Q^T, so that
  // v^T C_u^-1 v = |R v|^2. Cholesky in double; dropped eigen-components make
  // the matrix semi-definite, so a vanishing pivot zeroes its row.
  float r_f[kIwMaxN * kIwMaxN] = {};
  if (var::iwssim_cholesky_projection) {
    double a[kIwMaxN * kIwMaxN];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) sum += q[i * n + k] * inv_eig_d[k] * q[j * n + k];
        a[i * n + j] = sum;
      }
    }
    double r[kIwMaxN * kIwMaxN] = {};
    for (int i = 0; i < n; i++) {
      double d = a[i * n + i];
      for (int k = 0; k < i; k++) d -= r[k * n + i] * r[k * n + i];
      if (d <= 1e-12 * a[i * n + i]) continue;   // row stays zero
      const double rii = sqrt(d);
      r[i * n + i] = rii;
      for (int j = i + 1; j < n; j++) {
        double sum = a[i * n + j];
        for (int k = 0; k < i; k++) sum -= r[k * n + i] * r[k * n + j];
        r[i * n + j] = sum / rii;
      }
    }
    for (int i = 0; i < n * n; i++) r_f[i] = float(r[i]);
  }

  const float sigma_nsq = float(kIwSigmaNsq);
  const double nsq2 = kIwSigmaNsq * kIwSigmaNsq;
  const float inv_nsq2 = float(1.0 / nsq2);
  const float var_eps = float(kIwVarEps);
  const float weight_eps = float(kIwWeightEps);

  double wcs_sum = 0.0, w_sum = 0.0, cs_sum = 0.0;
  for (isize y = 1; y < h - 1; y++) {
    float row_wcs = 0.0f, row_w = 0.0f, row_cs = 0.0f;
    isize x = 1;

#if IC_HAS_F32X8
    if (var::iwssim_simd_weights && w - 2 >= 8) {
      // Same math as the scalar loop below, eight pixels per step. The
      // parent coefficient is upsampled once per row (the covariance pass
      // is done with row_scratch by now).
      float* prow_f = (float*)row_scratch;
      if (parent) {
        for (isize px = 1; px < w - 1; px++) prow_f[px] = parent_sample(*parent, px, y);
      }
      const f32x8 zero = set1(0.0f);
      const f32x8 one = set1(1.0f);
      const f32x8 veps = set1(var_eps);
      const f32x8 vsigma = set1(sigma_nsq);
      f32x8 acc_wcs = zero, acc_w = zero, acc_cs = zero;

      for (; x + 8 <= w - 1; x += 8) {
        f32x8 v[kIwMaxN];
        f32x8 d[9];
        for (int k = 0; k < 9; k++) {
          const isize dy = k / 3 - 1, dx = k % 3 - 1;
          v[k] = load(ref.Row(y + dy) + x + dx);
          d[k] = load(dst.Row(y + dy) + x + dx);
        }
        if (parent) v[9] = load(prow_f + x);

        // s^2 = |R v|^2 / n.
        f32x8 ss = zero;
        for (int i = 0; i < n; i++) {
          if (r_f[i * n + i] == 0.0f) continue;
          f32x8 acc = mul(v[i], r_f[i * n + i]);
          for (int j = i + 1; j < n; j++) acc = fma(acc, v[j], r_f[i * n + j]);
          ss = fma(ss, acc, acc);
        }
        ss = mul(ss, 1.0f / float(n));

        // Two-pass block statistics.
        f32x8 mr = v[0], md = d[0];
        for (int k = 1; k < 9; k++) { mr = add(mr, v[k]); md = add(md, d[k]); }
        mr = mul(mr, 1.0f / 9.0f);
        md = mul(md, 1.0f / 9.0f);
        f32x8 var_r = zero, var_d = zero, cov_rd = zero;
        for (int k = 0; k < 9; k++) {
          const f32x8 a = sub(v[k], mr), b = sub(d[k], md);
          var_r = fma(var_r, a, a);
          var_d = fma(var_d, b, b);
          cov_rd = fma(cov_rd, a, b);
        }
        var_r = mul(var_r, 1.0f / 9.0f);
        var_d = mul(var_d, 1.0f / 9.0f);
        cov_rd = mul(cov_rd, 1.0f / 9.0f);
        f32x8 g = div(cov_rd, add(var_r, veps));
        f32x8 vv = sub(var_d, mul(g, cov_rd));
        const f32x8 r_flat = lt(var_r, veps);
        g = select(r_flat, zero, g);
        vv = select(r_flat, var_d, vv);
        const f32x8 d_flat = lt(var_d, veps);
        g = select(d_flat, zero, g);
        vv = select(d_flat, zero, vv);

        // Weight: sum_k log2(1 + x_k) as the sum of the factors' exponents
        // plus log2 of the product of their mantissas (ten values in
        // [1, 2), so the product stays below 1024).
        const f32x8 a = mul(add(vv, mul(add(mul(g, g), one), sigma_nsq)), ss);
        const f32x8 b = mul(vv, vsigma);
        f32x8 esum = zero, mprod = one;
        for (int k = 0; k < n; k++) {
          const f32x8 term = add(one, mul(fma(b, a, eig_f[k]), inv_nsq2));
          esum = add(esum, exponent(term));
          mprod = mul(mprod, mantissa(term));
        }
        f32x8 wgt = add(esum, log2_approx(mprod));
        wgt = select(lt(wgt, set1(weight_eps)), zero, wgt);

        // CS from the blurred moments.
        const isize i0 = y * w + x;
        const f32x8 m1 = load(mu1.data + i0), m2 = load(mu2.data + i0);
        const f32x8 s1sq = sub(load(p11.data + i0), mul(m1, m1));
        const f32x8 s2sq = sub(load(p22.data + i0), mul(m2, m2));
        const f32x8 s12  = sub(load(p12.data + i0), mul(m1, m2));
        const f32x8 cs = div(fma(set1(c2), s12, 2.0f), add(add(s1sq, s2sq), set1(c2)));

        acc_wcs = fma(acc_wcs, wgt, cs);
        acc_w = add(acc_w, wgt);
        acc_cs = add(acc_cs, cs);
      }
      row_wcs += hsum(acc_wcs);
      row_w   += hsum(acc_w);
      row_cs  += hsum(acc_cs);
    }
#endif

    for (; x < w - 1; x++) {
      float v[kIwMaxN];
      iw_gather(ref, parent, x, y, v);

      // GSM multiplier: s^2 = v^T C_u^-1 v / n.
      float ss = 0.0f;
      if (var::iwssim_cholesky_projection) {
        for (int i = 0; i < n; i++) {
          if (r_f[i * n + i] == 0.0f) continue;   // dropped component, whole row is zero
          float acc = 0.0f;
          for (int j = i; j < n; j++) acc += r_f[i * n + j] * v[j];
          ss += acc * acc;
        }
      }
      else {
        for (int k = 0; k < n; k++) {
          if (inv_eig[k] == 0.0f) continue;
          float proj = 0.0f;
          for (int i = 0; i < n; i++) proj += q_f[i * n + k] * v[i];
          ss += proj * proj * inv_eig[k];
        }
      }
      ss /= float(n);

      // Distortion channel d = g*c + v over the 3x3 block: g by regression of
      // the distorted band on the reference, vv the residual variance. Two
      // passes (means, then centered moments) so there is no cancellation.
      float r[9], d[9];
      float mr = 0.0f, md = 0.0f;
      for (isize dy = -1; dy <= 1; dy++) {
        const float* rr = ref.Row(y + dy) + x;
        const float* rd = dst.Row(y + dy) + x;
        for (isize dx = -1; dx <= 1; dx++) {
          const int k = int((dy + 1) * 3 + (dx + 1));
          r[k] = rr[dx]; d[k] = rd[dx];
          mr += r[k]; md += d[k];
        }
      }
      mr /= 9.0f; md /= 9.0f;
      float var_r = 0.0f, var_d = 0.0f, cov_rd = 0.0f;
      for (int k = 0; k < 9; k++) {
        const float a = r[k] - mr, b = d[k] - md;
        var_r += a * a; var_d += b * b; cov_rd += a * b;
      }
      var_r /= 9.0f; var_d /= 9.0f; cov_rd /= 9.0f;
      float g, vv;
      if (var_r < var_eps) {
        g = 0.0f;
        vv = var_d;
      }
      else {
        g = cov_rd / (var_r + var_eps);
        vv = var_d - g * cov_rd;
      }
      if (var_d < var_eps) {
        g = 0.0f;
        vv = 0.0f;
      }

      // Mutual information between (c, d) and their noisy perceived versions,
      // summed over eigen-components.
      const float a = (vv + (1.0f + g * g) * sigma_nsq) * ss;
      const float b = sigma_nsq * vv;
      float wgt;
      if (var::iwssim_log_product) {
        // sum_k log2(1 + (a*eig_k + b)/s4) = log2((c/s4)^n * prod_k (1 + (a/c)*eig_k))
        // with c = s4 + b, and the product is the precomputed polynomial in
        // t = a/c. Everything is positive, so no cancellation anywhere.
        const double c = double(b) + nsq2;
        const double t = double(a) / c;
        double poly = coef[n];
        for (int j = n - 1; j >= 0; j--) poly = poly * t + coef[j];
        const double cs4 = c * double(inv_nsq2);   // >= 1
        double cs4n = cs4;
        for (int k = 1; k < n; k++) cs4n *= cs4;
        wgt = float(log2(cs4n * poly));
      }
      else {
        wgt = 0.0f;
        for (int k = 0; k < n; k++) wgt += log2f(1.0f + (a * eig_f[k] + b) * inv_nsq2);
      }
      if (wgt < weight_eps) wgt = 0.0f;

      const isize i = y * w + x;
      float l;
      const float cs = ssim_cs_terms(mu1.data[i], mu2.data[i], p11.data[i], p22.data[i], p12.data[i], c1, c2, &l);

      row_wcs += wgt * cs;
      row_w   += wgt;
      row_cs  += cs;
    }
    wcs_sum += row_wcs;
    w_sum   += row_w;
    cs_sum  += row_cs;
  }

  // No information anywhere (flat reference and no added noise): fall back to
  // the plain mean so we don't divide by zero.
  if (w_sum <= 0.0) return cs_sum * inv_count;
  return wcs_sum / w_sum;
}

static inline void swap_images(ImageF* a, ImageF* b) {
  ImageF t = a[0]; a[0] = b[0]; b[0] = t;
  t = a[1]; a[1] = b[1]; b[1] = t;
}

size_t ic_iwssim_score_scratch_size(int w, int h) {
  // Full-size planes as in ic_msssim_score: lo/band pair, mu1, mu2, tmp,
  // sigma1_sq, sigma2_sq, sigma12, blur.temp = 9. Plus the two pyramid levels
  // below the current one for both images: lowpass and bandpass pairs at
  // half size and a lowpass pair at quarter size. Coarser scales reuse the
  // same buffers via ShrinkTo.
  // Plus four double rows (three reference rows and the parent row) for the
  // covariance dot products, with slack to align them.
  const usize full = usize(w) * usize(h);
  const usize w2 = (usize(w) + 1) / 2, h2 = (usize(h) + 1) / 2;
  const usize w4 = (w2 + 1) / 2, h4 = (h2 + 1) / 2;
  return (9 * full + 4 * w2 * h2 + 2 * w4 * h4) * sizeof(float) + 4 * usize(w) * sizeof(double) + 8;
}

double ic_iwssim_score(int w, int h, const unsigned char* orig, const unsigned char* dist, void* scratch_ptr, unsigned char* error_map) {
  JXL_PROFILE_FUNC

  // Multi-scale: same minimum as ic_msssim_score.
  ic_assert(w >= 8 && h >= 8);
  if (w < 8 || h < 8) {
    return NAN;
  }

  ScratchBuffer scratch = { scratch_ptr, ic_iwssim_score_scratch_size(w, h), 0 };

  // Same constants and scale weights as ic_msssim_score.
  static const float kC1 = 0.0001f;
  static const float kC2 = 0.0009f;
  static const double kWeights[5] = { 0.0448, 0.2856, 0.3001, 0.2363, 0.1333 };
  static constexpr int kScales = 5;

  const size_t w2 = (size_t(w) + 1) / 2, h2 = (size_t(h) + 1) / 2;
  const size_t w4 = (w2 + 1) / 2, h4 = (h2 + 1) / 2;

  // Pyramid state for the current scale j, [0] = orig, [1] = dist. Each
  // descent swaps buffer pairs instead of copying, so a pair's capacity is
  // whatever it was allocated with; sizes only shrink so that always fits.
  ImageF cur[2]       = { ImageF(w, h, scratch),   ImageF(w, h, scratch) };   // lowpass j, then band j
  ImageF next[2]      = { ImageF(w2, h2, scratch), ImageF(w2, h2, scratch) }; // lowpass j+1
  ImageF band_next[2] = { ImageF(w2, h2, scratch), ImageF(w2, h2, scratch) }; // band j+1 (the parent)
  ImageF next2[2]     = { ImageF(w4, h4, scratch), ImageF(w4, h4, scratch) }; // lowpass j+2

  // RGBA8 -> Rec.601 luma in [0,1]. Same path as ic_ssim_score.
  constexpr float kR = 0.299f / 255.0f;
  constexpr float kG = 0.587f / 255.0f;
  constexpr float kB = 0.114f / 255.0f;
  {
    const size_t n0 = size_t(w) * size_t(h);
    float* row1 = cur[0].data;
    float* row2 = cur[1].data;
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

  // Double rows for the covariance pass, 8-byte aligned (planes are only
  // float aligned).
  double* row_scratch;
  {
    uintptr_t p = (uintptr_t)scratch.allocate(4 * size_t(w) * sizeof(double) + 8);
    row_scratch = (double*)((p + 7) & ~uintptr_t(7));
  }

  double log_iwssim = 0.0;
  double weights_used = 0.0;

  for (int j = 0; j < kScales; j++) {
    const size_t sx = cur[0].xsize();
    const size_t sy = cur[0].ysize();
    const size_t nx = (sx + 1) / 2;
    const size_t ny = (sy + 1) / 2;
    const size_t n = sx * sy;

    // Same rule as ic_msssim_score: the coarsest scale is the last one whose
    // successor would be too small.
    const bool is_last_scale = (j == kScales - 1) || nx < 8 || ny < 8;

    blur     .ShrinkTo(sx, sy);
    mu1      .ShrinkTo(sx, sy);
    mu2      .ShrinkTo(sx, sy);
    tmp      .ShrinkTo(sx, sy);
    sigma1_sq.ShrinkTo(sx, sy);
    sigma2_sq.ShrinkTo(sx, sy);
    sigma12  .ShrinkTo(sx, sy);

    const ImageF* parent = nullptr;
    bool next_is_last = true;

    if (!is_last_scale) {
      // band j = lowpass j - expand(lowpass j+1), in place. Scale 0 has to
      // build lowpass 1 first; every other scale inherited it from above.
      for (int c = 0; c < 2; c++) {
        next[c].ShrinkTo(nx, ny);
        if (j == 0) PyrReduce(cur[c], &next[c], tmp.data);
        PyrExpand(next[c], &mu1, tmp.data);   // mu1 is free until the blur below
        float* dst = cur[c].data;
        const float* src = mu1.data;
        for (size_t i = 0; i < n; i++) dst[i] -= src[i];
      }

      // When scale j+1 is bandpass too, build its band now: it is this
      // scale's parent and the next scale's input. When it is the lowpass
      // residual it is not used as a parent, as in the reference.
      const size_t nnx = (nx + 1) / 2;
      const size_t nny = (ny + 1) / 2;
      next_is_last = (j + 1 == kScales - 1) || nnx < 8 || nny < 8;
      if (!next_is_last) {
        for (int c = 0; c < 2; c++) {
          next2[c].ShrinkTo(nnx, nny);
          PyrReduce(next[c], &next2[c], tmp.data);
          band_next[c].ShrinkTo(nx, ny);
          PyrExpand(next2[c], &band_next[c], tmp.data);
          float* dst = band_next[c].data;
          const float* src = next[c].data;
          for (size_t i = 0; i < nx * ny; i++) dst[i] = src[i] - dst[i];
        }
        parent = &band_next[0];
      }
    }

    // SSIM moments of the current band (or lowpass residual), as in ic_msssim_score.
    blur(cur[0], &mu1);
    blur(cur[1], &mu2);
    {
      const float* p1 = cur[0].data;
      const float* p2 = cur[1].data;
      float* pt = tmp.data;
      for (size_t i = 0; i < n; i++) pt[i] = p1[i] * p1[i];
      blur(tmp, &sigma1_sq);
      for (size_t i = 0; i < n; i++) pt[i] = p2[i] * p2[i];
      blur(tmp, &sigma2_sq);
      for (size_t i = 0; i < n; i++) pt[i] = p1[i] * p2[i];
      blur(tmp, &sigma12);
    }

    double contribution;
    if (is_last_scale) {
      // Coarsest scale: plain mean of l*cs on the lowpass residual, uniform
      // weights, as in the reference.
      double ssim_sum = 0.0;
      uint32_t* err_out = (error_map != nullptr && j == 0) ? (uint32_t*)error_map : nullptr;
      for (size_t i = 0; i < n; i++) {
        float l;
        const float cs = ssim_cs_terms(mu1.data[i], mu2.data[i], sigma1_sq.data[i], sigma2_sq.data[i], sigma12.data[i], kC1, kC2, &l);
        const float ssim = l * cs;
        ssim_sum += ssim;
        if (err_out) {
          int value = int(255 * clamp(1.0f - ssim, 0.0f, 1.0f));
          err_out[i] = MagmaMap[value];
        }
      }
      contribution = ssim_sum / double(n);
    }
    else {
      if (error_map != nullptr && j == 0) {
        // Finest scale error map, same 1-SSIM mapping as ic_msssim_score. The
        // band is near zero-mean so this is essentially 1-CS.
        uint32_t* err_out = (uint32_t*)error_map;
        for (size_t i = 0; i < n; i++) {
          float l;
          const float cs = ssim_cs_terms(mu1.data[i], mu2.data[i], sigma1_sq.data[i], sigma2_sq.data[i], sigma12.data[i], kC1, kC2, &l);
          int value = int(255 * clamp(1.0f - l * cs, 0.0f, 1.0f));
          err_out[i] = MagmaMap[value];
        }
      }
      contribution = iw_pooled_cs(cur[0], cur[1], parent, mu1, mu2, sigma1_sq, sigma2_sq, sigma12, kC1, kC2, row_scratch);
    }

    if (contribution < 1e-10) contribution = 1e-10;  // guard the log
    log_iwssim += kWeights[j] * log(contribution);
    weights_used += kWeights[j];

    if (is_last_scale) break;

    // Descend. Buffers swap pairwise so nothing is copied.
    if (!next_is_last) {
      swap_images(cur, band_next);   // band j+1 becomes the current band
      swap_images(next, next2);      // lowpass j+2 becomes the next lowpass
    }
    else {
      swap_images(cur, next);        // the residual runs full SSIM on the lowpass
    }
  }

  if (weights_used <= 0.0) return 0.0;
  return exp(log_iwssim / weights_used);
}
