// Copyright (c) Ignacio Castaño <castano@gmail.com>
// Copyright (c) Jon Sneyers, Cloudinary. All rights reserved.
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

#include "ic_ssimulacra2.h"

#include <stddef.h> // size_t
#include <stdlib.h> // malloc
#include <string.h> // memcpy
#include <stdio.h> // printf
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

IC_VAR_BOOL(ssimu2_alpha_blend, false);
IC_VAR_BOOL(ssimu2_slow_blur, true);
IC_VAR_BOOL(ssimu2_fast_blur, true);


#define JXL_RESTRICT __restrict


namespace {

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

template <typename T> inline constexpr T max(const T & a, const T & b) {
  return (b < a) ? a : b;
}

template <typename T> inline constexpr T min(const T & a, const T & b) {
  return (a < b) ? a : b;
}

template <typename T> inline constexpr T clamp(const T & x, const T & lo, const T & hi) {
  return max(min(x, hi), lo);
}

template <typename T> inline constexpr T abs(T f) {
  return f < 0 ? -f : f;
}

template <typename T> inline constexpr T lerp(T a, T b, T t) {
    //return a + t * (b - a);
    return a * (1 - t) + b * t;
}


struct ImageF {
  ImageF() {}
  ImageF(size_t xs, size_t ys) : xs(xs), ys(ys), orig_xs(xs), orig_ys(ys) {
    data = (float*)malloc(xs * ys * sizeof(float));
  }
  ImageF(ImageF&& other) noexcept : xs(other.xs), ys(other.ys), orig_xs(other.orig_xs), orig_ys(other.orig_ys), data(other.data) {
    other.data = nullptr; // avoid double delete
    other.xs = other.ys = other.orig_xs = other.orig_ys = 0;
  }
  ~ImageF() {
    free(data);
  }
  void operator=(ImageF&& other) noexcept {
      if (this != &other) {
          delete[] data; // free old data if any

          xs = other.xs;
          ys = other.ys;
          orig_xs = other.orig_xs;
          orig_ys = other.orig_ys;
          data = other.data;

          other.data = nullptr;
          other.xs = other.ys = other.orig_xs = other.orig_ys = 0;
      }
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

struct Image3F {
  Image3F(size_t xs, size_t ys) {
    for (int i = 0; i < 3; i++)
      planes[i] = ImageF(xs, ys);
  }
  Image3F(Image3F&& other) noexcept {
    for (int i = 0; i < 3; i++)
      planes[i] = static_cast<ImageF&&>(other.planes[i]);
  }
  void operator=(Image3F&& other) noexcept {
    for (int i = 0; i < 3; i++)
      planes[i] = static_cast<ImageF&&>(other.planes[i]);
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

static bool SameSize(const ImageF& a, const ImageF& b) { return a.xsize() == b.xsize() && a.ysize() == b.ysize(); }
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

inline void OpsinAbsorbance(const float r, const float g, const float b,
                            const float* JXL_RESTRICT premul_absorb,
                            float* JXL_RESTRICT mixed0, float* JXL_RESTRICT mixed1, float* JXL_RESTRICT mixed2)
{
  const auto m0 = premul_absorb[0];
  const auto m1 = premul_absorb[1];
  const auto m2 = premul_absorb[2];
  const auto m3 = premul_absorb[3];
  const auto m4 = premul_absorb[4];
  const auto m5 = premul_absorb[5];
  const auto m6 = premul_absorb[6];
  const auto m7 = premul_absorb[7];
  const auto m8 = premul_absorb[8];
  *mixed0 = m0 * r + m1 * g + m2 * b + kOpsinAbsorbanceBias;
  *mixed1 = m3 * r + m4 * g + m5 * b + kOpsinAbsorbanceBias;
  *mixed2 = m6 * r + m7 * g + m8 * b + kOpsinAbsorbanceBias;
}
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

inline void StoreXYB(float r, float g, float b,
                     float* JXL_RESTRICT valx, float* JXL_RESTRICT valy, float* JXL_RESTRICT valz) {
  *valx = (r - g) * 0.5f;
  *valy = (r + g) * 0.5f;
  *valz = b;
}

static void LinearRGBToXYB(float r, float g, float b, const float* JXL_RESTRICT premul_absorb,
                           float* JXL_RESTRICT valx, float* JXL_RESTRICT valy, float* JXL_RESTRICT valz)
{
  float mixed0, mixed1, mixed2;
  OpsinAbsorbance(r, g, b, premul_absorb, &mixed0, &mixed1, &mixed2);

  // mixed* should be non-negative even for wide-gamut, so clamp to zero.
  mixed0 = ZeroIfNegative(mixed0);
  mixed1 = ZeroIfNegative(mixed1);
  mixed2 = ZeroIfNegative(mixed2);

  mixed0 = CubeRootAndAdd(mixed0, premul_absorb[9]);
  mixed1 = CubeRootAndAdd(mixed1, premul_absorb[10]);
  mixed2 = CubeRootAndAdd(mixed2, premul_absorb[11]);
  StoreXYB(mixed0, mixed1, mixed2, valx, valy, valz);
}

static void LinearRGBToXYB(const Image3F& linear,
                           const float* JXL_RESTRICT premul_absorb,
                           Image3F* JXL_RESTRICT xyb)
{
  const size_t xsize = linear.xsize();
  const size_t ysize = linear.ysize();

  for (size_t y = 0; y < ysize; y ++) {
    const float* JXL_RESTRICT row_in0 = linear.ConstPlaneRow(0, y);
    const float* JXL_RESTRICT row_in1 = linear.ConstPlaneRow(1, y);
    const float* JXL_RESTRICT row_in2 = linear.ConstPlaneRow(2, y);
    float* JXL_RESTRICT row_xyb0 = xyb->PlaneRow(0, y);
    float* JXL_RESTRICT row_xyb1 = xyb->PlaneRow(1, y);
    float* JXL_RESTRICT row_xyb2 = xyb->PlaneRow(2, y);

    for (size_t x = 0; x < xsize; x ++) {
      const auto in_r = row_in0[x];
      const auto in_g = row_in1[x];
      const auto in_b = row_in2[x];

      LinearRGBToXYB(in_r, in_g, in_b, premul_absorb, row_xyb0 + x, row_xyb1 + x, row_xyb2 + x);
    }
  }
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

static void ToXYB(const Image3F& src, Image3F* dst) {
  JXL_PROFILE_FUNC
  JXL_CHECK(SameSize(src, *dst));

  // Pre-broadcasted constants
  float premul_absorb[12];
  for (size_t i = 0; i < 9; ++i) {
    //const auto absorb = kOpsinAbsorbanceMatrix[i] * (in.metadata()->IntensityTarget() / 255.0f);
    const float absorb = kOpsinAbsorbanceMatrix[i];
    premul_absorb[i] = absorb;
  }
  const float neg_bias_cbrt = -cbrtf(kOpsinAbsorbanceBias);
  for (size_t i = 0; i < 3; ++i) {
    premul_absorb[9 + i] = neg_bias_cbrt;
  }

  LinearRGBToXYB(src, premul_absorb, dst);
}


#pragma pack(push, 1)
struct RecursiveGaussian {
  RecursiveGaussian(float sigma);

  // For k={1,3,5} in that order, each broadcasted 4x for LoadDup128. Used only
  // for vertical passes.
  //float n2[3 * 4];
  //float d1[3 * 4];

  // We unroll horizontal passes 4x - one output per lane. These are each lane's
  // multiplier for the previous output (relative to the first of the four
  // outputs). Indexing: 4 * 0..2 (for {1,3,5}) + 0..3 for the lane index.
  float mul_prev[3];
  // Ditto for the second to last output.
  //float mul_prev2[3];

  // We multiply a vector of inputs 0..3 by a vector shifted from this array.
  // in=0 uses all 4 (nonzero) terms; for in=3, the lower three lanes are 0.
  float mul_in[3];

  int radius;

  float kernel[11]; // sigma=15 -> radius=5 -> size=11
};
#pragma pack(pop)

// IC: jxl is using Kramer's rule, but maybe that's not the best approach.
inline bool Inv3x3Matrix(double matrix[9]) {
  // Intermediate computation is done in double precision.
  double temp[9];
  temp[0] = matrix[4] * matrix[8] - matrix[5] * matrix[7];
  temp[1] = matrix[2] * matrix[7] - matrix[1] * matrix[8];
  temp[2] = matrix[1] * matrix[5] - matrix[2] * matrix[4];
  temp[3] = matrix[5] * matrix[6] - matrix[3] * matrix[8];
  temp[4] = matrix[0] * matrix[8] - matrix[2] * matrix[6];
  temp[5] = matrix[2] * matrix[3] - matrix[0] * matrix[5];
  temp[6] = matrix[3] * matrix[7] - matrix[4] * matrix[6];
  temp[7] = matrix[1] * matrix[6] - matrix[0] * matrix[7];
  temp[8] = matrix[0] * matrix[4] - matrix[1] * matrix[3];
  double det = matrix[0] * temp[0] + matrix[1] * temp[3] + matrix[2] * temp[6];
  if (fabs(det) < 1e-10) {
    return false;// JXL_FAILURE("Matrix determinant is too close to 0");
  }
  double idet = 1.0 / det;
  for (int i = 0; i < 9; i++) {
    matrix[i] = temp[i] * idet;
  }
  return true;
}

// Computes A = B * C, with sizes rows*cols: A=ha*wa, B=wa*wb, C=ha*wb
inline void MatMul(const double* a, const double* b, int ha, int wa, int wb, double* c) {
  //std::vector<double> temp(wa);  // Make better use of cache lines
  JXL_CHECK(wa <= 3);
  double temp[3];
  for (int x = 0; x < wb; x++) {
    for (int z = 0; z < wa; z++) {
      temp[z] = b[z * wb + x];
    }
    for (int y = 0; y < ha; y++) {
      double e = 0;
      for (int z = 0; z < wa; z++) {
        e += a[y * wa + z] * temp[z];
      }
      c[y * wb + x] = e;
    }
  }
}


inline void GaussianKernel(int radius, float sigma, float kernel[11]) {
  JXL_CHECK(sigma > 0.0);

  size_t size = 2 * radius + 1;
  JXL_CHECK(size <= 11);

  const float scaler = -1.0f / (2.0f * sigma * sigma);
  float sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const float val = expf(scaler * i * i);
    kernel[i + radius] = val;
    sum += val;
  }
  for (size_t i = 0; i < size; ++i) {
    kernel[i] /= sum;
  }
}

// Implements "Recursive Implementation of the Gaussian Filter Using Truncated Cosine Functions" by Charalampidis [2016].
// https://discovery.researcher.life/article/recursive-implementation-of-the-gaussian-filter-using-truncated-cosine-functions/dcf24675f5eb30dba93c5205cdae3c40
RecursiveGaussian::RecursiveGaussian(float sigma) {
  JXL_PROFILE_FUNC
  constexpr double kPi = 3.141592653589793238;

  const double r = round(3.2795 * sigma + 0.2546);  // (57), "N"

  // Table I, first row
  const double pi_div_2r = kPi / (2.0 * r);
  const double omega[3] = {pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r};

  // (37), k={1,3,5}
  const double p_1 = +1.0 / tan(0.5 * omega[0]);
  const double p_3 = -1.0 / tan(0.5 * omega[1]);
  const double p_5 = +1.0 / tan(0.5 * omega[2]);

  // (44), k={1,3,5}
  const double r_1 = +p_1 * p_1 / sin(omega[0]);
  const double r_3 = -p_3 * p_3 / sin(omega[1]);
  const double r_5 = +p_5 * p_5 / sin(omega[2]);

  // (50), k={1,3,5}
  const double neg_half_sigma2 = -0.5 * sigma * sigma;
  const double recip_r = 1.0 / r;
  double rho[3];
  for (size_t i = 0; i < 3; ++i) {
    rho[i] = exp(neg_half_sigma2 * omega[i] * omega[i]) * recip_r;
  }

  // second part of (52), k1,k2 = 1,3; 3,5; 5,1
  const double D_13 = p_1 * r_3 - r_1 * p_3;
  const double D_35 = p_3 * r_5 - r_3 * p_5;
  const double D_51 = p_5 * r_1 - r_5 * p_1;

  // (52), k=5
  const double recip_d13 = 1.0 / D_13;
  const double zeta_15 = D_35 * recip_d13;
  const double zeta_35 = D_51 * recip_d13;

  double A[9] = {p_1,     p_3,     p_5,  //
                 r_1,     r_3,     r_5,  //  (56)
                 zeta_15, zeta_35, 1};
  JXL_CHECK(Inv3x3Matrix(A));

  const double gamma[3] = {1, r * r - sigma * sigma,  // (55)
                           zeta_15 * rho[0] + zeta_35 * rho[1] + rho[2]};
  double beta[3];
  MatMul(A, gamma, 3, 3, 1, beta);  // (53)

  // Sanity check: correctly solved for beta (IIR filter weights are normalized)
  const double sum = beta[0] * p_1 + beta[1] * p_3 + beta[2] * p_5;  // (39)
  JXL_CHECK(abs(sum - 1) < 1E-12);

  this->radius = int(r);

  for (size_t i = 0; i < 3; ++i) {
    double n2 = -beta[i] * cos(omega[i] * (radius + 1.0));  // (33)
    double d1 = -2.0 * cos(omega[i]);                       // (33)

    this->mul_prev[i] = (float)-d1;
    this->mul_in[i] = (float)n2;
  }

  if (var::ssimu2_slow_blur) {
    GaussianKernel(this->radius, sigma, this->kernel);
  }
}


// Scalar implementation for simplicity.
static void Gaussian1D(const RecursiveGaussian& rg, const float* JXL_RESTRICT in, intptr_t width, intptr_t stride, float* JXL_RESTRICT out) {
  // Note, here we are only using the first of 4 terms, the additional terms exist because the original code was unrolled 4x. Here's the original comment:
  // | Although the current output depends on the previous output, we can unroll
  // | up to 4x by precomputing up to fourth powers of the constants. Beyond that,
  // | numerical precision might become a problem.
  const float mul_in_1 = *(rg.mul_in + 0 * 4);
  const float mul_in_3 = *(rg.mul_in + 1 * 4);
  const float mul_in_5 = *(rg.mul_in + 2 * 4);
  const float mul_prev_1 = *(rg.mul_prev + 0 * 4);
  const float mul_prev_3 = *(rg.mul_prev + 1 * 4);
  const float mul_prev_5 = *(rg.mul_prev + 2 * 4);
  float prev_1 = 0.0f;
  float prev_3 = 0.0f;
  float prev_5 = 0.0f;
  float prev2_1 = 0.0f;
  float prev2_3 = 0.0f;
  float prev2_5 = 0.0f;

  const intptr_t N = rg.radius;

  for (intptr_t n = -N + 1; n < width; n++) {
    const intptr_t left = n - N - 1;
    const intptr_t right = n + N - 1;
    const float left_val = left >= 0 ? in[left * stride] : 0.0f;
    const float right_val = right < width ? in[right * stride] : 0.0f;
    const float sum = left_val + right_val;

    float out_1 = sum * mul_in_1 - prev2_1 + mul_prev_1 * prev_1;
    float out_3 = sum * mul_in_3 - prev2_3 + mul_prev_3 * prev_3;
    float out_5 = sum * mul_in_5 - prev2_5 + mul_prev_5 * prev_5;

    prev2_1 = prev_1;
    prev2_3 = prev_3;
    prev2_5 = prev_5;
    prev_1 = out_1;
    prev_3 = out_3;
    prev_5 = out_5;

    if (n >= 0) {
      out[n * stride] = out_1 + out_3 + out_5;
    }
  }
}


void GaussianHorizontal(const RecursiveGaussian& rg, const ImageF& in, ImageF* JXL_RESTRICT out) {
  JXL_PROFILE_FUNC
  JXL_CHECK(SameSize(in, *out));
  const intptr_t xsize = in.xsize();
  const intptr_t ysize = in.ysize();
  for (intptr_t y = 0; y < ysize; y++) {
    const float* row_in = in.Row(y);
    float* JXL_RESTRICT row_out = out->Row(y);
    Gaussian1D(rg, row_in, xsize, 1, row_out);
  }
}

static void GaussianVertical(const RecursiveGaussian& rg, const ImageF& in, ImageF* JXL_RESTRICT out) {
  JXL_PROFILE_FUNC
  JXL_CHECK(SameSize(in, *out));
  const intptr_t xsize = in.xsize();
  const intptr_t ysize = in.ysize();
  for (intptr_t x = 0; x < xsize; x++) {
    const float* col_in = in.data + x;
    float* JXL_RESTRICT col_out = out->data + x;
    Gaussian1D(rg, col_in, ysize, xsize, col_out);
  }
}

// @@ All optimizations undone to remove HWY dependency.
static void Gaussian(const RecursiveGaussian& rg, const ImageF& in, ImageF* JXL_RESTRICT temp, ImageF* JXL_RESTRICT out) {
  JXL_PROFILE_FUNC
  GaussianHorizontal(rg, in, temp);
  GaussianVertical(rg, *temp, out);
}


void ConvolveXSampleAndTranspose(const ImageF& in, ImageF* out, const float* JXL_RESTRICT kernel, int size) {
  JXL_CHECK(size % 2 == 1);

  const int r = size / 2;
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* const JXL_RESTRICT rowp = in.Row(y);
    for (size_t x = 0; x < in.xsize(); x++) {
      float sum = 0.0f;
      for (int i = -r; i <= r; ++i) {
        // Clamp to border.
        JXL_CHECK(i+r >= 0 && i+r < size);
        sum += rowp[clamp(int(x) + i, 0, (int)in.xsize()-1)] * kernel[i+r];
      }
      out->Row(x)[y] = sum;
    }
  }
}

static void ConvolveHorizontal(const ImageF& in, ImageF* JXL_RESTRICT out, const float* JXL_RESTRICT kernel, int r) {
  const intptr_t w = in.xsize();
  const intptr_t h = in.ysize();

  for (intptr_t y = 0; y < h; ++y) {
    const float* JXL_RESTRICT rowp = in.Row(y);
    float* JXL_RESTRICT rowout = out->Row(y);

    // Left border: clamp on the left side.
    for (intptr_t x = 0; x < min((intptr_t)r, w); ++x) {
      float sum = 0.0f;
      for (int i = -r; i <= r; ++i) {
        sum += rowp[clamp((int)x + i, 0, (int)w - 1)] * kernel[i + r];
      }
      rowout[x] = sum;
    }

    // Interior: no bounds checks needed.
    for (intptr_t x = r; x < w - r; ++x) {
      float sum = 0.0f;
      for (int i = -r; i <= r; ++i) {
        sum += rowp[x + i] * kernel[i + r];
      }
      rowout[x] = sum;
    }

    // Right border: clamp on the right side.
    for (intptr_t x = max(w - r, (intptr_t)r); x < w; ++x) {
      float sum = 0.0f;
      for (int i = -r; i <= r; ++i) {
        sum += rowp[clamp((int)x + i, 0, (int)w - 1)] * kernel[i + r];
      }
      rowout[x] = sum;
    }
  }
}

static void ConvolveVertical(const ImageF& in, ImageF* JXL_RESTRICT out, const float* JXL_RESTRICT kernel, int r) {
  const intptr_t w = in.xsize();
  const intptr_t h = in.ysize();

  // Process in vertical strips for cache locality.
  const intptr_t kStripWidth = 64;

  for (intptr_t x0 = 0; x0 < w; x0 += kStripWidth) {
    const intptr_t x1 = min(x0 + kStripWidth, w);

    // Top border.
    for (intptr_t y = 0; y < min((intptr_t)r, h); ++y) {
      float* JXL_RESTRICT rowout = out->Row(y);
      for (intptr_t x = x0; x < x1; ++x) {
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) {
          sum += in.Row(clamp((int)y + i, 0, (int)h - 1))[x] * kernel[i + r];
        }
        rowout[x] = sum;
      }
    }

    // Interior: no bounds checks.
    for (intptr_t y = r; y < h - r; ++y) {
      float* JXL_RESTRICT rowout = out->Row(y);
      for (intptr_t x = x0; x < x1; ++x) {
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) {
          sum += in.Row(y + i)[x] * kernel[i + r];
        }
        rowout[x] = sum;
      }
    }

    // Bottom border.
    for (intptr_t y = max(h - r, (intptr_t)r); y < h; ++y) {
      float* JXL_RESTRICT rowout = out->Row(y);
      for (intptr_t x = x0; x < x1; ++x) {
        float sum = 0.0f;
        for (int i = -r; i <= r; ++i) {
          sum += in.Row(clamp((int)y + i, 0, (int)h - 1))[x] * kernel[i + r];
        }
        rowout[x] = sum;
      }
    }
  }
}

static void SlowGaussian(const RecursiveGaussian& rg, const ImageF& in, ImageF* JXL_RESTRICT temp, ImageF* JXL_RESTRICT out) {
  ConvolveXSampleAndTranspose(in, temp, rg.kernel, rg.radius * 2 + 1);
  ConvolveXSampleAndTranspose(*temp, out, rg.kernel, rg.radius * 2 + 1);
}

static void FastGaussian(const RecursiveGaussian& rg, const ImageF& in, ImageF* JXL_RESTRICT temp, ImageF* JXL_RESTRICT out) {
  ConvolveHorizontal(in, temp, rg.kernel, rg.radius);
  ConvolveVertical(*temp, out, rg.kernel, rg.radius);
}


static Image3F Downsample(const Image3F &in, size_t fx, size_t fy) {
  JXL_PROFILE_FUNC
  const size_t out_xsize = (in.xsize() + fx - 1) / fx;
  const size_t out_ysize = (in.ysize() + fy - 1) / fy;
  Image3F out(out_xsize, out_ysize);
  const float normalize = 1.0f / (fx * fy);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t oy = 0; oy < out_ysize; ++oy) {
      float *JXL_RESTRICT row_out = out.PlaneRow(c, oy);
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
  return out;
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


// Temporary storage for Gaussian blur, reused for multiple images.
struct Blur {
  Blur(size_t xsize, size_t ysize) :
      rg(1.5), temp(var::ssimu2_slow_blur && !var::ssimu2_fast_blur ? ysize : xsize,
                     var::ssimu2_slow_blur && !var::ssimu2_fast_blur ? xsize : ysize) {}
  // temp is transposed only for the old SlowGaussian path (ConvolveXSampleAndTranspose).

  void operator()(const ImageF& in, ImageF* JXL_RESTRICT out) {
    if (var::ssimu2_fast_blur) {
      FastGaussian(rg, in, &temp, out);
    } else if (var::ssimu2_slow_blur) {
      SlowGaussian(rg, in, &temp, out);
    } else {
      Gaussian(rg, in, &temp, out);
    }
  }

  Image3F operator()(const Image3F& in) {
    Image3F out(in.xsize(), in.ysize());
    operator()(in.Plane(0), &out.Plane(0));
    operator()(in.Plane(1), &out.Plane(1));
    operator()(in.Plane(2), &out.Plane(2));
    return out;
  }

  // Allows reusing across scales.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
      if (var::ssimu2_slow_blur && !var::ssimu2_fast_blur) {
          temp.ShrinkTo(ysize, xsize); // transposed for SlowGaussian
      }
      else {
          temp.ShrinkTo(xsize, ysize);
      }
  }

  RecursiveGaussian rg;
  ImageF temp;
};

inline double tothe4th(double x) {
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
                    const Image3F &s22, const Image3F &s12, double *plane_averages, ImageF &error_map, int scale) {
  JXL_PROFILE_FUNC
  static const float kC2 = 0.0009f;
  const double onePerPixels = 1.0 / (m1.ysize() * m1.xsize());
  for (int c = 0; c < 3; ++c) {
    double sum1[2] = {};
    for (size_t y = 0; y < m1.ysize(); ++y) {
      const float *JXL_RESTRICT row_m1 = m1.PlaneRow(c, y);
      const float *JXL_RESTRICT row_m2 = m2.PlaneRow(c, y);
      const float *JXL_RESTRICT row_s11 = s11.PlaneRow(c, y);
      const float *JXL_RESTRICT row_s22 = s22.PlaneRow(c, y);
      const float *JXL_RESTRICT row_s12 = s12.PlaneRow(c, y);
      float *JXL_RESTRICT error_row = error_map.Row(y);
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

        error_row[x] += float(get_weight(c, scale, 0, 0) * d);
        error_row[x] += float(get_weight(c, scale, 0, 1) * d);
        JXL_CHECK(isfinite(error_row[x]));
      }
    }
    plane_averages[c * 2] = onePerPixels * sum1[0];
    plane_averages[c * 2 + 1] = sqrt(sqrt(onePerPixels * sum1[1]));
  }
}

static void EdgeDiffMap(const Image3F &img1, const Image3F &mu1, const Image3F &img2,
                        const Image3F &mu2, double *plane_averages, ImageF &error_map, int scale) {
  JXL_PROFILE_FUNC
  const double onePerPixels = 1.0 / (img1.ysize() * img1.xsize());
  for (int c = 0; c < 3; ++c) {
    double sum1[4] = {0.0};
    for (size_t y = 0; y < img1.ysize(); ++y) {
      const float *JXL_RESTRICT row1 = img1.PlaneRow(c, y);
      const float *JXL_RESTRICT row2 = img2.PlaneRow(c, y);
      const float *JXL_RESTRICT rowm1 = mu1.PlaneRow(c, y);
      const float *JXL_RESTRICT rowm2 = mu2.PlaneRow(c, y);
      float *JXL_RESTRICT error_row = error_map.Row(y);
      for (size_t x = 0; x < img1.xsize(); ++x) {
        double d1 = (1.0 + fabsf(row2[x] - rowm2[x])) / (1.0 + fabsf(row1[x] - rowm1[x])) - 1.0;

        // d1 > 0: distorted has an edge where original is smooth
        //         (indicating ringing, color banding, blockiness, etc)
        double artifact = max(d1, 0.0);
        double artifact4 = tothe4th(artifact);
        sum1[0] += artifact;
        sum1[1] += artifact4;

        // d1 < 0: original has an edge where distorted is smooth
        //         (indicating smoothing, blurring, smearing, etc)
        double detail_lost = max(-d1, 0.0);
        double detail_lost4 = tothe4th(detail_lost);
        sum1[2] += detail_lost;
        sum1[3] += detail_lost4;

        error_row[x] += float(get_weight(c, scale, 1, 0) * abs(artifact));
        error_row[x] += float(get_weight(c, scale, 1, 1) * abs(artifact));
        error_row[x] += float(get_weight(c, scale, 2, 0) * abs(detail_lost));
        error_row[x] += float(get_weight(c, scale, 2, 1) * abs(detail_lost));
        JXL_CHECK(isfinite(error_row[x]));
      }
    }
    plane_averages[c * 4] = onePerPixels * sum1[0];
    plane_averages[c * 4 + 1] = sqrt(sqrt(onePerPixels * sum1[1]));
    plane_averages[c * 4 + 2] = onePerPixels * sum1[2];
    plane_averages[c * 4 + 3] = sqrt(sqrt(onePerPixels * sum1[3]));
  }
}

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
static void MakePositiveXYB(Image3F &img) {
  for (size_t y = 0; y < img.ysize(); ++y) {
    float *JXL_RESTRICT rowY = img.PlaneRow(1, y);
    float *JXL_RESTRICT rowB = img.PlaneRow(2, y);
    float *JXL_RESTRICT rowX = img.PlaneRow(0, y);
    for (size_t x = 0; x < img.xsize(); ++x) {
      rowB[x] = (rowB[x] - rowY[x]) + 0.55f;
      rowX[x] = rowX[x] * 14.f + 0.42f;
      rowY[x] += 0.01f;
    }
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


static Msssim ComputeSSIMULACRA2(Image3F &orig, Image3F &dist, unsigned char* error_map) {
  JXL_PROFILE_FUNC
  Msssim msssim = {};

  size_t w = orig.xsize();
  size_t h = orig.ysize();

  Image3F img1(w, h);
  Image3F img2(w, h);
  ImageF error_accum(w, h);
  ImageF error_scale(w, h);
  error_scale.Clear();

  // This assumes the input is in srgb.
  ToXYB(orig, &img1);
  ToXYB(dist, &img2);
  MakePositiveXYB(img1);
  MakePositiveXYB(img2);

  Image3F mul(img1.xsize(), img1.ysize());
  Blur blur(img1.xsize(), img1.ysize());

  for (int scale = 0; scale < kNumScales; scale++) {
    if (img1.xsize() < 8 || img1.ysize() < 8) {
      break;
    }
    if (scale) {
      orig = Downsample(orig, 2, 2);
      dist = Downsample(dist, 2, 2);
      img1.ShrinkTo(orig.xsize(), orig.ysize());
      img2.ShrinkTo(orig.xsize(), orig.ysize());
      ToXYB(orig, &img1);
      ToXYB(dist, &img2);
      MakePositiveXYB(img1);
      MakePositiveXYB(img2);
      error_scale.ShrinkTo(orig.xsize(), orig.ysize());
      error_scale.Clear();
    }
    mul.ShrinkTo(img1.xsize(), img1.ysize());
    blur.ShrinkTo(img1.xsize(), img1.ysize());

    Multiply(img1, img1, &mul);
    Image3F sigma1_sq = blur(mul);  // blur(img1 * img1)

    Multiply(img2, img2, &mul);
    Image3F sigma2_sq = blur(mul);  // blur(img2 * img2)

    Multiply(img1, img2, &mul);
    Image3F sigma12 = blur(mul);    // blur(img1 * img2)

    Image3F mu1 = blur(img1);       // blur(img1)
    Image3F mu2 = blur(img2);       // blur(img2)

    SSIMMap(mu1, mu2, sigma1_sq, sigma2_sq, sigma12, msssim.scales[scale].avg_ssim, error_scale, scale);
    EdgeDiffMap(img1, mu1, img2, mu2, msssim.scales[scale].avg_edgediff, error_scale, scale);

    if (error_map != nullptr) {
      if (scale == 0) {
        memcpy(error_accum.data, error_scale.data, sizeof(float) * error_scale.xs * error_scale.ys);
      }
      else if (scale < 3) {
        UpscaleAndAccumulate(error_scale, error_accum);
      }
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
  char ch[] = "XYB";
  const bool verbose = false;
  if (verbose) printf("\n");
  for (size_t c = 0; c < 3; ++c) {
    for (size_t scale = 0; scale < kNumScales; ++scale) {
      for (size_t n = 0; n < 2; n++) {
        if (verbose) {
          printf("%f from channel %c ssim, scale 1:%i, %zu-norm (weight %f)\n",
                 weight[i] * abs(scales[scale].avg_ssim[c * 2 + n]), ch[c], 1 << scale, n * 3 + 1, weight[i]);
        }
        ssim += weight[i++] * abs(scales[scale].avg_ssim[c * 2 + n]);
        if (verbose) {
          printf("%f from channel %c ringing, scale 1:%i, %zu-norm (weight %f)\n",
              weight[i] * abs(scales[scale].avg_edgediff[c * 4 + n]), ch[c], 1 << scale, n * 3 + 1, weight[i]);
        }
        ssim += weight[i++] * abs(scales[scale].avg_edgediff[c * 4 + n]);
        if (verbose) {
          printf("%f from channel %c blur, scale 1:%i, %zu-norm (weight %f)\n",
                 weight[i] * abs(scales[scale].avg_edgediff[c * 4 + n + 2]), ch[c], 1 << scale, n * 3 + 1, weight[i]);
        }
        ssim += weight[i++] * abs(scales[scale].avg_edgediff[c * 4 + n + 2]);
      }
    }
  }
  if (verbose) {
      printf("raw ssim: %f\n", ssim);
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


Msssim ComputeSSIMULACRA2(int w, int h, const unsigned char* orig, const unsigned char* dist, unsigned char* error_map) {
  JXL_PROFILE_FUNC
  Image3F orig_img(w, h);
  Image3F dist_img(w, h);

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

  return ComputeSSIMULACRA2(orig_img, dist_img, error_map);
}

} // namespace


double ComputeSSIMULACRA2Score(int w, int h, const unsigned char* orig, const unsigned char* dist, unsigned char* error_map) {
  return ComputeSSIMULACRA2(w, h, orig, dist, error_map).Score();
}


double ComputeSSIMScore(int w, int h, const unsigned char* orig, const unsigned char* dist, unsigned char* error_map) {
  JXL_PROFILE_FUNC

  // Standard SSIM constants for [0,1] range (Wang et al. 2004).
  // C1 = (K1*L)^2, C2 = (K2*L)^2, with K1=0.01, K2=0.03, L=1.
  static const float kC1 = 0.0001f;
  static const float kC2 = 0.0009f;

  ImageF img1(w, h);
  ImageF img2(w, h);

  // Convert to [0,1] float.
  for (int y = 0; y < h; y++) {
    float* row1 = img1.Row(y);
    float* row2 = img2.Row(y);
    for (int x = 0; x < w; x++) {
      row1[x] = float(orig[y * w + x]) / 255.0f;
      row2[x] = float(dist[y * w + x]) / 255.0f;
    }
  }

  Blur blur(w, h);

  // mu1 = blur(img1), mu2 = blur(img2)
  ImageF mu1(w, h);
  ImageF mu2(w, h);
  blur(img1, &mu1);
  blur(img2, &mu2);

  // sigma1_sq = blur(img1*img1) - mu1*mu1
  // sigma2_sq = blur(img2*img2) - mu2*mu2
  // sigma12   = blur(img1*img2) - mu1*mu2
  ImageF tmp(w, h);
  ImageF sigma1_sq(w, h);
  ImageF sigma2_sq(w, h);
  ImageF sigma12(w, h);

  for (int y = 0; y < h; y++) {
    const float* r1 = img1.Row(y);
    float* rt = tmp.Row(y);
    for (int x = 0; x < w; x++) rt[x] = r1[x] * r1[x];
  }
  blur(tmp, &sigma1_sq);

  for (int y = 0; y < h; y++) {
    const float* r2 = img2.Row(y);
    float* rt = tmp.Row(y);
    for (int x = 0; x < w; x++) rt[x] = r2[x] * r2[x];
  }
  blur(tmp, &sigma2_sq);

  for (int y = 0; y < h; y++) {
    const float* r1 = img1.Row(y);
    const float* r2 = img2.Row(y);
    float* rt = tmp.Row(y);
    for (int x = 0; x < w; x++) rt[x] = r1[x] * r2[x];
  }
  blur(tmp, &sigma12);

  // Compute SSIM map and accumulate mean.
  double ssim_sum = 0.0;
  const double one_per_pixels = 1.0 / (w * h);

  for (int y = 0; y < h; y++) {
    const float* rm1 = mu1.Row(y);
    const float* rm2 = mu2.Row(y);
    const float* rs11 = sigma1_sq.Row(y);
    const float* rs22 = sigma2_sq.Row(y);
    const float* rs12 = sigma12.Row(y);
    for (int x = 0; x < w; x++) {
      float m1 = rm1[x];
      float m2 = rm2[x];
      float m1m2 = m1 * m2;
      float m1sq = m1 * m1;
      float m2sq = m2 * m2;
      float s1sq = rs11[x] - m1sq;
      float s2sq = rs22[x] - m2sq;
      float s12  = rs12[x] - m1m2;

      float num   = (2.0f * m1m2 + kC1) * (2.0f * s12 + kC2);
      float denom = (m1sq + m2sq + kC1) * (s1sq + s2sq + kC2);
      float ssim  = num / denom;

      ssim_sum += ssim;

      if (error_map != nullptr) {
        // Map 1-SSIM error to magma palette, same as ssimulacra2.
        float err = 1.0f - ssim;
        if (err < 0.0f) err = 0.0f;
        if (err > 1.0f) err = 1.0f;
        int value = int(255 * err);
        ((uint32_t*)error_map)[y * w + x] = MagmaMap[value];
      }
    }
  }

  return ssim_sum * one_per_pixels;
}
