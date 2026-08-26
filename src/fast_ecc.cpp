/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
//
// Modifications (warp-count reduction via chain-rule gradient reconstruction)
// Copyright (c) 2026, yosh-shimizu. Distributed under the same BSD-3 terms above.
//
// Derived from OpenCV modules/video/src/ecc.cpp.  The algorithmic change is in
// the per-iteration loop: instead of warping the two precomputed gradient
// images, we warp the image once, take finite-difference gradients of the WARPED
// image, and map them back into the image frame with the inverse-transpose of the
// warp's linear part.  That recombination, the masking and the zero-mean
// centring all happen per pixel inside the fused Gauss-Newton pass
// (gn_fused.inc); the only other pass over the planes is maskedStats() below.
// Not bit-identical to cv::findTransformECC (resampling and differentiation do
// not commute), but equally accurate against ground truth — see README.md.

// AVX2 + FMA.  FASTECC_AVX2 makes this translation unit the eight-lane
// instance: it needs AVX2 code generation (/arch:AVX2 or -mavx2 -mfma) and
// tells OpenCV's headers that the 256-bit universal intrinsics may be used.
// Outside OpenCV's own build the headers enable SSE2 alone, whatever the
// compiler was given; these are the feature macros its own dispatch would
// set.  The vector code below is written width-agnostically (vx_load,
// vlanes()), so nothing else changes: eight lanes and fused multiply-adds
// instead of four lanes and separate ones.
//
// Two ways to get it.  FAST_ECC_AVX2=ON compiles the whole library this way,
// and it then needs an AVX2 CPU (2013 on).  The default build instead
// compiles src/fast_ecc_avx2.cpp -- this file again, with FASTECC_AVX2_TU --
// as a second instance next to the 128-bit one, and findTransformECC() at
// the end of this file picks it at run time when the CPU has AVX2 and FMA
// (FASTECC_HAVE_AVX2_TU tells the first instance that the second exists).
#if defined(FASTECC_AVX2) && FASTECC_AVX2
#  if !defined(__AVX2__)
#    error "FASTECC_AVX2 needs AVX2 code generation: /arch:AVX2 (MSVC) or -mavx2 -mfma"
#  endif
#  include <immintrin.h>
#  define CV_SSE3 1
#  define CV_SSSE3 1
#  define CV_SSE4_1 1
#  define CV_SSE4_2 1
#  define CV_AVX 1
#  define CV_AVX2 1
#  define CV_FMA3 1
#endif

#include "fast_ecc.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

// The vector paths -- the sampler, the derivative and moment loops of the
// stripe, and the Gauss-Newton kernels -- are written with OpenCV's
// universal intrinsics in their function form (v_add, v_lut, ...), which
// exist from OpenCV 4.7: as wrappers over the operators until 4.10, native
// from 4.11, where the operators went away.  An older OpenCV, or
// -DFASTECC_NO_SIMD, gets the scalar loops.
#if !defined(FASTECC_NO_SIMD) && (CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7))
#  include <opencv2/core/hal/intrin.hpp>
#  if CV_SIMD
#    define FASTECC_SIMD 1
#  endif
#endif
#ifndef FASTECC_SIMD
#  define FASTECC_SIMD 0
#endif
// The Gauss-Newton kernels have a vector path on the same footing
// (gn_fused.inc); -DFASTECC_NO_SIMD_GN keeps their scalar loops alone,
// which is how the two are compared.
#if FASTECC_SIMD && !defined(FASTECC_NO_SIMD_GN)
#  define FASTECC_SIMD_GN 1
#else
#  define FASTECC_SIMD_GN 0
#endif

using namespace cv;

namespace {

// Masked first and second moments of the warped image and the template, in
// one pass over three planes.  Everything the normalisation needs falls out
// of these six sums: the two means, the two norms (sum of squares about the
// mean) and the correlation of the centred vectors.  OpenCV computes the
// same quantities with meanStdDev x2, subtract x2, countNonZero x2 and a dot
// product, each a separate masked pass that also has to write a plane.
struct MaskedStats {
    double n, si, sii, st, stt, sit;
    MaskedStats() : n(0), si(0), sii(0), st(0), stt(0), sit(0) {}
    void add(const MaskedStats& o) {
        n += o.n; si += o.si; sii += o.sii; st += o.st; stt += o.stt; sit += o.sit;
    }
};

class MaskedStatsBody : public ParallelLoopBody {
public:
    MaskedStatsBody(const Mat& im, const Mat& tm, const Mat& mask,
                    std::vector<MaskedStats>& out, Mutex& m)
        : im_(im), tm_(tm), mask_(mask), out_(out), mtx_(m) {}
    void operator()(const Range& r) const CV_OVERRIDE {
        MaskedStats s;
        const int w = im_.cols;
        for (int y = r.start; y < r.end; ++y) {
            const float* pim = im_.ptr<float>(y);
            const float* ptm = tm_.ptr<float>(y);
            const uchar* pmk = mask_.ptr<uchar>(y);
            double n = 0, si = 0, sii = 0, st = 0, stt = 0, sit = 0;
            for (int x = 0; x < w; ++x) {
                if (pmk[x] == 0) continue;
                const double v = pim[x], t = ptm[x];
                n += 1; si += v; sii += v * v; st += t; stt += t * t; sit += v * t;
            }
            s.n += n; s.si += si; s.sii += sii; s.st += st; s.stt += stt; s.sit += sit;
        }
        AutoLock lk(mtx_);
        out_.push_back(s);
    }
private:
    const Mat &im_, &tm_, &mask_;
    std::vector<MaskedStats>& out_;
    Mutex& mtx_;
};

static MaskedStats maskedStats(const Mat& im, const Mat& tm, const Mat& mask) {
    CV_Assert(im.type() == CV_32FC1 && tm.type() == CV_32FC1 && mask.type() == CV_8UC1);
    CV_Assert(im.size() == tm.size() && im.size() == mask.size());
    std::vector<MaskedStats> parts;
    Mutex m;
    MaskedStatsBody body(im, tm, mask, parts, m);
    parallel_for_(Range(0, im.rows), body);
    MaskedStats s;
    for (size_t i = 0; i < parts.size(); ++i) s.add(parts[i]);
    return s;
}

// The finite-difference gradient of the warped image is g = A^T (grad I)(W(x)),
// so the image-domain gradient the jacobian expects is A^{-T} g.  This returns
// A^{-T} as (r00, r01; r10, r11), applied per pixel inside the fused pass.
// For euclidean the linear part is orthonormal, so A^{-T} == A.  For
// homography this is still an approximation: the true jacobian of a
// projective warp varies per pixel; A is its linear part.
static void recombination(const Mat& map, int motionType, float r[4])
{
    CV_Assert(map.isContinuous() && map.cols == 3);
    const float* hptr = map.ptr<float>(0);
    if (motionType == MOTION_EUCLIDEAN) {
        const float h0 = hptr[0];   // cos(theta)
        const float h1 = hptr[3];   // sin(theta)
        r[0] = h0; r[1] = -h1;
        r[2] = h1; r[3] =  h0;
        return;
    }
    const float a = hptr[0], b = hptr[1], c = hptr[3], d = hptr[4];
    const float det = a*d - b*c;
    if (std::fabs(det) > 1e-12f) {
        const float s = 1.f/det;
        r[0] =  d*s; r[1] = -c*s;
        r[2] = -b*s; r[3] =  a*s;
    } else {
        // degenerate linear part: fall back to A rather than blowing up
        r[0] = a; r[1] = b;
        r[2] = c; r[3] = d;
    }
}

// Under an empty input mask, preMask is the rectangle [2, wd-3] x [2, hd-3]
// and its nearest-neighbour warp is the set of template pixels whose rounded
// warped coordinate lands in that rectangle.  Along a row that set is an
// interval, for an affine warp and for a projective one with a positive
// denominator alike, so it can be solved for instead of resampled: two
// bounds per constraint, four constraints, per row.  The ring of the
// template border is applied at the same time.
//
// rowInterval() solves one row into [x0, x1) and returns false when the
// projective denominator changes sign along it; analyticMask() fills a whole
// mask plane with it, and the single-pass iteration below calls it per
// stripe.
struct WarpCoef {
    double a0, a1, a2, a3, a4, a5, a6, a7, a8;
    bool homo;
    WarpCoef(const Mat& map, int motionType) {
        CV_Assert(map.isContinuous() && map.type() == CV_32FC1 && map.cols == 3);
        const float* m = map.ptr<float>(0);
        homo = motionType == MOTION_HOMOGRAPHY;
        a0 = m[0]; a1 = m[1]; a2 = m[2];
        a3 = m[3]; a4 = m[4]; a5 = m[5];
        a6 = homo ? m[6] : 0.0;
        a7 = homo ? m[7] : 0.0;
        a8 = homo ? m[8] : 1.0;
    }
    // the projective denominator is affine in (x, y): positive at the four
    // corners means positive on the whole template
    bool denominatorPositive(int ws, int hs) const {
        if (!homo) return true;
        const double c[4] = { a8, a6 * (ws - 1) + a8, a7 * (hs - 1) + a8, a6 * (ws - 1) + a7 * (hs - 1) + a8 };
        return c[0] > 0 && c[1] > 0 && c[2] > 0 && c[3] > 0;
    }
};

static bool rowInterval(const WarpCoef& c, int y, int ws, int wd, int hd, int ring, int& x0, int& x1)
{
    // round(x') in [2, wd-3]  <=>  1.5 <= x' < wd-2.5, and the same for y'
    const double xlo = 1.5, xhi = wd - 2.5, ylo = 1.5, yhi = hd - 2.5;
    const double eps = 1e-12;
    const double bx = c.a1 * y + c.a2, by = c.a4 * y + c.a5, dy = c.a7 * y + c.a8;
    // each constraint is  p*x + q >= 0  (or > 0); keep [lo, hi)
    double lo = ring, hi = ws - ring;
    bool empty = false;
    const double P[5] = { c.a0 - xlo * c.a6, -(c.a0 - xhi * c.a6), c.a3 - ylo * c.a6, -(c.a3 - yhi * c.a6), c.a6 };
    const double Q[5] = { bx - xlo * dy, -(bx - xhi * dy), by - ylo * dy, -(by - yhi * dy), dy };
    if (c.homo) {
        // the denominator must stay positive across the row we keep
        const double d0 = c.a6 * lo + dy, d1 = c.a6 * (hi - 1) + dy;
        if (d0 <= 0.0 || d1 <= 0.0) {
            if (d0 <= 0.0 && d1 <= 0.0) { x0 = x1 = ws; return true; }
            return false;
        }
    }
    for (int k = 0; k < (c.homo ? 5 : 4); ++k) {
        const double pk = P[k], qk = Q[k];
        if (pk > eps)       lo = std::max(lo, -qk / pk);
        else if (pk < -eps) hi = std::min(hi, -qk / pk);
        else if (qk < 0.0)  { empty = true; break; }
    }
    x0 = empty ? ws : (int)std::ceil(lo - 1e-9);
    x1 = empty ? ws : (int)std::ceil(hi - 1e-9);
    x0 = std::max(ring, std::min(ws - ring, x0));
    x1 = std::max(x0, std::min(ws - ring, x1));
    return true;
}

static bool analyticMask(Mat& mask, const Mat& map, int motionType, int wd, int hd, int ring)
{
    CV_Assert(mask.type() == CV_8UC1);
    const int hs = mask.rows, ws = mask.cols;
    const WarpCoef c(map, motionType);
    for (int y = 0; y < hs; ++y) {
        uchar* row = mask.ptr<uchar>(y);
        if (y < ring || y >= hs - ring) { std::memset(row, 0, ws); continue; }
        int x0, x1;
        if (!rowInterval(c, y, ws, wd, hd, ring, x0, x1)) return false;
        std::memset(row, 0, x0);
        std::memset(row + x0, 1, x1 - x0);
        std::memset(row + x1, 0, ws - x1);
    }
    return true;
}

// The bilinear combination of the four taps at index k (top-left) with the
// fractions u, v, as one expression shared by the vector and the scalar code
// so that the two agree bit for bit and a row does not depend on where the
// vector part ends.  Weighted-sum form rather than two lerps: the weights
// depend only on (u, v) and are ready before the taps arrive, so the chain
// after the gathers is a multiply and two adds.
#define FASTECC_BILINEAR(T00, T01, T10, T11, W00, W01, W10, W11) \
    (((T00) * (W00) + (T01) * (W01)) + ((T10) * (W10) + (T11) * (W11)))

// Bilinear taps of one span of n pixels whose four taps are all inside the
// source, at (a0*i + bx, a3*i + by) for i = 0..n-1.
//
// The vector path runs two passes over chunks of CH pixels: the coordinates,
// truncated and split into tap index and fractions, are streamed into small
// buffers first, then gathered and combined.  Nothing but the gathers and
// the weighting is on the second pass's dependency chain, which is what the
// out-of-order window ends up waiting on.
static inline void bilinearSpan(const float* SRC, size_t STEP,
                                float a0, float a3, float bx, float by, float* out, int n)
{
    int i = 0;
#if FASTECC_SIMD
    {
        enum { CH = 64 };
        const int L = VTraits<v_float32>::vlanes();
        float CV_DECL_ALIGNED(64) lane[VTraits<v_float32>::max_nlanes];
        int   CV_DECL_ALIGNED(64) idx[CH];
        float CV_DECL_ALIGNED(64) fu[CH], fv[CH];
        for (int k = 0; k < L; ++k) lane[k] = (float)k;
        const v_float32 vL = vx_setall_f32((float)L), vone = vx_setall_f32(1.f);
        const v_float32 va0 = vx_setall_f32(a0), va3 = vx_setall_f32(a3);
        const v_float32 vbx = vx_setall_f32(bx), vby = vx_setall_f32(by);
        const v_int32 vstep = vx_setall_s32((int)STEP);
        const int nv = n - n % L;
        while (i < nv) {
            const int m = std::min((int)CH, nv - i);
            v_float32 vi = v_add(vx_load_aligned(lane), vx_setall_f32((float)i));
            for (int k = 0; k < m; k += L) {
                const v_float32 sx = v_add(v_mul(va0, vi), vbx), sy = v_add(v_mul(va3, vi), vby);
                const v_int32 ix = v_trunc(sx), iy = v_trunc(sy);    // both >= 0 here
                v_store_aligned(idx + k, v_add(v_mul(iy, vstep), ix));
                v_store_aligned(fu + k, v_sub(sx, v_cvt_f32(ix)));
                v_store_aligned(fv + k, v_sub(sy, v_cvt_f32(iy)));
                vi = v_add(vi, vL);
            }
            for (int k = 0; k < m; k += L) {
                const v_float32 u = vx_load_aligned(fu + k), v = vx_load_aligned(fv + k);
                const v_float32 omu = v_sub(vone, u), omv = v_sub(vone, v);
                const v_float32 w00 = v_mul(omu, omv), w01 = v_mul(u, omv);
                const v_float32 w10 = v_mul(omu, v),   w11 = v_mul(u, v);
                const v_float32 t00 = vx_lut(SRC, idx + k),        t01 = vx_lut(SRC + 1, idx + k);
                const v_float32 t10 = vx_lut(SRC + STEP, idx + k), t11 = vx_lut(SRC + STEP + 1, idx + k);
                v_store(out + i + k, v_add(v_add(v_mul(t00, w00), v_mul(t01, w01)),
                                           v_add(v_mul(t10, w10), v_mul(t11, w11))));
            }
            i += m;
        }
    }
#endif
    for (; i < n; ++i) {
        const float sx = a0 * (float)i + bx, sy = a3 * (float)i + by;
        const int x0 = (int)sx, y0 = (int)sy;
        const float u = sx - (float)x0, v = sy - (float)y0;
        const float omu = 1.f - u, omv = 1.f - v;
        const float* r0 = SRC + (size_t)y0 * STEP + x0;
        const float* r1 = r0 + STEP;
        out[i] = FASTECC_BILINEAR(r0[0], r0[1], r1[0], r1[1], omu * omv, u * omv, omu * v, u * v);
    }
}

// One row of a projective warp: (a0*x + bx, a3*x + by) / (a6*x + bd), x =
// 0..ws-1, zero where the denominator vanishes or any tap is outside the
// source.  Same two passes; the first clamps the tap index into the source
// so the gathers stay in bounds and records which lanes were inside, the
// second zeroes the rest.  The scalar tail tests first.
static inline void projectiveRow(const float* SRC, size_t STEP, int SW, int SH,
                                 float a0, float a3, float a6, float bx, float by, float bd,
                                 float* out, int ws)
{
    int x = 0;
#if FASTECC_SIMD
    {
        enum { CH = 64 };
        const int L = VTraits<v_float32>::vlanes();
        float CV_DECL_ALIGNED(64) lane[VTraits<v_float32>::max_nlanes];
        int   CV_DECL_ALIGNED(64) idx[CH], ok[CH];
        float CV_DECL_ALIGNED(64) fu[CH], fv[CH];
        for (int k = 0; k < L; ++k) lane[k] = (float)k;
        const v_float32 vL = vx_setall_f32((float)L), vone = vx_setall_f32(1.f), vzf = vx_setzero_f32();
        const v_float32 va0 = vx_setall_f32(a0), va3 = vx_setall_f32(a3), va6 = vx_setall_f32(a6);
        const v_float32 vbx = vx_setall_f32(bx), vby = vx_setall_f32(by), vbd = vx_setall_f32(bd);
        const v_int32 vstep = vx_setall_s32((int)STEP), vzi = vx_setzero_s32();
        const v_int32 vxmax = vx_setall_s32(SW - 2), vymax = vx_setall_s32(SH - 2);
        const int nv = ws - ws % L;
        while (x < nv) {
            const int m = std::min((int)CH, nv - x);
            v_float32 vx = v_add(vx_load_aligned(lane), vx_setall_f32((float)x));
            for (int k = 0; k < m; k += L) {
                const v_float32 d = v_add(v_mul(va6, vx), vbd);
                const v_float32 nz = v_ne(d, vzf);
                const v_float32 dd = v_select(nz, d, vone);
                const v_float32 sx = v_div(v_add(v_mul(va0, vx), vbx), dd);
                const v_float32 sy = v_div(v_add(v_mul(va3, vx), vby), dd);
                const v_int32 ix = v_floor(sx), iy = v_floor(sy);
                const v_int32 in = v_and(v_and(v_ge(ix, vzi), v_ge(iy, vzi)),
                                         v_and(v_le(ix, vxmax), v_le(iy, vymax)));
                const v_int32 cx = v_min(v_max(ix, vzi), vxmax), cy = v_min(v_max(iy, vzi), vymax);
                v_store_aligned(idx + k, v_add(v_mul(cy, vstep), cx));
                v_store_aligned(ok + k, v_and(in, v_reinterpret_as_s32(nz)));
                v_store_aligned(fu + k, v_sub(sx, v_cvt_f32(ix)));
                v_store_aligned(fv + k, v_sub(sy, v_cvt_f32(iy)));
                vx = v_add(vx, vL);
            }
            for (int k = 0; k < m; k += L) {
                const v_float32 u = vx_load_aligned(fu + k), v = vx_load_aligned(fv + k);
                const v_float32 omu = v_sub(vone, u), omv = v_sub(vone, v);
                const v_float32 w00 = v_mul(omu, omv), w01 = v_mul(u, omv);
                const v_float32 w10 = v_mul(omu, v),   w11 = v_mul(u, v);
                const v_float32 t00 = vx_lut(SRC, idx + k),        t01 = vx_lut(SRC + 1, idx + k);
                const v_float32 t10 = vx_lut(SRC + STEP, idx + k), t11 = vx_lut(SRC + STEP + 1, idx + k);
                const v_float32 res = v_add(v_add(v_mul(t00, w00), v_mul(t01, w01)),
                                            v_add(v_mul(t10, w10), v_mul(t11, w11)));
                v_store(out + x + k, v_select(v_reinterpret_as_f32(vx_load_aligned(ok + k)), res, vzf));
            }
            x += m;
        }
    }
#endif
    for (; x < ws; ++x) {
        const float d = a6 * (float)x + bd;
        if (d == 0.f) { out[x] = 0.f; continue; }
        const float sx = (a0 * (float)x + bx) / d, sy = (a3 * (float)x + by) / d;
        const int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
        if (x0 < 0 || y0 < 0 || x0 + 1 >= SW || y0 + 1 >= SH) { out[x] = 0.f; continue; }
        const float u = sx - (float)x0, v = sy - (float)y0;
        const float omu = 1.f - u, omv = 1.f - v;
        const float* r0 = SRC + (size_t)y0 * STEP + x0;
        const float* r1 = r0 + STEP;
        out[x] = FASTECC_BILINEAR(r0[0], r0[1], r1[0], r1[1], omu * omv, u * omv, omu * v, u * v);
    }
}

// One row of the warped image, sampled straight from the blurred input with
// exact bilinear weights (OpenCV's warp rounds the coordinate to 1/32 px).
// Zero where any of the four taps is outside the source, like
// BORDER_CONSTANT.  Rows outside the template (the derivative halo at the
// top and bottom) are zero: everything they would feed is masked out.
static void sampleRow(const Mat& src, const WarpCoef& c, int y, int hs, float* out, int ws)
{
    if (y < 0 || y >= hs) { std::memset(out, 0, ws * sizeof(float)); return; }
    const int SW = src.cols, SH = src.rows;
    const float* SRC = src.ptr<float>(0);
    const size_t STEP = src.step / sizeof(float);
    const double fy = (double)y;
    const double bx = c.a1 * fy + c.a2, by = c.a4 * fy + c.a5, bd = c.a7 * fy + c.a8;
    if (!c.homo) {
        // sx = a0*x + bx and sy = a3*x + by are linear in x, so the x whose
        // four taps are inside the source form an interval: solve for it once
        // and run the interior without a test per pixel.
        double lo = 0.0, hi = (double)ws;
        const double eps = 1e-12;
        if (c.a0 > eps)       { lo = std::max(lo, (0.0 - bx) / c.a0);        hi = std::min(hi, ((SW - 1) - bx) / c.a0); }
        else if (c.a0 < -eps) { lo = std::max(lo, ((SW - 1) - bx) / c.a0);   hi = std::min(hi, (0.0 - bx) / c.a0); }
        else if (bx < 0.0 || bx >= SW - 1) { lo = 1.0; hi = 0.0; }
        if (c.a3 > eps)       { lo = std::max(lo, (0.0 - by) / c.a3);        hi = std::min(hi, ((SH - 1) - by) / c.a3); }
        else if (c.a3 < -eps) { lo = std::max(lo, ((SH - 1) - by) / c.a3);   hi = std::min(hi, (0.0 - by) / c.a3); }
        else if (by < 0.0 || by >= SH - 1) { lo = 1.0; hi = 0.0; }
        int xlo = std::min(ws, std::max(0, (int)std::ceil(lo)));
        int xhi = std::max(xlo, std::min(ws, (int)std::floor(hi) + 1));
        // one pixel of slack: a coordinate that lands exactly on the last
        // valid tap must not read past it
        while (xhi > xlo && (c.a0 * (xhi - 1) + bx >= SW - 1 || c.a3 * (xhi - 1) + by >= SH - 1)) --xhi;
        while (xhi > xlo && (c.a0 * xlo + bx < 0.0 || c.a3 * xlo + by < 0.0)) ++xlo;
        for (int x = 0; x < xlo; ++x) out[x] = 0.f;
        for (int x = xhi; x < ws; ++x) out[x] = 0.f;
        // the coordinate is formed from the row start in float: 24 bits
        // over a few hundred pixels is 1e-5 px, below anything measured
        bilinearSpan(SRC, STEP, (float)c.a0, (float)c.a3,
                     (float)(bx + c.a0 * xlo), (float)(by + c.a3 * xlo), out + xlo, xhi - xlo);
    } else {
        projectiveRow(SRC, STEP, SW, SH, (float)c.a0, (float)c.a3, (float)c.a6,
                      (float)bx, (float)by, (float)bd, out, ws);
    }
}

// Per-stripe scratch for the single-pass iteration: the sampled rows with a
// derivative halo of R rows above and below, the derivative planes and the
// mask rows.  One set per stripe, reused across iterations and levels of the
// same size.
struct StripeBuffers { Mat im, gx, gy, lp, mk; };
struct StripePool {
    int nstripes = 0, rowsPer = 0, R = 0, ws = 0;
    bool lap = false;
    std::vector<StripeBuffers> buf;
    int prepare(int hs, int ws_, int R_, bool lap_) {
        const int T = std::max(1, getNumThreads());
        // stripes per thread: more than one lets the pool balance a worker
        // that wakes late, and keeps a stripe's planes in L2 between the
        // sampler, the derivative loop and the kernel, which is why one
        // thread gets them too (FASTECC_STRIPES overrides, for measurement)
        static int perThread = -1;
        if (perThread < 0) {
            const char* e = std::getenv("FASTECC_STRIPES");
            perThread = e ? std::max(1, std::atoi(e)) : 4;
        }
        int n = std::max(1, std::min(T * perThread, hs / 16));
        const int rows = (hs + n - 1) / n;
        n = (hs + rows - 1) / rows;
        if (n != nstripes || rows != rowsPer || R_ != R || lap_ != lap || ws_ != ws) {
            nstripes = n; rowsPer = rows; R = R_; lap = lap_; ws = ws_;
            buf.assign(n, StripeBuffers());
            for (size_t i = 0; i < buf.size(); ++i) {
                buf[i].im.create(rows + 2 * R, ws, CV_32F);
                buf[i].gx.create(rows, ws, CV_32F);
                buf[i].gy.create(rows, ws, CV_32F);
                if (lap) buf[i].lp.create(rows, ws, CV_32F);
                buf[i].mk.create(rows, ws, CV_8U);
            }
        }
        return nstripes;
    }
};

// Masked moments of one stripe, centred on the previous iteration's means.
struct StripeStats {
    double n = 0, sd = 0, sdd = 0, se = 0, see = 0, sde = 0;
    void add(const StripeStats& o) { n += o.n; sd += o.sd; sdd += o.sdd; se += o.se; see += o.see; sde += o.sde; }
};

#include "gn_fused.inc"

static void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType)
{
    CV_Assert (map_matrix.type() == CV_32FC1);
    CV_Assert (update.type() == CV_32FC1);

    CV_Assert (motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
        motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert (map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert (map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert (map_matrix.rows == 2 && update.rows == 2);

    CV_Assert (update.cols == 1);
    CV_Assert( map_matrix.isContinuous());
    CV_Assert( update.isContinuous() );

    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);

    if (motionType == MOTION_TRANSLATION){
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
    if (motionType == MOTION_HOMOGRAPHY) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[6] += updatePtr[2];
        mapPtr[1] += updatePtr[3];
        mapPtr[4] += updatePtr[4];
        mapPtr[7] += updatePtr[5];
        mapPtr[2] += updatePtr[6];
        mapPtr[5] += updatePtr[7];
    }
    if (motionType == MOTION_EUCLIDEAN) {
        double new_theta = updatePtr[0];
        new_theta += asin(mapPtr[3]);

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = (float) cos(new_theta);
        mapPtr[3] = (float) sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}

} // anonymous namespace

namespace fastecc {

namespace {

// One scale of the iteration.  `src` and `dst` are single-channel CV_8U or
// CV_32F images, `inputMaskMat` an optional CV_8U mask on `dst` (empty for
// none), `map` the warp in the coordinates of these images.  Everything
// findTransformECC used to do after validating its arguments lives here; the
// pyramid wrapper below calls it once per level, coarsest first.
double runSingleScale(const Mat& src, const Mat& dst, Mat& map, int motionType,
                      int numberOfIterations, double termination_eps,
                      const Mat& inputMaskMat, int gaussFiltSize, int flags)
{
    const bool useLap   = (flags & FASTECC_LAPLACIAN_COLUMN) != 0;
    const bool useGrad5 = (flags & FASTECC_GRAD5) != 0;

    int paramTemp = 6;//default: affine
    switch (motionType){
      case MOTION_TRANSLATION:
          paramTemp = 2;
          break;
      case MOTION_EUCLIDEAN:
          paramTemp = 3;
          break;
      case MOTION_HOMOGRAPHY:
          paramTemp = 8;
          break;
    }

    const int numberOfParameters = paramTemp;
    const int numberOfUnknowns   = paramTemp + ((flags & FASTECC_LAPLACIAN_COLUMN) ? 1 : 0);

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped input image
    Mat imageMask     = Mat(hs, ws, CV_8U); // to store the final mask

    //gaussian filtering is optional (sigma=0 -> derived from gaussFiltSize, matching cv::findTransformECC)
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    // The mask to warp, as upstream builds it: the user mask (or all ones)
    // blurred like the images and thresholded, with a two-pixel border.
    // Only the warp path needs it -- a user mask, or a row on which the
    // analytic mask gives up -- so it is built on first use rather than
    // costing a full-size blur on every call.
    Mat preMask;
    auto ensurePreMask = [&]() {
        if (!preMask.empty()) return;
        if(inputMaskMat.empty())
            preMask = Mat::ones(hd, wd, CV_8U);
        else
            threshold(inputMaskMat, preMask, 0, 1, THRESH_BINARY);
        Mat preMaskFloat;
        preMask.convertTo(preMaskFloat, CV_32F);
        GaussianBlur(preMaskFloat, preMaskFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);
        // Change threshold.
        preMaskFloat *= (0.5/0.95);
        // Rounding conversion.
        preMaskFloat.convertTo(preMask, preMask.type());
        preMask.row(0).setTo(0); preMask.row(preMask.rows - 1).setTo(0);
        preMask.col(0).setTo(0); preMask.col(preMask.cols - 1).setTo(0);
        preMask.row(1).setTo(0); preMask.row(preMask.rows - 2).setTo(0);
        preMask.col(1).setTo(0); preMask.col(preMask.cols - 2).setTo(0);
    };

    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    // gradients (and, optionally, the laplacian) of the warped image; raw,
    // masked per pixel in the fused pass
    Mat imageWarpedGradientX = Mat(hs, ws, CV_32FC1);
    Mat imageWarpedGradientY = Mat(hs, ws, CV_32FC1);
    Mat imageWarpedLaplacian;
    if (useLap) imageWarpedLaplacian = Mat(hs, ws, CV_32FC1);

    // first order image derivatives: 3-tap central difference, or the 4th-order
    // 5-tap.  The forward-additive step is the exact maximiser of the paper's
    // first-order model, so this estimate is the only thing in that model that
    // is not exact, and its bias sets the convergence rate.
    const Matx13f dx3(-0.5f, 0.0f, 0.5f);
    const Matx<float, 1, 5> dx5(1.f/12, -8.f/12, 0.f, 8.f/12, -1.f/12);

    // scale of the laplacian column: the pre-filter's sigma, as GaussianBlur
    // derives it from the kernel size.  Only conditioning depends on it.
    const float lapScale = (float)std::max(0.8, 0.3 * ((gaussFiltSize - 1) * 0.5 - 1) + 0.8);

    // matrices needed for solving linear equation system for maximizing ECC
    Mat hessian                 = Mat(numberOfUnknowns, numberOfUnknowns, CV_32F);
    Mat hessianInv              = Mat(numberOfUnknowns, numberOfUnknowns, CV_32F);
    Mat imageProjection         = Mat(numberOfUnknowns, 1, CV_32F);
    Mat templateProjection      = Mat(numberOfUnknowns, 1, CV_32F);
    Mat imageProjectionHessian  = Mat(numberOfUnknowns, 1, CV_32F);
    Mat errorProjection         = Mat(numberOfUnknowns, 1, CV_32F);

    Mat deltaP = Mat(numberOfUnknowns, 1, CV_32F);//parameter correction (+ the laplacian coefficient)

    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;

    // The single-pass iteration: per stripe of rows, sample the warped image
    // straight from the blurred input (with a derivative halo), take the
    // gradients and the laplacian in the stripe, solve for the mask rows, and
    // accumulate the masked moments and the Gauss-Newton sums -- nothing
    // full-size is written and the whole iteration is one parallel region.
    // The planes are centred on the previous iteration's means (the current
    // ones are not known until the pass is over) and the projections are
    // corrected exactly afterwards with the column sums VJ.  A user mask, or
    // a projective denominator that changes sign over the template, takes
    // the multi-pass path below instead; FASTECC_LEGACY_PIPELINE forces it.
    const bool canFuse = (flags & FASTECC_LEGACY_PIPELINE) == 0 && inputMaskMat.empty();
    double muPrevI = 0, muPrevT = 0;
    if (canFuse) { const Scalar m0 = mean(templateFloat); muPrevI = muPrevT = m0[0]; }
    StripePool pool;

    // iteratively update map_matrix
    double rho      = -1;
    double last_rho = - termination_eps;
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho-last_rho)>= termination_eps); i++)
    {
        // The Gaussian pre-filter above extrapolated the template border
        // (BORDER_REFLECT_101), so a ring of gaussFiltSize/2 px of
        // templateFloat is fabricated rather than measured.  Used at full
        // weight it biases the estimated linear part toward a smaller scale.
        // Drop it, the same way opencv#29775 does in ecc.cpp; a ring of 1
        // (the border row/column) is dropped in any case.
        int ring = (imageMask.rows > gaussFiltSize && imageMask.cols > gaussFiltSize)
                   ? std::max(1, gaussFiltSize / 2) : 1;
        // the 5-tap stencil must not be formed from reflected samples either
        if (useGrad5 && imageMask.rows > 4 && imageMask.cols > 4) ring = std::max(ring, 2);

        double imgMean = 0, tmpMean = 0, imgNorm = 0, tmpNorm = 0, correlation = 0;
        const bool fused = canFuse && WarpCoef(map, motionType).denominatorPositive(ws, hs);
        if (fused) {
            // the kernel centres on the previous means; the pass fixes the rest
            imgMean = muPrevI; tmpMean = muPrevT;
        } else {
            // Warp-back ONLY the image.  The gradient images are NOT warped: we
            // take gradients of the warped image and recombine them with the
            // warp's linear part inside the fused pass.  This removes two warps
            // per iteration; the mask below removes a third when it can be solved
            // for instead.
            if (motionType != MOTION_HOMOGRAPHY)
                warpAffine(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
            else
                warpPerspective(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);


            // The mask: solved for when there is no user mask (see analyticMask),
            // warped like the image otherwise.
            if (!(inputMaskMat.empty() && analyticMask(imageMask, map, motionType, wd, hd, ring)))
            {
                ensurePreMask();
                if (motionType != MOTION_HOMOGRAPHY)
                    warpAffine(preMask, imageMask, map, imageMask.size(), maskFlags);
                else
                    warpPerspective(preMask, imageMask, map, imageMask.size(), maskFlags);
                imageMask.rowRange(0, ring).setTo(0);
                imageMask.rowRange(imageMask.rows - ring, imageMask.rows).setTo(0);
                imageMask.colRange(0, ring).setTo(0);
                imageMask.colRange(imageMask.cols - ring, imageMask.cols).setTo(0);
            }

            // gradients of the WARPED image (chain rule: d/dx[I(W(x))] = J_W^T grad(I)).
            // Left as they are: the fused pass masks them per pixel.
            if (useGrad5) {
                filter2D(imageWarped, imageWarpedGradientX, -1, dx5);
                filter2D(imageWarped, imageWarpedGradientY, -1, dx5.t());
            } else {
                filter2D(imageWarped, imageWarpedGradientX, -1, dx3);
                filter2D(imageWarped, imageWarpedGradientY, -1, dx3.t());
            }
            if (useLap)
                Laplacian(imageWarped, imageWarpedLaplacian, CV_32F, 1);

            // The normalisation from one pass of masked moments.  With N pixels in
            // the mask and S1, S2 the masked sums of a plane and its square, the
            // sum of squares about the mean is S2 - S1^2/N, and the correlation of
            // the two centred planes is sum(I T) - S_I S_T / N.  These are the
            // quantities OpenCV forms with meanStdDev, countNonZero and a dot
            // product on explicitly centred, explicitly masked copies.
            const MaskedStats st = maskedStats(imageWarped, templateFloat, imageMask);
            if (st.n == 0) {
              CV_Error(Error::StsNoConv, "NaN encountered.");
            }
            imgMean = st.si / st.n;
            tmpMean = st.st / st.n;
            imgNorm = std::sqrt(std::max(st.sii - st.si * st.si / st.n, 0.0));
            tmpNorm = std::sqrt(std::max(st.stt - st.st * st.st / st.n, 0.0));
            correlation = st.sit - st.si * st.st / st.n;

        }

        // The Gram matrix and both projections come out of a single pass over
        // the raw planes, with no jacobian materialised and nothing centred,
        // masked or recombined in memory first.
        float r[4];
        recombination(map, motionType, r);
        const float* h = map.ptr<float>(0);
        // each motion type has a kernel with and without the laplacian column;
        // the coefficients are the same, so one generic lambda fills either
        // Single pass: stripes of sampled rows, derivatives in the stripe,
        // mask rows solved, moments and Gauss-Newton sums accumulated, then
        // the exact correction of the projections to the true means.
        auto fusedIteration = [&](auto& proto) {
            using Acc = typename std::decay<decltype(proto)>::type;
            const int R = useGrad5 ? 2 : 1;
            const int nstripes = pool.prepare(hs, ws, R, useLap);
            std::vector<Acc> parts(nstripes, proto);
            std::vector<StripeStats> stats(nstripes);
            const WarpCoef coef(map, motionType);
            const float mI = (float)muPrevI, mT = (float)muPrevT;
            parallel_for_(Range(0, nstripes), [&](const Range& rg) {
                for (int si = rg.start; si < rg.end; ++si) {
                    StripeBuffers& B = pool.buf[si];
                    const int y0 = si * pool.rowsPer, y1 = std::min(hs, y0 + pool.rowsPer);
                    if (y0 >= y1) continue;
                    const int hsub = y1 - y0;
                    // 1. the warped rows, halo included (buffer row k = image row y0 - R + k)
                    for (int y = y0 - R; y < y1 + R; ++y)
                        sampleRow(imageFloat, coef, y, hs, B.im.ptr<float>(y - (y0 - R)), ws);
                    // 2. mask rows, 3. derivatives, 4. moments
                    StripeStats& st = stats[si];
                    for (int y = y0; y < y1; ++y) {
                        const int k = y - y0;
                        uchar* mk = B.mk.ptr<uchar>(k);
                        int x0 = ws, x1 = ws;
                        if (y >= ring && y < hs - ring) {
                            if (!rowInterval(coef, y, ws, wd, hd, ring, x0, x1)) { x0 = x1 = ws; }
                        }
                        std::memset(mk, 0, ws);
                        if (x1 > x0) std::memset(mk + x0, 1, x1 - x0);

                        const float* r0 = B.im.ptr<float>(k + R);
                        const float* rm1 = r0 - B.im.step1();
                        const float* rp1 = r0 + B.im.step1();
                        const float* rm2 = rm1 - B.im.step1();   // read only under useGrad5
                        const float* rp2 = rp1 + B.im.step1();
                        float* gx = B.gx.ptr<float>(k);
                        float* gy = B.gy.ptr<float>(k);
                        float* lp = useLap ? B.lp.ptr<float>(k) : nullptr;
                        const float* tm = templateFloat.ptr<float>(y);
                        if (x1 <= x0) continue;
                        // the derivatives, the laplacian and the moments about the
                        // previous means, over the mask interval; the moments are
                        // float within the row, like the kernels, and double after
                        double sd = 0, sdd = 0, se = 0, see = 0, sde = 0;
                        int x = x0;
#if FASTECC_SIMD
                        {
                            const int VL = VTraits<v_float32>::vlanes();
                            const v_float32 vmI = vx_setall_f32(mI), vmT = vx_setall_f32(mT);
                            const v_float32 c8 = vx_setall_f32(8.f), c12 = vx_setall_f32(1.f / 12);
                            const v_float32 c4 = vx_setall_f32(4.f), ch = vx_setall_f32(0.5f);
                            v_float32 vsd = vx_setzero_f32(), vsdd = vx_setzero_f32(), vse = vx_setzero_f32();
                            v_float32 vsee = vx_setzero_f32(), vsde = vx_setzero_f32();
                            for (; x + VL <= x1; x += VL) {
                                const v_float32 c = vx_load(r0 + x), l1 = vx_load(r0 + x - 1), r1 = vx_load(r0 + x + 1);
                                const v_float32 u1 = vx_load(rm1 + x), d1 = vx_load(rp1 + x);
                                if (useGrad5) {
                                    v_store(gx + x, v_mul(v_sub(v_add(v_sub(vx_load(r0 + x - 2), v_mul(c8, l1)),
                                                                      v_mul(c8, r1)), vx_load(r0 + x + 2)), c12));
                                    v_store(gy + x, v_mul(v_sub(v_add(v_sub(vx_load(rm2 + x), v_mul(c8, u1)),
                                                                      v_mul(c8, d1)), vx_load(rp2 + x)), c12));
                                } else {
                                    v_store(gx + x, v_mul(ch, v_sub(r1, l1)));
                                    v_store(gy + x, v_mul(ch, v_sub(d1, u1)));
                                }
                                if (useLap)
                                    v_store(lp + x, v_sub(v_add(v_add(v_add(l1, r1), u1), d1), v_mul(c4, c)));
                                const v_float32 d = v_sub(c, vmI), e = v_sub(vx_load(tm + x), vmT);
                                vsd = v_add(vsd, d);  vsdd = v_fma(d, d, vsdd);
                                vse = v_add(vse, e);  vsee = v_fma(e, e, vsee);  vsde = v_fma(d, e, vsde);
                            }
                            sd = v_reduce_sum(vsd); sdd = v_reduce_sum(vsdd);
                            se = v_reduce_sum(vse); see = v_reduce_sum(vsee); sde = v_reduce_sum(vsde);
                        }
#endif
                        if (useGrad5) {
                            for (int xt = x; xt < x1; ++xt) {
                                gx[xt] = (r0[xt - 2] - 8.f * r0[xt - 1] + 8.f * r0[xt + 1] - r0[xt + 2]) * (1.f / 12);
                                gy[xt] = (rm2[xt] - 8.f * rm1[xt] + 8.f * rp1[xt] - rp2[xt]) * (1.f / 12);
                            }
                        } else {
                            for (int xt = x; xt < x1; ++xt) {
                                gx[xt] = 0.5f * (r0[xt + 1] - r0[xt - 1]);
                                gy[xt] = 0.5f * (rp1[xt] - rm1[xt]);
                            }
                        }
                        if (useLap)
                            for (int xt = x; xt < x1; ++xt)
                                lp[xt] = r0[xt - 1] + r0[xt + 1] + rm1[xt] + rp1[xt] - 4.f * r0[xt];
                        for (int xt = x; xt < x1; ++xt) {
                            const float d = r0[xt] - mI, e = tm[xt] - mT;
                            sd += d; sdd += d * d; se += e; see += e * e; sde += d * e;
                        }
                        st.n += x1 - x0; st.sd += sd; st.sdd += sdd; st.se += se; st.see += see; st.sde += sde;
                    }
                    // 5. the Gauss-Newton sums of this stripe
                    Mat imView(hsub, ws, CV_32F, B.im.ptr<float>(R), B.im.step);
                    Mat gxView = B.gx.rowRange(0, hsub), gyView = B.gy.rowRange(0, hsub);
                    Mat lpView = useLap ? B.lp.rowRange(0, hsub) : Mat();
                    Mat mkView = B.mk.rowRange(0, hsub);
                    parts[si].rows(gxView, gyView, imView, templateFloat, lpView, mkView, y0, y1, y0);
                }
            }, (double)nstripes);
            Acc total = proto;
            std::fill(total.H, total.H + Acc::P * (Acc::P + 1) / 2, 0.0);
            std::fill(total.VI, total.VI + Acc::P, 0.0);
            std::fill(total.VT, total.VT + Acc::P, 0.0);
            std::fill(total.VJ, total.VJ + Acc::P, 0.0);
            StripeStats st;
            for (int si = 0; si < nstripes; ++si) { total.add(parts[si]); st.add(stats[si]); }
            if (st.n == 0) {
              CV_Error(Error::StsNoConv, "NaN encountered.");
            }
            const double meanI = muPrevI + st.sd / st.n;
            const double meanT = muPrevT + st.se / st.n;
            imgNorm = std::sqrt(std::max(st.sdd - st.sd * st.sd / st.n, 0.0));
            tmpNorm = std::sqrt(std::max(st.see - st.se * st.se / st.n, 0.0));
            correlation = st.sde - st.sd * st.se / st.n;
            for (int q = 0; q < Acc::P; ++q) {
                total.VI[q] -= (meanI - muPrevI) * total.VJ[q];
                total.VT[q] -= (meanT - muPrevT) * total.VJ[q];
            }
            storeSystem(total, hessian, imageProjection, templateProjection);
            muPrevI = meanI; muPrevT = meanT;
        };
        auto run = [&](auto& acc) {
            if (fused)
                fusedIteration(acc);
            else
                runFusedGN(acc, imageWarpedGradientX, imageWarpedGradientY, imageWarped,
                           templateFloat, imageWarpedLaplacian, imageMask,
                           hessian, imageProjection, templateProjection);
        };
        if (motionType == MOTION_AFFINE)
        {
            auto setup = [&](auto& acc) {
                acc.r00 = r[0]; acc.r01 = r[1]; acc.r10 = r[2]; acc.r11 = r[3];
                acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
            };
            if (useLap) { GNAffineL acc; setup(acc); acc.lsc = lapScale; run(acc); }
            else        { GNAffine acc;  setup(acc); run(acc); }
        }
        else if (motionType == MOTION_TRANSLATION)
        {
            auto setup = [&](auto& acc) {
                acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
            };
            if (useLap) { GNTranslationL acc; setup(acc); acc.lsc = lapScale; run(acc); }
            else        { GNTranslation acc;  setup(acc); run(acc); }
        }
        else if (motionType == MOTION_EUCLIDEAN)
        {
            auto setup = [&](auto& acc) {
                acc.r00 = r[0]; acc.r01 = r[1]; acc.r10 = r[2]; acc.r11 = r[3];
                acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
                acc.h0 = h[0];   // cos(theta)
                acc.h1 = h[3];   // sin(theta)
            };
            if (useLap) { GNEuclideanL acc; setup(acc); acc.lsc = lapScale; run(acc); }
            else        { GNEuclidean acc;  setup(acc); run(acc); }
        }
        else if (motionType == MOTION_HOMOGRAPHY)
        {
            // the per-pixel projective division folds into the gradient side,
            // so the same face-splitting form applies with a third plane
            auto setup = [&](auto& acc) {
                acc.r00 = r[0]; acc.r01 = r[1]; acc.r10 = r[2]; acc.r11 = r[3];
                acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
                acc.h0 = h[0]; acc.h1 = h[3]; acc.h2 = h[6];
                acc.h3 = h[1]; acc.h4 = h[4]; acc.h5 = h[7];
                acc.h6 = h[2]; acc.h7 = h[5];
            };
            if (useLap) { GNHomographyL acc; setup(acc); acc.lsc = lapScale; run(acc); }
            else        { GNHomography acc;  setup(acc); run(acc); }
        }
        hessianInv = hessian.inv();

        // calculate enhanced correlation coefficient (ECC)->rho
        last_rho = rho;
        rho = correlation/(imgNorm*tmpNorm);
        if (cvIsNaN(rho)) {
          CV_Error(Error::StsNoConv, "NaN encountered.");
        }


        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv*imageProjection;
        const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0)
        {
            rho = -1;
            CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");
        }
        const double lambda = (lambda_n/lambda_d);

        // estimate the update step delta_p
        errorProjection = lambda * templateProjection - imageProjection;
        deltaP = hessianInv * errorProjection;

        // update warping matrix (the laplacian coefficient, if any, is not a
        // motion parameter and is simply discarded)
        update_warping_matrix_ECC( map, deltaP.rowRange(0, numberOfParameters), motionType);
    }

    // return final correlation coefficient
    return rho;
}

// Re-express a warp x' = W x in coordinates scaled by s (s = 2 going one
// pyramid level finer, 1/2^L going straight to level L): S^-1 W S with
// S = diag(s, s, 1).  The linear part is unchanged, the translation scales
// with s, the projective row against it.
void scaleWarp(Mat& map, double s)
{
    float* m = map.ptr<float>(0);
    m[2] = (float)(m[2] * s);
    m[5] = (float)(m[5] * s);
    if (map.rows == 3) {
        m[6] = (float)(m[6] / s);
        m[7] = (float)(m[7] / s);
    }
}

}  // namespace

static double findTransformECCImpl(InputArray templateImage,
                                   InputArray inputImage,
                                   InputOutputArray warpMatrix,
                                   int motionType,
                                   TermCriteria criteria,
                                   InputArray inputMask,
                                   int gaussFiltSize,
                                   int flags,
                                   int nlevels)
{
    Mat src = templateImage.getMat();//template image
    Mat dst = inputImage.getMat();  //input image (to be warped)
    Mat map = warpMatrix.getMat();  //warp (transformation)

    CV_Assert(!src.empty());
    CV_Assert(!dst.empty());

    // If the user passed an un-initialized warpMatrix, initialize to identity
    if(map.empty()) {
        int rowCount = 2;
        if(motionType == MOTION_HOMOGRAPHY)
            rowCount = 3;

        warpMatrix.create(rowCount, 3, CV_32FC1);
        map = warpMatrix.getMat();
        map = Mat::eye(rowCount, 3, CV_32F);
    }

    if( ! (src.type()==dst.type()))
        CV_Error( Error::StsUnmatchedFormats, "Both input images must have the same data type" );

    //accept only 1-channel images
    if( src.type() != CV_8UC1 && src.type()!= CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "Images must have 8uC1 or 32fC1 type");

    if( map.type() != CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

    CV_Assert (map.cols == 3);
    CV_Assert (map.rows == 2 || map.rows ==3);

    CV_Assert (motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
        motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);

    if (motionType == MOTION_HOMOGRAPHY){
        CV_Assert (map.rows ==3);
    }

    CV_Assert (criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps    = (criteria.type & TermCriteria::EPS)   ? criteria.epsilon  :  -1;
    CV_Assert (nlevels >= 1);

    Mat inputMaskMat = inputMask.getMat();
    if (nlevels == 1)
        return runSingleScale(src, dst, map, motionType, numberOfIterations, termination_eps,
                              inputMaskMat, gaussFiltSize, flags);

    // Coarse to fine.  The levels are pyrDown of the images (a 5-tap Gaussian
    // centred on the even pixels, so level-L coordinates are exactly the
    // full-resolution ones over 2^L); the pre-filter, the border ring, the
    // mask and the flags are all applied per level by runSingleScale, and the
    // stop criterion holds per level.  The coarsest level keeps at least 16 px
    // on the template's shorter side.
    int levels = nlevels;
    while (levels > 1 && (std::min(src.rows, src.cols) >> (levels - 1)) < 16) --levels;

    std::vector<Mat> srcPyr(levels), dstPyr(levels), maskPyr(levels);
    src.convertTo(srcPyr[0], CV_32F);
    dst.convertTo(dstPyr[0], CV_32F);
    if (!inputMaskMat.empty())
        threshold(inputMaskMat, maskPyr[0], 0, 255, THRESH_BINARY);
    for (int l = 1; l < levels; ++l) {
        pyrDown(srcPyr[l - 1], srcPyr[l]);
        pyrDown(dstPyr[l - 1], dstPyr[l]);
        if (!maskPyr[l - 1].empty()) {
            // a coarse pixel is valid only if everything under it was
            pyrDown(maskPyr[l - 1], maskPyr[l]);
            threshold(maskPyr[l], maskPyr[l], 254, 255, THRESH_BINARY);
        }
    }

    Mat mapL = map.clone();
    scaleWarp(mapL, 1.0 / (double)(1 << (levels - 1)));
    double rho = -1;
    for (int l = levels - 1; l >= 0; --l) {
        if (l < levels - 1) scaleWarp(mapL, 2.0);
        if (l > 0) {
            // A coarse level can have nothing left to align (a band-limited
            // scene, a small template): if it gives up, keep the warp it
            // started from and let the next finer level try.  Only the
            // finest level's failure is the call's failure.
            Mat before = mapL.clone();
            try {
                rho = runSingleScale(srcPyr[l], dstPyr[l], mapL, motionType, numberOfIterations,
                                     termination_eps, maskPyr[l], gaussFiltSize, flags);
            } catch (const cv::Exception&) {
                before.copyTo(mapL);
            }
        } else {
            rho = runSingleScale(srcPyr[l], dstPyr[l], mapL, motionType, numberOfIterations,
                                 termination_eps, maskPyr[l], gaussFiltSize, flags);
        }
    }
    mapL.copyTo(map);
    return rho;
}

#if FASTECC_AVX2_TU
// The eight-lane instance exports its entry point under another name; the
// first instance below reaches it through the dispatch.
namespace detail {
double findTransformECC_avx2(InputArray templateImage, InputArray inputImage,
                             InputOutputArray warpMatrix, int motionType,
                             TermCriteria criteria, InputArray inputMask,
                             int gaussFiltSize, int flags, int nlevels)
{
    return findTransformECCImpl(templateImage, inputImage, warpMatrix, motionType, criteria,
                                inputMask, gaussFiltSize, flags, nlevels);
}
}  // namespace detail
#else

#if FASTECC_HAVE_AVX2_TU
namespace detail {
double findTransformECC_avx2(InputArray, InputArray, InputOutputArray, int, TermCriteria,
                             InputArray, int, int, int);
}
// Decided once per process: the CPU must have AVX2 and FMA3, and
// FASTECC_NO_AVX2 in the environment keeps the run on this instance, so the
// two can be compared in one binary.
static bool useAvx2()
{
    static const bool ok = checkHardwareSupport(CV_CPU_AVX2) && checkHardwareSupport(CV_CPU_FMA3)
                           && std::getenv("FASTECC_NO_AVX2") == nullptr;
    return ok;
}
#endif

const char* vectorPath()
{
#if FASTECC_AVX2
    return "avx2";
#else
#if FASTECC_HAVE_AVX2_TU
    if (useAvx2()) return "avx2";
#endif
    return FASTECC_SIMD ? "simd128" : "scalar";
#endif
}

double findTransformECC(InputArray templateImage,
                        InputArray inputImage,
                        InputOutputArray warpMatrix,
                        int motionType,
                        TermCriteria criteria,
                        InputArray inputMask,
                        int gaussFiltSize,
                        int flags,
                        int nlevels)
{
#if FASTECC_HAVE_AVX2_TU
    if (useAvx2())
        return detail::findTransformECC_avx2(templateImage, inputImage, warpMatrix, motionType,
                                             criteria, inputMask, gaussFiltSize, flags, nlevels);
#endif
    return findTransformECCImpl(templateImage, inputImage, warpMatrix, motionType, criteria,
                                inputMask, gaussFiltSize, flags, nlevels);
}
#endif  // FASTECC_AVX2_TU

} // namespace fastecc
