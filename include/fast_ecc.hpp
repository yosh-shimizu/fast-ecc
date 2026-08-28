// fast_ecc.hpp
//
// A drop-in, faster variant of cv::findTransformECC (Evangelidis & Psarakis,
// PAMI 2008).  The per-iteration warps are reduced from three (image + two
// gradient images) to one by warping only the image and reconstructing the
// gradients from the *warped* image via the chain rule.  See README.md.
//
// API mirrors cv::findTransformECC so callers can switch by replacing the
// namespace (cv:: -> fastecc::).
//
// Derived from OpenCV modules/video/src/ecc.cpp (Intel/BSD-3 license).
// See LICENSE and NOTICE.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>  // cv::MOTION_* enum values

namespace fastecc {

// Optional behaviour, OR-ed into the `flags` argument.
enum {
    // Carry the laplacian of the warped image as one more Gauss-Newton column.
    // It absorbs the isotropic second-order term of the misalignment residual
    // and the bilinear interpolation error: the first step from a 1 px offset
    // lands at 0.07 px instead of 0.18, the basin widens by 10-40%, and on a
    // real image the fixed point moves toward the truth (2.4x lower corner
    // error on the bundled test).  Per iteration it costs one 3x3 filter and
    // one more column in the fused pass.
    FASTECC_LAPLACIAN_COLUMN = 1,
    // 4th-order 5-tap finite difference for the image gradient instead of the
    // 3-tap: the error contracts by ~0.06 per iteration instead of ~0.17, so
    // the accuracy floor takes 3 iterations instead of 4-5.
    FASTECC_GRAD5 = 2,
    // Force the multi-pass iteration (OpenCV warp, full-size gradient planes,
    // separate moment and Gauss-Newton passes) instead of the single-pass
    // one.  The two agree to the accuracy floor; the single pass samples with
    // exact bilinear weights where OpenCV's warp rounds to 1/32 px, and does
    // an iteration in one parallel region.  Kept for comparison.
    FASTECC_LEGACY_PIPELINE = 4,
    // Inverse compositional iteration (Baker & Matthews): the template side
    // is linearised instead of the warped image, so the jacobian rows, their
    // Gram matrix and their projection onto the template are computed once
    // per level from the template's gradient; an iteration then samples the
    // input once per pixel and projects it onto the fixed rows -- no
    // derivative of the warped image, no Gram matrix -- and updates the warp
    // by composition, W <- W o W(dp)^-1.  Same maximiser of rho as the
    // forward-additive iteration (the fixed point agrees to the accuracy
    // floor); the laplacian column, if any, is left out of the first
    // iteration of a level, where it spoils the first step.
    FASTECC_INVERSE_COMPOSITIONAL = 8,
    // what `flags` defaults to
    FASTECC_DEFAULT_FLAGS = FASTECC_LAPLACIAN_COLUMN | FASTECC_GRAD5
};
#define FASTECC_HAS_FLAGS 1
#define FASTECC_HAS_IC 1
#define FASTECC_HAS_PYRAMID 1
#define FASTECC_HAS_SINGLE_PASS 1

// Estimates the geometric transform `warpMatrix` (CV_32FC1, 2x3 or 3x3 for
// homography) that aligns `templateImage` to `inputImage`.  `motionType` is one
// of cv::MOTION_TRANSLATION / MOTION_EUCLIDEAN / MOTION_AFFINE / MOTION_HOMOGRAPHY.
// Returns the final enhanced correlation coefficient (rho).
//
// Not bit-identical to cv::findTransformECC (gradients are taken on the warped
// image, and resampling does not commute with differentiation), but recovers
// the transform with the same accuracy against ground truth — see README.md
// and the bundled equivalence test.
//
// `nlevels` > 1 runs coarse to fine over a pyrDown pyramid, like
// cv::findTransformECCMultiScale: each level runs the full single-scale
// iteration (pre-filter, border ring, flags, stop criterion) on the
// downsampled images and hands its warp to the next finer level.  The
// coarsest level keeps at least 16 px on the template's shorter side, so
// `nlevels` is reduced for small templates, and a coarse level that gives up
// is skipped.  The default is 3; 1 is the single-scale iteration.
double findTransformECC(
    cv::InputArray templateImage,
    cv::InputArray inputImage,
    cv::InputOutputArray warpMatrix,
    int motionType = cv::MOTION_AFFINE,
    cv::TermCriteria criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001),
    cv::InputArray inputMask = cv::noArray(),
    int gaussFiltSize = 5,
    int flags = FASTECC_DEFAULT_FLAGS,
    int nlevels = 3);

// The vector path this build takes on this machine: "avx2" (eight lanes with
// fused multiply-adds -- compiled in with FAST_ECC_AVX2, or picked at run
// time from the second instance the default build carries), "simd128"
// (OpenCV's universal intrinsics at 128 bits: SSE2 or NEON) or "scalar"
// (OpenCV before 4.7, or -DFASTECC_NO_SIMD).  FASTECC_NO_AVX2 in the
// environment keeps a run on the 128-bit instance, for comparison.
const char* vectorPath();

// The scratch of a call (the pyramid, the per-level planes, the stripe
// buffers) is kept per thread between calls and re-used whenever the sizes
// repeat -- touching fresh pages costs more than filling them, and from
// inside a parallel region it serialises the workers.  It grows to the
// largest call seen on the thread; this frees it.
void releaseWorkspace();

} // namespace fastecc
