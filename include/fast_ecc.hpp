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
    // what `flags` defaults to
    FASTECC_DEFAULT_FLAGS = FASTECC_LAPLACIAN_COLUMN | FASTECC_GRAD5
};
#define FASTECC_HAS_FLAGS 1
#define FASTECC_HAS_PYRAMID 1

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
// `nlevels` is reduced for small templates.  1 (the default) is the
// single-scale iteration exactly as before.
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
    int nlevels = 1);

} // namespace fastecc
