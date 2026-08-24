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

// Estimates the geometric transform `warpMatrix` (CV_32FC1, 2x3 or 3x3 for
// homography) that aligns `templateImage` to `inputImage`.  `motionType` is one
// of cv::MOTION_TRANSLATION / MOTION_EUCLIDEAN / MOTION_AFFINE / MOTION_HOMOGRAPHY.
// Returns the final enhanced correlation coefficient (rho).
//
// Not bit-identical to cv::findTransformECC (gradients are taken on the warped
// image, and resampling does not commute with differentiation), but recovers
// the transform with the same accuracy against ground truth — see README.md
// and the bundled equivalence test.
double findTransformECC(
    cv::InputArray templateImage,
    cv::InputArray inputImage,
    cv::InputOutputArray warpMatrix,
    int motionType = cv::MOTION_AFFINE,
    cv::TermCriteria criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001),
    cv::InputArray inputMask = cv::noArray(),
    int gaussFiltSize = 5);

// EXPERIMENT (branch sigma-param): the pre-filter's sigma as an optimisation
// parameter.  See src/fast_ecc_sigma.cpp for the modes and the caveats.
enum SigmaMode {
    SIGMA_FIXED = 0,       // sigma stays at sigma0 (control)
    SIGMA_JOINT = 1,       // sigma is a Gauss-Newton parameter (input-side column)
    SIGMA_JOINT_SYM = 2,   // same, with the template's sigma-derivative folded in
    SIGMA_SCHEDULE = 3,    // sigma <- max(sigmaMin, sigma * decay) per iteration
    SIGMA_NUISANCE = 4     // the sigma column is solved for but sigma is never changed
};

struct SigmaResult {
    double rho;        // final enhanced correlation coefficient
    double sigma;      // final sigma
    int iterations;    // iterations run
};

SigmaResult findTransformECCSigma(
    cv::InputArray templateImage,
    cv::InputArray inputImage,
    cv::InputOutputArray warpMatrix,
    int motionType,
    cv::TermCriteria criteria,
    double sigma0,
    int mode,
    double sigmaMin = 0.8,
    double sigmaMax = 6.0,
    double decay = 0.7);

} // namespace fastecc
