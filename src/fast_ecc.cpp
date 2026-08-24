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

#include "fast_ecc.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

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

double findTransformECC(InputArray templateImage,
                        InputArray inputImage,
                        InputOutputArray warpMatrix,
                        int motionType,
                        TermCriteria criteria,
                        InputArray inputMask,
                        int gaussFiltSize)
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

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped input image
    Mat imageMask     = Mat(hs, ws, CV_8U); // to store the final mask

    Mat inputMaskMat = inputMask.getMat();
    //to use it for mask warping
    Mat preMask;
    if(inputMask.empty())
        preMask = Mat::ones(hd, wd, CV_8U);
    else
        threshold(inputMask, preMask, 0, 1, THRESH_BINARY);

    //gaussian filtering is optional (sigma=0 -> derived from gaussFiltSize, matching cv::findTransformECC)
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

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

    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    // gradients of the warped image (raw: masked per pixel in the fused pass)
    Mat imageWarpedGradientX = Mat(hs, ws, CV_32FC1);
    Mat imageWarpedGradientY = Mat(hs, ws, CV_32FC1);

    // first order image derivatives (central difference)
    Matx13f dx(-0.5f, 0.0f, 0.5f);

    // matrices needed for solving linear equation system for maximizing ECC
    Mat hessian                 = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv              = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection         = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection      = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian  = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection         = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction

    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;

    // iteratively update map_matrix
    double rho      = -1;
    double last_rho = - termination_eps;
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho-last_rho)>= termination_eps); i++)
    {
        // Warp-back ONLY the image (and the mask).  The gradient images are NOT
        // warped: we take gradients of the warped image and recombine them with
        // the warp's linear part inside the fused pass.  This removes two warps
        // per iteration.
        if (motionType != MOTION_HOMOGRAPHY)
        {
            warpAffine(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
            warpAffine(preMask,    imageMask,   map, imageMask.size(),   maskFlags);
        }
        else
        {
            warpPerspective(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
            warpPerspective(preMask,    imageMask,   map, imageMask.size(),   maskFlags);
        }
        imageMask.row(0).setTo(0); imageMask.row(imageMask.rows - 1).setTo(0);
        imageMask.col(0).setTo(0); imageMask.col(imageMask.cols - 1).setTo(0);

        // The Gaussian pre-filter above extrapolated the template border
        // (BORDER_REFLECT_101), so a ring of gaussFiltSize/2 px of
        // templateFloat is fabricated rather than measured.  Used at full
        // weight it biases the estimated linear part toward a smaller scale.
        // Drop it, the same way opencv#29775 does in ecc.cpp.  The single
        // row/column above already covers a ring of 1.
        const int blurRing = gaussFiltSize / 2;
        if (blurRing > 1 && imageMask.rows > 2*blurRing && imageMask.cols > 2*blurRing) {
            imageMask.rowRange(0, blurRing).setTo(0);
            imageMask.rowRange(imageMask.rows - blurRing, imageMask.rows).setTo(0);
            imageMask.colRange(0, blurRing).setTo(0);
            imageMask.colRange(imageMask.cols - blurRing, imageMask.cols).setTo(0);
        }

        // gradients of the WARPED image (chain rule: d/dx[I(W(x))] = J_W^T grad(I)).
        // Left as they are: the fused pass masks them per pixel.
        filter2D(imageWarped, imageWarpedGradientX, -1, dx);
        filter2D(imageWarped, imageWarpedGradientY, -1, dx.t());

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
        const double imgMean = st.si / st.n;
        const double tmpMean = st.st / st.n;
        const double imgNorm = std::sqrt(std::max(st.sii - st.si * st.si / st.n, 0.0));
        const double tmpNorm = std::sqrt(std::max(st.stt - st.st * st.st / st.n, 0.0));
        const double correlation = st.sit - st.si * st.st / st.n;

        // The Gram matrix and both projections come out of a single pass over
        // the raw planes, with no jacobian materialised and nothing centred,
        // masked or recombined in memory first.
        float r[4];
        recombination(map, motionType, r);
        if (motionType == MOTION_AFFINE)
        {
            GNAffine acc;
            acc.r00 = r[0]; acc.r01 = r[1]; acc.r10 = r[2]; acc.r11 = r[3];
            acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
            runFusedGN(acc, imageWarpedGradientX, imageWarpedGradientY, imageWarped, templateFloat,
                       imageMask, hessian, imageProjection, templateProjection);
        }
        else if (motionType == MOTION_TRANSLATION)
        {
            GNTranslation acc;
            acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
            runFusedGN(acc, imageWarpedGradientX, imageWarpedGradientY, imageWarped, templateFloat,
                       imageMask, hessian, imageProjection, templateProjection);
        }
        else if (motionType == MOTION_EUCLIDEAN)
        {
            const float* h = map.ptr<float>(0);
            GNEuclidean acc;
            acc.r00 = r[0]; acc.r01 = r[1]; acc.r10 = r[2]; acc.r11 = r[3];
            acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
            acc.h0 = h[0];   // cos(theta)
            acc.h1 = h[3];   // sin(theta)
            runFusedGN(acc, imageWarpedGradientX, imageWarpedGradientY, imageWarped, templateFloat,
                       imageMask, hessian, imageProjection, templateProjection);
        }
        else if (motionType == MOTION_HOMOGRAPHY)
        {
            // the per-pixel projective division folds into the gradient side,
            // so the same face-splitting form applies with a third plane
            const float* h = map.ptr<float>(0);
            GNHomography acc;
            acc.r00 = r[0]; acc.r01 = r[1]; acc.r10 = r[2]; acc.r11 = r[3];
            acc.muI = (float)imgMean; acc.muT = (float)tmpMean;
            acc.h0 = h[0]; acc.h1 = h[3]; acc.h2 = h[6];
            acc.h3 = h[1]; acc.h4 = h[4]; acc.h5 = h[7];
            acc.h6 = h[2]; acc.h7 = h[5];
            runFusedGN(acc, imageWarpedGradientX, imageWarpedGradientY, imageWarped, templateFloat,
                       imageMask, hessian, imageProjection, templateProjection);
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

        // update warping matrix
        update_warping_matrix_ECC( map, deltaP, motionType);
    }

    // return final correlation coefficient
    return rho;
}

} // namespace fastecc
