// fast_ecc_sigma.cpp -- the Gaussian pre-filter's sigma as an optimisation
// parameter.  EXPERIMENT (branch sigma-param); not part of the library API
// proper and not tuned for speed.
//
// The question: if sigma is carried alongside the motion parameters, with its
// own jacobian column, and the iteration starts from a large sigma, does it
// reach the accuracy floor in fewer iterations than a fixed sigma does?
//
// Both images are blurred with the same sigma, so the warped input is
// W_sigma(x; p) = (G_sigma * I)(phi(x; p)) and the template is G_sigma * T.
// The heat equation gives the sigma-derivative of a Gaussian-blurred image,
// d/dsigma (G_sigma * I) = sigma * laplacian(G_sigma * I), so the extra
// jacobian column is sigma times the warped laplacian of the blurred input.
// The template moves with sigma too; SIGMA_JOINT_SYM moves that term to the
// input side (column = input's derivative minus the template's), SIGMA_JOINT
// ignores it, i.e. treats the template as fixed within a step, which is how
// the paper's closed form is derived.
//
// Four modes share the code so that the mechanism can be separated from the
// mere fact of starting at a large sigma:
//
//   SIGMA_FIXED      sigma stays at sigma0                       (control)
//   SIGMA_JOINT      sigma is a Gauss-Newton parameter, input-side column
//   SIGMA_JOINT_SYM  same, with the template's sigma-derivative folded in
//   SIGMA_SCHEDULE   sigma <- max(sigmaMin, sigma * decay) per iteration
//   SIGMA_NUISANCE   the column is in the system, sigma itself never moves:
//                    the column then only absorbs the part of the residual
//                    that looks like a blur change, i.e. the isotropic
//                    second-order term of a misalignment, and no re-blur
//                    is ever needed
//
// The layout is the upstream one (image, both gradients and the laplacian
// are warped; the jacobian columns are materialised) so that nothing about
// fast-ecc's own reformulations enters the comparison.  The pre-filter's
// fabricated border ring is masked out, as in main.
//
// Set FASTECC_SIGMA_TRACE=N to print the (sigma, delta sigma, rho) trajectory
// of the first N calls to stderr; a per-mode summary (mean final sigma, mean
// iteration count) is printed at exit.

#include "fast_ecc.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

using namespace cv;

namespace fastecc {

namespace {

// what GaussianBlur picks for a CV_32F image when only sigma is given
int ksizeFor(double sigma) { return cvRound(sigma * 4 * 2 + 1) | 1; }

struct Blurred {
    Mat img, gx, gy, lap;   // G_sigma * src, its gradients, sigma * laplacian
};

void blurAt(const Mat& src, double sigma, bool wantGradients, Blurred& b) {
    GaussianBlur(src, b.img, Size(0, 0), sigma, sigma);
    if (wantGradients) {
        const Matx13f dx(-0.5f, 0.0f, 0.5f);
        filter2D(b.img, b.gx, -1, dx);
        filter2D(b.img, b.gy, -1, dx.t());
    }
    Laplacian(b.img, b.lap, CV_32F, 1);
    b.lap *= (float)sigma;
}

// the mask of pixels whose pre-filter support is entirely real data, as upstream
void maskAt(int hd, int wd, double sigma, Mat& preMask) {
    Mat ones = Mat::ones(hd, wd, CV_32F);
    GaussianBlur(ones, ones, Size(0, 0), sigma, sigma);
    ones *= (0.5 / 0.95);
    ones.convertTo(preMask, CV_8U);
    preMask.row(0).setTo(0); preMask.row(preMask.rows - 1).setTo(0);
    preMask.col(0).setTo(0); preMask.col(preMask.cols - 1).setTo(0);
    preMask.row(1).setTo(0); preMask.row(preMask.rows - 2).setTo(0);
    preMask.col(1).setTo(0); preMask.col(preMask.cols - 2).setTo(0);
}

// upstream jacobian columns, one plane each
void jacobianColumns(int motionType, const Mat& map, const Mat& gx, const Mat& gy,
                     const Mat& X, const Mat& Y, std::vector<Mat>& cols) {
    cols.clear();
    const float* h = map.ptr<float>(0);
    switch (motionType) {
    case MOTION_TRANSLATION:
        cols.push_back(gx); cols.push_back(gy);
        break;
    case MOTION_EUCLIDEAN: {
        const float h0 = h[0], h1 = h[3];               // cos, sin
        Mat hatX = -(X * h1) - (Y * h0);
        Mat hatY =  (X * h0) - (Y * h1);
        cols.push_back(gx.mul(hatX) + gy.mul(hatY));
        cols.push_back(gx); cols.push_back(gy);
        break;
    }
    case MOTION_AFFINE:
        cols.push_back(gx.mul(X)); cols.push_back(gy.mul(X));
        cols.push_back(gx.mul(Y)); cols.push_back(gy.mul(Y));
        cols.push_back(gx); cols.push_back(gy);
        break;
    case MOTION_HOMOGRAPHY: {
        const float h0 = h[0], h1 = h[3], h2 = h[6];
        const float h3 = h[1], h4 = h[4], h5 = h[7];
        const float h6 = h[2], h7 = h[5];
        Mat den;
        addWeighted(X, h2, Y, h5, 1.0, den);
        Mat hatX, hatY;
        addWeighted(X, h0, Y, h3, 0.0, hatX); hatX += h6;
        addWeighted(X, h1, Y, h4, 0.0, hatY); hatY += h7;
        divide(-hatX, den, hatX);
        divide(-hatY, den, hatY);
        Mat gxd, gyd;
        divide(gx, den, gxd);
        divide(gy, den, gyd);
        Mat temp = hatX.mul(gxd) + hatY.mul(gyd);
        cols.push_back(gxd.mul(X)); cols.push_back(gyd.mul(X)); cols.push_back(temp.mul(X));
        cols.push_back(gxd.mul(Y)); cols.push_back(gyd.mul(Y)); cols.push_back(temp.mul(Y));
        cols.push_back(gxd); cols.push_back(gyd);
        break;
    }
    default:
        CV_Error(Error::StsBadArg, "unknown motion type");
    }
}

void updateWarp(Mat& map, const float* u, int motionType) {
    float* m = map.ptr<float>(0);
    switch (motionType) {
    case MOTION_TRANSLATION:
        m[2] += u[0]; m[5] += u[1];
        break;
    case MOTION_AFFINE:
        m[0] += u[0]; m[3] += u[1]; m[1] += u[2]; m[4] += u[3]; m[2] += u[4]; m[5] += u[5];
        break;
    case MOTION_HOMOGRAPHY:
        m[0] += u[0]; m[3] += u[1]; m[6] += u[2];
        m[1] += u[3]; m[4] += u[4]; m[7] += u[5];
        m[2] += u[6]; m[5] += u[7];
        break;
    case MOTION_EUCLIDEAN: {
        double th = u[0] + std::asin((double)m[3]);
        m[2] += u[1]; m[5] += u[2];
        m[0] = m[4] = (float)std::cos(th);
        m[3] = (float)std::sin(th);
        m[1] = -m[3];
        break;
    }
    }
}

// ---- trace / summary -------------------------------------------------------

struct Summary { long calls = 0; double sumSigma = 0; long sumIters = 0; };
std::map<std::pair<int, int>, Summary>& summaries() {
    static std::map<std::pair<int, int>, Summary> s;
    return s;
}
void printSummaries() {
    for (const auto& kv : summaries()) {
        const Summary& s = kv.second;
        std::fprintf(stderr, "[sigma] mode=%d sigma0=%.2f calls=%ld  mean final sigma=%.3f  mean iters=%.2f\n",
                     kv.first.first, kv.first.second / 100.0, s.calls,
                     s.sumSigma / std::max(1L, s.calls), (double)s.sumIters / std::max(1L, s.calls));
    }
}
int traceBudget() {
    static int n = -1;
    if (n < 0) {
        const char* e = std::getenv("FASTECC_SIGMA_TRACE");
        n = e ? std::atoi(e) : 0;
        std::atexit(printSummaries);
    }
    return n;
}

}  // namespace

SigmaResult findTransformECCSigma(InputArray templateImage,
                                  InputArray inputImage,
                                  InputOutputArray warpMatrix,
                                  int motionType,
                                  TermCriteria criteria,
                                  double sigma0,
                                  int mode,
                                  double sigmaMin,
                                  double sigmaMax,
                                  double decay)
{
    Mat src = templateImage.getMat();
    Mat dst = inputImage.getMat();
    Mat map = warpMatrix.getMat();

    CV_Assert(!src.empty() && !dst.empty());
    if (map.empty()) {
        const int rows = motionType == MOTION_HOMOGRAPHY ? 3 : 2;
        warpMatrix.create(rows, 3, CV_32FC1);
        map = warpMatrix.getMat();
        map = Mat::eye(rows, 3, CV_32F);
    }
    CV_Assert(src.type() == dst.type());
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);
    CV_Assert(map.type() == CV_32FC1 && map.cols == 3 && (map.rows == 2 || map.rows == 3));
    CV_Assert(motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
              motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);
    if (motionType == MOTION_HOMOGRAPHY) CV_Assert(map.rows == 3);
    CV_Assert(criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    CV_Assert(mode >= SIGMA_FIXED && mode <= SIGMA_NUISANCE);
    CV_Assert(sigma0 > 0 && sigmaMin > 0 && sigmaMax >= sigmaMin);

    const int    maxIters = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double eps      = (criteria.type & TermCriteria::EPS)   ? criteria.epsilon  : -1;
    const bool   joint    = mode == SIGMA_JOINT || mode == SIGMA_JOINT_SYM || mode == SIGMA_NUISANCE;
    const bool   moveSig  = mode == SIGMA_JOINT || mode == SIGMA_JOINT_SYM;

    const int P = motionType == MOTION_TRANSLATION ? 2 : motionType == MOTION_EUCLIDEAN ? 3 :
                  motionType == MOTION_HOMOGRAPHY ? 8 : 6;
    const int ws = src.cols, hs = src.rows, wd = dst.cols, hd = dst.rows;

    Mat templateFloat, imageFloat;
    src.convertTo(templateFloat, CV_32F);
    dst.convertTo(imageFloat, CV_32F);

    Mat X(hs, ws, CV_32F), Y(hs, ws, CV_32F);
    for (int y = 0; y < hs; ++y) {
        float* px = X.ptr<float>(y);
        float* py = Y.ptr<float>(y);
        for (int x = 0; x < ws; ++x) { px[x] = (float)x; py[x] = (float)y; }
    }

    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;

    double sigma = std::min(std::max(sigma0, sigmaMin), sigmaMax);
    double lastDSigma = 0;
    bool reblur = true;
    Blurred T, I;
    Mat preMask;
    int ring = 1;

    Mat imageWarped(hs, ws, CV_32F), gxW(hs, ws, CV_32F), gyW(hs, ws, CV_32F), lapW(hs, ws, CV_32F);
    Mat mask(hs, ws, CV_8U), templateZM(hs, ws, CV_32F);
    std::vector<Mat> cols;

    const bool trace = traceBudget() > 0;
    static int traced = 0;
    const bool traceThis = trace && traced < traceBudget();
    if (traceThis) ++traced;

    double rho = -1, lastRho = -eps;
    int iters = 0;
    for (int i = 1; i <= maxIters; ++i) {
        // the |delta rho| stop only means something once sigma has settled:
        // rho at two different sigmas are not comparable
        const bool sigmaSettled = !moveSig || i == 1 || std::fabs(lastDSigma) < 0.02;
        if (i > 1 && sigmaSettled && std::fabs(rho - lastRho) < eps) break;
        iters = i;

        if (reblur) {
            blurAt(templateFloat, sigma, false, T);
            blurAt(imageFloat, sigma, true, I);
            maskAt(hd, wd, sigma, preMask);
            ring = std::max(1, ksizeFor(sigma) / 2);
            reblur = false;
        }

        if (motionType != MOTION_HOMOGRAPHY) {
            warpAffine(I.img, imageWarped, map, imageWarped.size(), imageFlags);
            warpAffine(I.gx,  gxW,  map, gxW.size(),  imageFlags);
            warpAffine(I.gy,  gyW,  map, gyW.size(),  imageFlags);
            warpAffine(I.lap, lapW, map, lapW.size(), imageFlags);
            warpAffine(preMask, mask, map, mask.size(), maskFlags);
        } else {
            warpPerspective(I.img, imageWarped, map, imageWarped.size(), imageFlags);
            warpPerspective(I.gx,  gxW,  map, gxW.size(),  imageFlags);
            warpPerspective(I.gy,  gyW,  map, gyW.size(),  imageFlags);
            warpPerspective(I.lap, lapW, map, lapW.size(), imageFlags);
            warpPerspective(preMask, mask, map, mask.size(), maskFlags);
        }
        // the template's fabricated ring (opencv#29775), plus the border
        if (2 * ring < hs && 2 * ring < ws) {
            mask.rowRange(0, ring).setTo(0); mask.rowRange(hs - ring, hs).setTo(0);
            mask.colRange(0, ring).setTo(0); mask.colRange(ws - ring, ws).setTo(0);
        }
        const Mat outside = (mask == 0);
        gxW.setTo(0.f, outside);
        gyW.setTo(0.f, outside);
        lapW.setTo(0.f, outside);
        imageWarped.setTo(0.f, outside);

        Scalar imgMean, imgStd, tmpMean, tmpStd;
        meanStdDev(imageWarped, imgMean, imgStd, mask);
        meanStdDev(T.img, tmpMean, tmpStd, mask);
        subtract(imageWarped, imgMean, imageWarped, mask);
        templateZM = Mat::zeros(hs, ws, CV_32F);
        subtract(T.img, tmpMean, templateZM, mask);
        const int n = countNonZero(mask);
        const double tmpNorm = std::sqrt(n * tmpStd.val[0] * tmpStd.val[0]);
        const double imgNorm = std::sqrt(n * imgStd.val[0] * imgStd.val[0]);

        jacobianColumns(motionType, map, gxW, gyW, X, Y, cols);
        if (joint) {
            Mat w = lapW.clone();
            if (mode == SIGMA_JOINT_SYM) {
                Mat tl = Mat::zeros(hs, ws, CV_32F);
                T.lap.copyTo(tl, mask);
                w -= tl;
            }
            cols.push_back(w);
        }
        const int Q = (int)cols.size();

        Mat H(Q, Q, CV_64F), imageProjection(Q, 1, CV_64F), templateProjection(Q, 1, CV_64F);
        for (int a = 0; a < Q; ++a) {
            for (int b = a; b < Q; ++b) {
                const double v = cols[a].dot(cols[b]);
                H.at<double>(a, b) = v; H.at<double>(b, a) = v;
            }
            imageProjection.at<double>(a, 0)    = cols[a].dot(imageWarped);
            templateProjection.at<double>(a, 0) = cols[a].dot(templateZM);
        }
        Mat Hinv = H.inv();

        const double correlation = templateZM.dot(imageWarped);
        lastRho = rho;
        rho = correlation / (imgNorm * tmpNorm);
        if (cvIsNaN(rho)) CV_Error(Error::StsNoConv, "NaN encountered.");

        Mat iph = Hinv * imageProjection;
        const double lambda_n = imgNorm * imgNorm - imageProjection.dot(iph);
        const double lambda_d = correlation - templateProjection.dot(iph);
        if (lambda_d <= 0.0) {
            rho = -1;
            CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");
        }
        const double lambda = lambda_n / lambda_d;
        Mat delta = Hinv * (lambda * templateProjection - imageProjection);

        float u[9];
        for (int a = 0; a < P; ++a) u[a] = (float)delta.at<double>(a, 0);
        updateWarp(map, u, motionType);

        double sigmaNew = sigma;
        if (joint) {
            lastDSigma = delta.at<double>(P, 0);
            if (moveSig) sigmaNew = std::min(std::max(sigma + lastDSigma, sigmaMin), sigmaMax);
        } else if (mode == SIGMA_SCHEDULE) {
            sigmaNew = std::max(sigmaMin, sigma * decay);
        }
        if (traceThis)
            std::fprintf(stderr, "[sigma] call %d iter %2d  sigma %.3f -> %.3f  (dsigma %+.4f)  rho %.6f\n",
                         traced, i, sigma, sigmaNew, joint ? lastDSigma : sigmaNew - sigma, rho);
        if (sigmaNew != sigma) { sigma = sigmaNew; reblur = true; }
    }

    Summary& s = summaries()[std::make_pair(mode, cvRound(sigma0 * 100))];
    s.calls += 1; s.sumSigma += sigma; s.sumIters += iters;

    SigmaResult r;
    r.rho = rho; r.sigma = sigma; r.iterations = iters;
    return r;
}

}  // namespace fastecc
