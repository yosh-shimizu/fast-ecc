// test_equivalence.cpp -- does fastecc recover the transform as accurately as
// cv::findTransformECC?
//
// The warp reduction is NOT bit-identical to OpenCV (gradients of the warped
// image differ from warped gradients under bilinear resampling), so the two
// warps cannot be compared entry by entry.  Both are compared against the known
// ground truth instead.
//
// Three things this checks that a single "is fastecc worse than cv" comparison
// does not:
//
//   1. an ABSOLUTE threshold.  Comparing only against cv passes whenever both
//      regress together, and says nothing when cv itself is wrong.
//   2. a SWEEP over gaussFiltSize, checked for GROWTH.  The pre-filter
//      fabricates a border ring of gaussFiltSize/2 px; if it is ever used as
//      data, error grows with the filter width.  More smoothing making the
//      answer worse is the signature of that defect, and no single-width test
//      can see it.  (Wider filters are legitimately harder at gaussFiltSize=3
//      for both implementations, so the absolute check is applied at the
//      default width and the sweep is checked for trend instead.)
//   3. SEVERAL sub-pixel phases per case, averaged.  One alignment is one draw
//      from a distribution whose spread is comparable to the quantity measured.
//   4. every combination of the optional flags.  The defaults (laplacian
//      column + 5-tap gradient) are what a caller gets, but the plain kernels
//      must keep working too, and each flag on its own.
//   5. a MARGIN of real data around the template.  The input is rendered from a
//      source larger than the template, so the template's support in the input
//      is surrounded by kMargin px of genuine content.  Without that, an input
//      made by warping the template onto itself carries a zero-filled strip
//      along two edges, the pre-filter smears it inward by gaussFiltSize/2, and
//      the sweep ends up measuring how far each width's mask ring happens to
//      sit from that strip rather than the estimator (affine at gauss 3 read
//      0.23 px that way, 0.002 px with a margin).
//
// The tolerances below are roughly 3x the measured capability.  They are meant
// to catch a real regression, not to be decoration: a 3x accuracy loss fails.

#include "fast_ecc.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>  // cv::findTransformECC, MOTION_*

#include <cstdio>
#include <vector>

using namespace cv;

namespace {

const int    kSize       = 256;  // side of the square template
const int    kMargin     = 16;   // px of real data around the template's support
const int    kPhases     = 8;    // sub-pixel phases averaged per case
const int    kDefaultGauss = 5;  // where the absolute tolerance applies
const double kVsCvSlack  = 1.6;  // fastecc may be this much worse than cv...
const double kVsCvFloor  = 0.004;// ...plus this, to absorb Monte-Carlo spread
const double kGrowthSlack = 1.3; // widening the filter may cost this much...
const double kGrowthFloor = 0.002;// ...plus this, before it counts as growth

Mat syntheticImage(int n) {
    Mat img(n, n, CV_32F);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const float v = 0.5f + 0.25f * std::sin(x * 0.07f) * std::cos(y * 0.053f)
                                 + 0.15f * std::sin((x + y) * 0.021f)
                                 + 0.10f * std::cos((x - 2 * y) * 0.013f);
            img.at<float>(y, x) = v * 255.f;
        }
    }
    GaussianBlur(img, img, Size(5, 5), 0);
    return img;
}

// RMS displacement of the four image corners between two warps, in pixels
double cornerRMS(const Mat& A, const Mat& B, int n) {
    auto to3 = [](const Mat& M) {
        Mat m;
        if (M.rows == 2) {
            m = (Mat_<double>(3,3) << M.at<float>(0,0), M.at<float>(0,1), M.at<float>(0,2),
                                      M.at<float>(1,0), M.at<float>(1,1), M.at<float>(1,2),
                                      0, 0, 1);
        } else {
            M.convertTo(m, CV_64F);
        }
        return m;
    };
    const Mat A3 = to3(A), B3 = to3(B);
    const double pts[4][2] = {{0,0}, {(double)n,0}, {0,(double)n}, {(double)n,(double)n}};
    double squaredSum = 0;
    for (int i = 0; i < 4; ++i) {
        const Mat v = (Mat_<double>(3,1) << pts[i][0], pts[i][1], 1.0);
        Mat a = A3 * v, b = B3 * v;
        a /= a.at<double>(2);
        b /= b.at<double>(2);
        const double dx = a.at<double>(0) - b.at<double>(0);
        const double dy = a.at<double>(1) - b.at<double>(1);
        squaredSum += dx*dx + dy*dy;
    }
    return std::sqrt(squaredSum / 4.0);
}

struct MotionCase {
    int         motion;
    const char* name;
    double      tolerancePx;  // absolute, against ground truth
};

// template coords -> input coords.  The nominal placement is the margin
// translation; the motion to recover on top of it is (3, -2) px plus the
// sub-pixel phase, and the linear part below.
Mat groundTruth(int motion, double shiftX, double shiftY) {
    const int rows = (motion == MOTION_HOMOGRAPHY) ? 3 : 2;
    Mat W = Mat::eye(rows, 3, CV_32F);
    W.at<float>(0, 2) = (float)(kMargin + 3.0 + shiftX);
    W.at<float>(1, 2) = (float)(kMargin - 2.0 + shiftY);
    if (motion == MOTION_EUCLIDEAN) {
        const double th = 0.4 * CV_PI / 180.0;
        W.at<float>(0,0) =  (float)std::cos(th); W.at<float>(0,1) = -(float)std::sin(th);
        W.at<float>(1,0) =  (float)std::sin(th); W.at<float>(1,1) =  (float)std::cos(th);
    } else if (motion == MOTION_AFFINE || motion == MOTION_HOMOGRAPHY) {
        W.at<float>(0,0) = 1.01f; W.at<float>(1,1) = 0.99f; W.at<float>(0,1) = 0.02f;
    }
    if (motion == MOTION_HOMOGRAPHY) {
        // an actual projective component, so the case is not just affine in a
        // 3x3 container
        W.at<float>(2,0) = 2e-5f;
        W.at<float>(2,1) = -1e-5f;
    }
    return W;
}

}  // namespace

struct FlagSet { int flags; const char* name; };

int main() {
    const Mat source = syntheticImage(kSize + 2 * kMargin);
    const Mat templ  = source(Rect(kMargin, kMargin, kSize, kSize)).clone();
    const TermCriteria crit(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6);

    const FlagSet flagSets[] = {
        {fastecc::FASTECC_DEFAULT_FLAGS,                          "default (laplacian column + 5-tap)"},
        {0,                                                       "plain"},
        {fastecc::FASTECC_LAPLACIAN_COLUMN,                       "laplacian column"},
        {fastecc::FASTECC_GRAD5,                                  "5-tap"},
    };

    const MotionCase cases[] = {
        // ~3x what the margin test measures (0.016 / 0.002 / 0.0036 / 0.005 px
        // at gauss 5; translation sits on the 1/32 px coordinate quantisation
        // of OpenCV's bilinear warp, the others below it)
        {MOTION_TRANSLATION, "TRANSLATION", 0.05},
        {MOTION_EUCLIDEAN,   "EUCLIDEAN",   0.006},
        {MOTION_AFFINE,      "AFFINE",      0.012},
        {MOTION_HOMOGRAPHY,  "HOMOGRAPHY",  0.015},
    };
    const int gaussSizes[] = {3, 5, 9};
    const int kNGauss = 3;

    int failures = 0;
    for (int fi = 0; fi < 4; ++fi) {
    const int flags = flagSets[fi].flags;
    std::printf("\n== flags = %d: %s ==\n", flags, flagSets[fi].name);
    std::printf("%-12s %6s %11s %11s %9s %8s   %s\n",
                "motion", "gauss", "errCv px", "errFast px", "rho", "tol px", "verdict");

    for (int ci = 0; ci < 4; ++ci) {
        const MotionCase& c = cases[ci];
        double errFastByGauss[3] = {0, 0, 0};
        bool   measured[3] = {false, false, false};

        for (int gi = 0; gi < kNGauss; ++gi) {
            const int gauss = gaussSizes[gi];
            RNG rng(7654321);  // same phases for every configuration, and for CI
            double sumCv = 0, sumFast = 0, worstRho = 1.0;
            int done = 0;

            for (int phase = 0; phase < kPhases; ++phase) {
                const double sx = rng.uniform(-0.5, 0.5), sy = rng.uniform(-0.5, 0.5);
                const Mat Wgt = groundTruth(c.motion, sx, sy);

                // I(p) = S(W^-1 p + margin), so that I(W x) = S(x + margin) = T(x)
                Mat W3 = Mat::eye(3, 3, CV_64F);
                Wgt.convertTo(W3.rowRange(0, Wgt.rows), CV_64F);
                Mat shift = Mat::eye(3, 3, CV_64F);
                shift.at<double>(0, 2) = kMargin;
                shift.at<double>(1, 2) = kMargin;
                Mat M = shift * W3.inv();
                Mat input;
                if (c.motion != MOTION_HOMOGRAPHY) {
                    warpAffine(source, input, M.rowRange(0, 2), source.size(),
                               INTER_LINEAR + WARP_INVERSE_MAP);
                } else {
                    warpPerspective(source, input, M, source.size(),
                                    INTER_LINEAR + WARP_INVERSE_MAP);
                }

                // start from the nominal placement: the margin translation only
                const int rows = (c.motion == MOTION_HOMOGRAPHY) ? 3 : 2;
                Mat Wcv   = Mat::eye(rows, 3, CV_32F);
                Wcv.at<float>(0, 2) = kMargin;
                Wcv.at<float>(1, 2) = kMargin;
                Mat Wfast = Wcv.clone();
                double rho = 0;
                try {
                    cv::findTransformECC(templ, input, Wcv, c.motion, crit, noArray(), gauss);
                    rho = fastecc::findTransformECC(templ, input, Wfast, c.motion, crit,
                                                    noArray(), gauss, flags);
                } catch (const cv::Exception& e) {
                    std::printf("%-12s %6d   EXCEPTION: %s\n", c.name, gauss, e.what());
                    ++failures;
                    break;
                }
                sumCv   += cornerRMS(Wcv,   Wgt, kSize);
                sumFast += cornerRMS(Wfast, Wgt, kSize);
                if (rho < worstRho) worstRho = rho;
                ++done;
            }
            if (done != kPhases) continue;

            const double errCv   = sumCv   / done;
            const double errFast = sumFast / done;
            errFastByGauss[gi] = errFast;
            measured[gi] = true;

            // (1) absolute, at the default width: fastecc must actually be
            //     accurate, not merely no worse than whatever cv did.
            const bool checkAbs   = (gauss == kDefaultGauss);
            const bool okAbsolute = !checkAbs || errFast <= c.tolerancePx;
            // (2) relative: and it must not fall behind cv by a real margin.
            const bool okVsCv     = errFast <= errCv * kVsCvSlack + kVsCvFloor;
            const bool okRho      = worstRho >= 0.99;
            const bool ok = okAbsolute && okVsCv && okRho;

            std::printf("%-12s %6d %11.4f %11.4f %9.6f %8s   %s%s%s%s\n",
                        c.name, gauss, errCv, errFast, worstRho,
                        checkAbs ? cv::format("%.3f", c.tolerancePx).c_str() : "-",
                        ok ? "PASS" : "FAIL",
                        okAbsolute ? "" : " [over absolute tolerance]",
                        okVsCv     ? "" : " [behind cv]",
                        okRho      ? "" : " [rho too low]");
            if (!ok) ++failures;
        }

        // (3) trend: a wider pre-filter must not make the answer worse.  It
        //     does exactly that when the fabricated border ring is used as
        //     data, so this is what guards against that defect returning.
        for (int gi = 0; gi + 1 < kNGauss; ++gi) {
            if (!measured[gi] || !measured[gi + 1]) continue;
            const double a = errFastByGauss[gi], b = errFastByGauss[gi + 1];
            if (b > a * kGrowthSlack + kGrowthFloor) {
                std::printf("%-12s   gauss %d -> %d: error GREW %.4f -> %.4f px"
                            "  [border ring used as data?]\n",
                            c.name, gaussSizes[gi], gaussSizes[gi + 1], a, b);
                ++failures;
            }
        }
    }
    }

    std::printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
