// test_equivalence.cpp — asserts fastecc recovers the transform as accurately
// as cv::findTransformECC.
//
// The warp-reduction trick is NOT bit-identical to OpenCV (computing gradients
// of the warped image differs from warping precomputed gradients under bilinear
// resampling). So instead of comparing the two warps directly, we compare each
// against the known ground-truth warp: fastecc must be at least as accurate as
// cv (within a small pixel margin) and reach a high correlation.
#include "fast_ecc.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>  // cv::findTransformECC, MOTION_*
#include <cstdio>

using namespace cv;

static Mat syntheticImage(int n) {
    Mat img(n, n, CV_32F);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            float v = 0.5f + 0.25f * std::sin(x * 0.07f) * std::cos(y * 0.053f)
                           + 0.15f * std::sin((x + y) * 0.021f)
                           + 0.10f * std::cos((x - 2 * y) * 0.013f);
            img.at<float>(y, x) = v * 255.f;
        }
    GaussianBlur(img, img, Size(5, 5), 0);
    return img;
}

// corner RMS (pixels) between two warps over an n x n image
static double cornerRMS(const Mat& A, const Mat& B, int n) {
    auto to3 = [](const Mat& M) {
        Mat m;
        if (M.rows == 2)
            m = (Mat_<double>(3,3) << M.at<float>(0,0),M.at<float>(0,1),M.at<float>(0,2),
                                      M.at<float>(1,0),M.at<float>(1,1),M.at<float>(1,2), 0,0,1);
        else { M.convertTo(m, CV_64F); }
        return m;
    };
    Mat A3 = to3(A), B3 = to3(B);
    double pts[4][2] = {{0,0},{(double)n,0},{0,(double)n},{(double)n,(double)n}};
    double se = 0;
    for (auto& p : pts) {
        Mat v = (Mat_<double>(3,1) << p[0], p[1], 1.0);
        Mat a = A3 * v, b = B3 * v;
        a /= a.at<double>(2); b /= b.at<double>(2);
        double dx = a.at<double>(0)-b.at<double>(0), dy = a.at<double>(1)-b.at<double>(1);
        se += dx*dx + dy*dy;
    }
    return std::sqrt(se / 4.0);
}

int main() {
    const int n = 256;
    const double marginPx = 0.5;   // fastecc may differ from cv, but not be worse than this
    Mat templ = syntheticImage(n);
    auto crit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6);

    struct Case { int motion; const char* name; };
    Case cases[] = {
        {MOTION_TRANSLATION, "TRANSLATION"},
        {MOTION_EUCLIDEAN,   "EUCLIDEAN"},
        {MOTION_AFFINE,      "AFFINE"},
        {MOTION_HOMOGRAPHY,  "HOMOGRAPHY"},
    };

    int failures = 0;
    for (const auto& c : cases) {
        Mat Wgt = Mat::eye(c.motion == MOTION_HOMOGRAPHY ? 3 : 2, 3, CV_32F);
        Wgt.at<float>(0, 2) = 3.0f;
        Wgt.at<float>(1, 2) = -2.0f;
        if (c.motion == MOTION_AFFINE || c.motion == MOTION_HOMOGRAPHY) {
            Wgt.at<float>(0,0)=1.01f; Wgt.at<float>(1,1)=0.99f; Wgt.at<float>(0,1)=0.02f;
        }
        Mat input;
        if (c.motion != MOTION_HOMOGRAPHY) warpAffine(templ, input, Wgt, templ.size(), INTER_LINEAR);
        else                               warpPerspective(templ, input, Wgt, templ.size(), INTER_LINEAR);

        Mat Wcv   = Mat::eye(c.motion == MOTION_HOMOGRAPHY ? 3 : 2, 3, CV_32F);
        Mat Wfast = Mat::eye(c.motion == MOTION_HOMOGRAPHY ? 3 : 2, 3, CV_32F);
        cv::findTransformECC(templ, input, Wcv, c.motion, crit);
        double rho = fastecc::findTransformECC(templ, input, Wfast, c.motion, crit);

        double errCv   = cornerRMS(Wcv,   Wgt, n);
        double errFast = cornerRMS(Wfast, Wgt, n);
        bool ok = (errFast <= errCv + marginPx) && (rho >= 0.99);
        std::printf("[%-11s] errCv=%.4f px  errFast=%.4f px  rho(fast)=%.6f  -> %s\n",
                    c.name, errCv, errFast, rho, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }
    std::printf("%s\n", failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
