// bench.cpp — time and accuracy of fastecc::findTransformECC vs cv::findTransformECC.
//
// Generates a synthetic textured image, applies a known warp to make the
// "input", then recovers the warp with both implementations. Reports per-call
// time, the recovered correlation rho, and the max |warp difference| between
// the two implementations (a small number means "same answer, just faster").
//
// Usage: bench [imageSize] [repeats]
#include "fast_ecc.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>  // cv::findTransformECC, MOTION_*
#include <chrono>
#include <cstdio>
#include <vector>

using namespace cv;
using clock_type = std::chrono::high_resolution_clock;

static Mat syntheticImage(int n) {
    // smooth, non-periodic texture so ECC has a well-defined optimum
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

static Mat groundTruth(int motionType) {
    Mat W = Mat::eye(motionType == MOTION_HOMOGRAPHY ? 3 : 2, 3, CV_32F);
    switch (motionType) {
        case MOTION_TRANSLATION:
            W.at<float>(0, 2) = 3.5f; W.at<float>(1, 2) = -2.5f; break;
        case MOTION_EUCLIDEAN: {
            float t = 0.03f;
            W.at<float>(0,0)=std::cos(t); W.at<float>(0,1)=-std::sin(t);
            W.at<float>(1,0)=std::sin(t); W.at<float>(1,1)= std::cos(t);
            W.at<float>(0,2)=3.0f; W.at<float>(1,2)=-2.0f; break;
        }
        case MOTION_AFFINE:
            W.at<float>(0,0)=1.01f; W.at<float>(0,1)=0.02f; W.at<float>(0,2)=3.0f;
            W.at<float>(1,0)=-0.015f; W.at<float>(1,1)=0.99f; W.at<float>(1,2)=-2.0f; break;
        case MOTION_HOMOGRAPHY:
            W.at<float>(0,0)=1.01f; W.at<float>(0,1)=0.02f; W.at<float>(0,2)=3.0f;
            W.at<float>(1,0)=-0.015f; W.at<float>(1,1)=0.99f; W.at<float>(1,2)=-2.0f;
            W.at<float>(2,0)=1e-5f; W.at<float>(2,1)=2e-5f; W.at<float>(2,2)=1.0f; break;
    }
    return W;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 512;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 20;

    Mat templ = syntheticImage(n);
    auto crit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 1e-4);

    const int motions[] = {MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY};
    const char* names[]  = {"TRANSLATION", "EUCLIDEAN", "AFFINE", "HOMOGRAPHY"};

    // corner RMS (in pixels) between two warps over the image extent
    auto cornerRMS = [&](const Mat& A, const Mat& B) {
        Mat A3 = A.rows == 2 ? (Mat_<double>(3,3) << A.at<float>(0,0),A.at<float>(0,1),A.at<float>(0,2),
                                                     A.at<float>(1,0),A.at<float>(1,1),A.at<float>(1,2), 0,0,1)
                             : Mat(A).clone();
        Mat B3 = B.rows == 2 ? (Mat_<double>(3,3) << B.at<float>(0,0),B.at<float>(0,1),B.at<float>(0,2),
                                                     B.at<float>(1,0),B.at<float>(1,1),B.at<float>(1,2), 0,0,1)
                             : Mat(B).clone();
        A3.convertTo(A3, CV_64F); B3.convertTo(B3, CV_64F);
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
    };

    std::printf("image %dx%d, %d repeats\n", n, n, repeats);
    std::printf("%-12s %8s %8s %9s %12s %12s\n",
                "motion", "cv(ms)", "fast(ms)", "speedup", "errCv(px)", "errFast(px)");

    for (int m = 0; m < 4; ++m) {
        Mat Wgt = groundTruth(motions[m]);
        // build the input image by warping the template by the GT warp
        Mat input;
        if (motions[m] != MOTION_HOMOGRAPHY)
            warpAffine(templ, input, Wgt, templ.size(), INTER_LINEAR);
        else
            warpPerspective(templ, input, Wgt, templ.size(), INTER_LINEAR);

        auto timeIt = [&](bool fast, Mat& outW, double& outRho) {
            double best = 1e30;
            for (int r = 0; r < repeats; ++r) {
                Mat W = Mat::eye(motions[m] == MOTION_HOMOGRAPHY ? 3 : 2, 3, CV_32F);
                auto t0 = clock_type::now();
                double rho = fast
                    ? fastecc::findTransformECC(templ, input, W, motions[m], crit)
                    : cv::findTransformECC(templ, input, W, motions[m], crit);
                auto t1 = clock_type::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best) { best = ms; outW = W.clone(); outRho = rho; }
            }
            return best;
        };

        Mat Wcv, Wfast; double rhoCv = 0, rhoFast = 0;
        double tcv   = timeIt(false, Wcv,   rhoCv);
        double tfast = timeIt(true,  Wfast, rhoFast);
        double errCv   = cornerRMS(Wcv,   Wgt);
        double errFast = cornerRMS(Wfast, Wgt);

        std::printf("%-12s %8.2f %8.2f %8.2fx %12.4f %12.4f\n",
                    names[m], tcv, tfast, tcv / tfast, errCv, errFast);
    }
    return 0;
}
