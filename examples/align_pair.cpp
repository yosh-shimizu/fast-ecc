// align_pair.cpp — minimal usage example.
//
// Aligns `input` to `template` and writes the warped result + the warp matrix.
//
// Usage: align_pair <template.png> <input.png> [out.png] [motion]
//   motion = translation | euclidean | affine | homography   (default: affine)
#include "fast_ecc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <string>

using namespace cv;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <template> <input> [out.png] [motion]\n", argv[0]);
        return 1;
    }
    Mat templ = imread(argv[1], IMREAD_GRAYSCALE);
    Mat input = imread(argv[2], IMREAD_GRAYSCALE);
    if (templ.empty() || input.empty()) { std::printf("could not read images\n"); return 1; }

    std::string outPath = argc > 3 ? argv[3] : "aligned.png";
    std::string motionStr = argc > 4 ? argv[4] : "affine";
    int motion = MOTION_AFFINE;
    if (motionStr == "translation") motion = MOTION_TRANSLATION;
    else if (motionStr == "euclidean") motion = MOTION_EUCLIDEAN;
    else if (motionStr == "homography") motion = MOTION_HOMOGRAPHY;

    Mat W = Mat::eye(motion == MOTION_HOMOGRAPHY ? 3 : 2, 3, CV_32F);
    double rho = fastecc::findTransformECC(
        templ, input, W, motion,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

    Mat aligned;
    int flags = INTER_LINEAR + WARP_INVERSE_MAP;
    if (motion != MOTION_HOMOGRAPHY)
        warpAffine(input, aligned, W, templ.size(), flags);
    else
        warpPerspective(input, aligned, W, templ.size(), flags);

    imwrite(outPath, aligned);
    std::printf("rho = %.6f, wrote %s\n", rho, outPath.c_str());
    std::printf("warp =\n");
    for (int r = 0; r < W.rows; ++r) {
        for (int c = 0; c < W.cols; ++c) std::printf("  % .6f", W.at<float>(r, c));
        std::printf("\n");
    }
    return 0;
}
