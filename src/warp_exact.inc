// ---------------------------------------------------------------------------
// A bilinear warp that does not quantise the sampling coordinate.
//
// NOT built by default.  Configure with -DFASTECC_EXACT_WARP=ON.
//
// OpenCV rounds the fractional part of the sampling coordinate to
// INTER_TAB_SIZE = 32 steps before it looks up the bilinear weights, and it
// does that in the coordinate arithmetic rather than in the pixel type.  So the
// sampled value is a step function of a sub-pixel shift with 1/32 px treads.
// Sweeping a shift across a span and counting distinct outputs (eval/dtest.cpp,
// 65 samples):
//
//     span        1/32 treads   warpAffine 32F   warpAffine 64F   remap 32F   exact
//     1/32 px               2                2                2           2      65
//     1/8  px               5                5                5           5      65
//     1    px              33               33               33          33      65
//
// Every OpenCV path quantises, including remap with an explicit float
// coordinate map, and CV_64F images do not help.
//
// Why it matters most for translation
// -----------------------------------
// Under a pure translation every pixel shares the same fractional coordinate,
// so the quantisation error is identical at every pixel and does not average
// down over the window.  Under affine and above the fraction differs per pixel,
// the error is effectively random, and it falls off as 1/sqrt(N).  That is
// exactly what the measurements show.
//
// A note on the comparison
// ------------------------
// OpenCV's warp has hand-written SIMD paths.  A scalar replacement would lose
// on that count alone, so the inner loop here is written to be
// auto-vectorisable: the captured values are copied into locals (MSVC will not
// treat closure members as loop-invariant), and the bounds test is lifted out
// of the loop by solving for the range of x whose four taps are all inside the
// source.  What remains is a gather, since consecutive x do not land on
// consecutive source pixels.

namespace {

// dst(x, y) = src(M * (x, y, 1)), bilinear, zero outside.  M is the inverse map,
// matching WARP_INVERSE_MAP.
void warpExactBilinear(const Mat& src, Mat& dst, const Mat& map32, bool homography) {
    CV_Assert(src.type() == CV_32FC1 && dst.type() == CV_32FC1);
    const float* p = map32.ptr<float>(0);
    const double h0 = p[0], h1 = p[1], h2 = p[2];
    const double h3 = p[3], h4 = p[4], h5 = p[5];
    const double h6 = homography ? p[6] : 0.0;
    const double h7 = homography ? p[7] : 0.0;
    const double h8 = homography ? p[8] : 1.0;

    const int sw = src.cols, sh = src.rows;
    const int dw = dst.cols;
    const float* srcData = src.ptr<float>(0);
    const size_t srcStep = src.step / sizeof(float);

    parallel_for_(Range(0, dst.rows), [&](const Range& range) {
        // locals: MSVC does not treat values read through the closure as
        // loop-invariant, and that alone stops the vectoriser
        const double a0 = h0, a1 = h1, a2 = h2, a3 = h3, a4 = h4, a5 = h5;
        const double a6 = h6, a7 = h7, a8 = h8;
        const int W = dw, SW = sw, SH = sh;
        const size_t STEP = srcStep;
        const float* SRC = srcData;
        const bool HOMO = homography;

        for (int y = range.start; y < range.end; ++y) {
            float* out = dst.ptr<float>(y);
            const double fy = (double)y;
            const double bx = a1 * fy + a2, by = a4 * fy + a5;

            int xlo = 0, xhi = W;
            if (!HOMO) {
                // sx = a0*x + bx and sy = a3*x + by are linear in x, so the set
                // of x whose four taps are inside the source is an interval.
                // Solve for it once instead of testing every pixel.
                double lo = 0.0, hi = (double)W;
                const double eps = 1e-9;
                if (a0 > eps)       { lo = std::max(lo, (0.0 - bx) / a0);
                                      hi = std::min(hi, ((SW - 1) - bx) / a0); }
                else if (a0 < -eps) { lo = std::max(lo, ((SW - 1) - bx) / a0);
                                      hi = std::min(hi, (0.0 - bx) / a0); }
                else if (bx < 0.0 || bx >= SW - 1) { lo = 1.0; hi = 0.0; }
                if (a3 > eps)       { lo = std::max(lo, (0.0 - by) / a3);
                                      hi = std::min(hi, ((SH - 1) - by) / a3); }
                else if (a3 < -eps) { lo = std::max(lo, ((SH - 1) - by) / a3);
                                      hi = std::min(hi, (0.0 - by) / a3); }
                else if (by < 0.0 || by >= SH - 1) { lo = 1.0; hi = 0.0; }
                xlo = std::min(W, std::max(0, (int)std::ceil(lo)));
                xhi = std::max(xlo, std::min(W, (int)std::floor(hi) + 1));
                // one pixel of slack: floor() of a value that lands exactly on
                // the boundary must not read past it
                while (xhi > xlo && (a0 * (xhi - 1) + bx >= SW - 1 ||
                                     a3 * (xhi - 1) + by >= SH - 1)) --xhi;
            }

            for (int x = 0; x < xlo; ++x) out[x] = 0.f;
            for (int x = xhi; x < W; ++x) out[x] = 0.f;

            if (!HOMO) {
                for (int x = xlo; x < xhi; ++x) {          // no branches inside
                    const double sx = a0 * x + bx, sy = a3 * x + by;
                    const int x0 = (int)sx, y0 = (int)sy;  // both are >= 0 here
                    const double u = sx - x0, v = sy - y0;
                    const float* r0 = SRC + (size_t)y0 * STEP + x0;
                    const float* r1 = r0 + STEP;
                    const double top = r0[0] + u * (r0[1] - r0[0]);
                    const double bot = r1[0] + u * (r1[1] - r1[0]);
                    out[x] = (float)(top + v * (bot - top));
                }
            } else {
                for (int x = xlo; x < xhi; ++x) {
                    const double fx = (double)x;
                    const double d = a6 * fx + a7 * fy + a8;
                    if (d == 0.0) { out[x] = 0.f; continue; }
                    const double sx = (a0 * fx + bx) / d, sy = (a3 * fx + by) / d;
                    const int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
                    if (x0 < 0 || y0 < 0 || x0 + 1 >= SW || y0 + 1 >= SH) { out[x] = 0.f; continue; }
                    const double u = sx - x0, v = sy - y0;
                    const float* r0 = SRC + (size_t)y0 * STEP + x0;
                    const float* r1 = r0 + STEP;
                    const double top = r0[0] + u * (r0[1] - r0[0]);
                    const double bot = r1[0] + u * (r1[1] - r1[0]);
                    out[x] = (float)(top + v * (bot - top));
                }
            }
        }
    });
}

}  // namespace
