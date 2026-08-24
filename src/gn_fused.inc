// ---------------------------------------------------------------------------
// Fused Gauss-Newton step.
//
// The upstream path materialises a hs x (ws*P) jacobian and then reads it back
// P^2 times for the Gram matrix plus 2*P times for each of the three vector
// projections -- none of it parallelised (Mat::dot has no parallel_for_).
//
// Every ECC jacobian is a column subset of an outer product.  Writing u for a
// short vector of gradient-like quantities and w for the coordinate vector:
//
//   translation  u = (gx, gy)                        w = (1)        2 = 2x1
//   euclidean    u = (gx*hatX + gy*hatY, gx, gy)     w = (1)        3 = 3x1
//   affine       u = (gx, gy)                        w = (x, y, 1)  6 = 2x3
//   homography   u = (a, b, e)                       w = (x, y, 1)  8 = 3x3-1
//
// where, for homography, den = h6*x + h7*y + 1, a = gx/den, b = gy/den,
// hatX = -x'/den, hatY = -y'/den and e = hatX*a + hatY*b.  Column (e,1) is the
// one the 8-parameter homography does not carry.
//
// The n-th row of G is u_n (x) w_n restricted to those columns, so
//
//   H_pq = sum_n (u_{k(p)} u_{k(q)}) (w_{l(p)} w_{l(q)})
//
// and the whole Gram matrix falls out of the K(K+1)/2 x L(L+1)/2 table of sums
// of u_k u_k' w_l w_l'.  One pass over gx, gy, i-bar and t-bar fills that table
// and both projections; x and y come from the loop indices, so Xgrid/Ygrid are
// never read.  float accumulators inside a row, promoted to double at the row
// end.
//
// The kernels below are written out with NAMED scalar accumulators rather than
// arrays: with `float m[KK][LL]` MSVC keeps the table in memory and every pixel
// pays indexed loads and stores (measured 0.67x against 1.11x for affine at
// window 256, single thread).  They are generated, not hand-written.
//
// All four are here.  translation and euclidean gain least: with w = (1) their
// Hessians have only 3 and 6 entries, so this stage is a small share of their
// work to begin with.


struct GNTranslation {
    enum { P = 2 };

    double H[3], VI[2], VT[2];
    GNTranslation() { std::fill(H, H + 3, 0.0); std::fill(VI, VI + 2, 0.0); std::fill(VT, VT + 2, 0.0); }
    void add(const GNTranslation& o) {
        for (int i = 0; i < 3; ++i) H[i] += o.H[i];
        for (int i = 0; i < 2; ++i) { VI[i] += o.VI[i]; VT[i] += o.VT[i]; }
    }

    void rows(const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,
              int y0, int y1) {
        const int w = gx.cols;
        double t_m0_0 = 0, t_m1_0 = 0, t_m2_0 = 0;
        double t_pi0 = 0, t_pi1 = 0, t_pt0 = 0, t_pt1 = 0;

        for (int y = y0; y < y1; ++y) {
            const float* pgx = gx.ptr<float>(y);
            const float* pgy = gy.ptr<float>(y);
            const float* pim = im.ptr<float>(y);
            const float* ptm = tm.ptr<float>(y);
            const float fy = (float)y;
            float m0_0 = 0, m1_0 = 0, m2_0 = 0;
            float pi0 = 0, pi1 = 0, pt0 = 0, pt1 = 0;

            for (int x = 0; x < w; ++x) {
                const float a = pgx[x], b = pgy[x];
                const float uu0 = a * a, uu1 = a * b, uu2 = b * b;
                m0_0 += uu0;
                m1_0 += uu1;
                m2_0 += uu2;
                const float ii = pim[x], tt = ptm[x];
                const float j0 = a;
                const float j1 = b;
                pi0 += j0 * ii;  pi1 += j1 * ii;
                pt0 += j0 * tt;  pt1 += j1 * tt;
            }

            t_m0_0 += m0_0;  t_m1_0 += m1_0;  t_m2_0 += m2_0;
            t_pi0 += pi0;  t_pi1 += pi1;  t_pt0 += pt0;  t_pt1 += pt1;
        }

        int n = 0;
        H[n++] += t_m0_0;  H[n++] += t_m1_0;
        H[n++] += t_m2_0;
        VI[0] += t_pi0;  VI[1] += t_pi1;
        VT[0] += t_pt0;  VT[1] += t_pt1;
    }
};

struct GNEuclidean {
    enum { P = 3 };
    float h0, h1;
    double H[6], VI[3], VT[3];
    GNEuclidean() { std::fill(H, H + 6, 0.0); std::fill(VI, VI + 3, 0.0); std::fill(VT, VT + 3, 0.0); }
    void add(const GNEuclidean& o) {
        for (int i = 0; i < 6; ++i) H[i] += o.H[i];
        for (int i = 0; i < 3; ++i) { VI[i] += o.VI[i]; VT[i] += o.VT[i]; }
    }

    void rows(const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,
              int y0, int y1) {
        const int w = gx.cols;
        double t_m0_0 = 0, t_m1_0 = 0, t_m2_0 = 0, t_m3_0 = 0, t_m4_0 = 0, t_m5_0 = 0;
        double t_pi0 = 0, t_pi1 = 0, t_pi2 = 0, t_pt0 = 0, t_pt1 = 0, t_pt2 = 0;

        for (int y = y0; y < y1; ++y) {
            const float* pgx = gx.ptr<float>(y);
            const float* pgy = gy.ptr<float>(y);
            const float* pim = im.ptr<float>(y);
            const float* ptm = tm.ptr<float>(y);
            const float fy = (float)y;
            // hatX, hatY are affine in (x, y): step them along the row
            float hatX = -(0.f * h1) - (fy * h0);
            float hatY =  (0.f * h0) - (fy * h1);
            float m0_0 = 0, m1_0 = 0, m2_0 = 0, m3_0 = 0, m4_0 = 0, m5_0 = 0;
            float pi0 = 0, pi1 = 0, pi2 = 0, pt0 = 0, pt1 = 0, pt2 = 0;

            for (int x = 0; x < w; ++x) {
                const float a = pgx[x], b = pgy[x];
                const float r = a * hatX + b * hatY;
                const float uu0 = r * r, uu1 = r * a, uu2 = r * b, uu3 = a * a, uu4 = a * b, uu5 = b * b;
                m0_0 += uu0;
                m1_0 += uu1;
                m2_0 += uu2;
                m3_0 += uu3;
                m4_0 += uu4;
                m5_0 += uu5;
                const float ii = pim[x], tt = ptm[x];
                const float j0 = r;
                const float j1 = a;
                const float j2 = b;
                pi0 += j0 * ii;  pi1 += j1 * ii;  pi2 += j2 * ii;
                pt0 += j0 * tt;  pt1 += j1 * tt;  pt2 += j2 * tt;
                hatX -= h1;  hatY += h0;
            }

            t_m0_0 += m0_0;  t_m1_0 += m1_0;  t_m2_0 += m2_0;  t_m3_0 += m3_0;  t_m4_0 += m4_0;  t_m5_0 += m5_0;
            t_pi0 += pi0;  t_pi1 += pi1;  t_pi2 += pi2;  t_pt0 += pt0;  t_pt1 += pt1;  t_pt2 += pt2;
        }

        int n = 0;
        H[n++] += t_m0_0;  H[n++] += t_m1_0;  H[n++] += t_m2_0;
        H[n++] += t_m3_0;  H[n++] += t_m4_0;
        H[n++] += t_m5_0;
        VI[0] += t_pi0;  VI[1] += t_pi1;  VI[2] += t_pi2;
        VT[0] += t_pt0;  VT[1] += t_pt1;  VT[2] += t_pt2;
    }
};

// ---- affine: K=2 L=3 P=6, 18 Gram sums + 12 projection sums ----
struct GNAffine {
    enum { P = 6 };

    double H[21], VI[6], VT[6];
    GNAffine() { std::fill(H, H + 21, 0.0); std::fill(VI, VI + 6, 0.0); std::fill(VT, VT + 6, 0.0); }
    void add(const GNAffine& o) {
        for (int i = 0; i < 21; ++i) H[i] += o.H[i];
        for (int i = 0; i < 6; ++i) { VI[i] += o.VI[i]; VT[i] += o.VT[i]; }
    }

    void rows(const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,
              int y0, int y1) {
        const int w = gx.cols;
        double t_m0_0 = 0, t_m0_1 = 0, t_m0_2 = 0, t_m0_3 = 0, t_m0_4 = 0, t_m0_5 = 0, t_m1_0 = 0, t_m1_1 = 0, t_m1_2 = 0, t_m1_3 = 0, t_m1_4 = 0, t_m1_5 = 0, t_m2_0 = 0, t_m2_1 = 0, t_m2_2 = 0, t_m2_3 = 0, t_m2_4 = 0, t_m2_5 = 0;
        double t_pi0 = 0, t_pi1 = 0, t_pi2 = 0, t_pi3 = 0, t_pi4 = 0, t_pi5 = 0, t_pt0 = 0, t_pt1 = 0, t_pt2 = 0, t_pt3 = 0, t_pt4 = 0, t_pt5 = 0;

        for (int y = y0; y < y1; ++y) {
            const float* pgx = gx.ptr<float>(y);
            const float* pgy = gy.ptr<float>(y);
            const float* pim = im.ptr<float>(y);
            const float* ptm = tm.ptr<float>(y);
            const float fy = (float)y;
            float m0_0 = 0, m0_1 = 0, m0_2 = 0, m0_3 = 0, m0_4 = 0, m0_5 = 0, m1_0 = 0, m1_1 = 0, m1_2 = 0, m1_3 = 0, m1_4 = 0, m1_5 = 0, m2_0 = 0, m2_1 = 0, m2_2 = 0, m2_3 = 0, m2_4 = 0, m2_5 = 0;
            float pi0 = 0, pi1 = 0, pi2 = 0, pi3 = 0, pi4 = 0, pi5 = 0, pt0 = 0, pt1 = 0, pt2 = 0, pt3 = 0, pt4 = 0, pt5 = 0;

            for (int x = 0; x < w; ++x) {
                const float fx = (float)x;
                const float a = pgx[x], b = pgy[x];
                const float uu0 = a * a, uu1 = a * b, uu2 = b * b;
                const float ww0 = fx * fx, ww1 = fx * fy, ww2 = fx, ww3 = fy * fy, ww4 = fy, ww5 = 1.f;
                m0_0 += uu0 * ww0;  m0_1 += uu0 * ww1;  m0_2 += uu0 * ww2;  m0_3 += uu0 * ww3;  m0_4 += uu0 * ww4;  m0_5 += uu0 * ww5;
                m1_0 += uu1 * ww0;  m1_1 += uu1 * ww1;  m1_2 += uu1 * ww2;  m1_3 += uu1 * ww3;  m1_4 += uu1 * ww4;  m1_5 += uu1 * ww5;
                m2_0 += uu2 * ww0;  m2_1 += uu2 * ww1;  m2_2 += uu2 * ww2;  m2_3 += uu2 * ww3;  m2_4 += uu2 * ww4;  m2_5 += uu2 * ww5;
                const float ii = pim[x], tt = ptm[x];
                const float j0 = a * fx;
                const float j1 = b * fx;
                const float j2 = a * fy;
                const float j3 = b * fy;
                const float j4 = a;
                const float j5 = b;
                pi0 += j0 * ii;  pi1 += j1 * ii;  pi2 += j2 * ii;  pi3 += j3 * ii;  pi4 += j4 * ii;  pi5 += j5 * ii;
                pt0 += j0 * tt;  pt1 += j1 * tt;  pt2 += j2 * tt;  pt3 += j3 * tt;  pt4 += j4 * tt;  pt5 += j5 * tt;
            }

            t_m0_0 += m0_0;  t_m0_1 += m0_1;  t_m0_2 += m0_2;  t_m0_3 += m0_3;  t_m0_4 += m0_4;  t_m0_5 += m0_5;  t_m1_0 += m1_0;  t_m1_1 += m1_1;  t_m1_2 += m1_2;  t_m1_3 += m1_3;  t_m1_4 += m1_4;  t_m1_5 += m1_5;  t_m2_0 += m2_0;  t_m2_1 += m2_1;  t_m2_2 += m2_2;  t_m2_3 += m2_3;  t_m2_4 += m2_4;  t_m2_5 += m2_5;
            t_pi0 += pi0;  t_pi1 += pi1;  t_pi2 += pi2;  t_pi3 += pi3;  t_pi4 += pi4;  t_pi5 += pi5;  t_pt0 += pt0;  t_pt1 += pt1;  t_pt2 += pt2;  t_pt3 += pt3;  t_pt4 += pt4;  t_pt5 += pt5;
        }

        int n = 0;
        H[n++] += t_m0_0;  H[n++] += t_m1_0;  H[n++] += t_m0_1;  H[n++] += t_m1_1;  H[n++] += t_m0_2;  H[n++] += t_m1_2;
        H[n++] += t_m2_0;  H[n++] += t_m1_1;  H[n++] += t_m2_1;  H[n++] += t_m1_2;  H[n++] += t_m2_2;
        H[n++] += t_m0_3;  H[n++] += t_m1_3;  H[n++] += t_m0_4;  H[n++] += t_m1_4;
        H[n++] += t_m2_3;  H[n++] += t_m1_4;  H[n++] += t_m2_4;
        H[n++] += t_m0_5;  H[n++] += t_m1_5;
        H[n++] += t_m2_5;
        VI[0] += t_pi0;  VI[1] += t_pi1;  VI[2] += t_pi2;  VI[3] += t_pi3;  VI[4] += t_pi4;  VI[5] += t_pi5;
        VT[0] += t_pt0;  VT[1] += t_pt1;  VT[2] += t_pt2;  VT[3] += t_pt3;  VT[4] += t_pt4;  VT[5] += t_pt5;
    }
};

struct GNHomography {
    enum { P = 8 };
    float h0, h1, h2, h3, h4, h5, h6, h7;
    double H[36], VI[8], VT[8];
    GNHomography() { std::fill(H, H + 36, 0.0); std::fill(VI, VI + 8, 0.0); std::fill(VT, VT + 8, 0.0); }
    void add(const GNHomography& o) {
        for (int i = 0; i < 36; ++i) H[i] += o.H[i];
        for (int i = 0; i < 8; ++i) { VI[i] += o.VI[i]; VT[i] += o.VT[i]; }
    }

    void rows(const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,
              int y0, int y1) {
        const int w = gx.cols;
        double t_m0_0 = 0, t_m0_1 = 0, t_m0_2 = 0, t_m0_3 = 0, t_m0_4 = 0, t_m0_5 = 0, t_m1_0 = 0, t_m1_1 = 0, t_m1_2 = 0, t_m1_3 = 0, t_m1_4 = 0, t_m1_5 = 0, t_m2_0 = 0, t_m2_1 = 0, t_m2_2 = 0, t_m2_3 = 0, t_m2_4 = 0, t_m2_5 = 0, t_m3_0 = 0, t_m3_1 = 0, t_m3_2 = 0, t_m3_3 = 0, t_m3_4 = 0, t_m3_5 = 0, t_m4_0 = 0, t_m4_1 = 0, t_m4_2 = 0, t_m4_3 = 0, t_m4_4 = 0, t_m4_5 = 0, t_m5_0 = 0, t_m5_1 = 0, t_m5_2 = 0, t_m5_3 = 0, t_m5_4 = 0, t_m5_5 = 0;
        double t_pi0 = 0, t_pi1 = 0, t_pi2 = 0, t_pi3 = 0, t_pi4 = 0, t_pi5 = 0, t_pi6 = 0, t_pi7 = 0, t_pt0 = 0, t_pt1 = 0, t_pt2 = 0, t_pt3 = 0, t_pt4 = 0, t_pt5 = 0, t_pt6 = 0, t_pt7 = 0;

        for (int y = y0; y < y1; ++y) {
            const float* pgx = gx.ptr<float>(y);
            const float* pgy = gy.ptr<float>(y);
            const float* pim = im.ptr<float>(y);
            const float* ptm = tm.ptr<float>(y);
            const float fy = (float)y;
            float m0_0 = 0, m0_1 = 0, m0_2 = 0, m0_3 = 0, m0_4 = 0, m0_5 = 0, m1_0 = 0, m1_1 = 0, m1_2 = 0, m1_3 = 0, m1_4 = 0, m1_5 = 0, m2_0 = 0, m2_1 = 0, m2_2 = 0, m2_3 = 0, m2_4 = 0, m2_5 = 0, m3_0 = 0, m3_1 = 0, m3_2 = 0, m3_3 = 0, m3_4 = 0, m3_5 = 0, m4_0 = 0, m4_1 = 0, m4_2 = 0, m4_3 = 0, m4_4 = 0, m4_5 = 0, m5_0 = 0, m5_1 = 0, m5_2 = 0, m5_3 = 0, m5_4 = 0, m5_5 = 0;
            float pi0 = 0, pi1 = 0, pi2 = 0, pi3 = 0, pi4 = 0, pi5 = 0, pi6 = 0, pi7 = 0, pt0 = 0, pt1 = 0, pt2 = 0, pt3 = 0, pt4 = 0, pt5 = 0, pt6 = 0, pt7 = 0;

            for (int x = 0; x < w; ++x) {
                const float fx = (float)x;
                const float gxv = pgx[x], gyv = pgy[x];
                const float den = fx * h2 + fy * h5 + 1.f;
                // cv::divide yields 0 where the denominator is 0; match that
                const float inv = den != 0.f ? 1.f / den : 0.f;
                const float a = gxv * inv;
                const float b = gyv * inv;
                const float hatX = (-fx * h0 - fy * h3 - h6) * inv;
                const float hatY = (-fx * h1 - fy * h4 - h7) * inv;
                const float e = hatX * a + hatY * b;
                const float uu0 = a * a, uu1 = a * b, uu2 = a * e, uu3 = b * b, uu4 = b * e, uu5 = e * e;
                const float ww0 = fx * fx, ww1 = fx * fy, ww2 = fx, ww3 = fy * fy, ww4 = fy, ww5 = 1.f;
                m0_0 += uu0 * ww0;  m0_1 += uu0 * ww1;  m0_2 += uu0 * ww2;  m0_3 += uu0 * ww3;  m0_4 += uu0 * ww4;  m0_5 += uu0 * ww5;
                m1_0 += uu1 * ww0;  m1_1 += uu1 * ww1;  m1_2 += uu1 * ww2;  m1_3 += uu1 * ww3;  m1_4 += uu1 * ww4;  m1_5 += uu1 * ww5;
                m2_0 += uu2 * ww0;  m2_1 += uu2 * ww1;  m2_2 += uu2 * ww2;  m2_3 += uu2 * ww3;  m2_4 += uu2 * ww4;  m2_5 += uu2 * ww5;
                m3_0 += uu3 * ww0;  m3_1 += uu3 * ww1;  m3_2 += uu3 * ww2;  m3_3 += uu3 * ww3;  m3_4 += uu3 * ww4;  m3_5 += uu3 * ww5;
                m4_0 += uu4 * ww0;  m4_1 += uu4 * ww1;  m4_2 += uu4 * ww2;  m4_3 += uu4 * ww3;  m4_4 += uu4 * ww4;  m4_5 += uu4 * ww5;
                m5_0 += uu5 * ww0;  m5_1 += uu5 * ww1;  m5_2 += uu5 * ww2;  m5_3 += uu5 * ww3;  m5_4 += uu5 * ww4;  m5_5 += uu5 * ww5;
                const float ii = pim[x], tt = ptm[x];
                const float j0 = a * fx;
                const float j1 = b * fx;
                const float j2 = e * fx;
                const float j3 = a * fy;
                const float j4 = b * fy;
                const float j5 = e * fy;
                const float j6 = a;
                const float j7 = b;
                pi0 += j0 * ii;  pi1 += j1 * ii;  pi2 += j2 * ii;  pi3 += j3 * ii;  pi4 += j4 * ii;  pi5 += j5 * ii;  pi6 += j6 * ii;  pi7 += j7 * ii;
                pt0 += j0 * tt;  pt1 += j1 * tt;  pt2 += j2 * tt;  pt3 += j3 * tt;  pt4 += j4 * tt;  pt5 += j5 * tt;  pt6 += j6 * tt;  pt7 += j7 * tt;
            }

            t_m0_0 += m0_0;  t_m0_1 += m0_1;  t_m0_2 += m0_2;  t_m0_3 += m0_3;  t_m0_4 += m0_4;  t_m0_5 += m0_5;  t_m1_0 += m1_0;  t_m1_1 += m1_1;  t_m1_2 += m1_2;  t_m1_3 += m1_3;  t_m1_4 += m1_4;  t_m1_5 += m1_5;  t_m2_0 += m2_0;  t_m2_1 += m2_1;  t_m2_2 += m2_2;  t_m2_3 += m2_3;  t_m2_4 += m2_4;  t_m2_5 += m2_5;  t_m3_0 += m3_0;  t_m3_1 += m3_1;  t_m3_2 += m3_2;  t_m3_3 += m3_3;  t_m3_4 += m3_4;  t_m3_5 += m3_5;  t_m4_0 += m4_0;  t_m4_1 += m4_1;  t_m4_2 += m4_2;  t_m4_3 += m4_3;  t_m4_4 += m4_4;  t_m4_5 += m4_5;  t_m5_0 += m5_0;  t_m5_1 += m5_1;  t_m5_2 += m5_2;  t_m5_3 += m5_3;  t_m5_4 += m5_4;  t_m5_5 += m5_5;
            t_pi0 += pi0;  t_pi1 += pi1;  t_pi2 += pi2;  t_pi3 += pi3;  t_pi4 += pi4;  t_pi5 += pi5;  t_pi6 += pi6;  t_pi7 += pi7;  t_pt0 += pt0;  t_pt1 += pt1;  t_pt2 += pt2;  t_pt3 += pt3;  t_pt4 += pt4;  t_pt5 += pt5;  t_pt6 += pt6;  t_pt7 += pt7;
        }

        int n = 0;
        H[n++] += t_m0_0;  H[n++] += t_m1_0;  H[n++] += t_m2_0;  H[n++] += t_m0_1;  H[n++] += t_m1_1;  H[n++] += t_m2_1;  H[n++] += t_m0_2;  H[n++] += t_m1_2;
        H[n++] += t_m3_0;  H[n++] += t_m4_0;  H[n++] += t_m1_1;  H[n++] += t_m3_1;  H[n++] += t_m4_1;  H[n++] += t_m1_2;  H[n++] += t_m3_2;
        H[n++] += t_m5_0;  H[n++] += t_m2_1;  H[n++] += t_m4_1;  H[n++] += t_m5_1;  H[n++] += t_m2_2;  H[n++] += t_m4_2;
        H[n++] += t_m0_3;  H[n++] += t_m1_3;  H[n++] += t_m2_3;  H[n++] += t_m0_4;  H[n++] += t_m1_4;
        H[n++] += t_m3_3;  H[n++] += t_m4_3;  H[n++] += t_m1_4;  H[n++] += t_m3_4;
        H[n++] += t_m5_3;  H[n++] += t_m2_4;  H[n++] += t_m4_4;
        H[n++] += t_m0_5;  H[n++] += t_m1_5;
        H[n++] += t_m3_5;
        VI[0] += t_pi0;  VI[1] += t_pi1;  VI[2] += t_pi2;  VI[3] += t_pi3;  VI[4] += t_pi4;  VI[5] += t_pi5;  VI[6] += t_pi6;  VI[7] += t_pi7;
        VT[0] += t_pt0;  VT[1] += t_pt1;  VT[2] += t_pt2;  VT[3] += t_pt3;  VT[4] += t_pt4;  VT[5] += t_pt5;  VT[6] += t_pt6;  VT[7] += t_pt7;
    }
};

// --- drivers --------------------------------------------------------------

template <class Acc>
class GNBody : public ParallelLoopBody {
public:
    GNBody(const Acc& proto, const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,
           std::vector<Acc>& out, Mutex& m)
        : proto_(proto), gx_(gx), gy_(gy), im_(im), tm_(tm), out_(out), mtx_(m) {}
    void operator()(const Range& r) const CV_OVERRIDE {
        Acc s = proto_;                      // carries the warp coefficients
        s.rows(gx_, gy_, im_, tm_, r.start, r.end);
        AutoLock lk(mtx_);
        out_.push_back(s);
    }
private:
    Acc proto_;
    const Mat &gx_, &gy_, &im_, &tm_;
    std::vector<Acc>& out_;
    Mutex& mtx_;
};

template <class Acc>
void runFusedGN(const Acc& proto, const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,
                Mat& hessian, Mat& imageProjection, Mat& templateProjection) {
    std::vector<Acc> parts;
    Mutex m;
    GNBody<Acc> body(proto, gx, gy, im, tm, parts, m);
    parallel_for_(Range(0, gx.rows), body);

    Acc s = proto;
    std::fill(s.H, s.H + Acc::P * (Acc::P + 1) / 2, 0.0);
    std::fill(s.VI, s.VI + Acc::P, 0.0);
    std::fill(s.VT, s.VT + Acc::P, 0.0);
    for (size_t i = 0; i < parts.size(); ++i) s.add(parts[i]);

    const bool f64 = hessian.type() == CV_64F;
    int n = 0;
    for (int p = 0; p < Acc::P; ++p)
        for (int q = p; q < Acc::P; ++q, ++n) {
            if (f64) { hessian.at<double>(p, q) = s.H[n]; hessian.at<double>(q, p) = s.H[n]; }
            else     { hessian.at<float>(p, q) = (float)s.H[n];
                       hessian.at<float>(q, p) = (float)s.H[n]; }
        }
    for (int p = 0; p < Acc::P; ++p) {
        if (f64) { imageProjection.at<double>(p, 0) = s.VI[p];
                   templateProjection.at<double>(p, 0) = s.VT[p]; }
        else     { imageProjection.at<float>(p, 0) = (float)s.VI[p];
                   templateProjection.at<float>(p, 0) = (float)s.VT[p]; }
    }
}
