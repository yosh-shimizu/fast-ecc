// ---------------------------------------------------------------------------
// Hand-vectorised variants of the fused Gauss-Newton kernels.
//
// NOT built by default.  Configure with -DFASTECC_SIMD=ON to select these in
// place of the generated scalar kernels; the two produce the same sums and can
// be compared in the same binary.  This file exists to show what explicit SIMD
// is worth, not because the library depends on it.
//
// Why it has to be written by hand
// --------------------------------
// The inner loops are floating-point reductions.  Under /fp:precise MSVC will
// not reassociate a floating-point sum, so it cannot vectorise them, and it
// does not: every one reports C5002 and the target emits no C5001 at all
// (configure with -DFASTECC_VEC_REPORT=ON to see it).
//
// It is not dependence analysis holding it back.  #pragma loop(ivdep) with
// __restrict on every pointer changes nothing (measured 0.394 ms against
// 0.393 ms at window 256).  Nor does spelling the reassociation out as four
// scalar partial sums -- that is 1.6x *slower*, because MSVC keeps the arrays
// in memory.  The only route is to write the reassociation ourselves.
//
// /fp:precise stays on.  It constrains what the compiler may do behind our
// back, not what the source may say.  The scalar kernels already reassociate
// deliberately -- float within a row, promoted to double at row end -- and one
// partial sum per lane is the same move one level further down.
//
// It is also more accurate, not less.  Against the same sums accumulated in
// double, worst relative error over windows 256..1024:
//
//     scalar   7.1e-05  1.3e-06  3.1e-05  6.6e-06
//     simd     1.9e-05  1.1e-06  1.7e-06  4.1e-07
//
// Four shorter dependency chains lose less than one long one.
//
// What it is worth, and why it is not on by default
// -------------------------------------------------
// The stage itself gets 3.4-3.8x faster.  The library does not, because the
// stage is already parallelised over row stripes, and SIMD and threads are
// substitutes for the same work.  Cost of one iteration, affine, analytic
// scene, against the scalar kernels:
//
//     threads      window 256   window 512   window 768
//     1                 1.36x        1.33x        1.35x
//     4                 1.07x        1.04x        1.14x
//     12                1.06x        1.06x        1.05x
//
// At four threads -- the everyday operating point -- a second implementation
// of four kernels, in intrinsics, buys 1.04-1.14x.  That is not worth
// maintaining.  On a single thread it buys 1.33-1.36x, which is worth having
// if you run ECC that way (embedded, or parallelised outside the call), and
// that is what the build option is for.
//
// The same shape as the warp reduction, for the same reason: an optimisation
// that competes with the thread pool for the same headroom looks much better
// on one core than on twelve.

// ---- affine: u = (gx, gy), w = (x, y, 1) ----------------------------------
struct GNAffineSimd {
    enum { P = 6 };

    double H[21], VI[6], VT[6];
    GNAffineSimd() { std::fill(H, H + 21, 0.0); std::fill(VI, VI + 6, 0.0); std::fill(VT, VT + 6, 0.0); }
    void add(const GNAffineSimd& o) {
        for (int i = 0; i < 21; ++i) H[i] += o.H[i];
        for (int i = 0; i < 6; ++i) { VI[i] += o.VI[i]; VT[i] += o.VT[i]; }
    }

    void rows(const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,
              int y0, int y1) {
        const int w = gx.cols;
        const int L = VTraits<v_float32>::vlanes();
        double t_m0_0 = 0, t_m0_1 = 0, t_m0_2 = 0, t_m0_3 = 0, t_m0_4 = 0, t_m0_5 = 0;
        double t_m1_0 = 0, t_m1_1 = 0, t_m1_2 = 0, t_m1_3 = 0, t_m1_4 = 0, t_m1_5 = 0;
        double t_m2_0 = 0, t_m2_1 = 0, t_m2_2 = 0, t_m2_3 = 0, t_m2_4 = 0, t_m2_5 = 0;
        double t_pi0 = 0, t_pi1 = 0, t_pi2 = 0, t_pi3 = 0, t_pi4 = 0, t_pi5 = 0;
        double t_pt0 = 0, t_pt1 = 0, t_pt2 = 0, t_pt3 = 0, t_pt4 = 0, t_pt5 = 0;

        std::vector<float> lane((size_t)L);
        for (int i = 0; i < L; ++i) lane[i] = (float)i;
        const v_float32 vlane = vx_load(&lane[0]);
        const v_float32 vstep = vx_setall_f32((float)L);

        for (int y = y0; y < y1; ++y) {
            const float* pgx = gx.ptr<float>(y);
            const float* pgy = gy.ptr<float>(y);
            const float* pim = im.ptr<float>(y);
            const float* ptm = tm.ptr<float>(y);
            const float fy = (float)y;
            const v_float32 vfy = vx_setall_f32(fy);

            v_float32 v00 = vx_setzero_f32(), v01 = vx_setzero_f32(), v02 = vx_setzero_f32(), v05 = vx_setzero_f32();
            v_float32 v10 = vx_setzero_f32(), v11 = vx_setzero_f32(), v12 = vx_setzero_f32(), v15 = vx_setzero_f32();
            v_float32 v20 = vx_setzero_f32(), v21 = vx_setzero_f32(), v22 = vx_setzero_f32(), v25 = vx_setzero_f32();
            v_float32 q0 = vx_setzero_f32(), q1 = vx_setzero_f32(), q4 = vx_setzero_f32(), q5 = vx_setzero_f32();
            v_float32 r0 = vx_setzero_f32(), r1 = vx_setzero_f32(), r4 = vx_setzero_f32(), r5 = vx_setzero_f32();

            v_float32 vfx = vlane;
            int x = 0;
            for (; x <= w - L; x += L) {
                const v_float32 a = vx_load(pgx + x), b = vx_load(pgy + x);
                const v_float32 uu0 = v_mul(a, a), uu1 = v_mul(a, b), uu2 = v_mul(b, b);
                const v_float32 ww0 = v_mul(vfx, vfx), ww1 = v_mul(vfx, vfy);
                v00 = v_fma(uu0, ww0, v00);  v01 = v_fma(uu0, ww1, v01);  v02 = v_fma(uu0, vfx, v02);  v05 = v_add(v05, uu0);
                v10 = v_fma(uu1, ww0, v10);  v11 = v_fma(uu1, ww1, v11);  v12 = v_fma(uu1, vfx, v12);  v15 = v_add(v15, uu1);
                v20 = v_fma(uu2, ww0, v20);  v21 = v_fma(uu2, ww1, v21);  v22 = v_fma(uu2, vfx, v22);  v25 = v_add(v25, uu2);
                const v_float32 ii = vx_load(pim + x), tt = vx_load(ptm + x);
                const v_float32 j0 = v_mul(a, vfx), j1 = v_mul(b, vfx);
                q0 = v_fma(j0, ii, q0);  q1 = v_fma(j1, ii, q1);  q4 = v_fma(a, ii, q4);  q5 = v_fma(b, ii, q5);
                r0 = v_fma(j0, tt, r0);  r1 = v_fma(j1, tt, r1);  r4 = v_fma(a, tt, r4);  r5 = v_fma(b, tt, r5);
                vfx = v_add(vfx, vstep);
            }

            float m0_0 = v_reduce_sum(v00), m0_1 = v_reduce_sum(v01), m0_2 = v_reduce_sum(v02), m0_5 = v_reduce_sum(v05);
            float m1_0 = v_reduce_sum(v10), m1_1 = v_reduce_sum(v11), m1_2 = v_reduce_sum(v12), m1_5 = v_reduce_sum(v15);
            float m2_0 = v_reduce_sum(v20), m2_1 = v_reduce_sum(v21), m2_2 = v_reduce_sum(v22), m2_5 = v_reduce_sum(v25);
            float pi0 = v_reduce_sum(q0), pi1 = v_reduce_sum(q1), pi4 = v_reduce_sum(q4), pi5 = v_reduce_sum(q5);
            float pt0 = v_reduce_sum(r0), pt1 = v_reduce_sum(r1), pt4 = v_reduce_sum(r4), pt5 = v_reduce_sum(r5);

            for (; x < w; ++x) {                       // tail, when the row is not a whole number of vectors
                const float fx = (float)x;
                const float a = pgx[x], b = pgy[x];
                const float uu0 = a * a, uu1 = a * b, uu2 = b * b;
                const float ww0 = fx * fx, ww1 = fx * fy;
                m0_0 += uu0 * ww0;  m0_1 += uu0 * ww1;  m0_2 += uu0 * fx;  m0_5 += uu0;
                m1_0 += uu1 * ww0;  m1_1 += uu1 * ww1;  m1_2 += uu1 * fx;  m1_5 += uu1;
                m2_0 += uu2 * ww0;  m2_1 += uu2 * ww1;  m2_2 += uu2 * fx;  m2_5 += uu2;
                const float ii = pim[x], tt = ptm[x];
                const float j0 = a * fx, j1 = b * fx;
                pi0 += j0 * ii;  pi1 += j1 * ii;  pi4 += a * ii;  pi5 += b * ii;
                pt0 += j0 * tt;  pt1 += j1 * tt;  pt4 += a * tt;  pt5 += b * tt;
            }

            const double dy = (double)fy, dyy = (double)fy * (double)fy;
            t_m0_0 += m0_0;  t_m0_1 += m0_1;  t_m0_2 += m0_2;  t_m0_3 += m0_5 * dyy;  t_m0_4 += m0_5 * dy;  t_m0_5 += m0_5;
            t_m1_0 += m1_0;  t_m1_1 += m1_1;  t_m1_2 += m1_2;  t_m1_3 += m1_5 * dyy;  t_m1_4 += m1_5 * dy;  t_m1_5 += m1_5;
            t_m2_0 += m2_0;  t_m2_1 += m2_1;  t_m2_2 += m2_2;  t_m2_3 += m2_5 * dyy;  t_m2_4 += m2_5 * dy;  t_m2_5 += m2_5;
            t_pi0 += pi0;  t_pi1 += pi1;  t_pi2 += pi4 * dy;  t_pi3 += pi5 * dy;  t_pi4 += pi4;  t_pi5 += pi5;
            t_pt0 += pt0;  t_pt1 += pt1;  t_pt2 += pt4 * dy;  t_pt3 += pt5 * dy;  t_pt4 += pt4;  t_pt5 += pt5;
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
