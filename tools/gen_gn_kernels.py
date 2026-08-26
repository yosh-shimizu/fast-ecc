# Emit the fused Gauss-Newton kernels of src/gn_fused.inc as straight-line
# C++ with *named* scalar accumulators.
#
#   python tools/gen_gn_kernels.py            # print to stdout
#   python tools/gen_gn_kernels.py --check    # diff against the checked-in file
#
# The generic array-based version regressed badly (affine 0.67x vs the
# hand-written 1.11x at window 256, single thread): MSVC keeps `float m[KK][LL]`
# and `float u[K]` in memory instead of registers, so every pixel pays indexed
# loads and stores.  Named locals fix that, at the cost of having to write out
# up to 36 + 16 accumulators -- hence this generator.
#
# Structure is identical for all four:  j = u (x) w restricted to a column list.
#
# The w = y^2 and w = y columns are constant along a row, so they are not
# accumulated per pixel: within one row, sum_x u*fy^2 == fy^2 * sum_x u.  Only
# the w = 1 column is summed in the loop and the other two are formed from it at
# row end.  That is exact, and it removes both arithmetic and live accumulators
# (affine 30 -> 20, homography 52 -> 36).
#
# Only the (u-pair, w-pair) sums that some entry of the Gram matrix actually
# needs are accumulated.  For the plain kernels that is the full table minus
# one dead entry in homography; for the kernels with the laplacian column it
# is what keeps the extra column at +5 accumulators instead of +12.
#
# The kernels read the RAW planes: the warped image, the blurred template, the
# two finite-difference gradients of the warped image, and the warped mask.
# Masking, the A^-T recombination of the gradient and the zero-mean centring
# happen per pixel inside the loop (a masked pixel is skipped; a = r00*gx +
# r01*gy, b = r10*gx + r11*gy; ii = im - muI, tt = tm - muT).  Nothing is
# materialised between filter2D and this pass.
#
# Each motion comes in two variants.  GN<Name>L carries one more column, the
# laplacian of the warped image scaled by lsc (the pre-filter's sigma): the
# nuisance direction that absorbs the isotropic second-order term of a
# misalignment residual and the bilinear interpolation error.  Its w is 1, so
# it adds one u entry and the Gram entries that pair it with the (., x) and
# (., 1) columns; the (., y) pairs are hoisted like everything else.
#
# Besides the Gram matrix and the two projections, every kernel also sums the
# jacobian columns themselves (VJ).  The single-pass iteration centres the
# planes on the previous iteration's means, since the current ones are not
# known until the pass is over, and corrects afterwards: for a column j and
# the true mean m, sum j (i - m) = sum j (i - m_prev) - (m - m_prev) sum j.
#
# rows() takes a `base`: the row index of the first row of the gx, gy, im, lp
# and mask planes.  The full-plane driver passes 0; the single-pass driver
# hands it stripe buffers whose row 0 is image row y0.  The template plane is
# always full-size and is indexed by the image row.
#
# Vector path.  Under FASTECC_SIMD_GN each row is first run in vectors of
# VL pixels with OpenCV's universal intrinsics (function form, OpenCV >= 4.7):
# one partial sum per lane for every live accumulator, reduced into the scalar
# row accumulators before the scalar loop takes the tail.  Float within a
# row, double at row end, as before -- the lanes are the same reassociation
# one level down, and four shorter chains lose less than one long one.  The
# mask is applied as a bitwise AND on the gradient inputs: every u, j and
# product downstream of a zeroed gradient is zero, so a masked lane
# contributes nothing to any sum, and the AND is safe on whatever the
# gradient planes hold outside the mask.  MSVC will not vectorise these
# reductions itself under /fp:precise (see the explicit-simd notes), which is
# why the lanes are written out.
import io
import sys

MOTIONS = [
    # name,        K, L, u-names,            column list (k, l)
    ('Translation', 2, 1, ['a', 'b'],        [(0, 0), (1, 0)]),
    ('Euclidean',   3, 1, ['r', 'a', 'b'],   [(0, 0), (1, 0), (2, 0)]),
    ('Affine',      2, 3, ['a', 'b'],        [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]),
    ('Homography',  3, 3, ['a', 'b', 'e'],   [(0, 0), (1, 0), (2, 0),
                                              (0, 1), (1, 1), (2, 1),
                                              (0, 2), (1, 2)]),
]

# per-motion preamble of the scalar loop (after the vector part, so `x` is
# where the tail starts) and per-pixel u
PRE = {
    'Translation': '',
    'Euclidean': (
        '        // hatX, hatY are affine in (x, y): step them along the row\n'
        '        float hatX = -((float)x * h1) - (fy * h0);\n'
        '        float hatY =  ((float)x * h0) - (fy * h1);\n'),
    'Affine': '',
    'Homography': '',
}

# translation's A^-T is the identity, so it reads the gradients directly
PIX = {
    'Translation':
        '            const float a = pgx[x], b = pgy[x];\n',
    'Euclidean':
        '            const float a = r00 * pgx[x] + r01 * pgy[x];\n'
        '            const float b = r10 * pgx[x] + r11 * pgy[x];\n'
        '            const float r = a * hatX + b * hatY;\n',
    'Affine':
        '            const float a = r00 * pgx[x] + r01 * pgy[x];\n'
        '            const float b = r10 * pgx[x] + r11 * pgy[x];\n',
    'Homography':
        '            const float gxv = r00 * pgx[x] + r01 * pgy[x];\n'
        '            const float gyv = r10 * pgx[x] + r11 * pgy[x];\n'
        '            const float den = fx * h2 + fy * h5 + 1.f;\n'
        '            // cv::divide yields 0 where the denominator is 0; match that\n'
        '            const float inv = den != 0.f ? 1.f / den : 0.f;\n'
        '            const float a = gxv * inv;\n'
        '            const float b = gyv * inv;\n'
        '            const float hatX = (-fx * h0 - fy * h3 - h6) * inv;\n'
        '            const float hatY = (-fx * h1 - fy * h4 - h7) * inv;\n'
        '            const float e = hatX * a + hatY * b;\n',
}
LAPPIX = '            const float l = lsc * pl[x];\n'

POST = {
    'Euclidean': '            hatX -= h1;  hatY += h0;\n',
}

# the same per pixel, VL pixels at a time.  gxv, gyv (and lv) are the masked
# gradient loads; vx the pixel column as float, vfy the row.
VPIX = {
    'Translation':
        '                const v_float32 a = gxv, b = gyv;\n',
    'Euclidean':
        '                const v_float32 a = v_add(v_mul(vr00, gxv), v_mul(vr01, gyv));\n'
        '                const v_float32 b = v_add(v_mul(vr10, gxv), v_mul(vr11, gyv));\n'
        '                const v_float32 hatX = v_sub(vnfyh0, v_mul(vx, vh1));\n'
        '                const v_float32 hatY = v_sub(v_mul(vx, vh0), vfyh1);\n'
        '                const v_float32 r = v_add(v_mul(a, hatX), v_mul(b, hatY));\n',
    'Affine':
        '                const v_float32 a = v_add(v_mul(vr00, gxv), v_mul(vr01, gyv));\n'
        '                const v_float32 b = v_add(v_mul(vr10, gxv), v_mul(vr11, gyv));\n',
    'Homography':
        '                const v_float32 gxr = v_add(v_mul(vr00, gxv), v_mul(vr01, gyv));\n'
        '                const v_float32 gyr = v_add(v_mul(vr10, gxv), v_mul(vr11, gyv));\n'
        '                const v_float32 den = v_add(v_add(v_mul(vx, vh2), vfyh5), vone);\n'
        '                const v_float32 inv = v_select(v_ne(den, vzf), v_div(vone, den), vzf);\n'
        '                const v_float32 a = v_mul(gxr, inv);\n'
        '                const v_float32 b = v_mul(gyr, inv);\n'
        '                const v_float32 hatX = v_mul(v_sub(v_sub(v_mul(vx, vnh0), vfyh3), vh6), inv);\n'
        '                const v_float32 hatY = v_mul(v_sub(v_sub(v_mul(vx, vnh1), vfyh4), vh7), inv);\n'
        '                const v_float32 e = v_add(v_mul(hatX, a), v_mul(hatY, b));\n',
}
VLAPPIX = '                const v_float32 l = v_mul(vlsc, lv);\n'

# broadcast once per rows() call (the warp coefficients) and once per row
# (what depends on fy)
VRECOMB = ('        const v_float32 vr00 = v_setall_f32(r00), vr01 = v_setall_f32(r01), '
           'vr10 = v_setall_f32(r10), vr11 = v_setall_f32(r11);\n')
VCONST = {
    'Translation': '',
    'Euclidean':   VRECOMB + '        const v_float32 vh0 = v_setall_f32(h0), vh1 = v_setall_f32(h1);\n',
    'Affine':      VRECOMB,
    'Homography':  VRECOMB +
        '        const v_float32 vh2 = v_setall_f32(h2), vh6 = v_setall_f32(h6), vh7 = v_setall_f32(h7);\n'
        '        const v_float32 vnh0 = v_setall_f32(-h0), vnh1 = v_setall_f32(-h1);\n'
        '        const v_float32 vone = v_setall_f32(1.f), vzf = v_setzero_f32();\n',
}
VPRE = {
    'Translation': '',
    'Euclidean':   '                const v_float32 vnfyh0 = v_setall_f32(-(fy * h0)), vfyh1 = v_setall_f32(fy * h1);\n',
    'Affine':      '',
    'Homography':  '                const v_float32 vfyh3 = v_setall_f32(fy * h3), vfyh4 = v_setall_f32(fy * h4), '
                   'vfyh5 = v_setall_f32(fy * h5);\n',
}

RECOMB = '    float r00, r01, r10, r11;   // A^-T of the current warp\n'
MEANS  = '    float muI, muT;             // masked means of warped image and template\n'
LAPSC  = '    float lsc;                  // scale of the laplacian column (the pre-filter sigma)\n'
MEMBERS = {
    'Translation': MEANS,
    'Euclidean':   RECOMB + MEANS + '    float h0, h1;\n',
    'Affine':      RECOMB + MEANS,
    'Homography':  RECOMB + MEANS + '    float h0, h1, h2, h3, h4, h5, h6, h7;\n',
}


def tri(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def vdecl(names, per_line=6):
    """v_float32 declarations of zeroed lane accumulators, a few per line"""
    lines = []
    for i in range(0, len(names), per_line):
        chunk = names[i:i + per_line]
        lines.append('                v_float32 ' + ', '.join('v%s = v_setzero_f32()' % n for n in chunk) + ';')
    return lines


def vreduce(names, per_line=6):
    lines = []
    for i in range(0, len(names), per_line):
        chunk = names[i:i + per_line]
        lines.append('                ' + '  '.join('%s = v_reduce_sum(v%s);' % (n, n) for n in chunk))
    return lines


out = []
for base, K0, L, U0, cols0 in MOTIONS:
    for lap in (False, True):
        name = base + ('L' if lap else '')
        K = K0 + (1 if lap else 0)
        U = U0 + (['l'] if lap else [])
        cols = cols0 + ([(K0, L - 1)] if lap else [])
        P = len(cols)
        kp = tri(K)
        lp = tri(L)
        wn = ['fx', 'fy', '1.f'] if L == 3 else ['1.f']
        vwn = ['vx', 'vfy', '1.f'] if L == 3 else ['1.f']

        # lp indices whose w-pair does not involve fx are constant along a row.
        # For L == 3 those are (fy,fy), (fy,1) and (1,1); the last is the one
        # that gets accumulated, the first two are it scaled by fy^2 and fy.
        ONE = lp.index((L - 1, L - 1))                 # the w = 1 column
        rowconst = [b for b, (i, j) in enumerate(lp) if i != 0 and b != ONE] if L == 3 else []

        # which (u-pair, w-pair) sums the Gram matrix needs, and which of those
        # are summed per pixel (a row-constant w-pair is formed from ONE)
        needed, live_set = [], []
        for p in range(P):
            for q in range(p, P):
                kpi = kp.index((min(cols[p][0], cols[q][0]), max(cols[p][0], cols[q][0])))
                lpi = lp.index((min(cols[p][1], cols[q][1]), max(cols[p][1], cols[q][1])))
                if (kpi, lpi) not in needed:
                    needed.append((kpi, lpi))
                lv = (kpi, ONE) if lpi in rowconst else (kpi, lpi)
                if lv not in live_set:
                    live_set.append(lv)
        needed.sort()
        live_set.sort()
        acc = ['m%d_%d' % ab for ab in needed]
        live = ['m%d_%d' % ab for ab in live_set]
        uu_used = sorted(set(a for a, b in live_set))
        ww_used = sorted(set(b for a, b in live_set if b != ONE and b not in rowconst))

        # a projection column carrying w = fy is the matching w = 1 column
        # scaled.  Where that partner column is not in the list (homography's
        # (e, 1)), one extra accumulator stands in for it.
        FY = 1 if L == 3 else None
        hoistp, extra = {}, []
        if L == 3:
            for s, (k, l) in enumerate(cols):
                if l != FY:
                    continue
                partner = next((q for q, (k2, l2) in enumerate(cols)
                                if k2 == k and l2 == L - 1), None)
                if partner is None:
                    nm = 'e%d' % k
                    if nm not in extra:
                        extra.append(nm)
                    hoistp[s] = nm
                else:
                    hoistp[s] = '%d' % partner
        proj = (['pi%d' % s for s in range(P)] + ['pt%d' % s for s in range(P)] +
                ['pj%d' % s for s in range(P)])
        liveproj = (['pi%d' % s for s in range(P) if s not in hoistp] +
                    ['pi' + e for e in extra] +
                    ['pt%d' % s for s in range(P) if s not in hoistp] +
                    ['pt' + e for e in extra] +
                    ['pj%d' % s for s in range(P) if s not in hoistp] +
                    ['pj' + e for e in extra])
        kept = [s for s in range(P) if s not in hoistp]

        o = []
        o.append('// ---- %s: K=%d L=%d P=%d, %d Gram sums (%d live) + %d projection sums ----'
                 % (name.lower(), K, L, P, len(acc), len(live), 3 * P))
        o.append('struct GN%s {' % name)
        o.append('    enum { P = %d };' % P)
        members = MEMBERS[base] + (LAPSC if lap else '')
        o.append(members.rstrip('\n') if members else '')
        o.append('    double H[%d], VI[%d], VT[%d], VJ[%d];' % (P * (P + 1) // 2, P, P, P))
        o.append('    GN%s() { std::fill(H, H + %d, 0.0); std::fill(VI, VI + %d, 0.0);'
                 ' std::fill(VT, VT + %d, 0.0); std::fill(VJ, VJ + %d, 0.0); }'
                 % (name, P * (P + 1) // 2, P, P, P))
        o.append('    void add(const GN%s& o) {' % name)
        o.append('        for (int i = 0; i < %d; ++i) H[i] += o.H[i];' % (P * (P + 1) // 2))
        o.append('        for (int i = 0; i < %d; ++i) { VI[i] += o.VI[i]; VT[i] += o.VT[i]; VJ[i] += o.VJ[i]; }'
                 % P)
        o.append('    }')
        o.append('')
        o.append('    void rows(const Mat& gx, const Mat& gy, const Mat& im, const Mat& tm,')
        o.append('              const Mat& lp, const Mat& mask, int y0, int y1, int base) {')
        o.append('        const int w = gx.cols;')
        if not lap:
            o.append('        (void)lp;')
        # double totals
        o.append('        double ' + ', '.join('t_%s = 0' % a for a in acc) + ';')
        o.append('        double ' + ', '.join('t_%s = 0' % p for p in proj) + ';')
        # vector constants of the call
        o.append('#if FASTECC_SIMD_GN')
        o.append('        const int VL = VTraits<v_float32>::vlanes();')
        o.append('        float CV_DECL_ALIGNED(64) lane[VTraits<v_float32>::max_nlanes];')
        o.append('        for (int k = 0; k < VL; ++k) lane[k] = (float)k;')
        o.append('        const v_float32 vlane = v_load_aligned(lane), vVL = v_setall_f32((float)VL);')
        o.append('        const v_float32 vmuI = v_setall_f32(muI), vmuT = v_setall_f32(muT);')
        o.append('        const v_int32 vzi = v_setzero_s32();')
        if VCONST[base]:
            o.append(VCONST[base].rstrip('\n'))
        if lap:
            o.append('        const v_float32 vlsc = v_setall_f32(lsc);')
        o.append('#endif')
        o.append('')
        o.append('        for (int y = y0; y < y1; ++y) {')
        o.append('            const float* pgx = gx.ptr<float>(y - base);')
        o.append('            const float* pgy = gy.ptr<float>(y - base);')
        o.append('            const float* pim = im.ptr<float>(y - base);')
        o.append('            const float* ptm = tm.ptr<float>(y);')
        if lap:
            o.append('            const float* pl = lp.ptr<float>(y - base);')
        o.append('            const uchar* pmk = mask.ptr<uchar>(y - base);')
        o.append('            const float fy = (float)y;')
        o.append('            float ' + ', '.join('%s = 0' % a for a in live) + ';')
        o.append('            float ' + ', '.join('%s = 0' % p for p in liveproj) + ';')
        o.append('            int x = 0;')

        # ---- the vector part of the row
        o.append('#if FASTECC_SIMD_GN')
        o.append('            {')
        if L == 3:
            o.append('                const v_float32 vfy = v_setall_f32(fy);')
        if VPRE[base]:
            o.append(VPRE[base].rstrip('\n'))
        o.extend(vdecl(live))
        o.extend(vdecl(liveproj))
        o.append('                v_float32 vx = vlane;')
        o.append('                for (; x + VL <= w; x += VL) {')
        o.append('                    const v_float32 mf = v_reinterpret_as_f32(v_ne(v_reinterpret_as_s32('
                 'v_load_expand_q(pmk + x)), vzi));')
        o.append('                    const v_float32 gxv = v_and(v_load(pgx + x), mf), gyv = v_and(v_load(pgy + x), mf);')
        if lap:
            o.append('                    const v_float32 lv = v_and(v_load(pl + x), mf);')
        vbody = VPIX[base] + (VLAPPIX if lap else '')
        o.append(vbody.replace('                ', '                    ').rstrip('\n'))
        up = []
        for idx, (i, j) in enumerate(kp):
            if idx in uu_used:
                up.append('uu%d = v_mul(%s, %s)' % (idx, U[i], U[j]))
        o.append('                    const v_float32 ' + ', '.join(up) + ';')
        if L == 3:
            wp = []
            for idx, (i, j) in enumerate(lp):
                if idx not in ww_used:
                    continue
                wp.append('ww%d = %s' % (idx, ('v_mul(%s, %s)' % (vwn[i], vwn[j])) if vwn[j] != '1.f'
                                          else vwn[i]))
            if wp:
                o.append('                    const v_float32 ' + ', '.join(wp) + ';')
        for a in range(len(kp)):
            line = []
            for b in range(len(lp)):
                if (a, b) not in live_set:
                    continue
                if L == 1 or b == ONE:
                    line.append('vm%d_%d = v_add(vm%d_%d, uu%d);' % (a, b, a, b, a))
                else:
                    line.append('vm%d_%d = v_fma(uu%d, ww%d, vm%d_%d);' % (a, b, a, b, a, b))
            if line:
                o.append('                    ' + '  '.join(line))
        o.append('                    const v_float32 ii = v_sub(v_load(pim + x), vmuI), '
                 'tt = v_sub(v_load(ptm + x), vmuT);')
        for s, (k, l) in enumerate(cols):
            if s in hoistp:
                continue
            jexpr = U[k] if L == 1 or vwn[l] == '1.f' else 'v_mul(%s, %s)' % (U[k], vwn[l])
            o.append('                    const v_float32 j%d = %s;' % (s, jexpr))
        for pre, src in (('pi', 'ii'), ('pt', 'tt')):
            terms = ['v%s%d = v_fma(j%d, %s, v%s%d);' % (pre, s, s, src, pre, s) for s in kept]
            terms += ['v%s%s = v_fma(%s, %s, v%s%s);' % (pre, e, U[int(e[1:])], src, pre, e) for e in extra]
            o.append('                    ' + '  '.join(terms))
        terms = ['vpj%d = v_add(vpj%d, j%d);' % (s, s, s) for s in kept]
        terms += ['vpj%s = v_add(vpj%s, %s);' % (e, e, U[int(e[1:])]) for e in extra]
        o.append('                    ' + '  '.join(terms))
        o.append('                    vx = v_add(vx, vVL);')
        o.append('                }')
        o.extend(vreduce(live))
        o.extend(vreduce(liveproj))
        o.append('            }')
        o.append('#endif')

        # ---- the scalar loop (the whole row without FASTECC_SIMD_GN, the tail with it)
        if PRE[base]:
            o.append(PRE[base].rstrip('\n').replace('        ', '            '))
        o.append('')
        o.append('            for (; x < w; ++x) {')
        o.append('                if (pmk[x] != 0) {')
        body_start = len(o)          # everything from here to the projections is indented once more
        if L == 3:
            o.append('                const float fx = (float)x;')
        elif base == 'Homography':
            o.append('                const float fx = (float)x;')
        body = PIX[base] + (LAPPIX if lap else '')
        body = body.replace('            ', '                ')
        o.append(body.rstrip('\n'))
        # u products
        up = []
        for idx, (i, j) in enumerate(kp):
            if idx in uu_used:
                up.append('uu%d = %s * %s' % (idx, U[i], U[j]))
        o.append('                const float ' + ', '.join(up) + ';')
        # w products
        if L == 3:
            wp = []
            for idx, (i, j) in enumerate(lp):
                if idx not in ww_used:
                    continue           # constant along the row, the bare sum, or unused
                wp.append('ww%d = %s' % (idx, ('%s * %s' % (wn[i], wn[j])) if wn[j] != '1.f'
                                          else wn[i]))
            if wp:
                o.append('                const float ' + ', '.join(wp) + ';')
        # gram
        for a in range(len(kp)):
            line = []
            for b in range(len(lp)):
                if (a, b) not in live_set:
                    continue
                term = 'uu%d' % a if (L == 1 or b == ONE) else 'uu%d * ww%d' % (a, b)
                line.append('m%d_%d += %s;' % (a, b, term))
            if line:
                o.append('                ' + '  '.join(line))
        # projections (centred on the fly)
        o.append('                const float ii = pim[x] - muI, tt = ptm[x] - muT;')
        for s, (k, l) in enumerate(cols):
            if s in hoistp:
                continue                       # formed at row end from its partner
            jexpr = U[k] if L == 1 or wn[l] == '1.f' else '%s * %s' % (U[k], wn[l])
            o.append('                const float j%d = %s;' % (s, jexpr))
        for pre, src in (('pi', 'ii'), ('pt', 'tt')):
            terms = ['%s%d += j%d * %s;' % (pre, s, s, src) for s in kept]
            terms += ['%s%s += %s * %s;' % (pre, e, U[int(e[1:])], src) for e in extra]
            o.append('                ' + '  '.join(terms))
        terms = ['pj%d += j%d;' % (s, s) for s in kept]
        terms += ['pj%s += %s;' % (e, U[int(e[1:])]) for e in extra]
        o.append('                ' + '  '.join(terms))
        for i in range(body_start, len(o)):
            o[i] = '\n'.join('    ' + ln for ln in o[i].split('\n'))
        o.append('                }')
        if base in POST:
            o.append(POST[base].rstrip('\n').replace('            ', '                '))
        o.append('            }')
        o.append('')
        if rowconst:
            o.append('            const double dy = (double)fy, dyy = (double)fy * (double)fy;')
            for a in range(len(kp)):
                line = []
                for b in range(len(lp)):
                    if (a, b) not in needed:
                        continue
                    if b in rowconst:
                        scale = 'dyy' if lp[b][1] == 1 else 'dy'   # (fy,fy) or (fy,1)
                        line.append('t_m%d_%d += m%d_%d * %s;' % (a, b, a, ONE, scale))
                    else:
                        line.append('t_m%d_%d += m%d_%d;' % (a, b, a, b))
                if line:
                    o.append('            ' + '  '.join(line))
        else:
            o.append('            ' + '  '.join('t_%s += %s;' % (a, a) for a in acc))

        if hoistp:
            for pre in ('pi', 'pt', 'pj'):
                line = []
                for s in range(P):
                    if s in hoistp:
                        line.append('t_%s%d += %s%s * dy;' % (pre, s, pre, hoistp[s]))
                    else:
                        line.append('t_%s%d += %s%d;' % (pre, s, pre, s))
                o.append('            ' + '  '.join(line))
        else:
            o.append('            ' + '  '.join('t_%s += %s;' % (p, p) for p in proj))
        o.append('        }')
        o.append('')
        # scatter into the packed upper triangle, (p,q) row-major
        o.append('        int n = 0;')
        for p in range(P):
            terms = []
            for q in range(p, P):
                kpi = kp.index((min(cols[p][0], cols[q][0]), max(cols[p][0], cols[q][0])))
                lpi = lp.index((min(cols[p][1], cols[q][1]), max(cols[p][1], cols[q][1])))
                terms.append('H[n++] += t_m%d_%d;' % (kpi, lpi))
            o.append('        ' + '  '.join(terms))
        o.append('        ' + '  '.join('VI[%d] += t_pi%d;' % (s, s) for s in range(P)))
        o.append('        ' + '  '.join('VT[%d] += t_pt%d;' % (s, s) for s in range(P)))
        o.append('        ' + '  '.join('VJ[%d] += t_pj%d;' % (s, s) for s in range(P)))
        o.append('    }')
        o.append('};')
        o.append('')
        out.append('\n'.join(x for x in o if x is not None))

text = '\n'.join(out)

if '--check' in sys.argv:
    shipped = io.open('src/gn_fused.inc', encoding='utf-8').read()
    stale = []
    for block in out:
        name = block.split('\n')[1]              # 'struct GNAffine {'
        body = block[block.index(name):].rstrip()
        if body not in shipped:
            stale.append(name.split()[1])
    if stale:
        print('src/gn_fused.inc is STALE for: ' + ', '.join(stale))
        sys.exit(1)
    print('src/gn_fused.inc matches the generator')
    sys.exit(0)

print(text)
