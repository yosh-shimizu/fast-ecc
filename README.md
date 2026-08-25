# fast-ecc

A research fork of OpenCV's `cv::findTransformECC` — the ECC image alignment of
**Evangelidis & Psarakis, PAMI 2008** — used to develop and measure improvements to
the algorithm's **speed** and **correctness**.

```cpp
// the signature mirrors cv::findTransformECC exactly
fastecc::findTransformECC(templ, input, W, cv::MOTION_AFFINE, criteria);
```

> Not affiliated with or endorsed by OpenCV. Derived from `modules/video/src/ecc.cpp`
> (BSD-3). See [NOTICE](NOTICE) and [LICENSE](LICENSE).

![before/after alignment overlay](docs/before_after.png)

*Template in green, input in magenta — colour fringes mean misalignment. After
`fastecc::findTransformECC` (affine, ρ = 0.9997) the overlay collapses to grey.
Reproduce with `examples/align_pair.cpp`.*

## What this repository is for

Two kinds of work happen here, and they have different destinations.

**Speed.** Changes that make ECC compute the same answer in less time. These live here,
because they restructure the inner loop in ways that are more intrusive than OpenCV
would want in a single patch.

**Correctness.** Bugs found while measuring the above. These go **upstream** — they
belong in OpenCV, not in a fork. See [Upstream contributions](#upstream-contributions).

Each substantive change lands as its own commit with its own measurement, and the
[Results](#results) table below grows as they do. Nothing is claimed here that is not
measured, and every measurement states its operating point — motion type, window size
and thread count — because in this code the same change can be 1.5× or 1.0× depending
on all three.

## Status

Early, but the accuracy defect that made the old benchmark meaningless is fixed and
the numbers below are taken against a corrected OpenCV baseline.

| | state |
|---|---|
| Warp reduction (3 warps → 1 per iteration) | landed |
| Gaussian pre-filter border fix | landed ([opencv#29775](https://github.com/opencv/opencv/pull/29775) ported) |
| Accuracy test with a criterion that can fail | landed |
| Fused Gauss–Newton stage, all four motion types | landed |
| Row-invariant columns hoisted out of the inner loop | landed |
| Normalisation folded into the fused pass; mask solved for, not warped | landed |
| 5-tap image gradient (3 iterations to the floor instead of 4–5) | landed, on by default |
| Laplacian column (wider basin, 2.6× lower error on a real image) | landed, on by default |
| Equivalence test with a margin of real data, run under every flag set | landed |

### What was wrong, and how it is guarded now

`findTransformECC` smooths the template with `GaussianBlur`, whose default
`BORDER_REFLECT_101` fabricates a ring `gaussFiltSize/2` px wide by mirroring the
interior. Used at full weight, that ring biases the estimated linear part toward a
smaller scale. This fork inherited the defect from `ecc.cpp` verbatim — it masked the
warped *input* border but never the template's fabricated ring.

It went unnoticed because the bundled test compared this fork only against a stock
OpenCV that had the same defect, with a 0.5 px tolerance on an error of 0.01–0.03 px.
Both are now fixed: the ring is dropped, and `test/test_equivalence.cpp` checks an
absolute threshold, agreement with cv, and — the part that actually catches this
class of bug — that **widening the pre-filter never makes the answer worse**. With the
fix reverted the suite reports sixteen failures and names the cause.

## Results

Every row states its operating point, because in this code the same change
measures 1.3x or 1.1x depending on it.

### Accuracy — equal to a fixed OpenCV with the plain kernels, better with the defaults

Measured against an OpenCV that carries the border fix, affine, window 200, 200
trials, started **at** the ground truth so that any movement is bias:

| | analytic scene | real image (fruits, resampled) | scale bias (t), analytic |
|---|---:|---:|---:|
| `cv::findTransformECC` (fixed) | 0.0017 px | 0.0072 px | +5.7e-07 (0.6) |
| `fastecc::findTransformECC`, `flags = 0` | 0.0017 px | 0.0072 px | +5.8e-07 (0.6) |
| `fastecc::findTransformECC` (default flags) | **0.0015 px** | **0.0028 px** | +3.1e-07 (0.3) |

With the plain kernels the two are statistically indistinguishable and neither shows a
systematic bias: the warp reduction, the fused stage and the folded normalisation are the
same algebra evaluated in a different order, and convergence counts over 150 trials at
five deformation steps match `cv` trial for trial. Before the border fix this fork sat at
0.0287 px with a −1.8e−04 (t = −63.9) scale bias at `gaussFiltSize=9`.

The defaults add the [laplacian column](#two-optional-flags-the-laplacian-column-and-the-5-tap-gradient),
which takes the bilinear interpolation error out of the fixed point: on the real image the
corner error drops 2.6×, and the fraction of trials that converge from the no-motion guess
goes from 69 / 26 / 4 % to 82 / 33 / 8 % at ×1 / ×2 / ×4 the default deformation.

### Speed

Cost of one iteration, affine, analytic scene, fixed 20 iterations, best of 3 runs,
default flags, against a border-fixed OpenCV:

| window | 4 threads | 1 thread |
|---:|---:|---:|
| 256 | **2.61×** | 1.90× |
| 512 | 2.80× | 2.10× |
| 768 | **3.53×** | 2.29× |

Three changes contribute, and they cover each other. The **warp reduction** takes the
three bilinear warps per iteration down to one; it pays most at small windows and
decays at large ones with a thread pool, because OpenCV parallelises the `warpAffine`
it removes but not the `filter2D` that replaces it. The **fused Gauss–Newton stage**
does the opposite: it builds the Hessian and both projections in one pass without
materialising the Jacobian, and the saving grows with the parameter count. The
**folded normalisation** removes the eleven masked OpenCV calls and the mask warp that
sat between the two, which were 45–60 % of an iteration once the other two had landed.

The bundled `./build/bench` (512×512, default thread count, median of 3) shows the
per-motion spread; the last column is the previous release for reference:

| motion | cv | fast | speedup | previous release |
|---|---:|---:|---:|---:|
| TRANSLATION | 23.4 ms | 8.2 ms | 2.88× | 1.71× |
| EUCLIDEAN | 36.0 ms | 11.7 ms | 3.08× | 1.93× |
| AFFINE | 46.5 ms | 17.6 ms | 2.64× | 2.11× |
| HOMOGRAPHY | 123.9 ms | 27.6 ms | **4.46×** | 3.24× |

Homography gains most because the Jacobian path re-reads its blocks P² times for the
Gram matrix, and P = 8. The bench counts iterations to convergence, so the 5-tap
gradient (3 iterations to the floor instead of 4–5) is part of these figures.

The fused kernels are generated by [`tools/gen_gn_kernels.py`](tools/gen_gn_kernels.py);
CI fails if `src/gn_fused.inc` stops matching its output.

Measured on an Intel Core i7-8700K (6C/12T), Windows 11, MSVC, Release, against
OpenCV 4.15.0-dev built with the border fix applied.

## Reports

The measurements behind each change, in more depth than the table above, are in
[`docs/reports/`](docs/reports/index.html):

- **Reducing the warps per iteration** — how three warps become one, and why it is
  allowed
- **The warp reduction against the border fix** — why the speed-up decays at large
  windows with a thread pool, and the `gaussFiltSize` re-tuning
- **Fusing the Gauss–Newton stage** — why all four motion types are the same
  face-splitting product, and why the saving grows with the parameter count
- **After the fusion** — the 5-tap gradient (the first-order model's one estimated
  quantity), the folded normalisation, the sigma experiment and the laplacian column it
  left behind

## Why the warp reduction is faster

OpenCV's ECC warps **three** images every iteration: the input image **and** its two
gradient images `∂I/∂x`, `∂I/∂y` (plus a cheap nearest-neighbour mask warp).
`warpAffine`/`warpPerspective` with bilinear interpolation is the dominant cost of the
inner loop.

This version warps **only the image**, then obtains the gradients from the *warped*
image with a `filter2D`; the `A⁻ᵀ` recombination is applied per pixel inside the fused
Gauss–Newton pass. So the two bilinear gradient warps per iteration disappear.

| | image warp | gradient warps | mask warp |
|---|:---:|:---:|:---:|
| `cv::findTransformECC` | 1 | 2 | 1 |
| `fastecc::findTransformECC` | 1 | **0** (filter2D) | **0** (solved for; 1 with a user mask) |

The gain from *this* change is **not** uniform. It is largest at small windows and on a
single thread, and falls to break-even at large windows with a thread pool — because
OpenCV parallelises `warpAffine` above roughly 384 px but not the
`filter2D` that replaces it. That region is covered by the fused
Gauss–Newton stage instead. Any speed figure quoted without a window size and a thread
count is meaningless.

### The math

ECC maximizes the normalized correlation between the template and the warped input
`I(W(x; p))`. The Gauss–Newton step needs the **steepest-descent images**, i.e. the
image gradient evaluated at the warped location. The key identity is the chain rule: for
a warp with linear part `A`,

```
∂/∂x [ I(W(x; p)) ] = Aᵀ · (∇I)(W(x; p))
```

So the finite-difference gradient of the **already-warped** image is the image-domain
gradient pre-multiplied by `Aᵀ`. Multiplying it by `A⁻ᵀ` (`recombination`, applied per pixel
inside the fused pass) undoes
that frame change and recovers the quantity the Jacobian assembly expects, **without
warping the gradient images separately**.

For euclidean the linear part is orthonormal, so `A⁻ᵀ = A` and the recombination is a
rotation of the gradient vectors; for affine it is the exact inverse frame change. For
homography it remains an approximation: a projective warp's Jacobian varies per pixel
and only its linear part is used.

Note that the recombination coefficient does **not** change the Gauss–Newton fixed
point — it is an invertible reparameterisation, so `Jᵀr = 0` is the same condition
either way. Correcting it from `A` to `A⁻ᵀ` was a correctness cleanup, not a measurable
accuracy improvement.

### The normalisation is folded into the fused pass

Everything between the `filter2D` and the fused Gauss–Newton pass used to be a chain of
masked OpenCV calls that each wrote a plane: three masked `setTo`, two `meanStdDev`,
two masked `subtract`, two `countNonZero`, a dot product and the two `addWeighted` of
the recombination. Profiling one iteration showed that chain, together with the mask
warp, at 45–60 % of the time on a single thread (window 200) — more than the warp and
the Gauss–Newton stage combined for translation.

It is now two passes that write nothing. `maskedStats` collects the six masked moments
(N, ΣI, ΣI², ΣT, ΣT², ΣIT) in one sweep; the means, both norms and the correlation of the
centred vectors are algebra on those. The fused kernels then read the *raw* planes and
do the masking (a masked pixel is skipped), the `A⁻ᵀ` recombination and the zero-mean
centring per pixel. The arithmetic is the same, only the summation order differs: the
fixed point agrees with the previous layout to four digits in all four motion types.

The mask warp goes the same way when there is no user mask. `preMask` is then a
rectangle, and its nearest-neighbour warp is the set of template pixels whose rounded
warped coordinate lands in it — along a row an interval, for affine and (with a
positive denominator) projective warps alike. `analyticMask` solves for that interval
per row instead of resampling; against the warped mask it differs on 0–2 pixels per
warp (the 1/1024 fixed-point ties of OpenCV's rounding), and the fixed point does not
move. A user-supplied mask still takes the warp.

Per iteration against the previous layout, best of three interleaved runs:

| motion | 200 px, 1 thread | | 512 px, 1 thread | 200 px, 4 threads |
|---|---:|---:|---:|---:|
| translation | 0.605 → 0.387 ms | 1.56× | | |
| euclidean | 0.678 → 0.481 ms | 1.41× | | |
| affine | 0.829 → 0.590 ms | 1.40× | 1.47× | 1.49× |
| homography | 1.707 → 1.278 ms | 1.34× | 1.35× | 1.88× |

(Of that, folding the normalisation alone is 1.26× / 1.21× / 1.17× / 1.12×; the rest
is the mask.)

### Two optional flags: the laplacian column and the 5-tap gradient

`findTransformECC` takes a trailing `flags` argument; both flags are on by default (`FASTECC_DEFAULT_FLAGS`), pass 0 for the plain kernels.

**`FASTECC_LAPLACIAN_COLUMN`** carries the laplacian of the warped image, scaled by the
pre-filter's sigma, as one more Gauss–Newton column. The coefficient it estimates is
discarded; the column is a *nuisance direction*. It came out of asking whether sigma
itself could be estimated alongside the motion (it cannot — the correlation coefficient
has no maximum in sigma while the images are misaligned, so a joint estimate runs sigma
up), but the column that estimate would use turns out to have the shape of two things
the first-order model leaves in the residual: the isotropic second-order term of a
misalignment, and the error of bilinear interpolation. With it in the system the first
step from a 1 px offset lands at 0.07 px instead of 0.18, the basin widens, and on a
real image the fixed point moves toward the truth. It costs one 3×3 filter and a
larger fused pass (affine 20 → 27 live accumulators).

**`FASTECC_GRAD5`** uses the 4th-order 5-tap finite difference `(1, −8, 0, 8, −1)/12`
for the image gradient instead of the 3-tap. The forward-additive step is the exact
maximiser of the paper's first-order model, so the gradient estimate is the only thing in
that model that is not exact, and its bias sets the convergence rate: the error contracts
by ~0.06 per iteration instead of ~0.17. The fixed point is unchanged and the cost is nil.

Measured against the plain kernels (window 200, one thread, 100 trials; `eval`):

| | plain | column | 5-tap | both |
|---|---:|---:|---:|---:|
| affine, error after 1 / 3 iterations from 1 px | 0.183 / 0.0049 | 0.074 / 0.0033 | 0.078 / 0.0017 | **0.030 / 0.0016** |
| affine floor, analytic scene | 0.0016 | 0.0014 | 0.0017 | 0.0014 |
| affine floor, real image (fruits) | 0.0073 | **0.0027** | 0.0070 | **0.0028** |
| homography floor, real image | 0.0129 | 0.0062 | 0.0123 | 0.0061 |
| affine floor, 5 grey levels of noise | 0.0216 | 0.0214 | 0.0208 | 0.0208 |
| basin ×1 / ×2, affine | 76 / 23 % | 86 / 30 % | 77 / 26 % | 86 / 31 % |
| basin ×1 / ×2, homography | 62 / 21 % | 80 / 29 % | 66 / 21 % | 80 / 28 % |
| ms per iteration, affine / homography | 0.580 / 1.218 | 0.644 / 1.243 | 0.584 / 1.235 | 0.647 / 1.243 |
| ms per call (|Δρ| < 1e-6), affine / homography | 4.27 / 7.84 | 4.44 / 7.94 | 3.38 / 6.41 | **3.42 / 6.38** |

The column pays for its 2–19 % per iteration with one iteration fewer, so on its own it
is time-neutral and buys accuracy and basin; with the 5-tap the call is 1.25× faster
*and* the most accurate of the four. For pure translation the column costs more time than
it saves (the plain kernel is already at its floor in 3 iterations) while still widening
the basin from 82 to 98 % at ×1.

### Not bit-identical to OpenCV

This method takes gradients **of the warped image**, whereas OpenCV warps gradients **of
the original image**. Bilinear resampling and finite differencing do not commute for
affine and projective warps, so the result is **not bit-identical** to
`cv::findTransformECC`. (For pure translation the two paths are exactly equivalent as
operators; any difference there is floating-point summation order and border handling,
not non-commutation.)

Which of the two is closer to ground truth is **not settled** and is not currently
claimed either way. Measuring it requires a fixed OpenCV baseline and many random
sub-pixel phases — a single alignment cannot distinguish the two.

## Quick start

```bash
git clone https://github.com/yosh-shimizu/fast-ecc.git
cd fast-ecc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # accuracy vs ground truth
./build/bench                                 # timing table
./build/align_pair template.png input.png aligned.png affine
```

Requires OpenCV `core`, `imgproc` and `video` (plus `imgcodecs` for the example) — any
4.x.  On Debian/Ubuntu: `sudo apt-get install libopencv-dev`.

### Use it in your project

The library is a single `.cpp` + header. Either add this repo via CMake
`add_subdirectory(fast-ecc)` and link `fast_ecc`, or drop `include/fast_ecc.hpp` and
`src/fast_ecc.cpp` into your build.

```cpp
#include <fast_ecc.hpp>

cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
double rho = fastecc::findTransformECC(
    templateGray, inputGray, W, cv::MOTION_AFFINE,
    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 1e-4));
```

The signature mirrors `cv::findTransformECC` exactly (same arguments, same defaults,
same `MOTION_*` types, same `warpMatrix` conventions).

## Upstream contributions

Correctness work found here is reported to OpenCV rather than kept as a fork advantage.

| | |
|---|---|
| [opencv#29774](https://github.com/opencv/opencv/issues/29774) | `findTransformECC` systematically under-estimates scale (the border defect above) |
| [opencv#29775](https://github.com/opencv/opencv/pull/29775) | the fix for it |
| [opencv#29776](https://github.com/opencv/opencv/issues/29776) | the ECC accuracy test averages warp-matrix entries of different units, so it cannot detect regressions |
| [opencv#29781](https://github.com/opencv/opencv/issues/29781) | `findTransformECCMultiScale` has the same border defect |
| [opencv_extra#1405](https://github.com/opencv/opencv_extra/pull/1405) | performance baseline update |

## Scope & limitations

- Inputs are single-channel `CV_8UC1` or `CV_32FC1`, like OpenCV.
- The speed-up is largest where warps dominate the inner loop. If you add heavy
  per-iteration work on top (robust losses, extra parameters), the relative gain shrinks.
- For homography the gradient recombination is an approximation (only the linear part of
  the per-pixel projective Jacobian is used). If you need bit-exact agreement with
  OpenCV for any motion type, prefer the stock function.
- Higher-order interpolation (`INTER_CUBIC` / `INTER_LANCZOS4`) in the warp does **not**
  improve registration accuracy in our measurements (corner RMS moves by less than the
  Monte-Carlo noise, and the iteration count is unchanged) while costing 1.4–2.6× per
  iteration — so, like OpenCV, this code stays on bilinear.
- Pairs well with a coarse-to-fine pyramid (align low-res first) to cut warped pixels.

## Citation

```bibtex
@article{evangelidis2008ecc,
  author  = {Evangelidis, Georgios D. and Psarakis, Emmanouil Z.},
  title   = {Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {30}, number = {10}, pages = {1858--1865}, year = {2008}
}
```

## License

BSD-3-Clause. Derived from OpenCV `ecc.cpp` (Copyright © 2000 Intel Corporation);
modifications © 2026 [yosh-shimizu](https://github.com/yosh-shimizu). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
