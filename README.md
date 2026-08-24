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

### Accuracy — equal to a fixed OpenCV

Measured against an OpenCV that carries the border fix, on the analytic scene,
affine, window 200, 200 trials, started **at** the ground truth so that any movement
is bias:

| | corner RMS | scale bias (t) |
|---|---:|---:|
| `cv::findTransformECC` (fixed) | 0.0017 px | +8.7e-07 (1.3) |
| `fastecc::findTransformECC` | 0.0017 px | +8.7e-07 (1.3) |

The fused Gauss–Newton stage does not move these: it is the same algebra evaluated in
a different order. Convergence counts over 150 trials at five deformation steps match
`cv` exactly for translation and affine, and differ by one or two borderline trials
out of 150 for euclidean and homography.

Statistically indistinguishable, and neither shows a systematic bias. Before the
border fix this fork sat at 0.0287 px with a −1.8e−04 (t = −63.9) scale bias at
`gaussFiltSize=9`.

**Accuracy is equal, not better.** Which of the two resampling orders is closer to
ground truth remains unsettled — see [Not bit-identical](#not-bit-identical-to-opencv).

### Speed

Cost of one iteration, affine, analytic scene, best of 3 runs, against a
border-fixed OpenCV:

| window | 4 threads | 1 thread |
|---:|---:|---:|
| 256 | **1.80×** | 1.35× |
| 512 | 1.68× | 1.38× |
| 768 | **1.90×** | 1.56× |

Two changes contribute, and they cover each other. The **warp reduction** takes the
three bilinear warps per iteration down to one; it pays most at small windows and
decays to nothing at large ones with a thread pool, because OpenCV parallelises the
`warpAffine` it removes but not the `filter2D` that replaces it. The
**fused Gauss–Newton stage** does the opposite: it builds the Hessian and both
projections in one pass without materialising the Jacobian, and the saving grows with
the parameter count, so it is largest exactly where the warp reduction has given up.

The bundled `./build/bench` (512×512, default thread count) shows the per-motion
spread, and how much of it is the fused stage:

| motion | cv | fast | speedup | before fusion |
|---|---:|---:|---:|---:|
| TRANSLATION | 22.9 ms | 13.4 ms | 1.71× | 1.09× |
| EUCLIDEAN | 35.4 ms | 18.4 ms | 1.93× | 1.06× |
| AFFINE | 46.6 ms | 22.1 ms | 2.11× | 1.11× |
| HOMOGRAPHY | 122.0 ms | 38.4 ms | **3.24×** | 1.21× |

Homography gains most because the Jacobian path re-reads its blocks P² times for the
Gram matrix, and P = 8.

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
| `fastecc::findTransformECC` | 1 | **0** (filter2D) | 1 |

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

Per iteration, one thread, window 200 (best of three interleaved runs):

| motion | before | after | |
|---|---:|---:|---:|
| translation | 0.591 ms | 0.468 ms | 1.26× |
| euclidean | 0.692 ms | 0.571 ms | 1.21× |
| affine | 0.811 ms | 0.691 ms | 1.17× |
| homography | 1.743 ms | 1.556 ms | 1.12× |

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
