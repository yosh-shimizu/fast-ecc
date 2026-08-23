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

Early. The warp reduction described below is implemented; the rest of the roadmap is
measured but not yet committed.

| | state |
|---|---|
| Warp reduction (3 warps → 1 per iteration) | landed |
| Gaussian pre-filter border fix | **known defect, fix pending** — see below |
| Fused Gauss–Newton stage | measured, not committed |
| Meaningful accuracy regression test | **known defect, fix pending** |

### Known defect: the Gaussian pre-filter border

`findTransformECC` smooths the template with `GaussianBlur`, whose default
`BORDER_REFLECT_101` fabricates a ring `gaussFiltSize/2` px wide by mirroring the
interior. That ring is then used at full weight, which biases the estimated linear part
toward a smaller scale. **This fork inherits the defect from `ecc.cpp` verbatim**: it
masks the warped *input* border but never the *template's* fabricated ring.

Measured against an OpenCV build that has the fix (window 512, affine): OpenCV 0.027 px
corner RMS, this fork 0.294 px. Porting the fix brings it to 0.019 px. It is not yet
committed only because it is queued behind the upstream review.

Until then, **do not use this fork where absolute accuracy matters**; the speed
comparison is unaffected.

### Known defect: the equivalence test proves little

`test/test_equivalence.cpp` asserts `errFast <= errCv + 0.5px`. The estimator's own
accuracy floor is around 0.01–0.03 px, so a **10× accuracy regression passes**. The same
flaw was reported against OpenCV's own ECC tests
([opencv#29776](https://github.com/opencv/opencv/issues/29776)); this fork has its own
version of it and needs a criterion with a real threshold.

## Results

Populated as commits land. Every row states the operating point.

*(none yet — the current benchmark compares against an OpenCV that has the border
defect, which flatters this fork for the wrong reason. It will be re-measured against a
fixed baseline once the fix above is committed.)*

## Why the warp reduction is faster

OpenCV's ECC warps **three** images every iteration: the input image **and** its two
gradient images `∂I/∂x`, `∂I/∂y` (plus a cheap nearest-neighbour mask warp).
`warpAffine`/`warpPerspective` with bilinear interpolation is the dominant cost of the
inner loop.

This version warps **only the image**, then obtains the gradients from the *warped*
image with a `filter2D` and a single `addWeighted` recombination — so the two bilinear
gradient warps per iteration disappear.

| | image warp | gradient warps | mask warp |
|---|:---:|:---:|:---:|
| `cv::findTransformECC` | 1 | 2 | 1 |
| `fastecc::findTransformECC` | 1 | **0** (filter2D + addWeighted) | 1 |

The gain is **not** uniform. It is largest at small windows and on a single thread, and
falls to break-even at large windows with a thread pool — because OpenCV parallelises
`warpAffine` above roughly 384 px but not the `filter2D`/`addWeighted` that replace it.
Any speed figure quoted without a window size and a thread count is meaningless.

### The math

ECC maximizes the normalized correlation between the template and the warped input
`I(W(x; p))`. The Gauss–Newton step needs the **steepest-descent images**, i.e. the
image gradient evaluated at the warped location. The key identity is the chain rule: for
a warp with linear part `A`,

```
∂/∂x [ I(W(x; p)) ] = Aᵀ · (∇I)(W(x; p))
```

So the finite-difference gradient of the **already-warped** image is the image-domain
gradient pre-multiplied by `Aᵀ`. Multiplying it by `A⁻ᵀ` (`warp_gradients_*_ECC`) undoes
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
ctest --test-dir build --output-on-failure   # see the caveat above
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
