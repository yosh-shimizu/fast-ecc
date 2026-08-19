# fast-ecc — a faster, drop-in `cv::findTransformECC`

A near drop-in replacement for OpenCV's `cv::findTransformECC` (the ECC image
alignment of **Evangelidis & Psarakis, PAMI 2008**) that **reduces the per-iteration
warps from three to one**. Same iteration math, fewer resampling passes.

```cpp
// before
cv::findTransformECC(templ, input, W, cv::MOTION_AFFINE, criteria);
// after — just change the namespace
fastecc::findTransformECC(templ, input, W, cv::MOTION_AFFINE, criteria);
```

> Not affiliated with or endorsed by OpenCV. Derived from `modules/video/src/ecc.cpp`
> (BSD-3). See [NOTICE](NOTICE) and [LICENSE](LICENSE).

![before/after alignment overlay](docs/before_after.png)

*Template in green, input in magenta — color fringes mean misalignment. After
`fastecc::findTransformECC` (affine, ρ = 0.9997) the overlay collapses to gray.
Reproduce with `examples/align_pair.cpp`.*

## Why it's faster

OpenCV's ECC warps **three** images every iteration: the input image **and** its two
gradient images `∂I/∂x`, `∂I/∂y` (plus a cheap nearest-neighbour mask warp).
`warpAffine`/`warpPerspective` with bilinear interpolation is the dominant cost of the
inner loop.

This version warps **only the image**, then obtains the gradients it needs from the
*warped* image with a `filter2D` and a single `addWeighted` recombination — so the two
bilinear gradient warps per iteration disappear.

| | image warp | gradient warps | mask warp |
|---|:---:|:---:|:---:|
| `cv::findTransformECC` | 1 | 2 | 1 |
| `fastecc::findTransformECC` | 1 | **0** (filter2D + addWeighted) | 1 |

## How it works (the math)

ECC maximizes the normalized correlation between the template and the warped input
`I(W(x; p))`. The Gauss–Newton step needs the **steepest-descent images**, i.e. the
image gradient evaluated at the warped location.

The key identity is the chain rule. For a warp with linear part `A`,

```
∂/∂x [ I(W(x; p)) ] = Aᵀ · (∇I)(W(x; p))
```

So the finite-difference gradient of the **already-warped** image is the warped-image
gradient pre-multiplied by `Aᵀ`. Re-combining it with the warp's linear part
(`warp_gradients_*_ECC`) recovers the quantity the Jacobian assembly expects, **without
warping the gradient images separately**. For translation/euclidean the linear part is
orthonormal so the recombination introduces no directional distortion; for
affine/homography it uses the warp's global linear part (a close approximation).

### Not bit-identical to OpenCV — but just as accurate

This method takes gradients **of the warped image**, whereas OpenCV warps gradients
**of the original image**. Bilinear resampling and finite differencing do not commute
exactly, so the result is **not bit-identical** to `cv::findTransformECC`, even for pure
translation. What matters in practice is the recovered transform: across motion types it
matches OpenCV's accuracy against ground truth — and in the bundled benchmark it is
**consistently a little closer to ground truth**, because the gradients are consistent
with the actually-resampled intensities. The shipped equivalence test asserts fastecc is
**no less accurate than `cv::findTransformECC`** (corner RMS vs ground truth) for all four
motion types.

## Benchmark

Synthetic 512×512 image, best of 25 runs, Release build (MSVC, OpenCV 4.x).
Numbers are hardware-dependent — reproduce on your machine with `./build/bench`.

<!-- BENCH:START -->
| motion | cv (ms) | fast (ms) | speedup | errCv (px) | errFast (px) |
|---|---:|---:|---:|---:|---:|
| TRANSLATION |  18.3 |  16.4 | 1.12× | 0.667 | 0.172 |
| EUCLIDEAN   |  38.6 |  35.1 | 1.10× | 0.710 | 0.169 |
| AFFINE      |  55.8 |  44.6 | 1.25× | 0.921 | 0.293 |
| HOMOGRAPHY  | 115.9 | 104.7 | 1.11× | 1.066 | 0.380 |
<!-- BENCH:END -->

`errCv` / `errFast` are the corner RMS (in pixels) of the recovered warp against the
known ground-truth warp — i.e. accuracy, lower is better. The speed-up comes from
dropping two bilinear warps per iteration; the exact factor depends on image size,
motion type and how warp-bound your build is.

## Quick start

```bash
git clone https://github.com/yosh-shimizu/fast-ecc.git
cd fast-ecc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # equivalence vs cv::findTransformECC
./build/bench                                 # timing table
./build/align_pair template.png input.png aligned.png affine
```

Requires OpenCV (`core`, `imgproc`, and `video` for the `MOTION_*` enum) — any 4.x.
On Debian/Ubuntu: `sudo apt-get install libopencv-dev`.

### Use it in your project

The library is a single `.cpp` + header. Either add this repo via CMake
`add_subdirectory(fast-ecc)` and link `fast_ecc`, or just drop `include/fast_ecc.hpp`
and `src/fast_ecc.cpp` into your build.

```cpp
#include <fast_ecc.hpp>

cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
double rho = fastecc::findTransformECC(
    templateGray, inputGray, W, cv::MOTION_AFFINE,
    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 1e-4));
```

The signature mirrors `cv::findTransformECC` exactly (same arguments, same defaults,
same `MOTION_*` types, same `warpMatrix` conventions).

## Scope & limitations

- Inputs are single-channel `CV_8UC1` or `CV_32FC1`, like OpenCV.
- The speed-up is largest where warps dominate the inner loop (i.e. plain ECC). If you
  add heavy per-iteration work on top (robust losses, extra parameters), the relative
  gain shrinks.
- For affine/homography the gradient recombination is an approximation; if you need
  bit-exact agreement with OpenCV for those motions, prefer the stock function.
- Higher-order interpolation (`INTER_CUBIC` / `INTER_LANCZOS4`) in the warp does **not**
  improve registration accuracy in our benchmarks (corner RMS moves by less than the
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
