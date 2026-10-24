# ic-metrics

[![build](https://github.com/Ludicon/ic-metrics/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/Ludicon/ic-metrics/actions/workflows/build.yml?query=branch%3Amain)

A small C++ library for image-quality metrics.

This code is based on [Cloudinary's SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) by Jon Sneyers, but refactored to:

- build as a **single compilation unit** (`ic_metrics.cpp` + `ic_metrics.h`),
- contain **no dependencies outside the C standard library** + compiler's SIMD intrinsics,
- performs **no heap memory allocations** (caller passes a scratch buffer),
- also provide **SSIM**, **MS-SSIM**, and **IW-SSIM** implementations.

More metrics are planned.

The upstream code is BSD-3-Clause (copyright Cloudinary); attribution and the same license are preserved. see [`LICENSE`](LICENSE).

## API

```c
#include "ic_metrics.h"

// Allocate scratch space
size_t n = ic_ssimulacra2_score_scratch_size(w, h);
void* scratch = malloc(n);

// Compute score
double score = ic_ssimulacra2_score(w, h, orig_rgba, dist_rgba, scratch);

// Optionally, provide error map argument:
unsigned char* error_map = (unsigned char*)malloc(w * h * 4);
score = ic_ssimulacra2_score(w, h, orig_rgba, dist_rgba, scratch, error_map);

// Free scratch space after done.
free(scratch);
```

`ic_ssim_score`, `ic_msssim_score` and `ic_iwssim_score` have exactly the same API, but compute the SSIM, MS-SSIM and IW-SSIM (information content weighted SSIM) metrics respectively.

`ic_ssim_score_channel` computes single-channel SSIM on one plane of the RGBA8 input (channel 0-3) instead of Rec.601 luma. This is intended for non-color data (normal XY, roughness, alpha).

## Performance

Measured on an Apple M4 Pro (10 P-core + 4 E-core).

**Vs. other SSIMULACRA 2 implementations:**

| implementation           | 1024 × 1024 | 4096 × 4096 | notes |
|---|---:|---:|---|
| [Vship]                  |      7 ms   |    101 ms   | GPU, Vulkan/MoltenVK (see note) |
| `ic-metrics` (this)      |     29 ms   |    510 ms   | NEON / AVX2 path, default weight pruning |
| [fssimu2]                |     43 ms   |    757 ms   | Zig port |
| [vszip]                  |     63 ms   |   1010 ms   | Zig, from the vapoursynth-zip plugin, weight pruning |
| [cloudinary/ssimulacra2] |     71 ms   |   1212 ms   | reference C++, SIMD via Highway |
| [fast-ssim2]             |     48 ms   |   1280 ms   | Rust, SIMD-accelerated |
| [rust-av/ssimulacra2]    |     82 ms   |   1575 ms   | Rust port |

Median across `self` + JPEG q40/q70/q90 distortions, 10 iterations per cell, `bench --threads 1`. The CPU implementations are single-thread; `ic-metrics`, `vszip` and `Vship` use weight pruning (see [Differences](#differences)), the rest compute every sub-score. [Vship] is a **GPU** implementation, timed on the M4 Pro through its experimental Vulkan backend on MoltenVK (time includes host-device transfer).

[cloudinary/ssimulacra2]: https://github.com/cloudinary/ssimulacra2
[rust-av/ssimulacra2]:    https://github.com/rust-av/ssimulacra2
[fssimu2]:                https://github.com/gianni-rosato/fssimu2
[fast-ssim2]:             https://github.com/imazen/fast-ssim2
[vszip]:                  https://github.com/dnjulek/vapoursynth-zip
[Vship]:                  https://codeberg.org/Line-fr/Vship
[fast-ssimu2-paper]:      https://www.researchgate.net/publication/401347718_Fast_implementation_of_SSIMULACRA2_for_image_quality_assessment

## Differences

Scores are *not* bit-exact with [cloudinary/ssimulacra2]; in practice they match to within ~0.01 across our test set, but a few deliberate deviations are responsible for the drift.

- **Filter form.** Cloudinary uses a single-pass recursive (IIR) Gaussian approximation. We use the textbook separable two-pass FIR. It's easier to understand and trivial to optimize. The two filters have the same intent but produce slightly different results.
- **Border handling.** Cloudinary samples out-of-image taps as zero (clamp-to-border), which artificially attenuates error along the image edges. The error map fades toward black in a 4-pixel rim regardless of what's actually there. We mirror-extend instead, so border pixels are scored the same way interior pixels are. This is the larger of the two contributions to the score drift.
- **Weight pruning.** SSIMULACRA2 sums 108 weighted sub-scores; many of the trained weights are ~0. We skip the blurs and map kernels for any sub-score whose weight is below `ssimu2_prune_threshold` (default `0.01`, matching [vszip]), which is ~25–30% faster than computing everything. The measured score shift is at most +0.0006 across our set; set the threshold to `0.0` for the lossless subset (exact-zero weights only) or a negative value to disable it. Skipping near-zero-weight sub-scores was proposed in [*Fast Implementation of SSIMULACRA2 for Image Quality Assessment*][fast-ssimu2-paper].

## Error map

`ic_ssimulacra2_score` can fill an RGBA error map sized `w × h`, coloring where the metric thinks the perceptual difference lives. Example output, JPEG q40 vs. its source (paving-stone texture, [CC0 from AmbientCG][ambientcg]):

![error map example](docs/error_map_example.png)

Brighter regions correspond to higher per-pixel error. Note how the blocking and ringing JPEG introduces along the stone edges lights up most of the map.

[ambientcg]: https://ambientcg.com/view?id=PavingStones065

## Building

```sh
cmake -B build
cmake --build build
```

CMake (≥ 3.20) and a C++ compiler are the only requirements.

The repo also ships two example tools:

- **`ssimudiff`** — score two PNGs and write an error-map PNG: `ssimudiff orig.png dist.png error.png`
- **`bench` / `compare`** — perf and cross-implementation correctness harnesses (compare against [cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2), [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2), [fssimu2](https://github.com/gianni-rosato/fssimu2), [fast-ssim2](https://github.com/imazen/fast-ssim2), [vszip](https://github.com/dnjulek/vapoursynth-zip), and [Vship](https://codeberg.org/Line-fr/Vship) (GPU) when their submodules are checked out under `extern/`).

## AI Disclaimer

The initial port of SSIMULACRA 2 was hand-authored. The surrounding scaffolding (CMake build, `bench`/`compare` harnesses, and this README), weight pruning optimizations, and the IW-SSIM ipmlementation were developed with help from Claude.

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).
