# ic-metrics

[![build](https://github.com/Ludicon/ic-metrics/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/Ludicon/ic-metrics/actions/workflows/build.yml?query=branch%3Amain)

A small C++ library for image-quality metrics.

This code is based on [Cloudinary's SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) by Jon Sneyers, but refactored to:

- build as a **single compilation unit** (`ic_metrics.cpp` + `ic_metrics.h`),
- contain **no dependencies outside the C standard library** + compiler's SIMD intrinsics,
- performs **no heap memory allocations** (caller passes a scratch buffer),
- also provide **SSIM** and **MS-SSIM** implementations.

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

`ic_ssim_score` and `ic_msssim_score` have exactly the same API, but compute the SSIM and MS-SSIM metrics respectively.

## Performance

Measured on an Apple M4 Pro (10 P-core + 4 E-core).

**Scaling of this implementation:**

| size        | 1 thread | 4 threads | 8 threads |
| ----------- | -------: | --------: | --------: |
| 1024 × 1024 |    43 ms |     16 ms |     16 ms |
| 4096 × 4096 |   771 ms |    241 ms |    193 ms |

The multi-thread numbers above are from the `omp-experiment` branch, which adds `#pragma omp parallel for` to the row loops and reductions to the two map kernels. Past ~4 threads the inner blur passes become memory-bandwidth-bound, so scaling tapers off. These figures predate the default weight pruning, so they isolate thread scaling alone; current `main` single-thread is faster (see the comparison table below).

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

Median across `self` + JPEG q40/q70/q90 distortions, 10 iterations per cell, `bench --threads 1`. The CPU implementations are single-thread; `ic-metrics` and `vszip` use their default weight pruning (see [Differences](#differences)), the rest compute every sub-score. [Vship] is a **GPU** implementation, timed on the M4 Pro through its experimental Vulkan backend on MoltenVK (its time includes our RGBA8→planar deinterleave + host↔device transfer); its scores track the cluster to ~0.2–0.3, with identical images landing near 99.9 rather than 100.

[cloudinary/ssimulacra2]: https://github.com/cloudinary/ssimulacra2
[rust-av/ssimulacra2]:    https://github.com/rust-av/ssimulacra2
[fssimu2]:                https://github.com/gianni-rosato/fssimu2
[fast-ssim2]:             https://github.com/imazen/fast-ssim2
[vszip]:                  https://github.com/dnjulek/vapoursynth-zip
[Vship]:                  https://codeberg.org/Line-fr/Vship

## Differences

Scores are *not* bit-exact with [cloudinary/ssimulacra2]; in practice they match to within ~0.01 across our test set, but a few deliberate deviations are responsible for the drift.

- **Filter form.** Cloudinary uses a single-pass recursive (IIR) Gaussian approximation. We use the textbook separable two-pass FIR. It's easier to understand and trivial to optimize. The two filters have the same intent but produce slightly different results.
- **Border handling.** Cloudinary samples out-of-image taps as zero (clamp-to-border), which artificially attenuates error along the image edges. The error map fades toward black in a 4-pixel rim regardless of what's actually there. We mirror-extend instead, so border pixels are scored the same way interior pixels are. This is the larger of the two contributions to the score drift.
- **Weight pruning.** SSIMULACRA2 sums 108 weighted sub-scores; many of the trained weights are ~0. We skip the blurs and map kernels for any sub-score whose weight is below `ssimu2_prune_threshold` (default `0.01`, matching [vszip]), which is ~25–30% faster than computing everything. The measured score shift is at most +0.0006 across our set; set the threshold to `0.0` for the lossless subset (exact-zero weights only) or a negative value to disable it.

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

CMake (≥ 3.20) is the only requirement. The `omp-experiment` branch additionally needs an OpenMP runtime. On macOS: `brew install libomp`.

The repo also ships two example tools:

- **`ssimudiff`** — score two PNGs and write an error-map PNG: `ssimudiff orig.png dist.png error.png`
- **`bench` / `compare`** — perf and cross-implementation correctness harnesses (compare against [cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2), [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2), [fssimu2](https://github.com/gianni-rosato/fssimu2), [fast-ssim2](https://github.com/imazen/fast-ssim2), [vszip](https://github.com/dnjulek/vapoursynth-zip), and [Vship](https://codeberg.org/Line-fr/Vship) (GPU) when their submodules are checked out under `extern/`).

## AI Disclaimer

The initial port of SSIMULACRA 2 was hand-authored. The surrounding scaffolding (CMake build, `bench`/`compare`/`ssimudiff` harnesses, parallelization experiments, and this README) was developed with help from Claude (Anthropic).

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).
