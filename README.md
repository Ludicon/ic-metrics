# ic-metrics

[![build](https://github.com/Ludicon/ic-metrics/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/Ludicon/ic-metrics/actions/workflows/build.yml?query=branch%3Amain)

A small C++ library for image-quality metrics.

This code is based on [Cloudinary's SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) by Jon Sneyers, but refactored to:

- build as a **single compilation unit** (one `.cpp` + one `.h`),
- contain **no dependencies outside the C standard library** + compiler's SIMD intrinsics,
- do **zero internal heap allocations** (caller passes a scratch buffer),
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

`ic_ssim_score` has exactly the same API, but computes the SSIM metric instead.

## Performance

Measured on an Apple M4 Pro (10 P-core + 4 E-core).

**Scaling of this implementation:**

| size        | 1 thread | 4 threads | 8 threads |
| ----------- | -------: | --------: | --------: |
| 1024 × 1024 |    43 ms |     16 ms |     16 ms |
| 4096 × 4096 |   767 ms |    241 ms |    193 ms |

Single-thread is what `main` ships. The multi-thread numbers above are from the `omp-experiment` branch, which adds `#pragma omp parallel for` to the row loops and reductions to the two map kernels. Past ~4 threads the inner blur passes become memory-bandwidth-bound, so scaling tapers off.

**Vs. other SSIMULACRA 2 implementations** (single-thread):

| implementation           | 1024 × 1024 | 4096 × 4096 | notes |
|---|---:|---:|---|
| `ic-metrics` (this)      |     43 ms   |    767 ms   | NEON / AVX2 path |
| [fssimu2]                |     43 ms   |    759 ms   | Zig port |
| [cloudinary/ssimulacra2] |     73 ms   |   1203 ms   | reference C++, SIMD via Highway |
| [rust-av/ssimulacra2]    |     84 ms   |   1541 ms   | Rust port |

Median across `self` + JPEG q40/q70/q90 distortions, 10 iterations per cell, `bench --impl all --threads 1` on the `omp-experiment` branch.

[cloudinary/ssimulacra2]: https://github.com/cloudinary/ssimulacra2
[rust-av/ssimulacra2]:    https://github.com/rust-av/ssimulacra2
[fssimu2]:                https://github.com/gianni-rosato/fssimu2

## Differences

Scores are *not* bit-exact with [cloudinary/ssimulacra2]; in practice they match to within ~0.01 across our test set, but two deliberate deviations in the Gaussian blur are responsible for the drift.

- **Filter form.** Cloudinary uses a single-pass recursive (IIR) Gaussian approximation. We use the textbook separable two-pass FIR. It's easier to understand and trivial to optimize. The two filters have the same intent but produce slightly different results.
- **Border handling.** Cloudinary samples out-of-image taps as zero (clamp-to-border), which artificially attenuates error along the image edges. The error map fades toward black in a 4-pixel rim regardless of what's actually there. We mirror-extend instead, so border pixels are scored the same way interior pixels are. This is the larger of the two contributions to the score drift.

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
- **`bench` / `compare`** — perf and cross-implementation correctness harnesses (compare against [cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2), [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2), and [fssimu2](https://github.com/gianni-rosato/fssimu2) when their submodules are checked out under `extern/`).

## AI Disclaimer

The initial port of SSIMULACRA 2 was hand-authored. The surrounding scaffolding (CMake build, `bench`/`compare`/`ssimudiff` harnesses, parallelization experiments, and this README) was developed with help from Claude (Anthropic).

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).
