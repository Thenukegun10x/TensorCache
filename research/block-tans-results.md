# Dictionary Block-tANS — Results Log

Research track, `src/tensorcache/tans.py` + `tans_dict.py`. Nothing in the
existing cache paths imports these modules; no existing file was modified.

## Setup

- Data: COCO val @336 `balanced` (`data/coco_val`, 5000 JPEGs).
  Calibration = first 60 (sorted), eval = disjoint later slices.
- Dictionary: K=8 tables per section kind, learned by deterministic k-means
  over sqrt-probabilities of block histograms (block=2048), Laplace-smoothed.
- Hardware: RTX 4050 Laptop (6 GB), performance governor; the GPU idles at
  ~210 MHz and needs ~1 s of sustained kernels before timing (the benchmark
  does this; without it you measure DVFS ramp).

## M0 — coder parity and dictionary ratio

Single-table tANS vs rANS, real XS sections (336 balanced):

| section | rANS | tANS (1 block) | tANS B=2048 |
|---|---|---|---|
| smooth i8 | 4.925x | 4.916x | 4.836x |
| mid i8 | 2.059x | 2.058x | 2.042x |
| textured i8 | 1.526x | 1.525x | 1.517x |
| ll4 | 2.90x | 2.90x | 2.91x |

Single-block deltas are ≤ 0 — the coder is at parity with rANS; every
positive delta is the 8 B/block header. (Confirmed by sweeping B: at
B = section length, mid i8 is -0.02%, smooth i8 -0.15%.)

Dictionary tANS, 12 held-out images, referenced tables (E=0) amortized
over a 5000-section cache:

| | rANS | tANS-1 | dict, per-section table | dict, per-block table |
|---|---|---|---|---|
| u8 | 1.028x | 1.025x | 1.078x | 1.105x |
| i8 | 2.022x | 2.006x | 2.027x | 2.133x |
| total (u8+i8) | 1.595x | 1.586x | 1.630x | **1.696x (+6.0%)** |

- Per-block selection beats per-section by ~4%: the heterogeneity the
  dictionary is meant to capture is real.
- Generalization gap is nil: calibration +0.068 vs eval +0.079 bits/sym
  (K=8). K=16 gives +0.070 (capacity saturated at K=8).
- The `u8` masks (which rANS only got 1.03x on) are where the dictionary
  helps most: 1.028x -> 1.105x (+7.5%).

## M1 — Triton GPU decode

Kernel: one warp per 32 blocks, one lane per block, table resident per
launch; payload grouped by selector at write time so each launch has a
single table. Bit reads window ≤ 3 bytes per symbol. 128 COCO i8 sections,
4.21 MB decoded output, steady state, 60 reps:

| B | blocks | stored | ratio | GPU GB/s | vs 1-core numba rANS |
|---|---|---|---|---|---|
| 64 | 65824 | 2513.9 KB | 1.635x | 10.1 | ~50x |
| 128 | 32945 | 2223.4 KB | 1.849x | 8.6 | ~45x |
| **256** | 16509 | 2082.5 KB | **1.974x** | **7.1** | **~37x** |
| 512 | 8287 | 2017.6 KB | 2.037x | 5.3 | ~28x |
| 1024 | 4176 | 1991.0 KB | 2.064x | 3.4 | ~16x |
| 2048 | 2124 | 1985.1 KB | 2.070x | 1.9 | ~9x |

For reference, CPU numba on one core: rANS 0.20 GB/s, tANS 0.14 GB/s.
So the GPU is 35-50x a single core and roughly 4-5x an 8-worker CPU pool.
**B=256 is the operating point**: it beats rANS on ratio (1.974x vs 1.958x)
and still runs at 7.1 GB/s.

Correctness: GPU batch decode is bit-identical to the CPU reference for
every B tested (asserted in the benchmark); the Triton two-pass encoder
(lengths + pack) reproduces the CPU payload byte-for-byte.

Encode (CPU numba, 60 real COCO i8 sections, warmed): rANS 0.33 ms/img,
tANS 0.50 ms/img — tANS is 1.5x slower. Both are far below the wavelet
encode (~35 ms/img), so the build pipeline is not entropy-bound, but the
codec itself is slower on CPU (see Downsides).

## M2 (partial) — end-to-end encode/decode

`benchmarks/e2e_tans.py`: images → batched GPU wavelet+pack → CPU numba tANS
encode → **GPU tANS decode → GPU wavelet decode → pixels**, verified
pixel-exact against the untampered arena decode at every setting. Decode
plan (blob parsing, table grouping, payload/meta upload, output buffers) is
built once, as a loader would at cache open; the timed path is kernel
launches only.

Decode, 256 COCO images/batch, steady state, dictionary from 120 disjoint
images:

| B | ratio | wavelet only | tANS kernels only | e2e | entropy overhead | H2D saved |
|---|---|---|---|---|---|---|
| 128 | 10.60x | 10358 img/s | 121653 img/s | 10206 img/s | **+1%** | 34% |
| 256 | 11.14x | 10307 img/s | 86304 img/s | 9468 img/s | +9% | 37% |

At production batch sizes the entropy layer is essentially free — the tANS
kernels (86-122k img/s) are hidden behind the wavelet decode. The real win
is H2D: only the compressed sections cross PCIe (~28-30 KB/img instead of
the ~46 KB/img decoded arena), and with the blobs resident on device it is
zero after the first epoch.

Encode (one-time build): wavelet+pack ~17-20 ms/img, tANS ~1.8-2.7 ms/img
on CPU. Note tANS encode is 1.5x rANS encode, so build time grows slightly;
the Triton GPU encoder is validated correct but not wired into this path.

Fixed en route: `_cached_codec` was `maxsize=16`, but a 4-kind × K=8
dictionary has 32 tables, so it thrashed and rebuilt tables (~9 ms each) on
every section — encode dropped from 37 ms/img to 1.8 ms/img once sized to
the working set. Any multi-kind dictionary use hits this.

## v3.1 — u16 per-block header (format FMT_VERSION=2)

`bit_len` and `final_state` both fit u16 for R <= 15 and B*R <= 65535, so the
per-block header dropped 8 B -> 4 B. Measured (60 COCO eval, dict from 120):

| | u32 rows | u16 rows |
|---|---|---|
| tANS total | 1628.9 KB | **1587.4 KB** |
| vs Shannon (order-0) | +1.11% | **−1.47%** |
| i8 | 4.098 b/sym | 3.972 b/sym |
| ll4_hi | 82.8% of H | 94.9% |

692 B/img saved; the whole section mix now codes *below* the global order-0
bound (the dictionary's per-block selection is a conditional model).

Decoupling effect (ratio was the reason to avoid small B):

| B | ratio now | ratio before | GPU kernels | e2e overhead |
|---|---|---|---|---|
| 128 | 11.64x | 11.09x | 55.7k img/s | +14% |
| 256 | 11.95x | 11.66x | 36.0k img/s | +28% |
| 512 | 12.08x | 11.92x | 24.1k img/s | +48% |

B=128 is now the training sweet spot: near-fastest GPU, lowest overhead, and
no ratio penalty versus the old B=256 point.

Compatibility: no released cache uses method id 2 (PixelCacheWriter writes
only raw/rANS), so this cannot affect existing caches. A `fmt` byte was added
to the section header so a mismatched container fails with "unsupported tANS
section format" instead of misparsing; goldens were regenerated and the
layout tests updated.

## M2 hardening (done before wiring)

- **Cache-embeddable dictionary schema** (`tans_dict.dictionary_to_meta` /
  `dictionary_from_meta`): compact per-kind (sym, freq) pairs, a version and
  a `codec: "tans"` tag, plus full validation on load (sums to 2**R, no
  duplicate symbols, declared K matches, expected kinds present). A corrupt
  or foreign meta raises `TansDictError` instead of decoding garbage. The
  decoder trusts only in-cache tables, so a cache stays reproducible.
- **ll4 byte-planar tANS** (`pack_byte_planar_tans` /
  `unpack_byte_planar_tans`): same outer wrapper and method ids as the rANS
  version, inner section method 2 marks the codec, so a reader can dispatch
  on it. Ratio-neutral or better, and closes the format gap for integration.
- **Decode capability gate** (`tans_gpu_decode_available` /
  `require_tans_gpu_decode`): tANS decode is GPU-only in practice (CPU is
  1.5x slower than rANS), so callers get an explicit error with the
  workaround rather than a silently slow path. CPU never claims GPU decode.
- Tests: 27 in `test_tans*.py` (schema roundtrip, corruption rejection,
  planar roundtrip + raw fallback, gate behaviour on CPU and CUDA).

## VRAM: the resident-compressed claim, measured

`benchmarks/vram_tans.py`, real allocations on the 4050:

| resident on device | per image | COCO-5000 |
|---|---|---|
| decoded arenas (u8+i8+ll4+meta) | 48.9 KB | 251 MB |
| tANS blobs + tables | 29.9 KB | 153 MB |

So residency is **1.64x**, saving ~98 MB at COCO-5000 — not an
order-of-magnitude win. Correction to an earlier verbal estimate: the
decoded arena is ~49 KB/img (the sparse arena is already compact), not
~140 KB, and raw RGB is a third representation that never needs to be on
the device. The VRAM story is real but modest at this scale; it grows
linearly with image count/resolution (ImageNet-1k, higher res). The
stronger systems wins are the 39% PCIe reduction and dropping the
`num_workers>=4` CPU decode requirement.

## Downsides (measured, not speculative)

1. **CPU/serial paths are ~1.5x slower than rANS.** On real COCO i8:
   decode rANS 0.18 GB/s vs tANS 0.12 GB/s; encode 0.33 vs 0.50 ms/img.
   The +5.6% ratio win only pays if decode runs on the GPU. A CPU-only
   deployment (no Triton/CUDA) would get smaller files but a slower
   per-sample fetch — a net regression unless extra workers absorb it.
2. **Not in the shipped loader yet.** `xs_entropy="tans"` and the in-loader
   GPU decode path are not wired into `pixel_cache.py`; the e2e harness
   measures the exact path a loader would take, but the API/flag, meta
   schema and DataLoader integration are still M2 work.
3. **Dictionary is an input, not free.** Requires a representative
   calibration sample and a versioned artifact. Wavelet-mode shift is
   fine (dict trained on `balanced`, applied to `compress` +7.15% and
   `ultra` +5.34% vs rANS), but genuinely different data (medical,
   satellite, line art) is untested and the Laplace-smoothed tables would
   degrade toward — not catastrophically below — rANS parity.
4. **Referenced tables (E=0) make the cache non-self-describing.** The
   blobs are undecodable without `_pixel_meta.json` tables. Sections that
   embed tables cost ~5 KB each, which is why they don't.
5. **Margins are uneven.** p5 +1.42%, worst +0.12%: some images gain
   almost nothing. The aggregate is carried by the dense `u8` (+8.7%) and
   `i8` (+5.4%) sections; small `ll4` sections mostly stay raw via the
   gain gate.
6. **New format, new surface.** Second codec + format + learning pipeline
   + Triton kernels to maintain; old readers cannot read method id 2
   (forward incompatible, same policy as the rANS rollout); pure-Python
   fallback decode is very slow (numba is effectively required on CPU).
7. **One GPU, one dataset.** Speed was measured on a throttling laptop
   4050 under sustained load; ROCm/9070 XT is unvalidated, as is behavior
   on a throttled long training run.

## M1b — does it hold across real images?

Disjoint split: dictionaries learned on 300 COCO images, evaluated on 200
**different** images. Full v2 section mix (u8 + i8 + ll4 byte-planar + raw
meta), dictionary amortized (charged its 59.7 KB at this cache size).

- **200/200 images beat rANS.** mean +5.74%, p50 +5.96%, p95 +8.71%,
  worst +0.12% (i.e. never worse), best +10.75%.
- Byte-weighted aggregate: rANS **10.540x**, dict-tANS **11.166x** (+5.61%),
  both measured against raw RGB.
- Per-image ratio spread (vs raw RGB): rANS p5 6.44x / p50 11.20x /
  p95 22.30x; dict-tANS p5 6.86x / p50 11.80x / p95 24.33x. The win tracks
  the whole distribution, so it is not driven by easy images.
- Where it comes from: `u8` densest sections +8.7% (rANS 1.004x ->
  1.100x), `i8` +5.4%, `ll4_lo` +5.4%, `ll4_hi` +1.6%. The sections rANS
  handled worst are where the dictionary helps most.
- At a 5000-image cache the dictionary share drops from 299 B/img to
  ~12 B/img, so the +5.61% becomes ~+6.5%.

## Interpretation

- The coder is ratio-neutral vs rANS and strictly better with a dictionary;
  the win is +6% bytes at the u8+i8 level, concentrated on the sections rANS
  handled worst.
- The GPU kernel is fast enough that entropy stops being a pipeline
  bottleneck: 7.1 GB/s vs the ~1.7 GB/s the ~12k img/s wavelet decode
  consumes. Fusion into the wavelet mega-kernel is therefore optional.
- The remaining structural win is the loader: with entropy on GPU, blobs
  can be decoded on-device, so the `num_workers>=4` CPU requirement and the
  ~140 KB/img D2H of decoded arenas both disappear.

## Open items (M2/M3)

- numba CPU fallback for `decode` exists; Triton path is CUDA/ROCm only.
- Cache integration: `xs_entropy="tans"` writer flag, E=0 tables in
  `_pixel_meta.json`, method id 2 in `enc` rows, v1/v2 readers untouched.
- Fusion measurement (expected small, per above).
- Full COCO-5000 eval + ablations (K, R, B, per-kind vs shared dictionary).
- ROCm validation on the 9070 XT.

## Reproduce

```
.venv/bin/python TensorCache/tests/gen_tans_goldens.py
.venv/bin/python -m pytest TensorCache/tests/test_tans.py TensorCache/tests/test_tans_dict.py -q
.venv/bin/python TensorCache/benchmarks/calibrate_tans_dict.py 60 12
.venv/bin/python TensorCache/benchmarks/eval_tans_coco.py 300 200
.venv/bin/python TensorCache/benchmarks/bench_tans_gpu.py 128 60
.venv/bin/python TensorCache/benchmarks/bench_tans_ratio.py
```
