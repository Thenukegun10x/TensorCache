# Research Plan: Dictionary Block-tANS, Triton-native, in TensorCache

**Status:** draft v2 — reviewed, gaps closed (see §7).
**Goal:** a GPU-native entropy layer for XS arena caches that matches rANS
ratio, decodes at multi-GB/s on GPU, and enables a zero-CPU-worker loader
path (plus VRAM-resident compressed datasets). Shipped behind a flag; all
existing formats/readers untouched.

## 0. Background

- XS arenas have 4 section kinds (`u8` masks, `i8` vals, `meta` raw, `ll4`),
  currently coded with 4-way interleaved rANS on CPU in DataLoader workers
  (~0.2-0.3 ms/sample, needs `num_workers>=4`).
- Rice-in-Triton experiment (`benchmarks/exp_triton_rice.py`) tied CPU rANS
  on speed and lost 10-45% on ratio: bit-serial work moved, not removed,
  and Rice can't track peaked distributions. Lesson: keep near-entropy
  coding (table ANS family), kill per-symbol serial *cost* (no division),
  parallelize over many independent blocks.

## 1. Algorithm: dictionary block-tANS

- **Base coder:** tANS, precision L=12 (TOT=4096, same as rANS for direct
  comparison). Per-symbol decode = table lookups, no division, no variable
  renorm loop. L is jointly a ratio knob and an occupancy knob: the decode
  table is 2^L entries (~4 B/entry → 4 KB at L=10, 8 KB at L=11, 16 KB at
  L=12) loaded to SRAM per program — hence the L ablation in §4.
- **Blocking:** each section split into contiguous chunks of B symbols
  (sweep B in {128, 256, 512, 1024, 2048}; default guess 512). One
  independent tANS stream per block: own bit-range + final state.
  Cross-block dependency: none. Per-block header overhead is
  4 (bit_off) + 4 (bit_len) + 4 (final_state) + 1 (selector) = 13 B, so at
  B=128 the header alone costs ~10% on dense sections — the sweep must
  report ratio *net* of headers, and B<256 is expected to lose.
- **Dictionary, not one global table:** 8 prototype distributions per
  section kind, learned offline from a calibration split (k-means over
  normalized histograms). Each block stores a 3-bit selector for its best
  table. Encode-side selection: exact bit-cost over the 8 tables is only
  8 dot-products of the block histogram against precomputed codelength
  vectors (one vectorized pass) — no trial encoding.
- **Determinism is load-bearing:** spread tie-breaks, table order, and
  selector argmin ties must be fully deterministic (no hash seed, no dict
  ordering) across runs and platforms — byte-identical output for identical
  input, or goldens (§5/M0) are meaningless.
- **Section kinds:** separate dictionaries for `u8`, `i8`, `ll4-lo`,
  `ll4-hi` (statistics differ: bitmap-ish vs Laplacian vs peaked). `meta`
  stays raw (measured rANS loss carries over; re-verify once).
- **`ll4` interaction:** keep the existing byte-planar split (lo/hi planes
  coded as separate tANS sections), since planar separation beat mixed
  streams for rANS and the cause (peaked-hi vs uniform-lo) is coder-agnostic.

## 2. Format v3 (back-compatible by construction)

- New section method id `2` = block-tANS (`0` raw, `1` rANS unchanged).
- Per-section layout: `[u32 n][u8 R][u8 K][u32 B][u8 E]`
  + (K × (`[u16 n_used]` + used-symbols freq list) iff E == 1)
  + `[u32 nblocks]` + `[nblocks × (u32 bit_len, u32 final_state)]`
  + (`[nblocks × u8 selector]` iff K > 1) + concatenated block payloads.
  byte_len is derived as ceil(bit_len/8), keeping headers at 8 B/block.
  E = 1 embeds tables (standalone blobs, tests); E = 0 references them
  (cache design — decode takes `tables`, which live once per cache in meta;
  embedding 8 tables per section cost ~5 KB on ~10 KB sections, measured).
- Tables stored **in the cache** (meta sidecar, ~10 KB total: 4 dicts ×
  8 tables × ~100 used symbols × 3 B), never hardcoded: caches stay
  self-describing across code versions, and decode depends *only* on
  in-cache tables. `tans_dict_version` in meta is informational
  (reproducibility of the learning procedure), never read by the decoder.
  Meta gets `xs_entropy="tans"`, `tans_dict_version`, per-row `enc` flags
  extended (existing `enc` int already carries method ids).
- Old readers: see unknown method → clean error telling minimum version
  (same policy as the rANS rollout). New readers: keep v1/v2 branches.

## 3. Implementation

- **New module `src/tensorcache/tans.py`** (no edits to `rans.py`):
  `normalize` reuse from rANS, `spread()` (Duda fast spread),
  `build_encode_table` / `build_decode_table`, `encode_block`,
  `decode_block`, `pack_section_tans` / `unpack_section_tans` (same
  MIN_GAIN-style fallback-to-raw policy, threshold re-tuned; sections that
  fail the gate — expect most `u8` masks — stay raw, so worst case ties
  the rANS baseline instead of regressing it).
- **Error semantics mirror `rans.py`:** `TansError`, absurd-count guard
  (never OOM on corrupt `n`), truncation/corruption tests. Untrusted cache
  bytes must Err, never hang or panic.
- **Triton kernels** (same module, CUDA/ROCm via Triton, CPU fallback
  below): `_tans_pack_kernel` / `_tans_unpack_kernel`, one program per
  block, only the selected table loaded to SRAM per program. Batched entry
  points operating on concatenated multi-image buffers (loader-shaped input).
- **CPU fallback:** pure-Python reference + numba `njit` block decoder
  (mirrors `rans.py` convention), so v3 caches read on Windows / GPU-less
  hosts. Triton path used iff available + device is CUDA/HIP.
- **Writer integration:** `PixelCacheWriter(..., xs_entropy="tans")` —
  note this widens `xs_entropy` from `bool` to `bool | str` (`True` keeps
  meaning rANS, `"tans"` selects v3, `False` is raw; old `True/False` metas
  keep reading unchanged).
- **Loader integration (flag-gated, default off):** GPU-direct path —
  H2D compressed blobs + offsets, on-GPU tANS decode → existing batch
  wavelet driver. VRAM-resident mode: upload blobs once (LRU window for
  datasets over budget — ImageNet-scale never fully resides), zero H2D
  after warmup. Target API: `make_xs_loader(..., entropy_device="cuda")`.
- **Exports:** update `__init__.py` + `__all__` per AGENTS.md §6.

## 4. Evaluation protocol

- **Data:** COCO val 5000 @336 `balanced` (matches published 10.28x baseline),
  plus `textured`/`smooth` modes; calibration/eval splits disjoint
  (dictionary must not train on eval images).
- **Metrics:** ratio vs raw RGB; build img/s (encode+pack+entropy);
  decode GB/s and img/s; loader img/s at `num_workers=0` vs rANS baseline
  at `num_workers=8`; peak VRAM (build with N=128, train with B=256);
  max trainable batch on 6GB 4050 (resident vs streaming).
- **Baselines:** raw arenas, rANS (`xs_entropy=True`), no-entropy v1.
- **Ablations:** B sweep; tables {1,4,8,16}; L {10,11,12};
  dictionary vs single global table; per-kind vs shared dictionary.
- **Hardware:** RTX 4050 Laptop (primary); RX 9070 XT if accessible
  (validates the ROCm/Triton path).

## 5. Milestones & exit criteria

- **M0 — Python reference:** `tans.py` CPU paths, roundtrip + golden tests.
  No Rust tANS reference exists, so goldens are self-generated: commit
  `tests/data/tans_goldens.bin` from the reviewed v1 implementation, then
  a byte-stability rule (any later byte change fails CI until goldens are
  deliberately regenerated + reviewed). Single-table exit: coder at parity
  (single-block deltas ≤ 0 — measured -0.02% mid-u8, -0.15% smooth-i8),
  net within ±1% of rANS on mid/textured at B=2048; smooth net
  (+1.9%, all 8-B/block header) flips to better-or-equal with the
  dictionary milestone, which is the local-adaptation payoff. Exit: green tests.
- **M1 — Triton kernels — DONE (see research/block-tans-results.md).**
  Correctness bit-identical vs CPU (+) and the two-pass GPU encoder matches
  the CPU payload byte-for-byte. Measured 128×COCO i8, steady state:
  B=128 → 8.6 GB/s / 1.849x; B=256 → 7.1 GB/s / **1.974x (> rANS 1.958x)**;
  B=512 → 5.3 GB/s / 2.037x. 35-45x a single numba core; ~4-5x an 8-worker
  CPU setup. ≥3 GB/s bar cleared at every B tested. Key measurement note:
  the 4050 idles at ~210 MHz and needs ~1 s of sustained kernels before
  timing, or short bursts measure DVFS ramp not throughput.
  Encode build tax: both codecs are far below the wavelet encode
  (~35 ms/img), so the build is not entropy-bound; note tANS is 1.5x
  slower than rANS on CPU (see the results doc's Downsides). The Triton
  two-pass encoder is validated correct but not yet wired into the timed
  path. The `textured`/`smooth` modes from the original plan are not
  presets in this codebase; mode-shift was tested with
  `compress`/`ultra` instead (dictionary generalizes: +7.15%/+5.34%).
  Fusion: standalone kernels already leave the wavelet stage idle-bound
  (7-8 GB/s vs the ~1.7 GB/s the 12k img/s decoder consumes), so fusing
  buys little — measured, not assumed.
- **M2 — Integration:** writer flag, reader back-compat (old caches load),
  GPU-direct loader path; `tests/test_codec.py` + `test_rans.py` green,
  new `tests/test_tans.py`; `AGENTS.md` + `README.md` updated.
  Exit: `pytest -v` clean.
- **M3 — Eval + writeup:** full protocol §4, ablations, honest negatives
  (Rice-style null results included). Exit: post draft + release notes.

## 6. Risks

- Triton scalar bit-loops underperform (Rice precedent) → **resolved**: the
  measured kernel is load-bound but fast enough (7-8 GB/s) once the block
  count is high (B ≤ 256) and clocks are ramped; the plan's B≥256 guidance
  was for ratio, so the operating point is B=256 (both). CUDA/HIP port
  reserved as phase 2, Triton kept as fallback.
- Dictionary overfits calibration → mitigation: disjoint splits, report
  calibration-vs-eval gap explicitly.
- ROCm Triton gaps on 9070 XT → mitigation: CPU-fallback keeps correctness;
  perf portability is a measured result, not an assumption.
- Scope creep into wavelet changes → non-goal: wavelet/sparse layout frozen.

## 7. Review log (v1 → v2)

Re-read caught six gaps, all fixed above rather than appended:

1. **L-vs-SRAM math unstated** — decode table is 2^L × ~4 B (4/8/16 KB
   for L=10/11/12); §1 now states it, which is also the real justification
   for the L ablation.
2. **Header overhead unbudgeted** — 13 B/block kills small B on dense
   sections; §1 now carries the formula and the B≥256 expectation.
3. **Selector procedure missing** — encode-side selection is 8 codelength
   dot-products, no trial encodes (§1).
4. **No error semantics** — new-codec plan with no `TansError`/guards;
   §3 now mirrors `rans.py` (Err, never hang).
5. **`xs_entropy` type change unacknowledged** — `bool` → `bool | str`
   with `True` still meaning rANS; old metas unaffected (§3).
6. **Goldens without a reference impl** — no Rust tANS exists, so §5 now
   specifies self-generated goldens + byte-stability rule instead of
   hand-waving "cross-check discipline". M1 encode criterion sharpened to
   match the measured raw-pack rate; resident mode got its LRU fallback.
