# AGENTS.md — TensorCache

> Guidance for AI coding agents working in this repo. Keep this file concise, factual, and up-to-date.

## 1. Project Overview

**TensorCache** (`tcache` on PyPI, import name `tensorcache`, v0.7.4) — ultra-fast, high-fidelity block-wise INT8 feature & pixel cache engine for PyTorch.

Three bottlenecks solved:
1. **Feature Cache Bloat:** AMO-BQ G32 stores `1.09375 B/value` (`1.83x` vs BF16); opt-in `quality` mode + G16 reaches `0.3652%` rel RMSE (vs `0.5594%` balanced G32) at `1.1875 B/value` (`1.68x`). See `README.md` feature-fidelity table and `research/amo-quality-results.md`.
2. **JPEG/PNG CPU Decode:** zero-copy `np.memmap` + GPU stream prefetch — `>2,000 MB/s`, `iter_batches` ~27k samp/s, `make_xs_loader` ~10k img/s @336.
3. **Pixel Cache Tunable Wavelet XS:** CDF 5/3 + RCT + adaptive RDO (`ultra`/`high`/`balanced`/`compress`/`ultra_comp` presets) — `33dB 13.6x` <-> `44dB 3.2x` (`balanced 37dB 7.8x`), GPU-decoded.

Installation: `pip install tcache` (import `tensorcache` or `tcache`) or `pip install -e .` from repo root. Python `>=3.9`, `torch>=2.0`, `numpy>=1.20`, `pillow>=9.0` (`pyproject.toml`). Extras:
- `entropy` — `numba` only (rANS/tANS hot loops). Without it, codecs fall back to pure Python (~50x slower).
- `fast` — `triton` (skipped on Windows), `zstandard`, `blosc2`, `safetensors`, `numba`.
- `dev` — `pytest>=7.0`, `tabulate`, `rich`.

## 2. Repo Structure

```
/
├── src/tensorcache/
│   ├── __init__.py        # public API re-exports (+ __all__)
│   ├── __main__.py        # `python -m tensorcache` -> cli.main
│   ├── cli.py             # `tensorcache` / `python -m tensorcache`: info, benchmark, cache-info, compress-demo
│   ├── codec.py           # INT8/INT4/INT3 codecs, AMO-BQ, wavelet 8x + adaptive, sparse bitstream, batched encoders
│   ├── fused_ops.py       # Triton fused kernels (guarded import; CPU fallbacks)
│   ├── feature_cache.py   # FeatureCacheWriter / FeatureCacheDataset (mmap .bin + .json; sharded)
│   ├── pixel_cache.py     # PixelCacheWriter / PixelCacheDataset (raw/int4/int3/xs; mono; XS loader)
│   ├── rans.py            # rANS entropy codec for XS sections (numba + Python ref)
│   ├── tans.py            # block-tANS entropy codec (FMT_VERSION=2, GPU-decoded)
│   ├── tans_dict.py       # tANS dictionary learning/validation/meta
│   ├── prefetcher.py      # AsyncGPUPrefetcher (double-buffered CUDA stream)
│   ├── streamer.py        # ZeroCopyTensorStreamer (pinned + ring-buffered)
│   ├── utils.py           # compress/decompress/estimate_compression/benchmark_tensor/auto_select_mode/help_text
│   └── _validate.py       # CacheValidationError + meta/file validation (feature, shards, pixel, xs)
├── src/tcache/            # alias package: `from tensorcache import *`
├── tests/
│   ├── test_codec.py      # codec + feature/pixel disk I/O + XS/entropy + append mode + validation
│   ├── test_amo_quality.py# AMO-BQ quality mode (exhaustive search, cache roundtrip, triton parity)
│   ├── test_fused_ops.py  # GPU-only, skipped if !torch.cuda.is_available()
│   ├── test_rans.py       # rANS tests + Rust golden cross-check (tests/data/rans_goldens.bin)
│   ├── test_tans.py       # block-tANS tests + golden cross-check (tests/data/tans_goldens.bin)
│   ├── test_tans_dict.py  # tANS dictionary learning/validation/meta/GPU-gate
│   ├── test_tans_cache.py # tANS cache roundtrip + parity + loader gates
│   ├── gen_tans_goldens.py# regenerate tests/data/tans_goldens.bin (reviewed, byte-stable)
│   └── data/              # rans_goldens.bin, tans_goldens.bin
├── benchmarks/            # 17 scripts: amo_quality/coco/accuracy, tans GPU/ratio/e2e, dataloader, dual-gpu, memory, vram, xs_gains, etc.
├── research/              # results md/json: amo-*, block-tans-*, tans_dicts
├── .github/workflows/     # ci.yml (pytest + build/twine), release.yml (PyPI, optional)
├── analyze_compression_error.py  # full pixel+feature benchmark (tabulate)
├── experiment_blockwise_int8.py  # G-sweep, symmetric/asymmetric, bitwidth
├── bench_optimal_arch.py  # AMO-BQ architecture sweep
├── run_sensitivity.py     # sensitivity analysis
├── download_test_dataset.py  # Imagenette-320 fetcher -> data/test_images (safe tar extraction)
├── pyproject.toml
└── README.md
```

Data dirs: `data/` (contains `imagenette2-320.tgz`, `flowers-102/102flowers.tgz` — do not commit large binaries). Cache files are runtime-generated (`*_int8.bin`, `*_scales.bin`, `*_zp.bin`, `*_pixels.bin`, `*_meta.json`, `*_pixel_meta.json`, `*_shards.json`) — gitignored.

## 3. Build / Run / Test Commands

```bash
# install (dev)
pip install -e ".[dev]"          # CPU-only
pip install -e ".[fast,dev]"     # + triton/zstd/blosc2/safetensors/numba (Linux; triton skipped on Windows)
pip install -e ".[entropy,dev]"  # lighter: just numba for rANS/tANS

# run all tests (fused tests auto-skip on CPU/Windows)
pytest -v
pytest tests/test_codec.py -v
pytest tests/test_fused_ops.py -v  # requires CUDA + Triton

# single-file direct run (no pytest)
python tests/test_codec.py

# CLI
python -m tensorcache info
python -m tensorcache benchmark --shape 16,446,768 --device cuda
python -m tensorcache cache-info --prefix ./cache/dinov3
python -m tensorcache compress-demo --mode balanced --group-size 16
tensorcache --help

# benchmarks / analysis (require data images)
python analyze_compression_error.py          # needs Plant dataset or data/test_images fallback
python experiment_blockwise_int8.py          # needs CUDA + timm + Plant data
python download_test_dataset.py              # fetches Imagenette-320 (~326 MB) to data/test_images
python benchmarks/bench_amo_quality.py --limit 5000 --repeats 3

# Windows note: Triton unavailable -> codec falls back to vectorized PyTorch path
```

CI: `.github/workflows/ci.yml` — pytest on ubuntu/windows × py3.9/3.13 (+`fast` on ubuntu py3.12), plus build + twine check. `release.yml` publishes to PyPI on release (optional, gated on `PYPI_API_TOKEN` secret). No Makefile.

## 4. Core Architecture — Read Before Editing

### 4.1 Codec (`src/tensorcache/codec.py`)
- **Symmetric INT8:** `quantize_int8_g32(x, group_size=32)` — flatten, pad, `view(-1,G)`, `amax`, `scales = max/127 -> BF16`, `round(x/scale) clamp [-128,127] -> int8`. Returns `(q_int8, scales BF16, orig_shape)`. `dequantize_int8_g32` — Triton path if `HAS_TRITON and device.type in (cuda,hip)` else padded `view(-1,G) * scales` + copy into `out_buffer` (preserve this API).
- **AMO-BQ (asymmetric MSE-optimal):** `quantize_int8_amo_bq(x, group_size, mode=...)` with presets `fast`/`balanced`/`accurate`/`max`/`quality` (`AMO_BQ_PRESETS`). UINT8 values + BF16 scales + UINT8 zero points (`1.09375 B/value` G32, `1.1875 B/value` G16). `quality` is opt-in: searches actual BF16 scales and scores the **decoded** BF16 values (`_quantize_amo_quality_torch`) — ~34.7% lower RMSE than balanced at G16. `dequantize_int8_amo_bq` decodes `(q - zp) * scale`.
- **INT4/INT3 (pixels only):** `pack_int4`/`unpack_int4`, `pack_int3`/`unpack_int3`, `quantize_int4_g32`/`dequantize_int4_g32`, `quantize_int3_g32`/`dequantize_int3_g32`. Feature caches reject `quant_bits<8` (>2% RMSE collapse); pixel caches use them.
- **Wavelet 8x:** `rct_forward`/`rct_inverse`, `quantize_pixel_wavelet8x`/`dequantize_pixel_wavelet8x` (static, `q_scale`).
- **Wavelet adaptive:** `quantize_pixel_wavelet_adaptive` (per-image, dispatch-bound) + `quantize_pixel_wavelet_adaptive_batched` (batched, bit-identical, ~28x). Presets `ultra`/`high`/`balanced`/`compress`/`ultra_comp`; `lamb`/`q_scale` tunable. `ADAPTIVE_CODEBOOK` (16-lvl), `SUBBAND_GAINS`, `WAVELET_ADAPTIVE_PRESETS`.
- **Sparse bitstream:** `sparse_pack_plane`/`sparse_pack_meta`/`sparse_pack_arena` (+ batched variants `*_batched`), `sparse_unpack_plane`/`sparse_unpack_meta`/`arena_to_sparse`, `arena_nbytes`/`sparse_nbytes`. Zero-block skip + hier nibble masks. Batched variants are byte-identical to per-image reference.
- **Batched GPU encode:** `quantize_pixel_wavelet_adaptive_batched(imgs [N,H,W,3], mode=...)` — one torch pass per plane-type; returns N packed_meta dicts bit-identical to per-image encoder.

**Invariants:**
- Scales stored as BF16 on disk as `uint16` bitcast; zero points as `uint8` (`feature_cache.py`). Do not change dtype without migrating file format.
- Padding is trimmed on return (`flatten()[:numel]`). Keep.
- Storage cost: sym `1 + 2/32 = 1.0625 B/elem`; AMO `1 + 3/32 = 1.09375 B/elem` (G32).
- Triton fallback must stay bit-identical to the PyTorch path — test on CPU.

### 4.2 Feature Cache (`src/tensorcache/feature_cache.py`)
- **Writer:** pre-allocates `np.memmap` `w+` (single) or per-shard (`num_shards>1`) with shape `(num_samples, seq_len, dim)` int8/uint8 and `(num_samples, scales_per_sample)` uint16 (+ `_zp.bin` uint8 for AMO). `append` handles `[seq_len,dim]` and `[B,seq_len,dim]` (batched GPU 64/chunk, CPU per-sample). `close()` flushes, closes mmaps, writes `_meta.json` (single) or `_shards.json` (sharded) with `num_samples = current_idx`. Must call `close()` — file lock otherwise.
- **Sharded:** `num_shards=8` → `feat_shard{i}_*.bin` + `feat_shards.json` manifest (`shard_prefixes`, `shard_sizes`, `num_samples`, `seq_len`, `dim`, `group_size`, `amo_bq`). Sequential split (e.g. 10k/8 → 1250 per shard). Dataset/Streamer/streamer accept `rank`/`world_size` or `shard_idx` for DDP.
- **Dataset:** read-only mmap `mode="r"`. `__getitem__` copies via `arr.copy()` then optional `auto_dequant_device` fused dequant. `iter_batches` does C-level batch slice → `torch.from_numpy(...).to(device)` (~27k samp/s contiguous, ~5.8k shuffled).
- **Append mode:** `FeatureCacheWriter.open_append(prefix, extra_samples, **expected_settings)` reopens a single-shard cache for growth (grow-only truncate, `mode="r+"`). Settings asserted against `_meta.json`; sharded caches raise `NotImplementedError`.
- **Validation:** `_validate.py` — `validate_feature_meta`, `validate_feature_shards_manifest`, `_check_file_size`. Raises `CacheValidationError` (a `ValueError`).
- `close()` releases mmap handles — critical on Windows.

### 4.3 Pixel Cache (`src/tensorcache/pixel_cache.py`)
- **Raw/int4/int3:** `quant="raw"/"int4"/"int3"` (or `quant_bits=8/4/3`). Raw uint8 mmap `(N,H,W,C)`; int4/int3 packed + scales. `append_image` accepts `np.ndarray | PIL.Image | torch.Tensor | str|Path`, resizes via BILINEAR. `PixelCacheDataset.__getitem__` returns `torch.uint8 [H,W,C]`.
- **Mono:** `channels=1` stores 1 sample/pixel (`3x` smaller). Color input rejected (no silent conversion). Mono `G=8` recommended (beats RGB G32 on both axes). `quant="xs"` is RGB-only.
- **XS arena cache (`quant="xs"`):** `_append_xs` writes 4 per-sample sections (u8/i8/meta/ll4) with an offset table in `_pixel_meta.json["xs_table"]`. `append_batch(images=|metas=)` routes through batched pack.
- **rANS entropy layer (`xs_entropy=True`, default, 0.6+):** u8/i8/ll4 rANS-coded via `rans.py` (`pack_section`, `pack_byte_planar` — ll4 byte-planar); meta stays raw; sections must shrink >=8% (`MIN_GAIN_PCT`) to be coded. Each row carries `enc=[u8,i8,meta,ll4]` method flags + `lb_off/lb_len`. `get_arena_packed` rANS-decodes worker-side back to byte-identical v1 arena bytes. `xs_entropy=False` writes exact pre-0.6 v1 layout.
- **Block-tANS entropy layer (`xs_entropy="tans"`, 0.7+, `tans.py` + `tans_dict.py`):** opt-in alternative. `PixelCacheWriter(xs_entropy="tans", tans_dict=..., tans_block=256)` codes the same 4 sections with **block-tANS** + a K-table **dictionary** learned from a calibration sample (`learn_dictionary_from_arenas`, k-means over block histograms) with a per-block selector. Section format `FMT_VERSION=2`: `[u32 n][u8 fmt][u8 R][u8 K][u32 B][u8 E]` + K tables (if E=1) + `[u32 nblocks]` + nblocks×u16 `bit_len` + nblocks×u16 `final_state` + selectors (if K>1) + payloads grouped by selector. Dictionary stored **once** in `_pixel_meta.json["tans"]` as base64 (`dictionary_to_meta`/`from_meta`, ~25KB for 4×K=8) and validated on load. **Decode is GPU-only by design** (`build_decode_plan`/`run_decode_plan`, one warp per 32 blocks) — `make_xs_loader` requires `num_workers=0` on a CUDA/ROCm device (`require_tans_gpu_decode` raises otherwise). `cache_images(..., xs_entropy="tans", tans_calib=256)` learns the dictionary in a pre-pass; `tans_dict=` accepts a path/`{kind: tables}`/meta dict to share one.
- **Batched pack:** `sparse_pack_plane_batched`/`sparse_pack_meta_batched`/`sparse_pack_arena_batched` do the whole pack in one torch pass per plane-type (~4x faster than per-image). Byte-identical to per-image reference. All samples in a batch must share H/W/mode/adaptive.
- **Loader:** `make_xs_loader(prefix, batch_size=..., device=..., num_workers=...)` yields `uint8 [B,H,W,3]` on CUDA. `cache_images(dir, prefix, ...)` one-liner build.

### 4.4 Fused Ops (`src/tensorcache/fused_ops.py`)
- **Requires Triton + CUDA/ROCm** — import is guarded (`HAS_TRITON`), CPU fallbacks everywhere. Kernels: `_fused_quant_kernel`, `_fused_amo_quant_kernel` (+ `_fused_amo_quality_kernel`), `_fused_dequant_kernel`, `_fused_dequant_int4_kernel`, `_fused_dequant_int3_kernel`, `_fused_dequant_matmul_kernel`, wavelet 8x + adaptive dequant, XS mega-kernel (`_xs_k0_occ`/`_xs_k1a_mode`/`_xs_k1b_words`/`_xs_k1c_hier`/`_xs_k2_gather` + compiled synthesis).
- `FusedDequantLinear` fuses `Dequant + GEMM + bias` in registers — zero intermediate VRAM.
- Tests skip if `!torch.cuda.is_available()`.

### 4.5 Prefetcher & Streamer
- `AsyncGPUPrefetcher` (`prefetcher.py`) — wraps any `DataLoader`, uses `torch.cuda.Stream` + `non_blocking=True` double buffer. `device.type in (cuda,hip)` check; `stream=None` on CPU.
- `ZeroCopyTensorStreamer` (`streamer.py`) — pinned CPU buffers + GPU ring buffers `out_bf16_0/1`, `np.copyto` into pinned, async `copy_(non_blocking=True)`, `wait_stream`, dequant into ring target. `low_vram=True` halves footprint (~128MB B128). `shard_idx`/`rank`/`world_size` for DDP. `close()` synchronizes and deletes buffers.

### 4.6 CLI & utils (`cli.py`, `utils.py`)
- CLI subcommands: `info`, `benchmark`, `cache-info`, `compress-demo`. Entry point `tensorcache` / `python -m tensorcache`.
- `utils.py`: `compress`/`decompress` (auto sym vs AMO), `estimate_compression`, `benchmark_tensor` (rich table, `include_quality=True` opt-in), `auto_select_mode` (quality requires opt-in), `help_text`.

## 5. Conventions & Pitfalls

- **Paths:** Linux dev machine. Use `pathlib.Path`, `os.path`, never hardcode `/`. Prefix handling: `str(prefix)+"_int8.bin"` etc.
- **Dtype discipline:** Features are `bfloat16` throughout; scales are BF16 on wire; raw pixels are `uint8`. Don't silently upcast to FP32 except in metrics.
- **Triton fallback:** Any change to `codec.py` dequant must keep both Triton and PyTorch fallback paths bit-identical. Test on CPU.
- **Memmap lifecycle:** Always provide `close()` and call it in tests. On Windows, open mmap prevents deletion.
- **No dynamic allocations in hot path:** Streamer/prefetcher are designed for zero allocation — avoid `torch.empty` inside loops without `out_buffer`.
- **rANS/tANS need numba (silent ~50x cliff):** `rans.py`/`tans.py` hot loops run under `@njit` when `HAS_NUMBA`, else fall back to pure-Python. This warns **loudly and once per process** from `warn_if_no_numba`, and from `make_xs_loader`. `numba` is in the `[fast]` and `[entropy]` extras.
- **XS entropy back-compat:** existing caches are never rewritten or invalidated — the format is per-cache, decided by `enc` row flags (+ `xs_entropy` marker) in `_pixel_meta.json`. New readers must keep the v1 branch (missing `enc` -> raw sections, ll4 offsets in int16 elements x2). New caches are unreadable by <0.6 readers (forward incompat, unavoidable). Keep `tests/test_rans.py` green: the Rust golden cross-check pins `encode`/`decode` byte-for-byte (`Texel/examples/rans_golden.rs` regenerates `tests/data/rans_goldens.bin`).
- **tANS goldens:** `tests/gen_tans_goldens.py` regenerates `tests/data/tans_goldens.bin` — once committed, goldens pin `encode()` output; any byte change fails `test_cross_version_stability` until deliberately regenerated AND reviewed.
- **Benchmark thresholds:** Rel RMSE `<1.0%` is the pass criterion (`tests/test_codec.py`, `tests/test_fused_ops.py`). Real sym G32 RMSE ~0.54%; AMO balanced G32 ~0.478%; AMO quality G16 ~0.3652%.
- **Security:** `download_test_dataset.py` uses `_safe_extract_tar` (manual member validation + `filter="data"` on 3.12+) — keep the guard; don't expand tarballs without it.
- **Formatting:** No enforced formatter; follow existing style (4-space indent, `from __future__ import annotations`, type hints).

## 6. Editing Guidelines for Agents

- Prefer `Read`/`Edit` over `bash` for files; use `bash` only for `pytest`, `python`, `git` ops.
- Verify with `pytest tests/test_codec.py -v` after codec/cache edits; run `python tests/test_codec.py` as quick sanity.
- If touching `fused_ops.py`, keep the `import triton` guard (`HAS_TRITON`) — do not hard-import.
- When adding new cache formats, bump `meta.json` fields, handle backward compat in `FeatureCacheDataset.__init__` / `PixelCacheDataset.__init__`, and validate via `_validate.py`.
- Don't commit `data/*.tgz`, `*_int8.bin`, `*_scales.bin`, `*_zp.bin`, `*_pixels.bin`, `*_meta.json`, `__pycache__`, `dist/`.
- Keep `src/tensorcache/__init__.py` exports in sync when adding public symbols (update `__all__`).
- `tcache` alias package re-exports `tensorcache` — keep it in sync automatically (it does `from tensorcache import *`).

## 7. Useful References

- API docs & benchmarks: `README.md`
- Public API: `src/tensorcache/__init__.py`
- Error metrics: `analyze_compression_error.py`, `experiment_blockwise_int8.py`
- Compression tables: `compression_benchmark_results.json` (pixel + feature)
- AMO-BQ results: `research/amo-quality-results.md`, `research/amo_quality_coco.json`
- tANS results: `research/block-tans-results.md`, `research/tans_dicts.json`
- Dataset prep: `download_test_dataset.py`

## 8. Agent Context — Current Environment

- **OS:** Linux (`/home/conorm/Desktop/Pipeline/TensorCache`), `bash` shell.
- **Python:** `>=3.9` (dev env ~3.14), `torch>=2.0` installed; Triton available on Linux but GPU may be absent — CPU fallback paths active.
- **Git:** repo initialized (`git log` works). No Makefile, no `opencode.json`.
- **CI:** GitHub Actions (`.github/workflows/ci.yml`) — pytest + build/twine check; release workflow publishes to PyPI on release.
