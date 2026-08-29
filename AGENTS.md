# AGENTS.md — TensorCache

> Guidance for AI coding agents working in this repo. Keep this file concise, factual, and up-to-date.

## 1. Project Overview

**TensorCache** (`tensorcache`, v0.1.0) — Ultra-fast, high-fidelity block-wise INT8 feature & pixel cache engine for PyTorch.

Two bottlenecks solved:
1. **Feature Cache Bloat:** Block-wise microscaled INT8 (`G=32`, BF16 scales) — `1.88×` compression vs BF16, `0.54%` rel RMSE (see `compression_benchmark_results.json:190-201`, `README.md:77-81`).
2. **JPEG/PNG CPU Decode:** Zero-copy `np.memmap` + GPU stream prefetching — `>2000 MB/s` throughput (`README.md:6-7`, `src/tensorcache/pixel_cache.py:95-137`, `src/tensorcache/feature_cache.py:112-210`).

Installation: `pip install tensorcache` or `pip install -e .` from repo root. Python `>=3.9`, `torch>=2.0`, `numpy>=1.20` (`pyproject.toml:16-18`). Optional `fast` extras: `triton`, `zstandard`, `blosc2`, `safetensors` (`pyproject.toml:21-27`) — note `triton` is skipped on Windows (`platform_system != 'Windows'`).

## 2. Repo Structure

```
/
├── src/tensorcache/
│   ├── __init__.py        # public API re-exports
│   ├── codec.py           # BlockwiseInt8Codec, quantize/dequantize (core)
│   ├── fused_ops.py       # Triton fused kernels (requires CUDA/ROCm + Triton)
│   ├── feature_cache.py   # FeatureCacheWriter / FeatureCacheDataset (mmap .bin + .json)
│   ├── pixel_cache.py     # PixelCacheWriter / PixelCacheDataset (raw uint8 mmap)
│   ├── prefetcher.py      # AsyncGPUPrefetcher (double-buffered CUDA stream)
│   └── streamer.py        # ZeroCopyTensorStreamer (pinned + ring-buffered)
├── tests/
│   ├── test_codec.py      # codec + feature/pixel disk I/O tests
│   └── test_fused_ops.py  # GPU-only, skipped if !torch.cuda.is_available()
├── benchmarks/
│   ├── bench_memory_opt.py
│   ├── benchmark_dual_gpu.py
│   ├── speed_test_rigorous.py
│   └── tune_dequant.py
├── analyze_compression_error.py  # full pixel+feature benchmark (tabulate)
├── experiment_blockwise_int8.py  # G-sweep, symmetric/asymmetric, bitwidth
├── download_test_dataset.py      # Imagenette-320 fetcher -> data/test_images
├── pyproject.toml
└── README.md
```

Data dirs: `data/` (contains `imagenette2-320.tgz`, `flowers-102/102flowers.tgz` — do not commit large binaries). Cache files are runtime-generated (`*_int8.bin`, `*_scales.bin`, `*_pixels.bin`, `*_meta.json`, `*_pixel_meta.json`) — gitignore them.

## 3. Build / Run / Test Commands

```bash
# install (dev)
pip install -e ".[fast,dev]"

# run all tests (fused tests auto-skip on CPU/Windows)
pytest -v
pytest tests/test_codec.py -v
pytest tests/test_fused_ops.py -v  # requires CUDA + Triton

# single-file direct run (no pytest)
python tests/test_codec.py

# benchmarks / analysis (require data images)
python analyze_compression_error.py          # needs Plant dataset or data/test_images fallback
python experiment_blockwise_int8.py          # needs CUDA + timm + Plant data
python download_test_dataset.py              # fetches Imagenette-320 (~326 MB) to data/test_images
python benchmarks/speed_test_rigorous.py
python benchmarks/tune_dequant.py

# Windows note: Triton unavailable -> codec falls back to vectorized PyTorch path (codec.py:167-177)
```

No `Makefile`, no `opencode.json` yet, no CI config in repo. Use `pytest>=7.0` (`pyproject.toml:29`).

## 4. Core Architecture — Read Before Editing

### 4.1 Codec (`src/tensorcache/codec.py:40-202`)
- `quantize_int8_g32(x, group_size=32)` — flatten, pad to multiple of `G`, `view(-1,G)`, `amax`, `scales = max/127 -> BF16`, `round(x/scale) clamp [-128,127] -> int8`. Returns `(q_int8 (flat 1D), scales (1D BF16), orig_shape)` (`codec.py:40-75`).
- `quantize_int8_adaptive` — AdaRound-style parallel candidate search over `linspace(0.90,1.05,31)` per block, picks min L2 error (`codec.py:78-126`). ~10% lower error, heavier.
- `dequantize_int8_g32` — Triton path if `HAS_TRITON and device.type in (cuda,hip)` using `_triton_dequant_kernel` (`codec.py:22-37`, `150-165`) else padded `view(-1,G) * scales` + copy into `out_buffer` (`codec.py:167-177`). `out_buffer` avoids allocator overhead — preserve this API.
- `BlockwiseInt8Codec` wrapper (`codec.py:180-202`) — `group_size` + `adaptive` flag. Keep `group_size=32` as default; benchmarks assume it.

**Invariants:**
- Scales stored as BF16 (`torch.bfloat16`) on disk as `uint16` bitcast: `scales.view(torch.int16).cpu().numpy().view(np.uint16)` (`feature_cache.py:80`) and restored via `view(np.int16).copy()).view(torch.bfloat16)` (`feature_cache.py:153`). Do not change dtype without migrating file format.
- Padding is trimmed on return (`flatten()[:numel]`). Keep.
- Storage cost: `1 + 2/32 = 1.0625 B/elem` (`analyze_compression_error.py:515`, `559`).

### 4.2 Feature Cache (`src/tensorcache/feature_cache.py:21-211`)
- **Writer:** pre-allocates `np.memmap` `w+` with shape `(num_samples, seq_len, dim)` int8 and `(num_samples, scales_per_sample)` uint16 (`feature_cache.py:53-60`). `append` handles both `[seq_len,dim]` and `[B,seq_len,dim]` (`feature_cache.py:63-81`). `close()` flushes, closes `_mmap`, writes `_meta.json` with `num_samples = current_idx` (`feature_cache.py:83-109`). Must call `close()` — Windows holds file lock otherwise.
- **Dataset:** read-only mmap `mode="r"` (`feature_cache.py:134-142`). `__getitem__` copies via `arr.copy()` before `torch.from_numpy` to avoid memmap lifetime issues, then optional `auto_dequant_device` fused dequant (`feature_cache.py:147-163`). `iter_batches` does C-level batch slice `mmap[batch_idx]` → `torch.from_numpy(...).to(device)` (`feature_cache.py:165-197`).
- `close()` releases `_mmap` handles (`feature_cache.py:199-210`) — critical on Windows.

### 4.3 Pixel Cache (`src/tensorcache/pixel_cache.py:25-137`)
- Raw `uint8` mmap `(N,H,W,C)` (`pixel_cache.py:48-51`). `append_image` accepts `np.ndarray | PIL.Image | torch.Tensor | str|Path` and resizes to `(width,height)` via `BILINEAR` (`pixel_cache.py:54-74`). `PixelCacheDataset.__getitem__` returns `torch.uint8 [H,W,C]` with `arr.copy()` (`pixel_cache.py:122-129`).

### 4.4 Fused Ops (`src/tensorcache/fused_ops.py:1-233`)
- **Requires Triton + CUDA/ROCm** — hard import `import triton` at top (`fused_ops.py:17-18`) will fail on Windows/CPU. Guard imports or make optional if editing.
- Three kernels: `_fused_quant_kernel` (`fused_ops.py:24-52`), `_fused_dequant_kernel` (`fused_ops.py:82-96`), `_fused_dequant_matmul_kernel` (`fused_ops.py:119-184`). `FusedDequantLinear` (`fused_ops.py:187-233`) fuses `Dequant + GEMM + bias` in registers — zero intermediate VRAM.
- Tests skip if `!torch.cuda.is_available()` (`tests/test_fused_ops.py:15,32`).

### 4.5 Prefetcher & Streamer
- `AsyncGPUPrefetcher` (`src/tensorcache/prefetcher.py:11-67`) — wraps any `DataLoader`, uses `torch.cuda.Stream` + `non_blocking=True` double buffer. `device.type in (cuda,hip)` check; `stream=None` on CPU.
- `ZeroCopyTensorStreamer` (`src/tensorcache/streamer.py:19-163`) — pinnned CPU buffers `pin_memory=True` + GPU ring buffers `out_bf16_0/1` (`streamer.py:62-77`), `np.copyto` into pinned, async `copy_(non_blocking=True)`, `wait_stream`, dequant into ring target (`streamer.py:96-117`). Fixed ~45 MB footprint. `close()` synchronizes and deletes buffers (`streamer.py:129-163`).

## 5. Conventions & Pitfalls

- **Paths:** Windows dev machine (`win32`, `C:\Users\armor\Desktop\AI pipeline\...`). Use `pathlib.Path`, `os.path`, never hardcode `/`. Prefix handling: `str(prefix)+"_int8.bin"` etc. (`feature_cache.py:48-50`).
- **Dtype discipline:** Features are `bfloat16` throughout; scales are BF16 on wire; raw pixels are `uint8`. Don't silently upcast to FP32 except in metrics (`analyze_compression_error.py:170-205`).
- **Triton fallback:** Any change to `codec.py` dequant must keep both Triton and PyTorch fallback paths bit-identical. Test on CPU.
- **Memmap lifecycle:** Always provide `close()` and call it in tests (`tests/test_codec.py:78-80,102-104`). On Windows, open mmap prevents deletion.
- **No dynamic allocations in hot path:** Streamer/prefetcher are designed for zero allocation — avoid `torch.empty` inside loops without `out_buffer`.
- **Benchmark thresholds:** Rel RMSE `<1.0%` is the pass criterion (`tests/test_codec.py:36,51`, `tests/test_fused_ops.py:28`). Real Block-32 RMSE ~0.54% vs Naive FP8 ~2.6-5.2% (`compression_benchmark_results.json:126-148`, `190-201`).
- **Security:** `tar.extractall` in `download_test_dataset.py:28` — keep as is for now but don't expand without validation if hardening.
- **Formatting:** No enforced formatter; follow existing style (4-space indent, `from __future__ import annotations`, type hints).

## 6. Editing Guidelines for Agents

- Prefer `Read`/`Edit` over `bash` for files; use `bash` only for `pytest`, `python`, `git` ops.
- Verify with `pytest tests/test_codec.py -v` after codec/cache edits; run `python tests/test_codec.py` as quick sanity.
- If touching `fused_ops.py`, guard `import triton` with try/except like `codec.py:13-19` or ensure CI skips gracefully.
- When adding new cache formats, bump `meta.json` fields and handle backward compat in `FeatureCacheDataset.__init__`.
- Don't commit `data/*.tgz`, `*_int8.bin`, `*_scales.bin`, `*_pixels.bin`, `__pycache__`.
- Keep `src/tensorcache/__init__.py:1-42` exports in sync when adding public symbols (update `__all__`).

## 7. Useful References

- API docs & benchmarks: `README.md:1-86`
- Public API: `src/tensorcache/__init__.py:5-42`
- Error metrics: `analyze_compression_error.py:170-205`, `experiment_blockwise_int8.py:20-40`
- Compression tables: `compression_benchmark_results.json:2-216` (pixel + feature)
- Dataset prep: `download_test_dataset.py:7-44`

## 8. Agent Context — Current Environment

- **OS:** Windows (`win32`), PowerShell 7+ (`pwsh`), `workdir` param preferred over `cd`.
- **Python:** `>=3.9`, `torch>=2.0` installed; Triton unavailable on Windows (fallback path active).
- **No git repo** at workspace root (`Is directory a git repo: no`) — `git` commands will fail until `git init`.
- **Today:** 2026-08-29 (use for searches).
