# TensorCache ⚡ [![PyPI](https://img.shields.io/pypi/v/tcache?label=tcache)](https://pypi.org/project/tcache/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Ultra-fast, high-fidelity block-wise INT8 feature & pixel cache engine for PyTorch.**

`pip install tcache` → `import tensorcache` or `import tcache` — [PyPI](https://pypi.org/project/tcache/).

**New in 0.7.3: opt-in AMO-BQ `quality`.** Search the stored BF16 scales and
score the actual decoded values; choose G16 for higher fidelity or G32 for the
same storage cost as balanced. **G16 quality delivers 34.7% lower feature RMSE
for 8.57% more cache bytes** on COCO/DINOv3 versus G32 balanced. See the
[COCO feature comparison](#amo-bq-feature-fidelity-on-coco-validation).
Install or upgrade with `pip install --upgrade tcache`.

`tensorcache` eliminates two bottlenecks:
1. **Feature Cache Bloat:** AMO-BQ G32 stores `1.09375 B/value` (`1.83x` compression vs BF16). Opt-in `quality` G16 reaches **`0.3652%` relative RMSE** on COCO/DINOv3, versus `0.5594%` for G32 balanced, at `1.1875 B/value`.
2. **JPEG/PNG CPU Decode:** Zero-copy `mmap` + `GPU` stream prefetch `>2,000 MB/s`, ring-buffer `6.8MB` `VRAM` `batch8 128x768`.
3. **Training Throughput (v0.2.2):** `iter_batches` `27k samp/s` `17.7GB/s` `B128 446x768` `~8.3x` vs `v0.2.0` `3.1k`, `Streamer` `21k` `5.1ms` `double-buffered` `pinned + async H2D` (`low_vram` `128MB` `B128`), `sharded 8x` for `H100 DDP`.
4. **Pixel Cache Tunable Wavelet XS (v0.3.0):** `CDF 5/3` `4-lvl` `RCT` `4:2:0` `batched [3,H,W]` `GPU Triton` `out_buffer`, `adaptive RDO 4b` `G32` `16-lvl codebook` `TCQ-lite` `subband gains` `sparse bitstream` `zero-block skip` `hier nibble masks` `tunable` `44dB 3.2x` <-> `33dB 13.6x` (`balanced 37dB 7.8x`, `static 34dB 11.5x` measured COCO val 336) `presets ultra/high/balanced/compress/ultra_comp` `q_scale 1-8 lamb 1-50`.

---

## 🚀 Key Features

* **AMO-BQ (Asymmetric MSE-Optimal):** UINT8 values + BF16 scales + UINT8 zero points. Presets `fast/balanced/accurate/max` plus opt-in **`quality`**, which searches actual BF16 scales and minimizes decoded BF16 error. Group size separately controls storage: G32 is `1.09375 B/value` (`1.83x` vs BF16); G16 is `1.1875 B/value` (`1.68x`, **8.57% more cache bytes**).
* **Microsecond Dequant (v0.2.2 opt):** `Triton` `shift >>5` vs `div`, `autotune BLOCK 512/1024/2048` `stages 2/3` `163 GB/s` `5.4M` (`codec.py:22` `0.19ms` `10M` `26x` for `1024`), `FusedDequantLinear` `0` intermediate `fused_ops.py:119`. `NaN/Inf` isolated `0` poison `1e-8` scale.
* **Training-Optimized Dataloader:** `iter_batches` `27k samp/s` `17.7GB/s` `B128` `4.5ms` (`8.3x` vs `3.1k`), `Streamer` `21.6k` `5.1ms` `double-buffered` `pinned` `async H2D` overlap `dequant`. `low_vram=True` `128MB` `B128` (`53 MB` `B32`) vs `256MB`.
* **Big Data Sharded:** `Writer(num_shards=8)` `->` `feat_shard{i}_*.bin` `+` `feat_shards.json`, `Dataset/Streamer(rank, world_size)` `DDP` `H100 8x` `~5.8k` random `27k` contiguous. Single shard `num_shards=1` unchanged `100%` compat.
* **Minimal VRAM:** `q 5.22MB + scales 0.32MB + zp 0.16MB + out 10.45MB` `5.4M`; `G64` halves `scales/zp`.
* **Cross-Platform:** `CUDA`/`ROCm` `Triton` else `PyTorch` fallback, `Windows` `mmap` safe `close()`.
* **Tunable Wavelet XS Pixel:** `quantize_pixel_wavelet_adaptive(mode="balanced")` `lamb` `D+lamb*R` `G32 4b` `codebook 16` `TCQ-lite` `subband gains` `4:2:0` `sparse + zero-block skip + hier masks` `37dB 7.8x` (`44dB 3.2x` <-> `33dB 13.6x`, COCO val 336) `vs static 34dB 11.5x`.
* **XS Image Pipeline (new):** `tc.cache_images(dir, prefix)` one-liner build → `tc.make_xs_loader(prefix, batch_size=256)` yields `uint8 [B,H,W,3]` on `CUDA` at `~10k img/s` sustained `@336` (`laptop 4050`, `file->GPU`) — `10.3x` on `5000x COCO` with rANS entropy-coded sections (0.6 default; `fetch ~4.4k img/s/core`, so use `workers>=4`), `GPU` mega-kernel batch decode (`K0/K1a/K1b/K1c/K2` + compiled synthesis, bit-exact), `spawn-safe` workers, `CPU` fallback bit-exact.
* **Mono Grayscale Caches (v0.4.2):** `PixelCacheWriter(..., channels=1, quant=raw/int4/int3)` stores 1 sample/pixel — `3x` smaller (`110KB` vs `331KB` raw `336²`), grayscale-only ingest (color rejected, no silent conversion), `ds[i] -> uint8 [H,W,1]`. Mono `G=8` beats `RGB G=32` on both axes: `int4` RMSE `4.19` `83KB` vs `4.67` `186KB` — pass `group_size=8` for mono.
* **CLI + Python one-liners:** `tc.compress` / `tc.benchmark_tensor` / `tensorcache benchmark`.

---

## 📦 Installation

```bash
pip install tcache                # PyPI (import tensorcache or tcache)
pip install -e .          # dev
pip install -e ".[fast]"  # triton+zstd+blosc2 (Linux)
```

---

## ⚡ Quick Start

### 1. In-Memory (one-liners)
```python
import torch, tensorcache as tc

x = torch.randn(16,446,768, dtype=torch.bfloat16, device="cuda")

# AMO-BQ defaults: balanced, G32. Higher-fidelity quality mode is opt-in.
q,s,zp,shape = tc.compress(x, mode="balanced")  # or "fast"/"accurate"/"max"/"sym"/"adaptive"
rec = tc.decompress(q,s,shape,zp)               # <0.1ms BF16
tc.benchmark_tensor(x)                          # rich table
tc.estimate_compression(x.shape, group_size=32) # 1.09375 B 1.83x
tc.auto_select_mode(x, target_rmse=0.5)         # -> "balanced"
tc.help()                                       # python help

# Codec object
codec = tc.BlockwiseInt8Codec(group_size=32, amo_bq=True, amo_mode="balanced")
print(codec) # G=32, amo_bq=balanced 1.0938B 1:1.83x
```

### 1b. Opt-in higher-fidelity AMO-BQ

`balanced` with G32 remains the default. Choose `quality` explicitly and use
G16 for higher fidelity, or G32 for better accuracy at the default byte rate.

```python
q, scales, zp, shape = tc.compress(x, mode="quality", group_size=16)
rec = tc.decompress(q, scales, shape, zp, group_size=16)

codec = tc.BlockwiseInt8Codec(amo_bq=True, amo_mode="quality", group_size=16)
tc.benchmark_tensor(x, group_size=16, include_quality=True)
```

Pass the same `group_size` to the low-level decoder. Disk caches store it and
the selected mode in their metadata, so datasets and streamers read the right
settings automatically. The UINT8/BF16/UINT8 layout and decode operation stay
the same; quality spends more work when creating the cache.

```python
writer = tc.FeatureCacheWriter(
    "./cache/dinov3_quality", num_samples=10000, seq_len=446, dim=768,
    amo_bq=True, amo_mode="quality", group_size=16,
)
writer.append(x)
writer.close()

dataset = tc.FeatureCacheDataset("./cache/dinov3_quality")
# Consume dataset, then release mmap handles:
dataset.close()
```

CLI: `python -m tensorcache compress-demo --mode quality --group-size 16`.
Add `--quality` to `tensorcache benchmark --group-size 16` to include the
quality search; automatic mode selection also requires `include_quality=True`.

### 2. Feature Cache to Disk (mmap) - Training Optimized
```python
import tensorcache as tc
from torch.utils.data import DataLoader

# Write (amo_bq, G32 default balanced, G64 for minimal VRAM 1.046B 0.55%)
# Single shard (default, unchanged)
writer = tc.FeatureCacheWriter("./cache/dinov3", 10000,446,768, group_size=32, amo_bq=True, amo_mode="balanced")
# Or sharded for H100 / >1M samples (8 shards ~1.25k each)
writer = tc.FeatureCacheWriter("./cache/dinov3", 10000,446,768, group_size=32, amo_bq=True, amo_mode="balanced", num_shards=8)
writer.append(x) # [seq,dim] or [B,seq,dim] (batched GPU 64/chunk, CPU per-sample)
writer.close()   # single: _int8.bin ... | sharded: _shard0_int8.bin ... + _shards.json

# Load - auto-detects sharded
ds = tc.FeatureCacheDataset("./cache/dinov3") # -> (q uint8, s BF16, zp uint8) all shards
# DDP per-rank (H100 8x)
ds = tc.FeatureCacheDataset("./cache/dinov3", rank=rank, world_size=8) # only shard rank
# or explicit
ds = tc.FeatureCacheDataset("./cache/dinov3", shard_idx=0)
ds = tc.FeatureCacheDataset("./cache/dinov3", auto_dequant_device="cuda") # -> BF16 directly

# Old path still works but slower (~5k samp/s)
for q,s,zp in tc.AsyncGPUPrefetcher(DataLoader(ds,batch_size=256,pin_memory=True), device="cuda"):
    batch = tc.dequantize_int8_amo_bq(q,s,zp, shape, group_size=32)

# Training fastest (27k samp/s 17.7GB/s B128, 8.3x vs v0.2.0)
for batch in ds.iter_batches(batch_size=128, shuffle=False, device="cuda"): # contiguous 27k, shuffled ~5.8k
    train(batch)

# Minimal VRAM streamer (double-buffered 64MB B32 / 256MB B128, low_vram 32MB/128MB)
streamer = tc.ZeroCopyTensorStreamer("./cache/dinov3", batch_size=128, device="cuda", shuffle=True)
streamer = tc.ZeroCopyTensorStreamer("./cache/dinov3", batch_size=128, device="cuda", low_vram=True) # 128MB B128
streamer = tc.ZeroCopyTensorStreamer("./cache/dinov3", batch_size=128, device="cuda", shard_idx=0) # per-rank
for batch in streamer: # BF16 [B,seq,dim] from ring buffer out_bf16_0/1
    train(batch)
streamer.close()

# Fused head (no BF16 intermediate)
from tensorcache import FusedDequantLinear
head = FusedDequantLinear(768, num_classes, group_size=32).cuda()
logits = head(q, s) # dequant+GEMM in regs
```

### 2b. XS Wavelet Image Cache (TensorCache's own wavelet codec, GPU-decoded)

> **Notation note:** "XS" is TensorCache's internal codec name for its original
> wavelet image codec (CDF 5/3 lifting + RCT + adaptive RDO). It is **not** the
> JPEG XS standard (ISO/IEC 21122), is not affiliated with or endorsed by any
> JPEG XS rightsholder, does not implement that standard, and claims no
> compliance with it.

```python
import tensorcache as tc
# One-liner build: dir of JPEGs/PNGs -> mmap cache (balanced default)
info = tc.cache_images("data/coco_val", "./cache/coco336")
# -> {"num_samples": 5000, "ratio_vs_raw": 10.28, "ms_per_img": 38, ...}

# Training loop: page-cache slices in, GPU batches out (workers optional)
for imgs in tc.make_xs_loader("./cache/coco336", batch_size=256,
                              device="cuda", num_workers=8):
    train(imgs)  # uint8 [B,H,W,3] on CUDA
```
Fidelity presets (`xs_mode`, measured `16x COCO val 336`, `1x` = raw `RGB`, ratios include the default rANS entropy layer — lossless, so `PSNR`/`SSIM` are unchanged):

| preset | PSNR | SSIM | ratio | enhances |
|---|:---:|:---:|:---:|---|
| ultra (`4:4:4`) | `44.4dB` | `0.993` | `4.5x` | transparent — invisible even flipped |
| high (`4:4:4`) | `39.8dB` | — | `8.2x` | high quality |
| balanced (`4:2:0`, default) | `36.8dB` | `0.97` | `10.7x` | invisible at normal viewing |
| compress (`4:2:0`) | `35.3dB` | — | `13.6x` | high compression |
| ultra_comp (`4:2:0`) | `33.0dB` | `0.92` | `19.3x` | visible softening in busy texture |

Throughput (`balanced`, `336²`, `laptop RTX 4050`, sustained medians; measured on the pre-0.6 dense-storage layout):

| path | img/s | notes |
|---|:---:|---|
| `make_xs_loader` `BS32` (`workers=0`) | `~10k` | `file->GPU`, `60s` windows, `1965MHz` |
| batch API `BS32` | `~12k` | `16.5k` peak-boost bursts |
| batch API `BS256` | `~10.4k` | saturated plateau (`20 SMs` full at `BS32`) |
| single image | `~1.2k` | `0.8ms`, `out_buffer` zero-alloc |
| `PIL JPEG` `8 workers` (status quo) | `~0.8-1k` | what training does today |

Since 0.6 the XS sections are rANS entropy-coded by default (`xs_entropy=True`): caches are `~1.5x` smaller, at the cost of a CPU front-end that decodes at `~4.4k img/s/core` (`~0.2-0.3ms/sample`). Use `num_workers>=4` in `make_xs_loader` so fetch hides behind GPU decode; `num_workers=0` becomes fetch-bound. Pass `xs_entropy=False` for the old layout/throughput (e.g. tiny caches where CPU time matters more than bytes).

**0.7 adds `xs_entropy="tans"`** — a dictionary block-tANS codec that decodes on the GPU:

```python
tc.cache_images("data/coco_val", "./cache/coco336", xs_entropy="tans")  # learns a dict from 256 imgs
for imgs in tc.make_xs_loader("./cache/coco336", batch_size=256, num_workers=0):  # main-process GPU decode
    ...
```

It is `+5-6%` smaller than rANS at cache scale (`10.56x -> 11.20x` on COCO-336/256 imgs) and the entropy decode runs on-device at `~50k img/s`, adding `~+15%` to the wavelet decode at `B=128` (`~9.4k img/s e2e`). Because the CPU path would be *slower* than rANS, tANS caches require a CUDA/ROCm device and `num_workers=0` (the main process does entropy + wavelet decode); the reader is validated bit-identical against the rANS cache. The build stores the dictionary once in `_pixel_meta.json` (`~25KB`), so it amortizes (`+6.1%` at 256 images, more at 5000). `xs_entropy` also accepts a prebuilt `tans_dict` (path or `{kind: tables}`) to share one dictionary across caches.

`ViT-B` burns `~1-1.5k img/s/GPU` — the cache feeds `~10x` headroom on a laptop card while moving `33KB/img` (`~330MB/s H2D`, `PCIe` idle). Bigger GPUs just want bigger batches (`RTX PRO` saturates ~`BS256+`); `GB10` fits whole datasets in `128GB` unified memory. `CPU` fallback is bit-exact (`Windows`-safe, no `Triton` needed).

Chroma is choosable: `cache_images(..., chroma420=False)` forces `4:4:4` (default auto: `ultra`/`high` → `4:4:4`, below → `4:2:0`), `chroma420=True` forces `4:2:0`. Same flag on `PixelCacheWriter`. Forcing `4:4:4` on `balanced` = `+1.4dB` for `+16%` bytes (`38.7dB` `96x` vs `37.3dB` `112x`, COCO `336`).

### 2c. Mono Grayscale Pixel Caches (BW / grayscale JPEG-PNG / medical)
```python
import tensorcache as tc
# 3x smaller than replicating gray to RGB; color input is rejected (no silent convert)
writer = tc.PixelCacheWriter("./cache/mri336", 10000, 336, 336, channels=1,
                             quant="int4", group_size=8)  # G=8 recommended for mono
writer.append_image(gray)          # [H,W], [H,W,1], PIL "L"/"I;16", or path
writer.close()

ds = tc.PixelCacheDataset("./cache/mri336")
img = ds[0]  # uint8 [H,W,1]; ViT wants 3ch? t.expand(H,W,3) is a free view
```
Measured (`1x COCO` img `336²`; RMSE of `0-255` values, bytes/img incl. scales):

| config | RMSE | B/img | vs RGB G32 |
|---|:---:|---:|:---:|
| raw mono | `0` | `110K` | `3.0x` smaller |
| int4 mono `G=32` | `5.88` | `62K` | `3.0x` smaller |
| **int4 mono `G=8`** | **`4.19`** | `83K` | `2.2x` smaller **+ better fidelity** |
| int4 mono `G=4` | `3.31` | `110K` | `1.7x` smaller, max fidelity |
| int4 RGB `G=32` (replicated) | `4.67` | `186K` | — |
| int3 mono `G=8` | `8.34` | `69K` | `2.1x` smaller + better |
| int3 RGB `G=32` (replicated) | `9.65` | `145K` | — |

Recommendation: mono caches should use `group_size=8` (or `4` for max fidelity) — blockwise scales adapt per `G` spatial pixels; mono `G=32` spans `3x` more pixels per block than RGB, so a smaller `G` recovers (and beats) RGB fidelity while staying far smaller. `quant="xs"` stays RGB-only (`~1%` overhead for true gray — chroma compresses to occupancy bits).

### 3. CLI
```bash
python -m tensorcache info
python -m tensorcache benchmark --shape 16,446,768 --device cuda
python -m tensorcache cache-info --prefix ./cache/dinov3
python -m tensorcache compress-demo --shape 4,197,768 --mode balanced
tensorcache --help
```

---

## 📊 Feature Benchmarks

### AMO-BQ feature fidelity on COCO validation

**All 5,000 COCO val images, all 1,712,640,000 DINOv3 ViT-B/16 feature
values**, 336×336 inputs, BF16 features, RTX 4050 Laptop GPU, batch size 16.
Relative RMSE compares decoded BF16 features with the original BF16 features.

| Mode | Group size | B/value | Compression vs BF16 | Relative RMSE | RMSE reduction vs default | Encode GPU ms | Decode GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced` (default) | 32 | 1.09375 | 1.83× | 0.5594% | — | 0.758 | 0.060 |
| **`quality` (opt-in)** | **32** | **1.09375** | **1.83×** | **0.5202%** | **7.0%** | 3.447 | 0.062 |
| `balanced` | 16 | 1.18750 | 1.68× | 0.4228% | 24.4% | 0.844 | 0.060 |
| **`quality` (opt-in)** | **16** | **1.18750** | **1.68×** | **0.3652%** | **34.7%** | 3.486 | 0.062 |

**Pick `quality`, G16 for higher fidelity:** about one-third less feature RMSE
than the default, while retaining 1.68× compression versus BF16. Pick
`quality`, G32 for better accuracy at the default storage size. The trade-off
is more encoding work when building the cache; the decode operation is the
same. Existing presets and G32 defaults retain their behavior.

Times are median GPU latency per batch, with three timed repetitions per
batch after warmup. Image loading, feature extraction, input transfer and
compilation are excluded; decode reuses an output buffer. These measure the
codec, not end-to-end training or file-to-GPU throughput. BF16 scale and
zero-point overhead is included in B/value. G16 uses **8.57% more total cache
bytes** than G32. Results depend on the feature distribution.

```bash
python benchmarks/bench_amo_quality.py --limit 5000 --repeats 3
```

[Method and validation](https://github.com/Thenukegun10x/TensorCache/blob/v0.7.3/research/amo-quality-results.md) ·
[Full numeric results](https://github.com/Thenukegun10x/TensorCache/blob/v0.7.3/research/amo_quality_coco.json) ·
[Opt-in examples](#1b-opt-in-higher-fidelity-amo-bq)

### Historical comparison (5.4M randn + RSNA hetero)

The older measurements below use different inputs and timing conditions;
compare rows within a table.

| Format | B/elem | vs BF16 | rel RMSE |Outlier 0.1%| Dequant |
|---|:---:|---|:---:|:---:|---|
| Raw BF16 |2.00|1.00x|0.167%|0.119%|—|
| Naive FP8 E5M2 |1.00|2.00x|5.24%|6.81%|—|
| Naive FP8 E4M3 |1.00|2.00x|2.63%|3.19%|—|
| MXFP8 G32 |1.09|1.83x|2.38%|—|—|
| **Sym G32** |**1.06**|**1.88x**|**0.540%**|**0.11%**|**0.036ms 435GB/s**|
| **AMO-BQ fast G32** |1.093|1.83x|0.490%|0.15%|0.06ms 337GB/s|
| **AMO-BQ balanced G32** |**1.093**|**1.83x**|**0.478%**|0.15%|13ms quant|
| **AMO-BQ accurate G32** |1.093|1.83x|0.473%|—|49ms quant|
| **AMO-BQ G16** |1.187|1.68x|**0.395%**|—|13ms|
| **Sym G64** |1.046|1.91x|0.55%|—|—|

---

## 📜 License
Apache 2.0 — see `LICENSE`.
