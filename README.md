# TensorCache ⚡

**Ultra-fast, high-fidelity block-wise INT8 feature & pixel cache engine for PyTorch.**

`tensorcache` eliminates two bottlenecks:
1. **Feature Cache Bloat:** `AMO-BQ` asymmetric MSE-optimal `G32` `1.09B` `1.83x` vs BF16 `0.47%` `rel RMSE` (`sym 1.06B 0.54%`) — near `G16` floor `0.39%`.
2. **JPEG/PNG CPU Decode:** Zero-copy `mmap` + `GPU` stream prefetch `>2,000 MB/s`, ring-buffer `6.8MB` `VRAM` `batch8 128x768`.

---

## 🚀 Key Features

* **AMO-BQ (Asymmetric MSE-Optimal, G32):** `min-max + zp uint8 + 48×` clipping search `[0.95,1.10]` — `0.47%` `hetero 0.54%` vs `sym 0.74%` (`-26%`), `per-token 1.73%` (`cache\dtype_comparison.json`). Presets `fast/balanced/accurate/max`.
* **Microsecond Dequant:** `Triton` `(q-zp)*scale -> BF16` `0.06ms 5.4M 337GB/s` (`codec.py:40`), `FusedDequantLinear` `0` intermediate `fused_ops.py:119`.
* **Minimal VRAM:** `q 5.22MB + scales 0.32MB + zp 0.16MB + out 10.45MB` `5.4M`; `Streamer` fixed `~43MB` `double-buffer` `out_buffer` reuse, `G64` halves `scales/zp`.
* **Cross-Platform:** `CUDA`/`ROCm` `Triton` else `PyTorch` fallback, `Windows` `mmap` safe `close()`.
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

# AMO-BQ presets: fast (16,0.95-1.05) 6.9ms 0.49%, balanced (32,0.95-1.05) 13ms 0.478% (default), accurate (48,0.95-1.10) 49ms 0.473%
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

### 2. Feature Cache to Disk (mmap)
```python
import tensorcache as tc
from torch.utils.data import DataLoader

# Write (amo_bq, G32 default balanced, G64 for minimal VRAM 1.046B 0.55%)
writer = tc.FeatureCacheWriter("./cache/dinov3", 10000,446,768, group_size=32, amo_bq=True, amo_mode="balanced")
writer.append(x) # [seq,dim] or [B,seq,dim]
writer.close()   # writes _int8.bin (uint8) _scales.bin _zp.bin _meta.json

# Load
ds = tc.FeatureCacheDataset("./cache/dinov3") # -> (q uint8, s BF16, zp uint8)
ds = tc.FeatureCacheDataset("./cache/dinov3", auto_dequant_device="cuda") # -> BF16 directly
for q,s,zp in tc.AsyncGPUPrefetcher(DataLoader(ds,batch_size=256,pin_memory=True), device="cuda"):
    batch = tc.dequantize_int8_amo_bq(q,s,zp, shape, group_size=32)

# Minimal VRAM streamer (fixed ~43MB, zero alloc)
streamer = tc.ZeroCopyTensorStreamer("./cache/dinov3", batch_size=32, device="cuda")
for batch in streamer: # BF16 [B,seq,dim] from ring buffer out_bf16_0/1
    train(batch)
streamer.close()

# Fused head (no BF16 intermediate)
from tensorcache import FusedDequantLinear
head = FusedDequantLinear(768, num_classes, group_size=32).cuda()
logits = head(q, s) # dequant+GEMM in regs
```

### 3. CLI
```bash
python -m tensorcache info
python -m tensorcache benchmark --shape 16,446,768 --device cuda
python -m tensorcache cache-info --prefix ./cache/dinov3
python -m tensorcache compress-demo --shape 4,197,768 --mode balanced
tensorcache --help
```

---

## 📊 Benchmark (DINOv3 ViT-Base, 5.4M randn + RSNA hetero)

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
