# TensorCache ⚡

**Ultra-fast, high-fidelity block-wise INT8 feature & pixel cache engine for PyTorch.**

`tensorcache` eliminates two of the most common bottlenecks in Computer Vision training pipelines:
1. **Feature Cache Bloat:** Compresses intermediate ViT/CNN floating-point feature caches by **$1.88\times$** with negligible **$0.54\%$ reconstruction error** (matching raw BF16 outlier fidelity).
2. **JPEG/PNG CPU Decode Bottleneck:** Eliminates slow CPU image decoding via zero-copy memory mapping and GPU stream prefetching, achieving **$>2,000\text{ MB/s}$ throughput**.

---

## 🚀 Key Features

* **Block-wise Microscaled INT8 ($G=32$):** Divides feature maps into 32-element blocks with local 16-bit scales. Drops relative RMSE error from $5.25\%$ (naive FP8) down to **$0.54\%$**.
* **Microsecond GPU Dequantization:** Dequantizes directly in GPU registers at **$>800\text{ GB/s}$** memory bandwidth saturation.
* **100% Cross-Platform & Cross-Vendor:** Works natively across NVIDIA (CUDA) and AMD (ROCm / RDNA) architectures with zero compilation required.
* **Zero-Copy Memory-Mapped Datasets:** Drop-in PyTorch `Dataset` and `DataLoader` support with asynchronous double-buffering prefetchers.

---

## 📦 Installation

```bash
pip install tensorcache
```

---

## ⚡ Quick Start

### 1. In-Memory Tensor Compression
```python
import torch
import tensorcache as tc

codec = tc.BlockwiseInt8Codec(group_size=32)

# Compress BF16 tensor
features_bf16 = torch.randn(128, 446, 768, dtype=torch.bfloat16, device="cuda")
q_int8, scales, shape = codec.quantize(features_bf16)

# Dequantize in VRAM (< 0.1 ms)
reconstructed_bf16 = codec.dequantize(q_int8, scales, shape)
```

### 2. Fast Feature Caching to Disk
```python
import tensorcache as tc
from torch.utils.data import DataLoader

# Write features to memory-mapped cache
writer = tc.FeatureCacheWriter(
    output_prefix="./cache/dinov3_features",
    num_samples=10000,
    seq_len=446,
    dim=768,
    group_size=32
)
writer.append(features_bf16)
writer.close()

# Load during training with Async Prefetching
dataset = tc.FeatureCacheDataset("./cache/dinov3_features")
loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
prefetcher = tc.AsyncGPUPrefetcher(loader, device="cuda")

for q_int8, scales in prefetcher:
    # Fused GPU dequantization
    batch_bf16 = tc.dequantize_int8_g32(q_int8, scales, orig_shape=q_int8.shape)
    logits = model_head(batch_bf16)
```

---

## 📊 Benchmark Results (DINOv3 ViT-Base on Real Dataset)

| Format | Storage / Val | Compression vs BF16 | Relative RMSE % | Outlier Error % (Top 0.1%) | GPU Dequant Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw BF16** | 2.00 B | $1.00\times$ | $0.167\%$ | $0.119\%$ | Baseline |
| **Naive FP8 (`E5M2`)** | 1.00 B | $2.00\times$ | **$5.248\%$** | **$6.813\%$** | Memory-bound |
| **Naive FP8 (`E4M3`)** | 1.00 B | $2.00\times$ | **$2.637\%$** | **$3.193\%$** | Memory-bound |
| **TensorCache INT8 ($G=32$)** | **1.06 B** | **$1.88\times$** | **$0.540\%$** | **$0.117\%$** | **$> 800\text{ GB/s}$** |

---

## 📜 License
MIT License.
