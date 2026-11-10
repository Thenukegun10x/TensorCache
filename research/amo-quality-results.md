# Production AMO-BQ quality mode — 2026-11-10

The opt-in `mode="quality"` is implemented in both Triton and portable PyTorch.
`group_size=32` remains the default; `group_size=16` offers higher fidelity at
8.57% more bytes. Existing presets retain their encoding behavior.

## Full COCO validation

All 5,000 sorted COCO val JPEGs, resized to 336×336 with bilinear interpolation,
ImageNet normalization, pretrained DINOv3 ViT-B/16 checkpoint
`c6a5fb7d12bbd3cf3b0079253141c3332aaed7da`, BF16 features. Every feature value is
scored: **1,712,640,000 values**, with batches `[16,446,768]` and a final batch
of 8. This supersedes the sampled-block accuracy pilot for production claims.

NVIDIA RTX 4050 Laptop GPU, PyTorch 2.13.0+cu130, Triton 3.7.1. The benchmark
rotates the measurement order across four configurations. Each has three
warmup calls, then three timed encode/decode repetitions per batch. The table
uses medians of per-batch GPU medians. Decode reuses a preallocated BF16 output
buffer. Input loading, transfers, feature extraction and compilation are
excluded. Wall latency, individual batch timings, source hashes and dataset
filename hash are recorded in the JSON output as well.

| Configuration | Relative RMSE | Reduction vs balanced G32 | B/value | Encode GPU ms | Decode GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced G32 | 0.559449% | — | 1.09375 | 0.758 | 0.060 |
| quality G32 | 0.520193% | 7.02% | 1.09375 | 3.447 | 0.062 |
| balanced G16 | 0.422812% | 24.42% | 1.18750 | 0.844 | 0.060 |
| quality G16 | **0.365182%** | **34.72%** | 1.18750 | 3.486 | 0.062 |

At fixed G16 storage, quality lowers RMSE by a further 13.63% versus balanced
G16. The larger 34.72% improvement includes both smaller groups and better
search. G16 stores 1.1875 B/value instead of 1.09375, hence 8.57% more total
bytes, with compression versus BF16 changing from 1.83× to 1.68×.

Median wall encode latencies were 0.782, 3.467, 0.867 and 3.516 ms respectively.
These are codec latencies for resident features, not end-to-end training or
image-processing throughput. GPU decode latency changes by about 0.002 ms in
this run; the same decode kernel and wire representation are used.

## Algorithm and compatibility

- For each group, include zero in the min/max range. This also handles
  one-sided and nonzero constant groups correctly.
- Enumerate the positive BF16 scales between rounded `0.85 * scale0` and
  `1.10 * scale0`, inclusive. A fixed limit of 64 covers the full interval.
  Unlike legacy presets, these candidates are stored BF16 values, not FP32
  scale multipliers scored before storage rounding.
- Compute a UINT8 zero point for each candidate, choose the nearest decoded
  BF16 value (checking adjacent codes for shifted BF16 output boundaries),
  and minimize squared reconstruction error over finite inputs.
- Encode NaN/Inf positions as the selected zero point so they reconstruct as
  zero; exclude these positions from candidate error.
- Triton performs sanitization, search and quantization in one launch with
  direct BF16/FP16 loads. No host scalar finite check or full FP32 copy.

The new mode uses the existing UINT8 values, BF16 scales and UINT8 zero
points. It changes no dequantization API, output-buffer behavior or file
layout. Cache metadata persists `amo_mode="quality"`, `group_size` and the
fixed search bounds; append validates requested settings before changing
files. Single-shard and sharded datasets read the existing layout.

`compress`, `BlockwiseInt8Codec`, `FeatureCacheWriter` and the CLI accept the
new mode. `benchmark_tensor(..., include_quality=True)`,
`auto_select_mode(..., include_quality=True)` and CLI `benchmark --quality`
explicitly opt into its additional search cost. No new public export is
needed; the existing preset registry exposes `quality`.

## Reproduction

```bash
python benchmarks/bench_amo_quality.py --limit 5000 --repeats 3 \
  --output research/amo_quality_coco.json
pytest tests/test_codec.py tests/test_amo_quality.py -v
pytest tests/test_fused_ops.py -v
python tests/test_codec.py
```

The local COCO images and pretrained checkpoint are required; the benchmark
does not download data or fall back to synthetic features. The reference
tests check decoded squared error against exhaustive enumeration of all 256
codes for every BF16 scale, including partial groups. Coverage also includes
FP16/BF16/FP32, G16/G32/G64, the non-power-of-two CPU fallback, noncontiguous
inputs, zero/constant/nonfinite inputs, CUDA graph capture, cache append,
sharding and opt-in helper/CLI behavior. CPU/GPU floating-point reductions
can choose different near-tied encodings; the decode representation is the
same.

Validation completed: **116 tests passed** across the codec, fused-op and
quality suites (15 + 74 + 27), plus the direct codec sanity run. This includes
GPU-created single-shard and sharded caches read through dataset batches and
the ring-buffered streamer, with a partial final batch.

Raw results: [amo_quality_coco.json](amo_quality_coco.json).
