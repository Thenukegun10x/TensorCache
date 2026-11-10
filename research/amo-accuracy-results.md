# AMO-BQ accuracy experiments — 2026-11-10

The subsequent opt-in production implementation and full-feature benchmark
are documented in [amo-quality-results.md](amo-quality-results.md). The
results below preserve the earlier sampled-block research stage.

**Result:** decode-aware BF16 scale search lowers G32 relative RMSE by about
8% at the current byte rate. Combining it with G16 lowers relative RMSE by
about 35%, costing 8.57% more bytes per feature value.

These are research reference implementations. Production encoding, decoding,
defaults, cache files and public APIs have not been changed by this experiment.
No production throughput improvement or preservation is claimed yet.

## Why there is room to improve

Current AMO-BQ evaluates candidate scales and reconstructs values in FP32,
then rounds the winning scale to BF16 for storage. The final codes were chosen
using the unrounded scale. The actual decoder uses the stored BF16 scale and
rounds its output to BF16, so the search objective differs from the output
whose error matters.

The reference rounds candidate scales to BF16 before choosing codes and scores
the actual BF16 reconstruction. It can also enumerate representable BF16 scale
values directly, avoiding candidate evaluations that collapse to the same
stored scale.

## Workload

- All 5,000 COCO validation images, sorted by filename, resized to 336×336.
- Same cached pretrained DINOv3 ViT-B/16, ImageNet normalization, BF16 features,
  RTX 4050 Laptop GPU and software environment as the encoding speed study.
- **256 evenly spaced G32 blocks per image**: 1,280,000 blocks, **40,960,000
  feature values**. Every method sees exactly the same sampled values. This is
  about 2.39% of each image's features, not an exhaustive evaluation of all
  1.713 billion values in the earlier throughput benchmark.
- Global relative RMSE is `100 * sqrt(total squared error / total squared
  input norm)` using the production decoder for every reconstruction.
- Production baseline is commit `2b44190`. Prototype search uses finite BF16
  inputs; its timings are not comparable with the optimized Triton encoder.

## Results

| Method | Relative RMSE | RMSE reduction vs G32 balanced | Bytes/value |
| --- | ---: | ---: | ---: |
| Current G32 balanced | 0.555286% | — | 1.09375 |
| Current G32 accurate | 0.546947% | 1.50% | 1.09375 |
| Current G32 max | 0.550218% | 0.91% | 1.09375 |
| BF16-aware scoring, same 32 scale multipliers | 0.523492% | 5.73% | 1.09375 |
| BF16 grid search, 0.85–1.10× base scale | 0.514863% | 7.28% | 1.09375 |
| Grid plus three local least-squares refinements | 0.514658% | 7.32% | 1.09375 |
| Wider BF16 grid, 0.50–2.00× base scale | 0.511042% | **7.97%** | 1.09375 |
| Current G16 balanced | 0.422254% | 23.96% | 1.18750 |
| G16 BF16 grid, 0.85–1.10× base scale | 0.362999% | **34.63%** | 1.18750 |

The G16 result uses twice as many scale/zero-point records, increasing total
storage by `(1.1875 / 1.09375 - 1) * 100 = 8.57%`. Compression versus BF16
changes from about 1.83× to 1.68×. G16 is already supported by the current
cache metadata and decoder; existing G32 caches do not change.

The separate 256-image pilot and next-256-image slice show similar G32 gains:
about 7.3–7.4% for grid/refinement and 7.98% for the wider grid on the second
slice. Adding adjacent-code checks to account for BF16 output boundaries
helps only slightly, as does local least-squares refinement. Most of the
improvement comes from scoring the stored representation and searching its
scale grid. More FP32 candidates alone is a less effective use of work.

## Reference checks and reproduction

On independent synthetic BF16 blocks, decoded nearest-code selection matched
an exhaustive search over all 256 codes for each candidate scale. The local
refinement was checked not to increase decoded squared error at G16 and G32.
Every method is reconstructed using `dequantize_int8_amo_bq`.

```bash
python benchmarks/bench_amo_accuracy.py --images 5000 \
  --blocks-per-image 256 --output research/amo_accuracy_coco.json
```

The reference is in `benchmarks/bench_amo_accuracy.py`; full numeric results
are in `research/amo_accuracy_coco.json`. It reuses the local checkpoint and
COCO dataset and performs no downloads.

## Practical direction

First port BF16-aware scoring to Triton and measure its accuracy/encoding-time
trade-off against `2b44190`. The same 32 candidates already recover most of the
G32 gain. An opt-in wider BF16-grid search can target higher accuracy, and G16
can offer a larger quality improvement when 8.57% extra storage is acceptable.
Decoder cost and format remain the same for the G32 experiments; encoder cost
must be measured after implementing the GPU search.
