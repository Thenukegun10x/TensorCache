# AMO-BQ encoding optimization — COCO val, 2026-11-10

Compared the working-tree encoder with commit
`6ef15d048ff608ff6732ff033e955ae72acc8843` (v0.7.2). Both use the same
candidate counts, ranges, G32 groups, BF16 scales, UINT8 values and zero points.

## Workload and measurement

- All 5,000 sorted JPEGs in `data/coco_val`, resized to 336×336 with bilinear
  interpolation and ImageNet normalization.
- Real pretrained DINOv3 ViT-B/16 features from the cached
  `timm/vit_base_patch16_dinov3.lvd1689m` checkpoint, revision
  `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da`.
- BF16 features `[16, 446, 768]`; final batch has 8 images.
- NVIDIA GeForce RTX 4050 Laptop GPU, PyTorch 2.13.0+cu130, Triton 3.7.1.
- Three warmup calls per method/preset, followed by three timed calls per
  batch. Old/new measurement order alternates between batches. Each batch's
  median is used; throughput divides all 5,000 images by the sum of those
  batch medians. Speedup is the ratio of those summed wall times.
- Timings cover encoding, including allocation, Python dispatch and GPU
  synchronization. Image decode, feature extraction, input transfer and JIT
  compilation are excluded. These are encoder throughputs, not end-to-end
  image-processing or training throughputs.

## Results

| Preset | Old median batch ms | New median batch ms | Wall speedup | Old img/s | New img/s | Old rel. RMSE | New rel. RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | 4.580 | 0.440 | 10.28× | 3,395 | 34,892 | 0.571323% | 0.571323% |
| balanced | 8.192 | 0.794 | 10.26× | 1,910 | 19,596 | 0.559452% | 0.559449% |
| accurate | 11.897 | 1.131 | 10.49× | 1,320 | 13,853 | 0.551036% | 0.551035% |
| max | 15.624 | 1.481 | 10.55× | 1,003 | 10,581 | 0.554508% | 0.554482% |

Peak additional allocated GPU memory for a full batch, including encoded
outputs, falls from **26.62 MiB to 5.72 MiB** (78.5% less). This excludes the
already-resident model, feature tensor and cached allocator reservation.

## Implementation and compatibility

1. Load BF16/FP16 inputs directly and convert in registers, removing the
   full-tensor FP32 allocation. Masked loads also eliminate input padding.
2. Sanitize NaN/Inf during the kernel load, exclude them from candidate error,
   and encode their positions as the zero point so they reconstruct as zero.
   This removes the separate `isfinite().all()` scan and CPU scalar read.
3. Process 32 independent groups per Triton program with two warps, amortizing
   scheduling overhead and using more lanes than the original one-group
   program. The candidate search remains exhaustive.

The file format and decode path are unchanged. Encodings are **not guaranteed
byte-identical**: the compiler changes floating-point reduction layout, and
rare values/candidate choices differ. Across 1,712,640,000 encoded values,
the four presets differ from the old code at 266, 349, 540 and 682 UINT8
positions respectively (at most 0.000040%). BF16 scale differences are 11,
14, 21 and 24 out of 53,520,000 groups. Aggregate reconstruction error is
effectively unchanged. ROCm performance has not been measured.

## Reproduce and validate

Use an environment with PyTorch, Triton, timm, torchvision, safetensors,
Pillow and NumPy, plus the cached model and COCO validation images:

```bash
python benchmarks/bench_amo_coco.py --baseline-ref 6ef15d0 --limit 5000 \
  --repeats 3 --modes fast balanced accurate max \
  --output research/amo_coco_results.json
pytest tests/test_codec.py -v
pytest tests/test_fused_ops.py -v
python tests/test_codec.py
```

Validation: **15 codec tests + 70 fused-op tests passed**, along with the
direct codec sanity run. AMO regressions cover FP16/BF16/FP32, G16/G32/G64,
empty inputs, partial groups/programs, noncontiguous inputs, custom candidate
counts, NaN/Inf, and CUDA graph capture (which forbids host scalar reads).
The codec tests were run with CUDA hidden to exercise CPU fallbacks. Runs
requiring GPU access or DataLoader worker IPC used the host environment,
outside the restrictive sandbox.

Full timings, exact model path, software versions, byte-difference counts and
unrounded errors are in [amo_coco_results.json](amo_coco_results.json).
