"""Production AMO-BQ quality/size/encode/decode comparison on COCO val.

    python benchmarks/bench_amo_quality.py --limit 5000 --repeats 3

All feature values are evaluated. Decode, transfer, feature extraction and
compilation are excluded from encoder timings. Decode is timed separately
with a reused output buffer. No synthetic-data or CPU timing fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tensorcache.codec import dequantize_int8_amo_bq, quantize_int8_amo_bq
from tensorcache import fused_ops
from bench_amo_coco import CocoInputs, ROOT, measure, peak_extra


CASES = {"balanced_g32": ("balanced", 32), "quality_g32": ("quality", 32),
         "balanced_g16": ("balanced", 16), "quality_g16": ("quality", 16)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/coco_val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "research/amo_quality_coco.json")
    args = parser.parse_args()
    if min(args.limit, args.batch_size, args.repeats) <= 0 or args.workers < 0:
        parser.error("limit, batch-size and repeats must be positive; workers must be nonnegative")
    if not (torch.cuda.is_available() and fused_ops.HAS_TRITON):
        parser.error("functional CUDA/ROCm GPU and Triton required")
    files = sorted(args.data.glob("*.jpg"))[:args.limit]
    if len(files) != args.limit:
        parser.error(f"requested {args.limit} COCO images, found {len(files)}")
    import timm
    from safetensors.torch import load_model
    checkpoint = args.checkpoint
    if checkpoint is None:
        cache = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch16_dinov3.lvd1689m"
        candidates = sorted(cache.glob("snapshots/*/model.safetensors"))
        if not candidates:
            parser.error("no cached DINOv3 checkpoint; pass --checkpoint")
        checkpoint = candidates[0]
    model = timm.create_model("vit_base_patch16_dinov3", pretrained=False, img_size=336)
    load_model(model, str(checkpoint))
    model = model.eval().cuda()
    loader = DataLoader(CocoInputs(files, 336, True), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "images": len(files),
        "batch_size": args.batch_size, "repeats": args.repeats,
        "size": 336, "dtype": "bfloat16", "input": "all DINOv3 feature values",
        "checkpoint": str(checkpoint), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "triton": fused_ops.triton.__version__,
        "source_sha256": {name: hashlib.sha256((ROOT / "src/tensorcache" / name).read_bytes()).hexdigest()
                          for name in ("codec.py", "fused_ops.py")},
        "file_names_sha256": hashlib.sha256("\n".join(p.name for p in files).encode()).hexdigest(),
        "timing_excludes": ["image decode", "input transfer", "feature extraction", "JIT compilation"],
        "cases": {},
    }
    rows = {name: {"encode_gpu": [], "encode_wall": [], "decode_gpu": [], "decode_wall": [], "sse": 0.0}
            for name in CASES}
    seen, elements, energy = 0, 0, 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = model.forward_features(batch.cuda(non_blocking=True)).bfloat16().contiguous()
            if not torch.isfinite(x).all():
                raise RuntimeError("non-finite benchmark input")
            output = torch.empty_like(x)
            energy += x.float().square().sum().item()
            elements += x.numel()
            names = list(CASES)
            names = names[batch_index % len(names):] + names[:batch_index % len(names)]
            for name in names:
                mode, group_size = CASES[name]
                row = rows[name]

                def encode(value, mode):
                    return quantize_int8_amo_bq(value, group_size=group_size, mode=mode)

                if batch_index == 0:
                    report["input_batch_shape"] = list(x.shape)
                    for _ in range(3):
                        encoded = encode(x, mode)
                        dequantize_int8_amo_bq(*encoded, group_size=group_size, out_buffer=output)
                    row["peak_extra_bytes"] = peak_extra(encode, x, mode)
                encoded, gpu_ms, wall_ms = measure(encode, x, mode, args.repeats)
                row["encode_gpu"].append(gpu_ms)
                row["encode_wall"].append(wall_ms)

                def decode(value, mode):
                    return dequantize_int8_amo_bq(*encoded, group_size=group_size, out_buffer=output)

                rec, gpu_ms, wall_ms = measure(decode, x, mode, args.repeats)
                row["decode_gpu"].append(gpu_ms)
                row["decode_wall"].append(wall_ms)
                row["sse"] += (x.float() - rec.float()).square().sum().item()
            seen += batch.shape[0]
            if batch_index % 20 == 0 or seen == len(files):
                print(f"COCO {seen}/{len(files)}", flush=True)

    import statistics
    report["elements"] = elements
    for name, row in rows.items():
        mode, group_size = CASES[name]
        summary = {"mode": mode, "group_size": group_size,
                   "relative_rmse_pct": 100 * (row["sse"] / energy) ** .5,
                   "bytes_per_element": 1 + 3 / group_size,
                   "compression_vs_bf16": 2 / (1 + 3 / group_size),
                   "peak_encode_extra_bytes": row["peak_extra_bytes"]}
        for operation in ("encode", "decode"):
            for clock in ("gpu", "wall"):
                key = operation + "_" + clock
                summary["median_batch_" + key + "_ms"] = statistics.median(row[key])
                summary[key + "_images_per_second"] = len(files) / (sum(row[key]) / 1000)
                summary["batch_" + key + "_ms"] = row[key]
        report["cases"][name] = summary
    baseline = report["cases"]["balanced_g32"]
    for name, summary in report["cases"].items():
        summary["rmse_reduction_vs_balanced_g32_pct"] = 100 * (1-summary["relative_rmse_pct"]/baseline["relative_rmse_pct"])
        print(name, {k: v for k, v in summary.items() if not isinstance(v, list)}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
