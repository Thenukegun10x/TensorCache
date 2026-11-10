"""Compare AMO-BQ against a Git revision on real COCO validation inputs.

Example (uses a locally cached pretrained DINOv3 checkpoint):
    python benchmarks/bench_amo_coco.py --baseline-ref 6ef15d0 --limit 5000

Feature extraction, image decode, transfers and compilation are outside the
encoder timings. Wall time includes dispatch/synchronization overhead; CUDA
events measure the GPU interval. No synthetic-data or CPU timing fallback.
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tensorcache.codec import dequantize_int8_amo_bq, quantize_int8_amo_bq


ROOT = Path(__file__).resolve().parents[1]


class CocoInputs(Dataset):
    def __init__(self, files, size, features):
        self.files, self.size, self.features = files, size, features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.array(image)).float().div_(255)
        if self.features:
            x = x.permute(2, 0, 1)
            x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
                [0.229, 0.224, 0.225])[:, None, None]
        return x


def load_baseline(ref, directory):
    """Isolate both old modules so their relative imports use the old code."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=ROOT, text=True).strip()
    for name in ("codec", "fused_ops"):
        source = subprocess.check_output(
            ["git", "show", f"{revision}:src/tensorcache/{name}.py"], cwd=ROOT)
        (Path(directory) / f"{name}.py").write_bytes(source)
    package = types.ModuleType("_tensorcache_amo_baseline")
    package.__path__ = [directory]
    sys.modules[package.__name__] = package
    module = importlib.import_module(package.__name__ + ".codec")
    return module.quantize_int8_amo_bq, revision


def measure(fn, x, mode, repeats):
    gpu_ms, wall_ms = [], []
    result = None
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        result = fn(x, mode=mode)
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000)
        gpu_ms.append(start.elapsed_time(end))
    return result, statistics.median(gpu_ms), statistics.median(wall_ms)


def peak_extra(fn, x, mode):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = fn(x, mode=mode)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    del result
    return extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/coco_val")
    parser.add_argument("--baseline-ref", default="6ef15d0")
    parser.add_argument("--input", choices=("features", "pixels"), default="features")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["balanced"],
                        choices=("fast", "balanced", "accurate", "max"))
    parser.add_argument("--output", type=Path, default=ROOT / "research/amo_coco_results.json")
    args = parser.parse_args()
    if min(args.limit, args.size, args.batch_size, args.repeats) <= 0:
        parser.error("limit, size, batch-size and repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("a working CUDA/ROCm GPU is required")
    from tensorcache import fused_ops
    if not fused_ops.HAS_TRITON:
        parser.error("Triton is required")
    files = sorted(args.data.glob("*.jpg"))[:args.limit]
    if len(files) != args.limit:
        parser.error(f"requested {args.limit} COCO images, found {len(files)} in {args.data}")

    model, checkpoint = None, None
    if args.input == "features":
        import timm
        from safetensors.torch import load_model
        checkpoint = args.checkpoint
        if checkpoint is None:
            cache = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch16_dinov3.lvd1689m"
            candidates = sorted(cache.glob("snapshots/*/model.safetensors"))
            if not candidates:
                parser.error("no cached pretrained DINOv3 checkpoint; pass --checkpoint")
            checkpoint = candidates[0]
        model = timm.create_model("vit_base_patch16_dinov3", pretrained=False, img_size=args.size)
        load_model(model, str(checkpoint))
        model = model.eval().to("cuda")

    loader = DataLoader(CocoInputs(files, args.size, model is not None),
                        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    report = {
        "baseline_ref": args.baseline_ref, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "triton": fused_ops.triton.__version__,
        "input": args.input, "checkpoint": str(checkpoint) if checkpoint else None,
        "dtype": "bfloat16", "size": args.size, "images": len(files),
        "batch_size": args.batch_size, "repeats": args.repeats,
        "timing_excludes": ["image decode", "feature extraction", "input transfer", "compilation"],
        "modes": {},
    }
    timings = {mode: {name: {"gpu": [], "wall": [], "sse": 0.0} for name in
                        ("baseline", "optimized")} for mode in args.modes}
    mismatches = {mode: [0, 0, 0] for mode in args.modes}
    energy, seen = 0.0, 0
    with tempfile.TemporaryDirectory(prefix="tensorcache-amo-baseline-") as directory:
        baseline, report["baseline_commit"] = load_baseline(args.baseline_ref, directory)
        encoders = {"baseline": baseline, "optimized": quantize_int8_amo_bq}
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                x = batch.to("cuda", non_blocking=True)
                if model is not None:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x = model.forward_features(x)
                x = x.to(torch.bfloat16).contiguous()
                if not torch.isfinite(x).all():
                    raise RuntimeError("non-finite benchmark inputs")
                if batch_index == 0:
                    report["input_batch_shape"] = list(x.shape)
                    for mode in args.modes:
                        for fn in encoders.values():
                            for _ in range(3):
                                fn(x, mode=mode)
                        timings[mode]["baseline"]["peak_extra_bytes"] = peak_extra(baseline, x, mode)
                        timings[mode]["optimized"]["peak_extra_bytes"] = peak_extra(quantize_int8_amo_bq, x, mode)
                energy += x.float().square().sum().item()
                for mode in args.modes:
                    outputs = {}
                    # Alternate order to reduce systematic clock/thermal bias.
                    order = ("baseline", "optimized") if batch_index % 2 == 0 else ("optimized", "baseline")
                    for name in order:
                        result, gpu_ms, wall_ms = measure(encoders[name], x, mode, args.repeats)
                        row = timings[mode][name]
                        row["gpu"].append(gpu_ms)
                        row["wall"].append(wall_ms)
                        rec = dequantize_int8_amo_bq(*result)
                        row["sse"] += (x.float() - rec.float()).square().sum().item()
                        outputs[name] = result
                    for index in range(3):
                        mismatches[mode][index] += torch.count_nonzero(
                            outputs["baseline"][index] != outputs["optimized"][index]).item()
                seen += batch.shape[0]
                if batch_index % 20 == 0 or seen == len(files):
                    print(f"COCO {seen}/{len(files)} | feature/input shape {tuple(x.shape)}", flush=True)

    for mode, methods in timings.items():
        summary = {"mismatched_elements_q_scale_zp": mismatches[mode]}
        for name, row in methods.items():
            summary[name] = {
                "median_batch_gpu_ms": statistics.median(row["gpu"]),
                "median_batch_wall_ms": statistics.median(row["wall"]),
                "images_per_second_wall": len(files) / (sum(row["wall"]) / 1000),
                "relative_rmse_pct": 100 * (row["sse"] / energy) ** 0.5,
                "peak_extra_bytes": row["peak_extra_bytes"],
                "batch_gpu_ms": row["gpu"], "batch_wall_ms": row["wall"],
            }
        summary["wall_speedup"] = sum(methods["baseline"]["wall"]) / sum(methods["optimized"]["wall"])
        summary["gpu_speedup"] = sum(methods["baseline"]["gpu"]) / sum(methods["optimized"]["gpu"])
        report["modes"][mode] = summary
        print(mode, json.dumps({k: v for k, v in summary.items() if k not in encoders}), flush=True)
        for name in encoders:
            print(name, {k: v for k, v in summary[name].items() if not isinstance(v, list)}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
