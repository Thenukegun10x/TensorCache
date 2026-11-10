"""Experimental AMO-BQ accuracy search using the existing feature cache format.

This is a PyTorch research reference, not a replacement production encoder.
Search scores are measured after BF16 scale storage and BF16 reconstruction.
Uses sampled real pretrained DINOv3 features from COCO val; no downloads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensorcache.codec import dequantize_int8_amo_bq, quantize_int8_amo_bq


def evaluate_scales(x, scales, decoder_aware=True):
    """Score [blocks,candidates] scales against [blocks,G] FP32 values."""
    scales = scales.to(torch.bfloat16).float()
    bmin = x.amin(1)
    zp = (-bmin[:, None] / scales).round().clamp(0, 255)
    unrounded = x[:, None, :] / scales[:, :, None] + zp[:, :, None]
    q = unrounded.round().clamp(0, 255)
    rec = ((q - zp[:, :, None]) * scales[:, :, None]).to(torch.bfloat16).float()
    if decoder_aware:
        # BF16 output rounding moves decision boundaries. Check adjacent codes
        # too, choosing the closest value that the actual decoder can emit.
        for delta in (-1, 1):
            alt_q = (unrounded.round() + delta).clamp(0, 255)
            alt_rec = ((alt_q - zp[:, :, None]) * scales[:, :, None]).to(torch.bfloat16).float()
            better = (x[:, None, :] - alt_rec).square() < (x[:, None, :] - rec).square()
            q = torch.where(better, alt_q, q)
            rec = torch.where(better, alt_rec, rec)
    error = (x[:, None, :] - rec).square().sum(2)
    index = error.argmin(1)
    row = torch.arange(x.shape[0], device=x.device)
    return q[row, index].to(torch.uint8), scales[row, index].to(torch.bfloat16), zp[row, index].to(torch.uint8)


def experimental_quantize(x, method, chunk=1024, group_size=32):
    """Keep UINT8 q/zp and BF16 scale, so the production decoder is reused."""
    original_shape = x.shape
    x = x.reshape(-1, group_size).float()
    out_q, out_s, out_z = [], [], []
    for xb in x.split(chunk):
        lower, upper = xb.amin(1), xb.amax(1)
        if method == "zero_range":
            lower = lower.clamp(max=0)
            upper = upper.clamp(min=0)
        scale0 = (upper - lower).clamp(min=1e-8) / 255
        if method in ("stored_scale", "decoded_score", "zero_range"):
            scales = scale0[:, None] * torch.linspace(.95, 1.05, 32, device=x.device)[None, :]
        else:
            # BF16 positive scale bit patterns are ordered. Search actual wire
            # scales instead of spending candidates on duplicate rounded scales.
            low, high = (.5, 2.0) if method == "wide_grid" else (.85, 1.10)
            lo = (scale0 * low).to(torch.bfloat16).view(torch.int16).int()
            hi = (scale0 * high).to(torch.bfloat16).view(torch.int16).int()
            count = int((hi - lo).max().item()) + 1
            bits = torch.minimum(lo[:, None] + torch.arange(count, device=x.device), hi[:, None])
            scales = bits.to(torch.int16).view(torch.bfloat16).float()
        q, s, z = evaluate_scales(xb, scales, decoder_aware=method not in ("stored_scale", "zero_range"))
        if method == "lattice_refine":
            # Alternating least squares proposes a scale for the chosen codes,
            # then searches its neighboring BF16 values. Keep the current scale
            # as a candidate, so actual decoded error cannot increase.
            for _ in range(3):
                codes = q.float() - z[:, None].float()
                ls = ((xb * codes).sum(1) / codes.square().sum(1).clamp(min=1)).clamp(min=1e-12)
                bits = ls.to(torch.bfloat16).view(torch.int16).int()
                neighbors = (bits[:, None] + torch.arange(-2, 3, device=x.device)).to(torch.int16).view(torch.bfloat16).float()
                candidates = torch.cat((s.float()[:, None], neighbors), dim=1)
                q, s, z = evaluate_scales(xb, candidates)
        out_q.append(q)
        out_s.append(s)
        out_z.append(z)
    return torch.cat(out_q).flatten(), torch.cat(out_s), torch.cat(out_z), original_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--blocks-per-image", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("research/amo_accuracy_pilot.json"))
    args = parser.parse_args()
    if args.images <= 0 or args.blocks_per_image <= 0 or args.offset < 0:
        parser.error("images and blocks-per-image must be positive; offset must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA/ROCm GPU required")
    import timm
    from safetensors.torch import load_model
    from torch.utils.data import DataLoader
    from bench_amo_coco import CocoInputs, ROOT

    cache = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch16_dinov3.lvd1689m"
    candidates = sorted(cache.glob("snapshots/*/model.safetensors"))
    if not candidates:
        parser.error("no cached pretrained DINOv3 checkpoint")
    checkpoint = candidates[0]
    model = timm.create_model("vit_base_patch16_dinov3", pretrained=False, img_size=336)
    load_model(model, str(checkpoint))
    model = model.eval().cuda()
    files = sorted((ROOT / "data/coco_val").glob("*.jpg"))[args.offset:args.offset+args.images]
    if len(files) != args.images:
        parser.error("insufficient COCO images")
    loader = DataLoader(CocoInputs(files, 336, True), batch_size=16, num_workers=4, pin_memory=True)
    totals = {}
    energy, blocks = 0.0, 0
    methods = ["stored_scale", "decoded_score", "zero_range", "wire_grid", "lattice_refine", "wide_grid"]
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.forward_features(batch.cuda()).to(torch.bfloat16)
            per_image = features.reshape(features.shape[0], -1, 32)
            if args.blocks_per_image > per_image.shape[1]:
                parser.error("blocks-per-image exceeds the available G32 blocks")
            index = torch.linspace(0, per_image.shape[1] - 1, args.blocks_per_image, device="cuda").long()
            x = per_image[:, index].reshape(-1, 32).contiguous()
            if not torch.isfinite(x).all():
                raise RuntimeError("non-finite experimental inputs")
            energy += x.float().square().sum().item()
            blocks += x.shape[0]
            for mode in ("fast", "balanced", "accurate", "max"):
                result = quantize_int8_amo_bq(x, mode=mode)
                rec = dequantize_int8_amo_bq(*result).float()
                key = "production_" + mode
                totals[key] = totals.get(key, 0.0) + (x.float() - rec).square().sum().item()
            result = quantize_int8_amo_bq(x, group_size=16, mode="balanced")
            rec = dequantize_int8_amo_bq(*result, group_size=16).float()
            totals["production_g16"] = totals.get("production_g16", 0.0) + (x.float() - rec).square().sum().item()
            for method in methods:
                rec = dequantize_int8_amo_bq(*experimental_quantize(x, method)).float()
                totals[method] = totals.get(method, 0.0) + (x.float() - rec).square().sum().item()
            result = experimental_quantize(x, "wire_grid", group_size=16)
            rec = dequantize_int8_amo_bq(*result, group_size=16).float()
            totals["wire_grid_g16"] = totals.get("wire_grid_g16", 0.0) + (x.float() - rec).square().sum().item()
            if batch_index % 20 == 0 or (batch_index+1)*16 >= len(files):
                print(f"COCO {min((batch_index+1)*16, len(files))}/{len(files)}", flush=True)
    rmse = {method: 100 * (sse / energy) ** .5 for method, sse in totals.items()}
    report = {"images": len(files), "offset": args.offset, "blocks_per_image": args.blocks_per_image, "blocks": blocks,
              "elements": blocks * 32, "gpu": torch.cuda.get_device_name(),
              "baseline_commit": "2b44190", "torch": torch.__version__,
              "checkpoint": str(checkpoint), "relative_rmse_pct": rmse,
              "rmse_reduction_vs_balanced_pct": {method: 100*(1-value/rmse['production_balanced']) for method,value in rmse.items()},
              "bytes_per_element": {method: 1+3/(16 if method.endswith("g16") else 32) for method in totals},
              "notes": "Research reference; sampled COCO/DINOv3 BF16 blocks, not a production speed benchmark. BF16/UINT8 format unchanged; G16 controls use 8.57% more bytes than G32."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
