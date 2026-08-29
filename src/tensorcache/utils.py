"""
User-friendly helpers for TensorCache — auto-selection, estimation, one-liners.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any

import torch

from .codec import (
    BlockwiseInt8Codec,
    quantize_int8_g32,
    dequantize_int8_g32,
    quantize_int8_amo_bq,
    dequantize_int8_amo_bq,
    AMO_BQ_PRESETS,
)


def estimate_compression(
    shape: Tuple[int, ...],
    group_size: int = 32,
    amo_bq: bool = True,
) -> Dict[str, Any]:
    """Estimate bytes / compression without quantizing."""
    numel = math.prod(shape)
    num_blocks = (numel + group_size - 1)//group_size
    if amo_bq:
        bpe = 1 + 3/group_size  # 1B q + 2B scale +1B zp
    else:
        bpe = 1 + 2/group_size
    return {
        "numel": numel,
        "num_blocks": num_blocks,
        "bytes_per_elem": bpe,
        "total_bytes": int(numel*bpe),
        "total_MB": numel*bpe/1024/1024,
        "ratio_vs_bf16": 2.0/bpe,
        "ratio_vs_fp32": 4.0/bpe,
        "group_size": group_size,
        "amo_bq": amo_bq,
    }


def benchmark_tensor(
    x: torch.Tensor,
    group_size: int = 32,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Quick error/latency benchmark for all modes on one tensor."""
    import time
    if device:
        x = x.to(device)
    results = {}
    # sym
    for name, fn in [
        ("sym_g32", lambda: quantize_int8_g32(x, group_size)),
        ("adaptive", lambda: __import__("tensorcache.codec", fromlist=["quantize_int8_adaptive"]).quantize_int8_adaptive(x, group_size)),
    ]:
        t0=time.perf_counter()
        q,s,sh=fn()
        if x.is_cuda:
            torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)*1000
        rec=dequantize_int8_g32(q,s,sh, group_size)
        rel=(torch.norm(x.float()-rec.float())/torch.norm(x.float())).item()*100
        results[name]= {"rel_rmse%": rel, "ms": dt}

    for mode in ["fast","balanced","accurate"]:
        t0=time.perf_counter()
        q,s,zp,sh=quantize_int8_amo_bq(x, group_size, mode=mode)
        if x.is_cuda:
            torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)*1000
        rec=dequantize_int8_amo_bq(q,s,zp,sh, group_size)
        rel=(torch.norm(x.float()-rec.float())/torch.norm(x.float())).item()*100
        results[f"amo_{mode}"]= {"rel_rmse%": rel, "ms": dt, **dict(zip(["N","lo","hi"], AMO_BQ_PRESETS[mode][:3]))}

    if verbose:
        try:
            from rich.table import Table
            from rich.console import Console
            console=Console()
            tbl=Table(title=f"TensorCache bench {tuple(x.shape)} {x.dtype} {x.device}")
            tbl.add_column("Mode"); tbl.add_column("rel RMSE%"); tbl.add_column("ms"); tbl.add_column("B/elem")
            for k,v in results.items():
                bpe = 1+3/group_size if "amo" in k else 1+2/group_size
                tbl.add_row(k, f"{v['rel_rmse%']:.4f}", f"{v['ms']:.2f}", f"{bpe:.4f}")
            console.print(tbl)
        except Exception:
            for k,v in results.items():
                print(f"{k:12s} rel={v['rel_rmse%']:.4f}% ms={v['ms']:.2f}")
    return results


def auto_select_mode(
    x: torch.Tensor,
    target_rmse: float = 0.5,
    group_size: int = 32,
) -> str:
    """Pick cheapest AMO-BQ preset meeting target_rmse (else 'max' or G16)."""
    res=benchmark_tensor(x, group_size, verbose=False)
    for mode in ["fast","balanced","accurate","max"]:
        if res[f"amo_{mode}"]["rel_rmse%"] <= target_rmse:
            return mode
    # If still > target, suggest smaller G
    if group_size>16:
        return f"G16_{auto_select_mode(x, target_rmse, 16)}"
    return "max"


def compress(
    x: torch.Tensor,
    mode: str = "balanced",
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int,...]]:
    """One-liner: amo_bq compress. Returns (q,s,zp,shape)."""
    if mode in AMO_BQ_PRESETS:
        return quantize_int8_amo_bq(x, group_size, mode=mode)
    elif mode in ("sym","adaptive"):
        if mode=="sym":
            return quantize_int8_g32(x, group_size)
        from tensorcache.codec import quantize_int8_adaptive
        return quantize_int8_adaptive(x, group_size)
    else:
        raise ValueError(f"mode {mode!r} unknown, choose from {list(AMO_BQ_PRESETS.keys())+['sym','adaptive']}")


def decompress(
    q: torch.Tensor,
    scales: torch.Tensor,
    shape: Tuple[int,...],
    zero_points: Optional[torch.Tensor]=None,
    group_size: int=32,
) -> torch.Tensor:
    """One-liner decompress (auto sym vs amo)."""
    if zero_points is not None:
        return dequantize_int8_amo_bq(q, scales, zero_points, shape, group_size)
    return dequantize_int8_g32(q, scales, shape, group_size)


def help_text() -> str:
    return """
TensorCache — Ultra-fast INT8 feature cache (AMO-BQ)

Python:
  import tensorcache as tc, torch
  x = torch.randn(16,446,768, dtype=torch.bfloat16, device='cuda')
  # 1. Codec one-liners
  q,s,zp,shape = tc.compress(x, mode='balanced')  # fast/balanced/accurate/max/sym/adaptive
  rec = tc.decompress(q,s,shape,zp)               # BF16
  tc.benchmark_tensor(x)                          # compare all modes
  tc.estimate_compression(x.shape)                # bytes / ratio
  tc.auto_select_mode(x, target_rmse=0.5)         # -> 'balanced'

  # 2. Cache to disk (mmap + GPU prefetch)
  writer = tc.FeatureCacheWriter('./cache/feat', num_samples=10000, seq_len=446, dim=768, amo_bq=True, amo_mode='balanced')
  writer.append(x)  # [seq,dim] or [B,seq,dim]
  writer.close()
  ds = tc.FeatureCacheDataset('./cache/feat')               # (q,s,zp)
  ds = tc.FeatureCacheDataset('./cache/feat', auto_dequant_device='cuda') # BF16 directly
  for batch in ds.iter_batches(batch_size=256, device='cuda'): ...
  for q,s,zp in tc.AsyncGPUPrefetcher(loader, device='cuda'): rec = tc.dequantize_int8_amo_bq(q,s,zp, shape)

CLI:
  python -m tensorcache info
  python -m tensorcache benchmark --shape 16,446,768 --device cuda
  python -m tensorcache cache-info --prefix ./cache/feat
  tensorcache --help

Presets G32: fast (16,0.95-1.05) 6.9ms 0.49%, balanced (32,0.95-1.05) 13ms 0.478%, accurate (48,0.95-1.10) 49ms 0.473%
Storage: amo 1.09375 B/elem 1.83x vs BF16 (+0.03 vs sym 1.0625), dequant 0.06ms 5.4M
G16 0.39% 1.1875B 1.68x for <0.4% needs.
"""
