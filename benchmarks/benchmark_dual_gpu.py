"""
Dual NVIDIA GPU Benchmark for TensorCache.
Benchmarks dequantization throughput, latency, and reconstruction accuracy across both GPUs.
"""

import time
import torch
import numpy as np
from tabulate import tabulate

from tensorcache.codec import (
    quantize_int8_g32,
    dequantize_int8_g32,
    quantize_int8_adaptive
)
from tensorcache.prefetcher import AsyncGPUPrefetcher


def benchmark_gpu(device_idx: int):
    device = f"cuda:{device_idx}"
    gpu_name = torch.cuda.get_device_name(device_idx)
    vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
    
    print("\n" + "="*80)
    print(f"[*] BENCHMARKING GPU {device_idx}: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print("="*80)
    
    # 1. Generate realistic ViT activation batch: [128, 446, 768] in BF16
    batch_size = 128
    seq_len = 446
    dim = 768
    total_elements = batch_size * seq_len * dim
    raw_mb = (total_elements * 2.0) / (1024 * 1024) # BF16 size
    
    print(f"[*] Batch shape: [{batch_size}, {seq_len}, {dim}] ({total_elements:,} values, {raw_mb:.2f} MB in BF16)")
    
    torch.manual_seed(42)
    feat_bf16 = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device=device)
    
    # 2. Quantize to Block-wise INT8
    q_int8, scales, orig_shape = quantize_int8_g32(feat_bf16, group_size=32)
    compressed_mb = (q_int8.numel() * 1.0 + scales.numel() * 2.0) / (1024 * 1024)
    print(f"[+] Compressed Size: {compressed_mb:.2f} MB ({raw_mb / compressed_mb:.2f}x compression vs BF16)")
    
    # 3. Accuracy check
    rec = dequantize_int8_g32(q_int8, scales, orig_shape, group_size=32)
    diff = feat_bf16.float() - rec.float()
    rel_rmse = (torch.norm(diff) / torch.norm(feat_bf16.float())).item() * 100.0
    print(f"[+] Relative Reconstruction RMSE: {rel_rmse:.4f}%\n")
    
    # 4. Dequantization Latency & Throughput Benchmark
    # Warmup
    for _ in range(10):
        _ = dequantize_int8_g32(q_int8, scales, orig_shape, group_size=32)
    torch.cuda.synchronize(device)
    
    # Timed runs
    num_runs = 50
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = dequantize_int8_g32(q_int8, scales, orig_shape, group_size=32)
    torch.cuda.synchronize(device)
    total_time = time.perf_counter() - t0
    
    avg_latency_ms = (total_time / num_runs) * 1000.0
    # Processed bytes: Read INT8 (1B) + Read Scale (2B/32) + Write BF16 (2B) = 3.0625 bytes/val
    effective_io_bytes = total_elements * 3.0625
    throughput_gb_s = (effective_io_bytes / (1024**3)) / (avg_latency_ms / 1000.0)
    
    return {
        "GPU": f"GPU {device_idx}: {gpu_name}",
        "VRAM": f"{vram_gb:.1f} GB",
        "Compression": f"{raw_mb / compressed_mb:.2f}x",
        "RMSE %": f"{rel_rmse:.3f}%",
        "Latency": f"{avg_latency_ms:.2f} ms",
        "Throughput": f"{throughput_gb_s:.1f} GB/s"
    }


def main():
    print("="*80)
    print("      TENSORCACHE DUAL NVIDIA GPU HARDWARE BENCHMARK")
    print("="*80)
    
    num_gpus = torch.cuda.device_count()
    print(f"[*] Detected {num_gpus} NVIDIA GPUs.\n")
    
    results = []
    for i in range(num_gpus):
        res = benchmark_gpu(i)
        results.append(res)
        
    print("\n" + "="*80)
    print("                     FINAL HARDWARE PERFORMANCE SUMMARY")
    print("="*80)
    headers = ["GPU Device", "VRAM", "Comp vs BF16", "Rel RMSE %", "Dequant Latency", "VRAM Bandwidth"]
    rows = [
        [r["GPU"], r["VRAM"], r["Compression"], r["RMSE %"], r["Latency"], r["Throughput"]]
        for r in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
