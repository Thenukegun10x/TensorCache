"""
Tuned High-Speed Benchmark Suite for TensorCache.
Tests:
  1. Tuned Fused Dequantizer Scaling across batch sizes.
  2. High-Throughput Batch Slicing vs Standard PyTorch DataLoader.
"""

import time
import math
import torch
from tabulate import tabulate
import tempfile
from pathlib import Path

from tensorcache.codec import quantize_int8_g32, dequantize_int8_g32
from tensorcache.feature_cache import FeatureCacheWriter, FeatureCacheDataset


def benchmark_dequant_scaling(device="cuda:0"):
    print("\n" + "="*85)
    print(f"[*] BENCHMARK 1: Tuned Triton Dequantizer vs Naive PyTorch")
    print(f"[*] Hardware: {torch.cuda.get_device_name(0)}")
    print("="*85)
    
    seq_len = 446
    dim = 768
    batch_sizes = [16, 64, 128, 256]
    table_data = []
    
    for B in batch_sizes:
        total_elements = B * seq_len * dim
        raw_bf16_mb = (total_elements * 2.0) / (1024 * 1024)
        compressed_mb = (total_elements * 1.0625) / (1024 * 1024)
        
        feat = torch.randn(B, seq_len, dim, dtype=torch.bfloat16, device=device)
        q_int8, scales, shape = quantize_int8_g32(feat, group_size=32)
        
        # 1. Naive PyTorch
        for _ in range(5):
            _ = (q_int8.view(-1, 32).float() * scales.unsqueeze(-1).float()).view(shape).to(torch.bfloat16)
        torch.cuda.synchronize()
        
        runs = 50
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = (q_int8.view(-1, 32).float() * scales.unsqueeze(-1).float()).view(shape).to(torch.bfloat16)
        torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) / runs * 1000.0
        
        # 2. Tuned Fused Triton Dequant
        out_buf = torch.empty(shape, dtype=torch.bfloat16, device=device)
        for _ in range(5):
            _ = dequantize_int8_g32(q_int8, scales, shape, group_size=32, out_buffer=out_buf)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = dequantize_int8_g32(q_int8, scales, shape, group_size=32, out_buffer=out_buf)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) / runs * 1000.0
        
        effective_io_bytes = total_elements * 3.0625
        throughput_gb_s = (effective_io_bytes / (1024**3)) / (fused_ms / 1000.0)
        speedup = naive_ms / (fused_ms + 1e-9)
        
        table_data.append([
            f"B = {B}",
            f"{total_elements:,}",
            f"{raw_bf16_mb:.1f} MB",
            f"{compressed_mb:.1f} MB",
            f"{naive_ms:.2f} ms",
            f"{fused_ms:.2f} ms",
            f"{speedup:.2f}x",
            f"{throughput_gb_s:.1f} GB/s"
        ])
        
    headers = ["Batch Size", "Elements", "Raw BF16", "Compressed", "Naive PyTorch", "Fused Triton", "Speedup", "VRAM Bandwidth"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))


def benchmark_high_throughput_ingestion(device="cuda:0"):
    print("\n" + "="*85)
    print(f"[*] BENCHMARK 2: High-Throughput Memory-Mapped Ingestion (5,000 DINOv3 Samples)")
    print("="*85)
    
    num_samples = 5000
    seq_len = 446
    dim = 768
    batch_size = 128
    
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "bench_features"
        
        writer = FeatureCacheWriter(prefix, num_samples=num_samples, seq_len=seq_len, dim=dim, group_size=32)
        dummy_batch = torch.randn(100, seq_len, dim, dtype=torch.bfloat16)
        for _ in range(50):
            writer.append(dummy_batch)
        writer.close()
        
        ds = FeatureCacheDataset(prefix)
        
        # Warmup
        for batch in ds.iter_batches(batch_size=batch_size, device=device):
            break
            
        t0 = time.perf_counter()
        count = 0
        for batch_bf16 in ds.iter_batches(batch_size=batch_size, device=device):
            count += batch_bf16.shape[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        fps = count / elapsed
        mb_per_sec = (count * seq_len * dim * 2.0 / (1024**2)) / elapsed
        
        ds.close()
        writer.close()
        
        table = [
            ["TensorCache iter_batches (C-level mmap slice + Fused GPU Dequant)", f"{fps:.1f} samples/sec", f"{mb_per_sec:.1f} MB/s", f"{elapsed:.2f} s"]
        ]
        headers = ["DataLoader Engine", "Throughput (FPS)", "Equivalent BF16 Bandwidth", "Time for 5,000 Samples"]
        print(tabulate(table, headers=headers, tablefmt="github"))


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    benchmark_dequant_scaling(device=device)
    benchmark_high_throughput_ingestion(device=device)
    print("\n" + "="*85)
    print("                     ALL SPEED TESTS COMPLETED!")
    print("="*85)


if __name__ == "__main__":
    main()
