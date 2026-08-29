"""
Triton Dequantizer Hardware Auto-Tuner.
Tests block sizes and warp counts to maximize VRAM memory bus saturation.
"""

import time
import torch
import triton
import triton.language as tl
from tabulate import tabulate


@triton.jit
def _opt_dequant_kernel(
    int8_ptr, scales_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Vectorized 1-pass load & convert
    vals_i8 = tl.load(int8_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scale_idx = offsets // GROUP_SIZE
    scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)

    out_bf16 = (vals_i8 * scales).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out_bf16, mask=mask)


def main():
    device = "cuda:0"
    print("="*80)
    print(f"[*] TUNING TRITON DEQUANT KERNEL ON: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    total_elements = 128 * 446 * 768 # 43.8M elements (83.6MB in BF16)
    q_int8 = torch.randint(-128, 127, (total_elements,), dtype=torch.int8, device=device)
    scales = torch.rand(total_elements // 32, dtype=torch.bfloat16, device=device)
    out = torch.empty(total_elements, dtype=torch.bfloat16, device=device)

    rows = []
    best_bw = 0.0
    best_config = None

    for bs in [128, 256, 512, 1024]:
        for warps in [2, 4, 8]:
            grid = (triton.cdiv(total_elements, bs),)
            # Warmup
            for _ in range(10):
                _opt_dequant_kernel[grid](q_int8, scales, out, total_elements, BLOCK_SIZE=bs, GROUP_SIZE=32, num_warps=warps)
            torch.cuda.synchronize()
            
            runs = 50
            t0 = time.perf_counter()
            for _ in range(runs):
                _opt_dequant_kernel[grid](q_int8, scales, out, total_elements, BLOCK_SIZE=bs, GROUP_SIZE=32, num_warps=warps)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / runs * 1000.0
            
            effective_io_bytes = total_elements * 3.0625
            gb_s = (effective_io_bytes / (1024**3)) / (dt / 1000.0)
            
            if gb_s > best_bw:
                best_bw = gb_s
                best_config = (bs, warps)
                
            rows.append([f"BLOCK_SIZE={bs}", f"warps={warps}", f"{dt:.3f} ms", f"{gb_s:.1f} GB/s"])

    headers = ["Tile Block Size", "Warp Count", "Latency (43.8M elements)", "Effective Bandwidth"]
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print(f"\n[+] Optimal Hardware Configuration: BLOCK_SIZE={best_config[0]}, num_warps={best_config[1]} ({best_bw:.1f} GB/s)")


if __name__ == "__main__":
    main()
