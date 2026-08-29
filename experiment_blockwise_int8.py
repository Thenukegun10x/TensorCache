"""
Targeted Experiment Suite: Optimizing Block-wise Quantization for DINOv3 Features.
Explores:
  1. Block Size Sweep (G = 8, 16, 32, 64, 128, 256, 768)
  2. Symmetric (Max-Abs) vs Asymmetric (Min-Max with Zero-Point)
  3. Axis Alignment (Per-Token vs Per-Channel vs Flat)
  4. Scale Precision (BF16 2-byte scale vs 8-bit scale)
  5. Bitwidth Tradeoffs (INT6, INT7, INT8)
"""

import sys
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tabulate import tabulate

def compute_metrics(gt_fp32: torch.Tensor, rec: torch.Tensor):
    gt = gt_fp32.float()
    rec = rec.float()
    diff = gt - rec
    
    gt_norm = torch.norm(gt).item()
    diff_norm = torch.norm(diff).item()
    rel_rmse = (diff_norm / (gt_norm + 1e-12)) * 100.0
    mape = (diff.abs().mean() / (gt.abs().mean() + 1e-12)).item() * 100.0
    
    data_range = (gt.max() - gt.min()).item()
    mse = (diff ** 2).mean().item()
    psnr = 20.0 * math.log10(data_range / math.sqrt(mse)) if mse > 0 else float("inf")
    
    k_outliers = max(1, int(gt.numel() * 0.001))
    top_indices = torch.topk(gt.abs().flatten(), k=k_outliers).indices
    outlier_gt = gt.flatten()[top_indices]
    outlier_rec = rec.flatten()[top_indices]
    outlier_err = ((outlier_gt - outlier_rec).abs().mean() / (outlier_gt.abs().mean() + 1e-12)).item() * 100.0
    
    return rel_rmse, mape, psnr, outlier_err

# 1. Symmetric Block-wise Quantization
def quant_symm_int(x: torch.Tensor, block_size: int, bits: int = 8, scale_bytes: int = 2):
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    
    x_flat = x.flatten().float()
    pad_len = (block_size - (x_flat.numel() % block_size)) % block_size
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len))
        
    x_blocks = x_flat.view(-1, block_size)
    block_max = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    
    if scale_bytes == 1:
        # 8-bit scale (FP8 E4M3 representation for scale)
        sc = (block_max / qmax).to(torch.float8_e4m3fn).float()
    else:
        sc = (block_max / qmax).to(torch.bfloat16).float()
        
    scaled = x_blocks / sc
    q = torch.round(scaled).clamp(qmin, qmax)
    rec_blocks = q * sc
    rec = rec_blocks.flatten()[:x.numel()].view(x.shape)
    
    bytes_per_elem = (bits / 8.0) + (scale_bytes / float(block_size))
    return rec, bytes_per_elem

# 2. Asymmetric (Min-Max + Zero-Point) Block-wise Quantization
def quant_asymm_int(x: torch.Tensor, block_size: int, bits: int = 8):
    qmax = (1 << bits) - 1
    
    x_flat = x.flatten().float()
    pad_len = (block_size - (x_flat.numel() % block_size)) % block_size
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len))
        
    x_blocks = x_flat.view(-1, block_size)
    b_min = x_blocks.amin(dim=-1, keepdim=True)
    b_max = x_blocks.amax(dim=-1, keepdim=True)
    
    sc = ((b_max - b_min) / float(qmax)).clamp(min=1e-8).to(torch.bfloat16).float()
    zp = torch.round(-b_min / sc).clamp(0, qmax).to(torch.uint8).float()
    
    q = torch.round(x_blocks / sc + zp).clamp(0, qmax)
    rec_blocks = (q - zp) * sc
    rec = rec_blocks.flatten()[:x.numel()].view(x.shape)
    
    # 1 byte data + 2 bytes scale / G + 1 byte zero-point / G
    bytes_per_elem = (bits / 8.0) + (3.0 / float(block_size))
    return rec, bytes_per_elem

# 3. Axis-Aligned Per-Token Block-wise Quantization ([B, N, D])
def quant_token_aligned_int8(x: torch.Tensor, block_size: int = 32):
    # x is [B, N, D] where D is channels (768)
    B, N, D = x.shape
    x_reshaped = x.view(-1, D).float()
    # Chunk D into blocks
    num_blocks = D // block_size
    x_blocks = x_reshaped.view(-1, num_blocks, block_size)
    b_max = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = (b_max / 127.0).to(torch.bfloat16).float()
    
    q = torch.round(x_blocks / sc).clamp(-128, 127)
    rec_blocks = q * sc
    rec = rec_blocks.view(B, N, D)
    bytes_per_elem = 1.0 + (2.0 / float(block_size))
    return rec, bytes_per_elem

def main():
    print("="*85)
    print("      DINOv3 FEATURE CACHE COMPRESSION OPTIMIZATION SWEEP")
    print("="*85)
    
    # 1. Load Pretrained DINOv3 Stem & Extract Features
    sys.path.append(r"..\Plant identifier")
    import timm
    model = timm.create_model("vit_base_patch16_dinov3", pretrained=False).cuda().eval()
    
    # Sample images
    plant_data_dir = Path(r"..\Plant identifier\data\wa_plants_200k\train")
    imgs = list(plant_data_dir.rglob("*.jpg"))[:64]
    
    batch = []
    for p in imgs:
        with Image.open(p) as im:
            im = im.convert("RGB").resize((336, 336), Image.Resampling.BILINEAR)
            arr = (np.array(im, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            batch.append(torch.from_numpy(arr).permute(2, 0, 1).float())
            
    x = torch.stack(batch).cuda()
    with torch.no_grad():
        stem = model.forward_features(x)
        if isinstance(stem, (tuple, list)): stem = stem[0]
        gt_fp32 = stem.float().cpu()
        
    print(f"[*] Extracted DINOv3 ViT features: Shape {list(gt_fp32.shape)} ({gt_fp32.numel():,} elements)\n")
    
    # -------------------------------------------------------------
    # Experiment 1: Block Size Sweep (Symmetric INT8)
    # -------------------------------------------------------------
    print("[*] EXPERIMENT 1: Group / Block Size Sweep (Symmetric INT8, 2-byte Scale)")
    exp1_rows = []
    for G in [8, 16, 32, 64, 128, 256, 768]:
        rec, b_elem = quant_symm_int(gt_fp32, block_size=G, bits=8, scale_bytes=2)
        rmse, mape, psnr, out_err = compute_metrics(gt_fp32, rec)
        ratio_bf16 = 2.0 / b_elem
        exp1_rows.append([
            f"Block Size = {G}", f"{b_elem:.4f} B", f"{ratio_bf16:.2f}x",
            f"{rmse:.3f}%", f"{mape:.3f}%", f"{psnr:.2f} dB", f"{out_err:.3f}%"
        ])
    print(tabulate(exp1_rows, headers=["Configuration", "Bytes/Val", "Ratio vs BF16", "Rel RMSE %", "MAPE %", "PSNR", "Outlier Err%"], tablefmt="github"))
    
    # -------------------------------------------------------------
    # Experiment 2: Symmetric vs Asymmetric (Min-Max + Zero Point)
    # -------------------------------------------------------------
    print("\n[*] EXPERIMENT 2: Symmetric (Max-Abs) vs Asymmetric (Min-Max + Zero Point)")
    exp2_rows = []
    for G in [16, 32, 64]:
        # Symmetric
        rec_s, b_s = quant_symm_int(gt_fp32, block_size=G, bits=8, scale_bytes=2)
        rmse_s, mape_s, psnr_s, out_s = compute_metrics(gt_fp32, rec_s)
        exp2_rows.append([f"Symmetric (G={G})", f"{b_s:.4f} B", f"{2.0/b_s:.2f}x", f"{rmse_s:.3f}%", f"{mape_s:.3f}%", f"{psnr_s:.2f} dB", f"{out_s:.3f}%"])
        
        # Asymmetric
        rec_a, b_a = quant_asymm_int(gt_fp32, block_size=G, bits=8)
        rmse_a, mape_a, psnr_a, out_a = compute_metrics(gt_fp32, rec_a)
        exp2_rows.append([f"Asymmetric (G={G})", f"{b_a:.4f} B", f"{2.0/b_a:.2f}x", f"{rmse_a:.3f}%", f"{mape_a:.3f}%", f"{psnr_a:.2f} dB", f"{out_a:.3f}%"])
    print(tabulate(exp2_rows, headers=["Configuration", "Bytes/Val", "Ratio vs BF16", "Rel RMSE %", "MAPE %", "PSNR", "Outlier Err%"], tablefmt="github"))
    
    # -------------------------------------------------------------
    # Experiment 3: Scale Precision (BF16 2-byte scale vs FP8 1-byte scale)
    # -------------------------------------------------------------
    print("\n[*] EXPERIMENT 3: Scale Factor Precision (BF16 2-byte scale vs 8-bit scale)")
    exp3_rows = []
    for G in [16, 32, 64]:
        rec_2b, b_2b = quant_symm_int(gt_fp32, block_size=G, bits=8, scale_bytes=2)
        rmse_2b, mape_2b, psnr_2b, out_2b = compute_metrics(gt_fp32, rec_2b)
        exp3_rows.append([f"BF16 Scale (G={G})", f"{b_2b:.4f} B", f"{2.0/b_2b:.2f}x", f"{rmse_2b:.3f}%", f"{mape_2b:.3f}%", f"{psnr_2b:.2f} dB"])
        
        rec_1b, b_1b = quant_symm_int(gt_fp32, block_size=G, bits=8, scale_bytes=1)
        rmse_1b, mape_1b, psnr_1b, out_1b = compute_metrics(gt_fp32, rec_1b)
        exp3_rows.append([f"FP8-E4M3 Scale (G={G})", f"{b_1b:.4f} B", f"{2.0/b_1b:.2f}x", f"{rmse_1b:.3f}%", f"{mape_1b:.3f}%", f"{psnr_1b:.2f} dB"])
    print(tabulate(exp3_rows, headers=["Configuration", "Bytes/Val", "Ratio vs BF16", "Rel RMSE %", "MAPE %", "PSNR"], tablefmt="github"))

    # -------------------------------------------------------------
    # Experiment 4: Bitwidth Scaling (INT6 vs INT7 vs INT8 with G=32)
    # -------------------------------------------------------------
    print("\n[*] EXPERIMENT 4: Bitwidth Comparison (INT6, INT7, INT8 at G=32)")
    exp4_rows = []
    for bits in [6, 7, 8]:
        rec, b_elem = quant_symm_int(gt_fp32, block_size=32, bits=bits, scale_bytes=2)
        rmse, mape, psnr, out_err = compute_metrics(gt_fp32, rec)
        ratio_bf16 = 2.0 / b_elem
        exp4_rows.append([
            f"INT{bits} (Block=32)", f"{b_elem:.4f} B", f"{ratio_bf16:.2f}x",
            f"{rmse:.3f}%", f"{mape:.3f}%", f"{psnr:.2f} dB", f"{out_err:.3f}%"
        ])
    print(tabulate(exp4_rows, headers=["Bitwidth", "Bytes/Val", "Ratio vs BF16", "Rel RMSE %", "MAPE %", "PSNR", "Outlier Err%"], tablefmt="github"))

if __name__ == "__main__":
    main()
