"""
Unit Tests for BlockwiseInt8Codec and TensorCache.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

import math
from tensorcache.codec import (
    BlockwiseInt8Codec,
    quantize_int8_g32,
    dequantize_int8_g32,
    quantize_int8_adaptive,
    quantize_int4_g32,
    dequantize_int4_g32,
    quantize_int3_g32,
    dequantize_int3_g32,
    quantize_pixel_wavelet8x,
    dequantize_pixel_wavelet8x,
)
from tensorcache.feature_cache import FeatureCacheWriter, FeatureCacheDataset
from tensorcache.pixel_cache import PixelCacheWriter, PixelCacheDataset
from tensorcache.prefetcher import AsyncGPUPrefetcher


def test_quantize_dequantize_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(16, 446, 768, dtype=torch.bfloat16)
    
    # 1. Standard INT8 G=32
    q, s, shape = quantize_int8_g32(x, group_size=32)
    rec = dequantize_int8_g32(q, s, shape, group_size=32)
    
    diff = x.float() - rec.float()
    rel_rmse = (torch.norm(diff) / torch.norm(x.float())).item() * 100.0
    
    assert rec.shape == x.shape
    assert rec.dtype == torch.bfloat16
    assert rel_rmse < 1.0  # Error must be < 1% (empirically ~0.54%)
    print(f"Standard G=32 Roundtrip RMSE: {rel_rmse:.3f}%")


def test_adaptive_quantize_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(8, 128, 768, dtype=torch.bfloat16)
    
    q, s, shape = quantize_int8_adaptive(x, group_size=32)
    rec = dequantize_int8_g32(q, s, shape, group_size=32)
    
    diff = x.float() - rec.float()
    rel_rmse = (torch.norm(diff) / torch.norm(x.float())).item() * 100.0
    
    assert rec.shape == x.shape
    assert rel_rmse < 1.0
    print(f"Adaptive G=32 Roundtrip RMSE: {rel_rmse:.3f}%")


def test_feature_cache_disk_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "test_feat_cache"
        num_samples = 20
        seq_len = 100
        dim = 256
        
        # 1. Write
        writer = FeatureCacheWriter(prefix, num_samples=num_samples, seq_len=seq_len, dim=dim, group_size=32)
        fake_features = torch.randn(num_samples, seq_len, dim, dtype=torch.bfloat16)
        writer.append(fake_features)
        writer.close()
        
        # 2. Read
        dataset = FeatureCacheDataset(prefix)
        assert len(dataset) == num_samples
        
        # Index single item
        q_int8, scales = dataset[0]
        rec = dequantize_int8_g32(q_int8, scales, (seq_len, dim), group_size=32)
        
        assert rec.shape == (seq_len, dim)
        assert rec.dtype == torch.bfloat16
        dataset.close()
        writer.close()
        print("[+] FeatureCache disk I/O test passed!")


def test_pixel_cache_disk_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "test_pixel_cache"
        num_samples = 10
        H, W, C = 64, 64, 3
        
        # 1. Write
        writer = PixelCacheWriter(prefix, num_samples=num_samples, height=H, width=W, channels=C)
        fake_imgs = np.random.randint(0, 256, size=(num_samples, H, W, C), dtype=np.uint8)
        for i in range(num_samples):
            writer.append_image(fake_imgs[i])
        writer.close()
        
        # 2. Read
        dataset = PixelCacheDataset(prefix)
        assert len(dataset) == num_samples
        
        img0 = dataset[0]
        assert img0.shape == (H, W, C)
        assert img0.dtype == torch.uint8
        assert np.array_equal(img0.numpy(), fake_imgs[0])
        dataset.close()
        writer.close()
        print("[+] PixelCache raw disk I/O test passed!")


def test_int4_int3_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(8, 64, 768, dtype=torch.bfloat16)

    # 1. INT4 G=32
    q4, s4, shape4 = quantize_int4_g32(x, group_size=32)
    rec4 = dequantize_int4_g32(q4, s4, shape4, group_size=32)
    diff4 = x.float() - rec4.float()
    rmse4 = (torch.norm(diff4) / torch.norm(x.float())).item() * 100.0
    assert rec4.shape == x.shape
    assert rec4.dtype == torch.bfloat16
    assert rmse4 < 10.0  # INT4 has ~4-6% rel RMSE
    print(f"[+] INT4 G=32 Roundtrip RMSE: {rmse4:.3f}%")

    # 2. INT3 G=32
    q3, s3, shape3 = quantize_int3_g32(x, group_size=32)
    rec3 = dequantize_int3_g32(q3, s3, shape3, group_size=32)
    diff3 = x.float() - rec3.float()
    rmse3 = (torch.norm(diff3) / torch.norm(x.float())).item() * 100.0
    assert rec3.shape == x.shape
    assert rec3.dtype == torch.bfloat16
    assert rmse3 < 25.0  # INT3 on Gaussian noise has ~22% rel RMSE
    print(f"[+] INT3 G=32 Roundtrip RMSE: {rmse3:.3f}%")


def test_pixel_cache_quantized_disk_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        num_samples = 5
        H, W, C = 64, 64, 3
        fake_imgs = np.random.randint(0, 256, size=(num_samples, H, W, C), dtype=np.uint8)

        # Test INT4 pixel cache
        p4 = Path(tmpdir) / "test_pixel_int4"
        w4 = PixelCacheWriter(p4, num_samples=num_samples, height=H, width=W, channels=C, quant="int4")
        for i in range(num_samples):
            w4.append_image(fake_imgs[i])
        w4.close()

        ds4 = PixelCacheDataset(p4)
        assert len(ds4) == num_samples
        img4 = ds4[0]
        assert img4.shape == (H, W, C)
        assert img4.dtype == torch.uint8
        ds4.close()

        # Test INT3 pixel cache
        p3 = Path(tmpdir) / "test_pixel_int3"
        w3 = PixelCacheWriter(p3, num_samples=num_samples, height=H, width=W, channels=C, quant="int3")
        for i in range(num_samples):
            w3.append_image(fake_imgs[i])
        w3.close()

        ds3 = PixelCacheDataset(p3)
        assert len(ds3) == num_samples
        img3 = ds3[0]
        assert img3.shape == (H, W, C)
        assert img3.dtype == torch.uint8
        ds3.close()

        print("[+] PixelCache INT4/INT3 disk I/O test passed!")


def test_wavelet_8x_codec():
    torch.manual_seed(42)
    H, W, C = 64, 64, 3
    # Natural image proxy with spatial correlation
    raw = torch.randint(0, 256, (H, W, C), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = torch.nn.functional.avg_pool2d(raw, kernel_size=3, stride=1, padding=1).squeeze(0).permute(1, 2, 0).byte()
    
    packed_meta, shape = quantize_pixel_wavelet8x(img, q_scale=1.0)
    rec = dequantize_pixel_wavelet8x(packed_meta, device="cpu")
    
    assert rec.shape == (H, W, C)
    assert rec.dtype == torch.uint8
    diff = img.float() - rec.float()
    mse = (diff ** 2).mean().item()
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')
    rmse = (math.sqrt(mse) / 255.0) * 100.0
    
    assert psnr > 35.0
    print(f"[+] 8x Wavelet Codec Verified! PSNR: {psnr:.2f} dB, Rel RMSE: {rmse:.2f}%")


if __name__ == "__main__":
    test_quantize_dequantize_roundtrip()
    test_adaptive_quantize_roundtrip()
    test_feature_cache_disk_io()
    test_pixel_cache_disk_io()
    test_int4_int3_roundtrip()
    test_pixel_cache_quantized_disk_io()
    test_wavelet_8x_codec()

    print("\n[+] ALL UNIT TESTS PASSED SUCCESSFULLY!")
