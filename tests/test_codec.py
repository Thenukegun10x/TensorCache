"""
Unit Tests for BlockwiseInt8Codec and TensorCache.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from tensorcache.codec import (
    BlockwiseInt8Codec,
    quantize_int8_g32,
    dequantize_int8_g32,
    quantize_int8_adaptive
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
        print("[+] PixelCache disk I/O test passed!")


if __name__ == "__main__":
    test_quantize_dequantize_roundtrip()
    test_adaptive_quantize_roundtrip()
    test_feature_cache_disk_io()
    test_pixel_cache_disk_io()
    print("\n[+] ALL UNIT TESTS PASSED SUCCESSFULLY!")
