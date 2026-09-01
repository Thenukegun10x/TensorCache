"""
TensorCache: Ultra-fast, high-fidelity block-wise INT8 feature & pixel cache engine for ML.
Licensed under the Apache License, Version 2.0.
"""

from .codec import (
    BlockwiseInt8Codec,
    quantize_int8_g32,
    dequantize_int8_g32,
    quantize_int8_adaptive,
    quantize_int8_amo_bq,
    dequantize_int8_amo_bq,
    AMO_BQ_PRESETS,
    pack_int4,
    unpack_int4,
    pack_int3,
    unpack_int3,
    quantize_int4_g32,
    dequantize_int4_g32,
    quantize_int3_g32,
    dequantize_int3_g32,
    rct_forward,
    rct_inverse,
    quantize_pixel_wavelet8x,
    dequantize_pixel_wavelet8x,
    quantize_pixel_wavelet_adaptive,
    dequantize_pixel_wavelet_adaptive,
    ADAPTIVE_CODEBOOK,
    WAVELET_ADAPTIVE_PRESETS,
)
from .utils import (
    compress,
    decompress,
    estimate_compression,
    benchmark_tensor,
    auto_select_mode,
    help_text,
)
from .fused_ops import (
    quantize_fused_gpu,
    dequantize_fused_gpu,
    quantize_fused_int4_gpu,
    dequantize_fused_int4_gpu,
    quantize_fused_int3_gpu,
    dequantize_fused_int3_gpu,
    quantize_fused_wavelet8x_gpu,
    dequantize_fused_wavelet8x_gpu,
    quantize_fused_wavelet_adaptive_gpu,
    dequantize_fused_wavelet_adaptive_gpu,
    FusedDequantLinear
)
from .prefetcher import AsyncGPUPrefetcher
from .feature_cache import (
    FeatureCacheWriter,
    FeatureCacheDataset
)
from .pixel_cache import (
    PixelCacheWriter,
    PixelCacheDataset
)
from .streamer import ZeroCopyTensorStreamer

__version__ = "0.3.0"
__all__ = [
    "BlockwiseInt8Codec",
    "quantize_int8_g32",
    "dequantize_int8_g32",
    "quantize_int8_adaptive",
    "quantize_int8_amo_bq",
    "dequantize_int8_amo_bq",
    "AMO_BQ_PRESETS",
    "pack_int4",
    "unpack_int4",
    "pack_int3",
    "unpack_int3",
    "quantize_int4_g32",
    "dequantize_int4_g32",
    "quantize_int3_g32",
    "dequantize_int3_g32",
    "rct_forward",
    "rct_inverse",
    "quantize_pixel_wavelet8x",
    "dequantize_pixel_wavelet8x",
    "quantize_pixel_wavelet_adaptive",
    "dequantize_pixel_wavelet_adaptive",
    "ADAPTIVE_CODEBOOK",
    "WAVELET_ADAPTIVE_PRESETS",
    "compress",
    "decompress",
    "estimate_compression",
    "benchmark_tensor",
    "auto_select_mode",
    "help_text",
    "quantize_fused_gpu",
    "dequantize_fused_gpu",
    "quantize_fused_int4_gpu",
    "dequantize_fused_int4_gpu",
    "quantize_fused_int3_gpu",
    "dequantize_fused_int3_gpu",
    "quantize_fused_wavelet8x_gpu",
    "dequantize_fused_wavelet8x_gpu",
    "quantize_fused_wavelet_adaptive_gpu",
    "dequantize_fused_wavelet_adaptive_gpu",
    "FusedDequantLinear",
    "AsyncGPUPrefetcher",
    "FeatureCacheWriter",
    "FeatureCacheDataset",
    "PixelCacheWriter",
    "PixelCacheDataset",
    "ZeroCopyTensorStreamer",
]

def help():
    """Print friendly help (also `python -m tensorcache info`)."""
    print(help_text())
