"""TensorCache CLI — `python -m tensorcache` or `tensorcache`."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

from . import __version__
from .codec import AMO_BQ_PRESETS
from .utils import estimate_compression, benchmark_tensor

def _parse_shape(s: str):
    return tuple(int(x) for x in s.split(","))

def cmd_info(args):
    import torch
    print(f"TensorCache {__version__}")
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"  device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    try:
        import triton
        print(f"triton {triton.__version__} available")
    except Exception:
        print("triton not available (CPU fallback active)")
    print("\nAMO-BQ presets (G32, 1.09375 B/elem, 1.83x vs BF16):")
    for k,(N,lo,hi,desc) in AMO_BQ_PRESETS.items():
        print(f"  {k:8s} N={N:2d} [{lo:.2f},{hi:.2f}]  {desc}")
    print("\nSym G32 1.0625 B 1.88x vs BF16 0.54%, G16 1.1875 B 0.39%")
    print("\nQuant guidance:")
    print("  Features (fragile, need <2% RMSE): INT8 G32 0.70% 1.88x or AMO 0.47% 1.83x, INT7 G32 1.35% 2.13x max sub-2%")
    print("    INT4/INT3 for features blocked (12%/28% >>2% collapse) -> use INT8/INT7 only")
    print("  Pixels (robust, PSNR>30dB): raw 1B 1.0x, INT4 G32 0.5B 2x PSNR 37dB, INT3 0.375B 2.29x PSNR 31dB")
    print("    PixelCacheWriter(..., quant='int4'/'int3'/'raw') or quant_bits=4/3/8, group_size=32")
    print("  Second stage (lossless on q): RLE+4b+outlier 1.45x boring 5.33x const, 1.0x diverse fallback")
    print("\nTry: tensorcache benchmark --shape 16,446,768 --device cuda")
    print("     tensorcache cache-info --prefix ./cache/pixels  # auto-detects raw/int4/int3")

def cmd_benchmark(args):
    shape=_parse_shape(args.shape)
    device=args.device
    if device=="cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device="cpu"
    # synthetic proxy if no real data
    torch.manual_seed(42)
    print(f"Benchmark {shape} device={device} group_size={args.group_size}")
    x=torch.randn(shape, dtype=torch.bfloat16, device=device)
    benchmark_tensor(x, group_size=args.group_size, device=None, verbose=True)
    est=estimate_compression(shape, args.group_size, amo_bq=True)
    print(f"\nEstimate: {est['total_MB']:.2f} MB total, {est['bytes_per_elem']:.4f} B/elem, ratio vs BF16 {est['ratio_vs_bf16']:.2f}x")

def cmd_cache_info(args):
    prefix=Path(args.prefix)
    # Try feature meta, then pixel meta, then shards
    candidates = [Path(str(prefix)+"_meta.json"), Path(str(prefix)+"_pixel_meta.json"), Path(str(prefix)+"_shards.json")]
    meta_path = next((p for p in candidates if p.exists()), None)
    if meta_path is None:
        # Try pixel meta without suffix
        alt = Path(str(prefix)+"_pixel_meta.json")
        if alt.exists():
            meta_path=alt
        else:
            print(f"Meta not found: tried {candidates}")
            sys.exit(1)
    meta=json.loads(meta_path.read_text())
    print(f"# {meta_path}")
    print(json.dumps(meta, indent=2))
    # Show quant info if present
    if "quant" in meta:
        print(f"\nQuant: {meta['quant']} bits={meta.get('quant_bits')} group_size={meta.get('group_size')}")
    if "quant_bits" in meta:
        print(f"Quant bits: {meta['quant_bits']}")
    # estimate files
    for key in ["int8_file","scales_file","zp_file","bin_file","q_file","scales_file"]:
        fname=meta.get(key)
        if fname:
            for cand in [prefix.parent / fname, Path(str(prefix)+"_int8.bin"), Path(str(prefix)+"_scales.bin"), Path(str(prefix)+"_zp.bin"), Path(str(prefix)+"_pixels.bin"), Path(str(prefix)+f"_pixels_int{meta.get('quant_bits')}.bin")]:
                if cand.exists():
                    print(f"{key}: {cand} {cand.stat().st_size/1024/1024:.2f} MB")
                    break
    # Also direct check for pixel quant files
    for suffix in ["_pixels.bin", "_pixels_int4.bin", "_pixels_int3.bin", "_pixels_int4_scales.bin", "_pixels_int3_scales.bin", "_int8.bin", "_scales.bin", "_zp.bin"]:
        p = Path(str(prefix)+suffix)
        if p.exists():
            print(f"{p.name}: {p.stat().st_size/1024/1024:.2f} MB")

def cmd_compress_demo(args):
    print("Demo compress/decompress (synthetic):")
    torch.manual_seed(0)
    shape=_parse_shape(args.shape)
    device=args.device
    x=torch.randn(shape, dtype=torch.bfloat16, device=device if (device!="cuda" or torch.cuda.is_available()) else "cpu")
    from .utils import compress, decompress
    q,s,zp,sh=compress(x, mode=args.mode, group_size=args.group_size)
    rec=decompress(q,s,sh,zp, args.group_size)
    rel=(torch.norm(x.float()-rec.float())/torch.norm(x.float())).item()*100
    print(f"mode={args.mode} G={args.group_size} shape={shape} rel={rel:.4f}% q={q.dtype} s={s.dtype} zp={zp.dtype if zp is not None else None}")

def build_parser():
    p=argparse.ArgumentParser(
        prog="tensorcache",
        description="TensorCache — ultra-fast INT8 feature & pixel cache (AMO-BQ, INT4/INT3 pixel)",
        epilog="Examples:\n"
               "  tensorcache info\n"
               "  tensorcache benchmark --shape 16,446,768 --device cuda --group-size 32\n"
               "  tensorcache compress-demo --shape 4,197,768 --mode balanced --device cuda\n"
               "  tensorcache cache-info --prefix ./cache/dinov3\n"
               "  python -c \"from tensorcache import PixelCacheWriter; w=PixelCacheWriter('./cache/pixels',1000,224,224,3,quant='int4')\"\n"
               "  # Feature caches: INT8 only (INT4/INT3 guarded: >2%% RMSE collapse, use PixelCache for INT4/INT3)\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=__version__)
    sub=p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    pi=sub.add_parser("info", help="show version, device, presets, and quant notes")
    pi.description = "Show version, CUDA/Triton, AMO-BQ presets, and quant guidance (INT8 for features, INT4/INT3 for pixels)."
    pi.set_defaults(func=cmd_info)

    pb=sub.add_parser("benchmark", help="bench synthetic tensor (INT8/AMO-BQ)")
    pb.description = "Benchmark synthetic BF16 tensor through Blockwise INT8 encode/decode and report RMSE/BW."
    pb.add_argument("--shape", default="16,446,768", help="comma shape e.g. 16,446,768 (default: %(default)s)")
    pb.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="device for bench (default: %(default)s)")
    pb.add_argument("--group-size", type=int, default=32, dest="group_size", help="group size G (default: %(default)s) - 32 for 1.88x 0.70%%, 16 for 0.39%%")
    pb.set_defaults(func=cmd_benchmark)

    pc=sub.add_parser("cache-info", help="inspect cache prefix meta (_meta.json / _pixel_meta.json)")
    pc.description = "Inspect feature or pixel cache meta and file sizes. Supports raw, int4, int3 pixel caches and int8 feature caches."
    pc.add_argument("--prefix", required=True, help="cache prefix e.g. ./cache/dinov3 or ./cache/pixels")
    pc.set_defaults(func=cmd_cache_info)

    pd=sub.add_parser("compress-demo", help="demo AMO-BQ compress on synthetic data")
    pd.description = "Demo compress/decompress with AMO-BQ presets (fast/balanced/accurate/max) or sym/adaptive."
    pd.add_argument("--shape", default="4,197,768", help="tensor shape (default: %(default)s)")
    pd.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="device")
    pd.add_argument("--mode", default="balanced", choices=list(AMO_BQ_PRESETS.keys())+["sym","adaptive"], help="AMO preset or sym/adaptive (default: %(default)s)")
    pd.add_argument("--group-size", type=int, default=32, dest="group_size", help="group size (default: %(default)s)")
    pd.set_defaults(func=cmd_compress_demo)

    return p

def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    args.func(args)

if __name__=="__main__":
    main()
