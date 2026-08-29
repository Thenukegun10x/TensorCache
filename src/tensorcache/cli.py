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
    print("\nTry: tensorcache benchmark --shape 16,446,768 --device cuda")

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
    meta_path=Path(str(prefix)+"_meta.json")
    if not meta_path.exists():
        print(f"Meta not found: {meta_path}")
        sys.exit(1)
    meta=json.loads(meta_path.read_text())
    print(json.dumps(meta, indent=2))
    # estimate files
    for key in ["int8_file","scales_file","zp_file"]:
        fname=meta.get(key)
        if fname:
            fpath=prefix.parent / fname if (prefix.parent / fname).exists() else Path(str(prefix).rsplit("_",1)[0]+f"_{key.split('_')[0]}.bin") if False else None
            # try direct
            cand=Path(str(prefix)+"_int8.bin") if key=="int8_file" else Path(str(prefix)+"_scales.bin") if key=="scales_file" else Path(str(prefix)+"_zp.bin")
            if cand.exists():
                print(f"{key}: {cand.stat().st_size/1024/1024:.2f} MB")

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
    p=argparse.ArgumentParser(prog="tensorcache", description="TensorCache — ultra-fast INT8 feature cache (AMO-BQ)")
    p.add_argument("--version", action="version", version=__version__)
    sub=p.add_subparsers(dest="cmd", required=True)

    pi=sub.add_parser("info", help="show version, device, presets")
    pi.set_defaults(func=cmd_info)

    pb=sub.add_parser("benchmark", help="bench synthetic tensor")
    pb.add_argument("--shape", default="16,446,768", help="comma shape e.g. 16,446,768")
    pb.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="device")
    pb.add_argument("--group-size", type=int, default=32, dest="group_size")
    pb.set_defaults(func=cmd_benchmark)

    pc=sub.add_parser("cache-info", help="inspect cache prefix meta")
    pc.add_argument("--prefix", required=True, help="cache prefix e.g. ./cache/feat")
    pc.set_defaults(func=cmd_cache_info)

    pd=sub.add_parser("compress-demo", help="demo AMO-BQ compress")
    pd.add_argument("--shape", default="4,197,768")
    pd.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    pd.add_argument("--mode", default="balanced", choices=list(AMO_BQ_PRESETS.keys())+["sym","adaptive"])
    pd.add_argument("--group-size", type=int, default=32, dest="group_size")
    pd.set_defaults(func=cmd_compress_demo)

    return p

def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    args.func(args)

if __name__=="__main__":
    main()
