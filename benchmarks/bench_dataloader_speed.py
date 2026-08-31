"""
Dataloader Speed Benchmark for TensorCache
Compares all ingestion paths end-to-end:

  1. DataLoader(FeatureCacheDataset) naive (num_workers=0)
  2. DataLoader(FeatureCacheDataset, num_workers=2/4) + collate
  3. DataLoader + AsyncGPUPrefetcher
  4. FeatureCacheDataset.iter_batches (C-level batch slice)
  5. ZeroCopyTensorStreamer (pinned + double-buffered)
  6. FeatureCacheDataset(auto_dequant_device="cuda") + DataLoader

Also benchmarks PixelCacheDataset raw vs batched.

Reports: samples/sec, MB/s (BF16 equiv), ms/batch, p50/p95 latency.

Usage:
  /home/conorm/Desktop/Pipeline/.venv/bin/python benchmarks/bench_dataloader_speed.py
  /home/conorm/Desktop/Pipeline/.venv/bin/python benchmarks/bench_dataloader_speed.py --quick
  /home/conorm/Desktop/Pipeline/.venv/bin/python benchmarks/bench_dataloader_speed.py --num-samples 10000 --batch-size 128 --amo-bq --device cuda
"""

import argparse
import time
import tempfile
import statistics
import os
import gc
from pathlib import Path

import torch
import numpy as np

# allow running as script from repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorcache.feature_cache import FeatureCacheWriter, FeatureCacheDataset
from tensorcache.streamer import ZeroCopyTensorStreamer
from tensorcache.prefetcher import AsyncGPUPrefetcher
from tensorcache.codec import dequantize_int8_g32, dequantize_int8_amo_bq, quantize_int8_g32
from tensorcache.pixel_cache import PixelCacheWriter, PixelCacheDataset

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

def sync_if_cuda(device):
    if device.type in ("cuda", "hip") and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def fmt(v, n=1):
    return f"{v:.{n}f}"

def benchmark_fn(name, fn, num_samples, seq_len, dim, batch_size, device, warmup=1, runs=3):
    """Run fn() that yields batches and returns timing stats."""
    # warmup
    for _ in range(warmup):
        cnt = 0
        for batch in fn():
            # consume - prevent optimization
            if isinstance(batch, torch.Tensor):
                cnt += batch.shape[0]
                # touch data
                _ = batch.sum()
            elif isinstance(batch, (list, tuple)):
                # q,scales or q,scales,zp
                cnt += batch[0].shape[0] if hasattr(batch[0], "shape") else len(batch[0])
            elif isinstance(batch, dict):
                cnt += next(iter(batch.values())).shape[0]
            else:
                cnt += 1
        sync_if_cuda(device)

    latencies = []
    throughputs = []
    all_batch_lat = []  # per-batch lat for p50/p95

    for _ in range(runs):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        batch_times = []
        cnt = 0
        last_t = time.perf_counter()
        for batch in fn():
            # per-batch latency
            now = time.perf_counter()
            # first batch includes init overhead, but we capture all
            batch_times.append((now - last_t)*1000)
            last_t = now
            if isinstance(batch, torch.Tensor):
                cnt += batch.shape[0]
                # simulate compute keep on gpu
                if batch.is_cuda:
                    _ = batch.sum()
            elif isinstance(batch, (list, tuple)):
                # dequantize if needed to simulate real training
                first = batch[0]
                if isinstance(first, torch.Tensor):
                    cnt += first.shape[0]
                else:
                    cnt += len(batch)
            else:
                cnt += batch_size
        sync_if_cuda(device)
        dt = time.perf_counter() - t0
        # drop first batch latency outlier (cold)
        if len(batch_times) > 2:
            batch_times = batch_times[1:]
        all_batch_lat.extend(batch_times)
        latencies.append(dt*1000)
        throughputs.append(cnt / dt if dt > 0 else 0)

    median_ms = statistics.median(latencies)
    median_fps = statistics.median(throughputs)
    # BF16 equiv MB/s = samples * seq*dim*2 / time
    median_mbs = median_fps * seq_len * dim * 2 / (1024*1024)
    mean_batch_ms = statistics.mean(all_batch_lat) if all_batch_lat else 0
    p50 = statistics.median(all_batch_lat) if all_batch_lat else 0
    p95 = sorted(all_batch_lat)[int(len(all_batch_lat)*0.95)] if len(all_batch_lat) > 5 else p50
    return {
        "name": name,
        "median_ms": median_ms,
        "median_fps": median_fps,
        "median_mbs": median_mbs,
        "mean_batch_ms": mean_batch_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="TensorCache Dataloader Speed Benchmark")
    parser.add_argument("--num-samples", type=int, default=5000, help="samples in synthetic cache")
    parser.add_argument("--seq-len", type=int, default=446)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-sizes", type=str, default="", help="comma list to sweep, e.g. 32,128,256 overrides --batch-size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--amo-bq", action="store_true", help="use AMO-BQ (uint8+zp) vs sym int8")
    parser.add_argument("--amo-mode", type=str, default="balanced", help="amo mode fast/balanced/accurate/max")
    parser.add_argument("--quick", action="store_true", help="tiny run for CI (500 samples, 1 run)")
    parser.add_argument("--num-workers-list", type=str, default="0,2", help="workers to test for DataLoader")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-pixel", action="store_true", help="skip pixel cache bench")
    args = parser.parse_args()

    if args.quick:
        args.num_samples = 1000
        args.runs = 2
        args.batch_sizes = "32,128"

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to cpu")
        device_str = "cpu"
    device = torch.device(device_str)
    if device.type == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    else:
        print(f"[*] Device: CPU | torch {torch.__version__}")

    batch_sizes = [args.batch_size]
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    num_workers_list = [int(x) for x in args.num_workers_list.split(",") if x.strip()]

    seq_len, dim, group_size = args.seq_len, args.dim, args.group_size
    num_samples = args.num_samples
    total_elem = seq_len * dim
    print(f"[*] Config: N={num_samples} seq={seq_len} dim={dim} (elem/sample={total_elem:,}) group={group_size} amo_bq={args.amo_bq} mode={args.amo_mode if args.amo_bq else 'sym'}")
    for bs in batch_sizes:
        print(f"[*] Batch sizes sweep: {batch_sizes}")

    # ------------------------------------------------------------------
    # 1. Create synthetic cache
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "bench_feat"
        print(f"\n[*] Writing cache {prefix} ({num_samples} samples)...")
        t0 = time.perf_counter()
        writer = FeatureCacheWriter(
            prefix, num_samples=num_samples, seq_len=seq_len, dim=dim,
            group_size=group_size, amo_bq=args.amo_bq, amo_mode=args.amo_mode if args.amo_bq else None
        )
        # batched write but writer currently per-sample loop (bench will show that)
        # generate in chunks of 100 to avoid OOM
        chunk = 100
        dummy = torch.randn(chunk, seq_len, dim, dtype=torch.bfloat16)
        n_chunks = (num_samples + chunk -1)//chunk
        for i in range(n_chunks):
            cur = chunk if (i+1)*chunk <= num_samples else num_samples - i*chunk
            writer.append(dummy[:cur])
            if i % 10 == 0:
                print(f"  wrote {min((i+1)*chunk, num_samples)}/{num_samples}", end="\r")
        writer.close()
        print(f"\n[+] Write done {time.perf_counter()-t0:.2f}s")
        # verify sizes
        for suffix in ["_int8.bin", "_scales.bin", "_zp.bin", "_meta.json"]:
            p = Path(str(prefix) + suffix)
            if p.exists():
                print(f"  {p.name}: {p.stat().st_size/1024/1024:.2f} MB")

        # Open dataset once for iterative reuse
        ds = FeatureCacheDataset(prefix)

        # Precompute shape for dequant
        scales_per_sample = (seq_len*dim + group_size -1)//group_size

        # ------------------------------------------------------------------
        # 2. Benchmark each batch_size
        # ------------------------------------------------------------------
        for bs in batch_sizes:
            print("\n" + "="*110)
            print(f"[*] BENCHMARK batch_size={bs} | device={device} | samples={num_samples}")
            print("="*110)
            results = []

            # Helper to get device for iter_batches/streamer
            dev_str = str(device)

            # ---- A. Naive DataLoader workers=0 ----
            for nw in num_workers_list:
                def make_loader_fn(nw_inner=nw):
                    def fn():
                        from torch.utils.data import DataLoader
                        # collate will stack q,scales (and zp)
                        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw_inner, pin_memory=False, drop_last=False,
                                            persistent_workers=False)
                        for batch in loader:
                            # simulate training dequant (most users will do this)
                            # batch is tuple of stacked tensors: (q [B,seq,dim], scales [B,scales])
                            if args.amo_bq:
                                q, s, zp = batch
                                # move to device and dequant
                                q = q.to(device, non_blocking=False)
                                s = s.to(device, non_blocking=False)
                                zp = zp.to(device, non_blocking=False)
                                rec = dequantize_int8_amo_bq(q, s, zp, (q.shape[0], seq_len, dim), group_size=group_size)
                                yield rec
                            else:
                                q, s = batch
                                q = q.to(device, non_blocking=False)
                                s = s.to(device, non_blocking=False)
                                rec = dequantize_int8_g32(q, s, (q.shape[0], seq_len, dim), group_size=group_size)
                                yield rec
                    return fn
                name = f"DataLoader nw={nw} + dequant"
                try:
                    res = benchmark_fn(name, make_loader_fn(), num_samples, seq_len, dim, bs, device, warmup=1, runs=args.runs)
                    results.append(res)
                except Exception as e:
                    print(f"[!] {name} failed: {e}")
                    import traceback; traceback.print_exc()

            # ---- B. DataLoader + AsyncGPUPrefetcher (overlap H2D) ----
            # only test nw=0 case, prefetcher wraps loader
            for nw in [0]:  # prefetcher typical uses nw=2+pin_memory, but we test base
                def make_prefetch_fn():
                    def fn():
                        from torch.utils.data import DataLoader
                        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
                        # Need custom collate to pin? prefetcher will move to device async
                        # But ds returns cpu tensors, need to ensure pin_memory true uses pinned staging
                        # Our ds tensors are not pinned, but DataLoader pin_memory will pin them
                        pre = AsyncGPUPrefetcher(loader, device=dev_str)
                        for batch in pre:
                            if args.amo_bq:
                                q, s, zp = batch
                                # batch already on device via prefetcher
                                rec = dequantize_int8_amo_bq(q, s, zp, (q.shape[0], seq_len, dim), group_size=group_size)
                                yield rec
                            else:
                                q, s = batch
                                rec = dequantize_int8_g32(q, s, (q.shape[0], seq_len, dim), group_size=group_size)
                                yield rec
                    return fn
                name = f"DataLoader+Prefetcher nw={nw} pin_mem"
                try:
                    res = benchmark_fn(name, make_prefetch_fn(), num_samples, seq_len, dim, bs, device, warmup=1, runs=args.runs)
                    results.append(res)
                except Exception as e:
                    print(f"[!] {name} failed: {e}")
                    import traceback; traceback.print_exc()

            # ---- C. iter_batches (C-level mmap batch slice) ----
            def make_iter_batches_fn():
                def fn():
                    # ds.iter_batches yields BF16 already on device if dequantize=True
                    for batch in ds.iter_batches(batch_size=bs, shuffle=False, device=dev_str, dequantize=True):
                        yield batch
                return fn
            try:
                res = benchmark_fn("iter_batches (C-slice+dequant)", make_iter_batches_fn(), num_samples, seq_len, dim, bs, device, warmup=1, runs=args.runs)
                results.append(res)
            except Exception as e:
                print(f"[!] iter_batches failed: {e}")
                import traceback; traceback.print_exc()

            # ---- D. ZeroCopyTensorStreamer (pinned ring) ----
            # Streamer creates its own mmap, so need to keep ds closed? streamer opens independent
            # We'll create new streamer per fn call to avoid state reuse issues inside benchmark_fn warmup/runs loop
            def make_streamer_fn():
                def fn():
                    streamer = ZeroCopyTensorStreamer(prefix, batch_size=bs, device=dev_str, shuffle=False)
                    for batch in streamer:
                        yield batch
                    streamer.close()
                return fn
            try:
                res = benchmark_fn("ZeroCopyStreamer (pinned ring)", make_streamer_fn(), num_samples, seq_len, dim, bs, device, warmup=1, runs=args.runs)
                results.append(res)
            except Exception as e:
                print(f"[!] Streamer failed: {e}")
                import traceback; traceback.print_exc()

            # ---- E. auto_dequant_device Dataset (dequant per sample) ----
            # This is the slow path many users might accidentally use - per-item dequant in __getitem__
            try:
                ds_auto = FeatureCacheDataset(prefix, auto_dequant_device=dev_str)
                def make_auto_fn():
                    def fn():
                        from torch.utils.data import DataLoader
                        loader = DataLoader(ds_auto, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
                        for batch in loader:
                            # batch is already BF16 [B,seq,dim] stacked
                            yield batch
                    return fn
                res = benchmark_fn("DataLoader auto_dequant (per-sample)", make_auto_fn(), num_samples, seq_len, dim, bs, device, warmup=1, runs=args.runs)
                results.append(res)
                ds_auto.close()
            except Exception as e:
                print(f"[!] auto_dequant failed: {e}")

            # ---- F. Raw dequant microbench (no IO) isolated ----
            # Measure pure dequant speed for reference
            try:
                # take one batch of q/s from ds
                from torch.utils.data import DataLoader
                loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
                q_sample, *rest = next(iter(loader))
                # move to device
                q_sample = q_sample.to(device)
                if args.amo_bq:
                    s_sample, zp_sample = rest[0].to(device), rest[1].to(device)
                    # warmup
                    for _ in range(5):
                        _ = dequantize_int8_amo_bq(q_sample, s_sample, zp_sample, (q_sample.shape[0], seq_len, dim), group_size=group_size)
                    sync_if_cuda(device)
                    runs_dq = 50
                    t0 = time.perf_counter()
                    for _ in range(runs_dq):
                        _ = dequantize_int8_amo_bq(q_sample, s_sample, zp_sample, (q_sample.shape[0], seq_len, dim), group_size=group_size)
                    sync_if_cuda(device)
                    dt = (time.perf_counter()-t0)/runs_dq*1000
                    # bandwidth effective IO bytes = B*seq*dim*3.0625 (if amo 1+2/32+1/32?) Actually amo 1+3/32=1.093
                    # but dequant reads q(1B)+scales(2B/32)+zp(1B/32) + writes BF16(2B) = 3.093
                    eff_bytes = bs*seq_len*dim * (3.09375 if args.amo_bq else 3.0625)
                    gb_s = (eff_bytes/(1024**3))/(dt/1000)
                    results.append({"name":"(ref) pure dequant (no IO)", "median_fps": 1000/dt*bs, "median_mbs": gb_s*1024, "median_ms": dt, "mean_batch_ms": dt, "p50_ms": dt, "p95_ms": dt})
                else:
                    s_sample = rest[0].to(device)
                    for _ in range(5):
                        _ = dequantize_int8_g32(q_sample, s_sample, (q_sample.shape[0], seq_len, dim), group_size=group_size)
                    sync_if_cuda(device)
                    runs_dq = 50
                    t0 = time.perf_counter()
                    for _ in range(runs_dq):
                        _ = dequantize_int8_g32(q_sample, s_sample, (q_sample.shape[0], seq_len, dim), group_size=group_size)
                    sync_if_cuda(device)
                    dt = (time.perf_counter()-t0)/runs_dq*1000
                    eff_bytes = bs*seq_len*dim*3.0625
                    gb_s = (eff_bytes/(1024**3))/(dt/1000)
                    results.append({"name":"(ref) pure dequant (no IO)", "median_fps": 1000/dt*bs, "median_mbs": gb_s*1024, "median_ms": dt, "mean_batch_ms": dt, "p50_ms": dt, "p95_ms": dt})
            except Exception as e:
                print(f"[!] pure dequant bench failed: {e}")

            # Pretty print for this batch_size
            headers = ["Ingestion Path", "samples/s", "BF16 MB/s", "total ms", "batch p50 ms", "p95 ms"]
            rows = []
            for r in results:
                rows.append([
                    r["name"],
                    f"{r['median_fps']:.1f}",
                    f"{r['median_mbs']:.1f}",
                    f"{r['median_ms']:.1f}",
                    f"{r['p50_ms']:.2f}",
                    f"{r['p95_ms']:.2f}",
                ])
            # sort by fps desc
            rows_sorted = sorted(rows, key=lambda x: float(x[1]), reverse=True)
            print("\n")
            if HAS_TABULATE:
                print(tabulate(rows_sorted, headers=headers, tablefmt="github"))
            else:
                print(f"{'Ingestion Path':<40} {'samples/s':>12} {'MB/s':>10} {'total':>10} {'p50':>10} {'p95':>10}")
                for row in rows_sorted:
                    print(f"{row[0]:<40} {row[1]:>12} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}")

            # Recommendation
            if rows_sorted:
                best = rows_sorted[0][0]
                print(f"\n[+] Fastest for B={bs}: {best}")
                if "Streamer" in best or "iter_batches" in best:
                    print("    -> Use ZeroCopyTensorStreamer / iter_batches for training (avoids DataLoader per-sample copy).")
                elif "Prefetcher" in best:
                    print("    -> Prefetcher helps but still pays per-sample collate cost.")
                else:
                    print("    -> DataLoader path unexpectedly fastest (small data / CPU).")

        # ------------------------------------------------------------------
        # 3. PixelCache microbench optional
        # ------------------------------------------------------------------
        if not args.skip_pixel and not args.quick:
            print("\n" + "="*110)
            print("[*] PixelCache benchmark (336x336x3 uint8 raw)")
            print("="*110)
            H=W=336; C=3; num_pix = min(2000, num_samples)
            with tempfile.TemporaryDirectory() as tmpdir2:
                pfx = Path(tmpdir2) / "pix"
                pw = PixelCacheWriter(pfx, num_samples=num_pix, height=H, width=W, channels=C)
                fake = np.random.randint(0,256,size=(H,W,C), dtype=np.uint8)
                for _ in range(num_pix):
                    pw.append_image(fake)
                pw.close()
                pds = PixelCacheDataset(pfx)
                # A: per-sample __getitem__
                def pix_getitem():
                    def fn():
                        from torch.utils.data import DataLoader
                        loader = DataLoader(pds, batch_size=bs, shuffle=False, num_workers=0)
                        for b in loader:
                            yield b
                    return fn
                # B: manual batched mmap (emulating future iter_batches)
                def pix_batched():
                    def fn():
                        # C-level slice like feature_cache iter_batches
                        idx = np.arange(num_pix)
                        for i in range(0, num_pix, bs):
                            arr = pds.mmap_pixels[idx[i:i+bs]]
                            t = torch.from_numpy(arr.copy())
                            yield t
                    return fn
                p_res = []
                for name, fn in [("Pixel DataLoader", pix_getitem()), ("Pixel batched mmap", pix_batched())]:
                    r = benchmark_fn(name, fn, num_pix, H, W, bs, device, warmup=1, runs=2)
                    # override MB/s to uint8 bytes
                    r["median_mbs"] = r["median_fps"]*H*W*C/(1024*1024)
                    p_res.append(r)
                headers = ["Pixel Path", "imgs/s", "MB/s", "total ms", "p50 ms", "p95 ms"]
                rows = [[r["name"], f"{r['median_fps']:.1f}", f"{r['median_mbs']:.1f}", f"{r['median_ms']:.1f}", f"{r['p50_ms']:.2f}", f"{r['p95_ms']:.2f}"] for r in p_res]
                if HAS_TABULATE:
                    print(tabulate(rows, headers=headers, tablefmt="github"))
                else:
                    for row in rows:
                        print(row)
                pds.close()

        ds.close()
        print("\n" + "="*110)
        print("                 BENCHMARK COMPLETE - use fastest path for training")
        print("="*110)


if __name__ == "__main__":
    main()
