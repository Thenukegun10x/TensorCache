"""Lightweight, dependency-free validation for cache metadata and files.

Raises :class:`CacheValidationError` (a ``ValueError``) with a clear message
when a cache meta or file is malformed, and ``FileNotFoundError`` when a
required file is missing. Kept deliberately small: it checks *shape and
bounds*, not codec correctness.
"""
from __future__ import annotations

import os
from pathlib import Path


class CacheValidationError(ValueError):
    """Raised when a cache meta or backing file is malformed."""


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _req_int(meta, key, lo=None, hi=None):
    v = meta.get(key) if isinstance(meta, dict) else None
    if not _is_int(v):
        raise CacheValidationError(f"{key} must be an int, got {v!r}")
    if lo is not None and v < lo:
        raise CacheValidationError(f"{key} must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise CacheValidationError(f"{key} must be <= {hi}, got {v}")
    return v


def _req_str(meta, key):
    v = meta.get(key) if isinstance(meta, dict) else None
    if not isinstance(v, str) or not v:
        raise CacheValidationError(f"{key} must be a non-empty string, got {v!r}")
    return v


def _file_size(path, what):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing {what}: {p}")
    return p.stat().st_size


def _check_file_size(path, expected_bytes, what):
    """Reject a file that is too small for its declared shape.

    Writers pre-allocate to their capacity but persist only the number of
    samples actually appended, so the backing file is commonly *larger* than
    this calculation.  ``np.memmap(mode="r", shape=...)`` simply maps a prefix
    in that case, which is safe; only an undersized file would fail.
    """
    actual = _file_size(path, what)
    if actual < expected_bytes:
        raise CacheValidationError(
            f"{what} is truncated: {path} is {actual} bytes, need >= {expected_bytes}"
        )


def validate_feature_meta(meta, context="feature cache"):
    """Validate a single-shard (or per-shard) feature cache meta."""
    if not isinstance(meta, dict):
        raise CacheValidationError(f"{context}: meta must be a dict, got {type(meta).__name__}")
    num_samples = _req_int(meta, "num_samples", lo=0)
    seq_len = _req_int(meta, "seq_len", lo=1)
    dim = _req_int(meta, "dim", lo=1)
    group_size = _req_int(meta, "group_size", lo=1)
    amo_bq = meta.get("amo_bq", False)
    if not isinstance(amo_bq, bool):
        raise CacheValidationError(f"{context}: amo_bq must be a bool, got {amo_bq!r}")
    return {
        "num_samples": num_samples,
        "seq_len": seq_len,
        "dim": dim,
        "group_size": group_size,
        "amo_bq": amo_bq,
    }


def validate_feature_shards_manifest(manifest, context="sharded feature cache"):
    """Validate the <prefix>_shards.json manifest."""
    if not isinstance(manifest, dict):
        raise CacheValidationError(f"{context}: manifest must be a dict")
    num_shards = _req_int(manifest, "num_shards", lo=1)
    shard_prefixes = manifest.get("shard_prefixes")
    if not isinstance(shard_prefixes, list) or len(shard_prefixes) != num_shards:
        raise CacheValidationError(
            f"{context}: shard_prefixes must be a list of {num_shards} entries"
        )
    for p in shard_prefixes:
        if not isinstance(p, str) or not p:
            raise CacheValidationError(f"{context}: shard prefix must be a non-empty string")
    num_samples = _req_int(manifest, "num_samples", lo=0)
    seq_len = _req_int(manifest, "seq_len", lo=1)
    dim = _req_int(manifest, "dim", lo=1)
    group_size = _req_int(manifest, "group_size", lo=1)
    shard_sizes = manifest.get("shard_sizes")
    if shard_sizes is not None:
        if not isinstance(shard_sizes, list) or len(shard_sizes) != num_shards:
            raise CacheValidationError(
                f"{context}: shard_sizes must be a list of {num_shards} entries"
            )
        for s in shard_sizes:
            if not _is_int(s) or s < 0:
                raise CacheValidationError(f"{context}: shard_sizes entries must be ints >= 0")
        # shard_sizes records per-shard *capacity* while num_samples records
        # the actual number appended. A partially-filled cache legitimately
        # has capacity > written, so only the reverse is corruption.
        if sum(shard_sizes) < num_samples:
            raise CacheValidationError(
                f"{context}: shard_sizes sum {sum(shard_sizes)} < num_samples {num_samples}"
            )
    amo_bq = manifest.get("amo_bq", False)
    if not isinstance(amo_bq, bool):
        raise CacheValidationError(f"{context}: amo_bq must be a bool")
    return {
        "num_shards": num_shards,
        "shard_prefixes": shard_prefixes,
        "num_samples": num_samples,
        "seq_len": seq_len,
        "dim": dim,
        "group_size": group_size,
        "shard_sizes": shard_sizes,
        "amo_bq": amo_bq,
    }


def validate_pixel_meta(meta, context="pixel cache"):
    """Validate a pixel cache meta. Returns the normalized dict."""
    if not isinstance(meta, dict):
        raise CacheValidationError(f"{context}: meta must be a dict, got {type(meta).__name__}")
    num_samples = _req_int(meta, "num_samples", lo=0)
    height = _req_int(meta, "height", lo=1)
    width = _req_int(meta, "width", lo=1)
    channels = _req_int(meta, "channels", lo=1)
    if channels not in (1, 3):
        raise CacheValidationError(f"{context}: channels must be 1 or 3, got {channels}")
    quant = meta.get("quant", "raw")
    if quant not in ("raw", "int4", "int3", "xs"):
        raise CacheValidationError(f"{context}: quant must be raw/int4/int3/xs, got {quant!r}")
    group_size = meta.get("group_size", 32)
    if not _is_int(group_size) or group_size < 1:
        raise CacheValidationError(f"{context}: group_size must be an int >= 1")
    return {
        "num_samples": num_samples,
        "height": height,
        "width": width,
        "channels": channels,
        "quant": quant,
        "group_size": group_size,
    }


def validate_xs_table(meta, context="xs cache"):
    """Validate the xs_table rows against the meta's declared sample count.

    Returns the list of rows. Bounds-checking against file sizes happens in
    the caller (which knows the actual file sizes).
    """
    table = meta.get("xs_table")
    if not isinstance(table, list):
        raise CacheValidationError(f"{context}: xs_table must be a list")
    num_samples = meta.get("num_samples")
    if _is_int(num_samples) and len(table) != num_samples:
        raise CacheValidationError(
            f"{context}: xs_table has {len(table)} rows but num_samples is {num_samples}"
        )
    for i, r in enumerate(table):
        if not isinstance(r, dict):
            raise CacheValidationError(f"{context}: xs_table row {i} must be a dict")
        n_planes = _req_int(r, "n_planes", lo=0)
        if not _is_int(r.get("meta_row")) or r["meta_row"] < 0:
            raise CacheValidationError(f"{context}: xs_table row {i} meta_row must be an int >= 0")
        enc = r.get("enc")
        if enc is not None:
            if not isinstance(enc, list) or len(enc) != 4:
                raise CacheValidationError(f"{context}: xs_table row {i} enc must be a list of 4")
            for m in enc:
                if not _is_int(m) or m < 0:
                    raise CacheValidationError(f"{context}: xs_table row {i} enc entries must be ints >= 0")
            for key in ("u8_off", "u8_len", "i8_off", "i8_len", "lb_off", "lb_len"):
                if not _is_int(r.get(key)) or r[key] < 0:
                    raise CacheValidationError(f"{context}: xs_table row {i} {key} must be an int >= 0")
        else:
            for key in ("u8_off", "u8_len", "i8_off", "i8_len", "ll4_off", "ll4_len"):
                if not _is_int(r.get(key)) or r[key] < 0:
                    raise CacheValidationError(f"{context}: xs_table row {i} {key} must be an int >= 0")
    return table


def check_xs_row_bounds(row, u8_size, i8_size, ll4_size, meta_row_count, context="xs cache"):
    """Check one xs_table row against actual file sizes."""
    enc = row.get("enc")
    if not _is_int(row.get("u8_off")) or not _is_int(row.get("u8_len")):
        raise CacheValidationError(f"{context}: row missing u8 offsets")
    if row["u8_off"] + row["u8_len"] > u8_size:
        raise CacheValidationError(f"{context}: u8 range out of bounds")
    if row["i8_off"] + row["i8_len"] > i8_size:
        raise CacheValidationError(f"{context}: i8 range out of bounds")
    if enc:
        if row["lb_off"] + row["lb_len"] > ll4_size:
            raise CacheValidationError(f"{context}: ll4 range out of bounds")
    else:
        # v1: ll4 offsets are int16 element indices
        if row["ll4_off"] * 2 + row["ll4_len"] * 2 > ll4_size:
            raise CacheValidationError(f"{context}: ll4 range out of bounds")
    if row["meta_row"] + row["n_planes"] > meta_row_count:
        raise CacheValidationError(f"{context}: meta_row range out of bounds")
