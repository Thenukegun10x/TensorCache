"""
Dictionary learning + per-block table selection for block-tANS (research).

Separate module: `tans.py` owns the codec and format, this owns the
statistics. Nothing in existing cache paths imports either module.

Pipeline: collect block histograms from calibration arenas -> k-means over
sqrt-probabilities (Hellinger-adjacent, deterministic) -> Laplace-smoothed
(+1) quantized tables -> per-block minimum-cross-entropy selection.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Union

import numpy as np

from .tans import HAS_TRITON, R_DEFAULT, normalize


def codelengths(freq: List[int], R: int) -> np.ndarray:
    """Ideal codelength -log2(p) per symbol for one table (all freqs >= 1)."""
    f = np.asarray(freq, dtype=np.float64)
    return R - np.log2(f)


def select_tables(H: "np.ndarray", freqs: List[List[int]], R: int) -> List[int]:
    """Per-block argmin cross-entropy table index. Deterministic (ties -> lowest).

    H: [nb x 256] block counts. Cost(block, table) = H . L_table, one matmul.
    Laplace smoothing in learn_dictionary guarantees no zero freqs, so no
    infinite costs ever occur.
    """
    L = np.stack([codelengths(f, R) for f in freqs], axis=1)  # [256 x K]
    costs = H.astype(np.float64) @ L
    return np.argmin(costs, axis=1).tolist()


def _kmeans_pp(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic k-means++ seeding (fixed rng => fixed centers)."""
    n = X.shape[0]
    centers = [rng.integers(n)]
    d2 = np.full(n, np.inf)
    for _ in range(1, K):
        d = np.sum((X - X[centers[-1]]) ** 2, axis=1)
        d2 = np.minimum(d2, d)
        s = d2.sum()
        if s <= 0:
            centers.append(rng.integers(n))
        else:
            centers.append(int(np.searchsorted(np.cumsum(d2) / s,
                                               rng.random())))
    return X[np.array(centers)]


def _lloyd(X: np.ndarray, C: np.ndarray, iters: int) -> np.ndarray:
    """Lloyd iterations with deterministic ties and empty-cluster repair."""
    K = C.shape[0]
    for _ in range(iters):
        d = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        a = np.argmin(d, axis=1)  # ties -> lowest index
        new_C = C.copy()
        for k in range(K):
            pts = X[a == k]
            if len(pts):
                new_C[k] = pts.mean(axis=0)
            else:  # re-seed at the worst-fit point (deterministic)
                worst = int(np.argmax(d[np.arange(len(X)), a]))
                new_C[k] = X[worst]
        if np.array_equal(new_C, C):
            break
        C = new_C
    return C


def learn_dictionary(block_hists: Union[np.ndarray, list],
                     K: int = 8, R: int = R_DEFAULT,
                     seed: int = 0, iters: int = 25) -> List[List[int]]:
    """Cluster block histograms into K quantized prototype tables.

    block_hists: [N x 256] counts (any scale; rows normalized inside).
    Returns K frequency lists, each summing to 2**R, Laplace-smoothed so
    every symbol keeps probability mass (selection never faces inf cost).
    """
    H = np.asarray(block_hists, dtype=np.float64)
    if H.ndim != 2 or H.shape[1] != 256 or len(H) == 0:
        raise ValueError("block_hists must be non-empty [N x 256]")
    P = H / H.sum(axis=1, keepdims=True)
    X = np.sqrt(np.clip(P, 0, None))
    rng = np.random.default_rng(seed)
    C = _lloyd(X, _kmeans_pp(X, min(K, len(X)), rng), iters)
    tables = []
    for k in range(C.shape[0]):
        probs = C[k] ** 2
        counts = (probs * 1_000_000 + 1).astype(np.int64).tolist()
        tables.append(normalize(counts, R))
    while len(tables) < K:  # fewer samples than K: pad with copies (valid tables)
        tables.append(list(tables[-1]))
    return tables


def collect_block_hists(sections: list, block: int) -> np.ndarray:
    """Stacked [N x 256] block histograms over raw section byte strings."""
    Hs = []
    for raw in sections:
        arr = np.frombuffer(raw, dtype=np.uint8)
        nb = (len(arr) + block - 1) // block
        H = np.zeros((nb, 256), dtype=np.int64)
        for b in range(nb):
            H[b] = np.bincount(arr[b * block:(b + 1) * block], minlength=256)
        Hs.append(H)
    return np.concatenate(Hs, axis=0) if Hs else np.zeros((0, 256))


def save_tables(tables: dict, path: Union[str, Path]) -> None:
    """Persist {kind: [freq lists]} + meta as JSON (research artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables, indent=1))


def load_tables(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text())


# The four tANS-coded XS section kinds: dense u8 arena stream, i8 vals, and
# the byte-planar ll4 planes. `meta` stays raw (measured loss).
TANS_KINDS = ("u8", "i8", "ll4_lo", "ll4_hi")


def arena_sections(arena: dict) -> dict:
    """XS arena dict (from `codec.sparse_pack_arena`) -> the four byte streams."""
    l4 = arena["ll4"].numpy().astype(np.int16).tobytes()
    return {"u8": arena["arena_u8"].numpy().tobytes(),
            "i8": arena["arena_i8"].numpy().tobytes(),
            "ll4_lo": l4[0::2], "ll4_hi": l4[1::2]}


def learn_dictionary_from_arenas(arenas, K: int = 8, R: int = R_DEFAULT,
                                 block: int = 256, seed: int = 0) -> dict:
    """Learn the four section-kind dictionaries from calibration arenas.

    This is the build-time step for `xs_entropy="tans"`: encode a
    representative sample of the dataset to arenas, then cluster their block
    histograms. Cost is paid once per cache.
    """
    import numpy as np  # local: keeps this module importable without numpy

    out = {}
    for kind in TANS_KINDS:
        secs = [arena_sections(a)[kind] for a in arenas]
        out[kind] = learn_dictionary(collect_block_hists(secs, block),
                                     K=K, R=R, seed=seed)
    return out


def coerce_dictionary(tans_dict, R_hint: int = R_DEFAULT, K_hint: int = 8):
    """Accept {kind: tables} | meta dict | JSON path -> (tables, R, K, B).

    Lets callers pass either a freshly learned dictionary or one loaded from
    a cache's `_pixel_meta.json["tans"]`.
    """
    if isinstance(tans_dict, (str, Path)):
        loaded = json.loads(Path(tans_dict).read_text())
        if isinstance(loaded, dict) and loaded.get("codec") == "tans":
            return dictionary_from_meta(loaded, expected_kinds=TANS_KINDS)
        tables = loaded.get("kinds", loaded)
        return tables, R_hint, len(next(iter(tables.values()))), 256
    if isinstance(tans_dict, dict) and tans_dict.get("codec") == "tans":
        return dictionary_from_meta(tans_dict, expected_kinds=TANS_KINDS)
    if isinstance(tans_dict, dict):
        kinds = set(tans_dict)
        if kinds != set(TANS_KINDS):
            raise TansDictError(
                f"dictionary kinds {sorted(kinds)} != {sorted(TANS_KINDS)}")
        return dict(tans_dict), R_hint, len(next(iter(tans_dict.values()))), 256
    raise TansDictError(
        "tans_dict must be {kind: tables}, a tans meta dict, or a JSON path")


class TansDictError(ValueError):
    """Invalid or incompatible tANS dictionary metadata."""


# ---------------------------------------------------------------------------
# Cache-embeddable dictionary (goes in _pixel_meta.json["tans"])
#
# Stored compactly as per-table (sym, freq) pairs for freq > 0, so a 4-kind
# x K=8 dictionary is ~10 KB of JSON instead of the ~60 KB full-matrix dump.
# The decoder trusts ONLY these tables (never anything learned at import), so
# a cache decoded years later is byte-for-byte reproducible.
# ---------------------------------------------------------------------------

DICT_VERSION = 1


def _pack_kind(tables) -> str:
    """K tables -> base64 of [u16 n_used][n_used x (u8 sym, u16 freq)]..."""
    import struct as _s
    b = bytearray()
    for t in tables:
        used = [(s, f) for s, f in enumerate(t) if f]
        b += _s.pack("<H", len(used))
        for s, f in used:
            b += _s.pack("<BH", s, f)
    return base64.b64encode(bytes(b)).decode("ascii")


def _unpack_kind(blob: str, K: int, R: int, kind: str):
    import struct as _s
    try:
        raw = base64.b64decode(blob, validate=True)
    except Exception as e:
        raise TansDictError(f"{kind}: undecodable dictionary blob") from e
    pos, out = 0, []
    for _ in range(K):
        if len(raw) < pos + 2:
            raise TansDictError(f"{kind}: truncated dictionary blob")
        (nu,) = _s.unpack_from("<H", raw, pos)
        pos += 2
        if nu == 0 or nu > 256 or len(raw) < pos + 3 * nu:
            raise TansDictError(f"{kind}: bad table entry count {nu}")
        freq = [0] * 256
        for _ in range(nu):
            sym, f = _s.unpack_from("<BH", raw, pos)
            pos += 3
            if freq[sym] != 0:
                raise TansDictError(f"{kind}: duplicate symbol {sym}")
            freq[sym] = f
        if sum(freq) != (1 << R):
            raise TansDictError(f"{kind}: table does not sum to 2**{R}")
        out.append(freq)
    if pos != len(raw):
        raise TansDictError(f"{kind}: trailing bytes in dictionary blob")
    return out


def dictionary_to_meta(tables: dict, R: int, K: int, B: int,
                       version: int = DICT_VERSION) -> dict:
    """{kind: [freq lists]} -> compact, cache-embeddable meta.

    Stored as base64 binary (3 B per used symbol), which keeps a 4-kind x
    K=8 dictionary at ~20 KB of JSON instead of ~460 KB of nested pairs.
    """
    if not tables:
        raise TansDictError("empty dictionary")
    kinds = {}
    for kind, tabs in tables.items():
        if len(tabs) != K:
            raise TansDictError(f"{kind}: expected {K} tables, got {len(tabs)}")
        for t in tabs:
            if len(t) != 256 or sum(t) != (1 << R):
                raise TansDictError(f"{kind}: table does not sum to 2**{R}")
        kinds[kind] = _pack_kind(tabs)
    return {"version": version, "codec": "tans", "R": int(R), "B": int(B),
            "K": int(K), "kinds": kinds}


def dictionary_from_meta(meta: dict, expected_kinds=None):
    """Inverse of dictionary_to_meta, with full validation.

    Returns (tables, R, K, B). Raises TansDictError on any inconsistency so
    a corrupt/foreign meta fails loudly rather than decoding garbage.
    """
    if not isinstance(meta, dict) or meta.get("codec") != "tans":
        raise TansDictError("not a tANS dictionary")
    if meta.get("version") != DICT_VERSION:
        raise TansDictError(f"unsupported dictionary version {meta.get('version')}")
    R, K, B = int(meta["R"]), int(meta["K"]), int(meta["B"])
    if not (1 <= R <= 16) or K < 1 or B < 1:
        raise TansDictError(f"bad dictionary params R={R} K={K} B={B}")
    kinds_meta = meta.get("kinds")
    if not isinstance(kinds_meta, dict) or not kinds_meta:
        raise TansDictError("dictionary has no kinds")
    if expected_kinds is not None and set(kinds_meta) != set(expected_kinds):
        raise TansDictError(
            f"dictionary kinds {sorted(kinds_meta)} != expected "
            f"{sorted(expected_kinds)}")
    tables = {}
    for kind, blob in kinds_meta.items():
        if not isinstance(blob, str):
            raise TansDictError(f"{kind}: malformed dictionary entry")
        tables[kind] = _unpack_kind(blob, K, R, kind)
    return tables, R, K, B


# ---------------------------------------------------------------------------
# Decode capability gate
#
# The ratio win only pays with GPU decode; CPU decode is ~1.5x slower than
# rANS. Callers must therefore opt in to a device explicitly and get a clear
# error (not a silent slow path) when Triton/CUDA is unavailable.
# ---------------------------------------------------------------------------

def tans_gpu_decode_available(device=None) -> bool:
    """True if the GPU block-tANS kernels can run on `device`."""
    if not HAS_TRITON:
        return False
    import torch
    dev = torch.device(device if device is not None else
                       ("cuda:0" if torch.cuda.is_available() else "cpu"))
    return dev.type in ("cuda", "hip")


def require_tans_gpu_decode(device=None) -> None:
    """Raise a clear error if GPU tANS decode is unavailable on `device`."""
    if tans_gpu_decode_available(device):
        return
    import torch
    dev = torch.device(device if device is not None else
                       ("cuda:0" if torch.cuda.is_available() else "cpu"))
    why = "Triton is not installed" if not HAS_TRITON else \
        f"device {dev} is not CUDA/ROCm"
    raise RuntimeError(
        f"block-tANS GPU decode unavailable ({why}). "
        "tANS caches decode on the GPU because the CPU path is ~1.5x slower "
        "than rANS; either decode on a CUDA/ROCm device or build the cache "
        "with xs_entropy=True (rANS)."
    )
