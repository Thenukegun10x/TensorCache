"""Generate tests/data/tans_goldens.bin from the reviewed tANS reference.

Byte-stability rule: once committed, goldens pin encode() output. Any later
byte change fails test_cross_version_stability until goldens are deliberately
regenerated AND reviewed (the blob bytes themselves are the spec).

Layout: [u32 nvec] then per vector [u32 dlen][data][u32 blen][blob],
blobs encoded with R=12, B=512 (defaults).
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorcache import tans


def _vectors():
    vecs = []
    vecs.append(bytes([7] * 1000 + [8] * 300 + [250] * 50) + bytes(range(256)))
    vecs.append(bytes(i % 256 for i in range(1024)))
    for n in range(33):
        vecs.append(bytes((i * 37 + 11) % 251 for i in range(n)))
    x = 0x123456789ABCDEF1
    out = bytearray()
    for _ in range(4096):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        x &= 0xFFFFFFFFFFFFFFFF
        out.append(x % 256)
    vecs.append(bytes(out))
    vecs.append(bytes([42] * 64))
    # block-boundary lengths at default B=512
    for n in (511, 512, 513, 1023, 1024, 1025):
        vecs.append(bytes((i * 53 + 7) % 251 for i in range(n)))
    return vecs


def main():
    vecs = _vectors()
    out = bytearray()
    out += struct.pack("<I", len(vecs))
    for v in vecs:
        blob = tans.encode(v)
        assert tans.decode(blob) == v, "reference failed its own roundtrip"
        out += struct.pack("<I", len(v)) + v
        out += struct.pack("<I", len(blob)) + blob
    dest = Path(__file__).parent / "data" / "tans_goldens.bin"
    dest.write_bytes(bytes(out))
    print(f"wrote {len(vecs)} vectors -> {dest} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
