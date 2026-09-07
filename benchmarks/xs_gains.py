"""5/3 synthesis energy gains via float reference (no rounding noise)."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/conorm/Desktop/Pipeline/TensorCache/src")


def idwt53_1d_pure(s, d):
    s, d = s.float(), d.float()
    Ns, Nd = s.shape[-1], d.shape[-1]
    dp = F.pad(d, (1, 1), mode="replicate")
    even = s - (dp[..., :-1] + dp[..., 1:])[..., :Ns] / 4.0
    ep = F.pad(even, (0, 1), mode="replicate")
    odd = d + (ep[..., :-1] + ep[..., 1:])[..., :Nd] / 2.0
    out = torch.empty(s.shape[:-1] + (Ns + Nd,))
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return out


def step53(LL, LH, HL, HH):
    s_r = idwt53_1d_pure(LL.transpose(-2, -1), LH.transpose(-2, -1)).transpose(-2, -1)
    d_r = idwt53_1d_pure(HL.transpose(-2, -1), HH.transpose(-2, -1)).transpose(-2, -1)
    return idwt53_1d_pure(s_r, d_r)


H = W = 336
SIZES = {4: (21, 21), 3: (42, 42), 2: (84, 84), 1: (168, 168)}
AIMP = 256.0


def empty():
    p = {"LL4": torch.zeros(1, *SIZES[4])}
    for lvl in (4, 3, 2, 1):
        for o in ("LH", "HL", "HH"):
            p[f"{o}{lvl}"] = torch.zeros(1, *SIZES[lvl])
    return p


def synth(p):
    r3 = step53(p["LL4"], p["LH4"], p["HL4"], p["HH4"])
    r2 = step53(r3, p["LH3"], p["HL3"], p["HH3"])
    r1 = step53(r2, p["LH2"], p["HL2"], p["HH2"])
    return step53(r1, p["LH1"], p["HL1"], p["HH1"])


gains = {}
for lvl in (4, 3, 2, 1):
    for o in ("LH", "HL", "HH"):
        n = f"{o}{lvl}"
        h, w = SIZES[lvl]
        p = empty()
        p[n][0, h // 2, w // 2] = AIMP
        r = synth(p)
        gains[n] = float((r ** 2).sum()) / AIMP ** 2

n1 = gains["LH1"]
print("SUBBAND_GAINS = {")
for lvl in (4, 3, 2, 1):
    for o in ("LH", "HL", "HH"):
        print(f'    "{o}{lvl}": {gains[f"{o}{lvl}"] / n1:.3f},')
print("}")
