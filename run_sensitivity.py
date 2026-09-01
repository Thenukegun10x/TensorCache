import pathlib, math, torch, torch.nn.functional as F, numpy as np
from PIL import Image
import sys
sys.path.insert(0,'src')
from tensorcache.codec import quantize_int8_g32, dequantize_int8_g32

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"device {device}")

# Load images
img_dir=pathlib.Path('data/test_images')
imgs=list(img_dir.glob('*.jpg'))[:256]
print(f"found {len(imgs)} images for sensitivity")

# Try to load ViT feature extractor via timm, fallback to synthetic if fails
features=None
try:
    import timm
    print("timm",timm.__version__)
    model=timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
    model.eval().to(device)
    # preprocess: timm data_config?
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    config=resolve_data_config({}, model=model)
    # use simple manual: resize 224, normalize imagenet
    import torchvision.transforms as T
    tfm=T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    feats=[]
    with torch.no_grad():
        for i,p in enumerate(imgs):
            with Image.open(p) as im:
                im=im.convert('RGB')
                t=tfm(im).unsqueeze(0).to(device)
                # forward features
                if hasattr(model,'forward_features'):
                    f=model.forward_features(t)
                else:
                    f=model(t)
                # f shape [1, seq, dim] or [1,dim]
                if f.ndim==3:
                    feats.append(f[0].float().cpu())
                elif f.ndim==2:
                    feats.append(f.float().cpu())
                else:
                    # handle tuple
                    if isinstance(f,(tuple,list)):
                        f=f[0]
                    feats.append(f.float().cpu())
            if i>=63:
                break
    features=torch.stack(feats)  # [N, seq, dim] or [N, dim]
    print(f"extracted real features {features.shape} dtype {features.dtype} range [{features.min():.3f},{features.max():.3f}]")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"timm failed {e}, fallback synthetic")
    torch.manual_seed(42)
    def make_features(n=64, seq=197, dim=384): # vit_small dim 384
        t=torch.randn(n, seq, dim)*0.5
        t+=torch.randn(n, seq,1)*0.35
        t+=torch.randn(n,1,dim)*0.35
        flat=t.flatten(); idx=torch.randperm(flat.numel())[:int(flat.numel()*0.001)]; flat[idx]*=6
        return t.to(torch.bfloat16)
    features=make_features(64,197,384)
    print(f"synthetic {features.shape}")

gt_f32=features.float()
total_elem=features.numel()
print(f"total_elem {total_elem:,} total MB FP32 {total_elem*4/1024/1024:.2f} BF16 {total_elem*2/1024/1024:.2f}")
# baseline flat G32
q,s,shape=quantize_int8_g32(features,32)
rec=dequantize_int8_g32(q,s,shape,32)
# compute metrics
def metrics(gt, rec):
    gt_f=gt.float(); rec_f=rec.float(); diff=gt_f-rec_f
    rel=diff.norm().item()/(gt_f.norm().item()+1e-12)*100
    mape=diff.abs().mean().item()/(gt_f.abs().mean().item()+1e-12)*100
    maxe=diff.abs().max().item()
    dr=(gt_f.max()-gt_f.min()).item()
    mse=(diff**2).mean().item()
    psnr=20*math.log10(dr/math.sqrt(mse)) if mse>0 else float('inf')
    cos=F.cosine_similarity(gt_f.flatten(), rec_f.flatten(), dim=0).item()
    k=max(1,int(gt.numel()*0.001))
    top=torch.topk(gt_f.abs().flatten(),k=k).indices
    out=(gt_f.flatten()[top]-rec_f.flatten()[top]).abs().mean().item()/(gt_f.flatten()[top].abs().mean().item()+1e-12)*100
    return dict(rel=rel,mape=mape,psnr=psnr,cos=cos,maxe=maxe,out=out)

base_m=metrics(gt_f32, rec.float())
print(f"BASELINE G32 BF16 1.0625 B/elem 1.88x BF16 rel {base_m['rel']:.4f}% mape {base_m['mape']:.4f}% psnr {base_m['psnr']:.2f} cos {base_m['cos']:.6f} max {base_m['maxe']:.4f} out {base_m['out']:.3f}%")

# helper for scale quantized variants
def sym_with_log8(x, group=32):
    orig_shape=x.shape
    x_flat=x.flatten().float()
    numel=x.numel()
    pad=(group - numel%group)%group
    if pad>0: x_flat=F.pad(x_flat,(0,pad))
    x_blocks=x_flat.view(-1,group)
    is_fin=torch.isfinite(x_blocks)
    x_clean=torch.where(is_fin,x_blocks,torch.zeros_like(x_blocks))
    block_max=x_clean.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales_fp32=(block_max/127.0).squeeze(-1) # [M]
    # per-tensor log quant 8b
    log_s=torch.log2(scales_fp32.clamp(min=1e-8))
    lmin=log_s.min(); lmax=log_s.max()
    q=torch.round((log_s - lmin)/(lmax-lmin+1e-12)*255).clamp(0,255)
    scales_q=torch.pow(2, q/255*(lmax-lmin)+lmin)
    # quant data
    scaled=x_clean/scales_q.unsqueeze(-1)
    q_blocks=torch.round(scaled).clamp(-128,127).to(torch.int8)
    q_flat=q_blocks.flatten()[:numel]
    # dequant
    q_p=F.pad(q_flat,(0,pad)) if pad>0 else q_flat
    deq=q_p.view(-1,group).float()*scales_q.unsqueeze(-1)
    rec=deq.flatten()[:numel].view(orig_shape).to(torch.bfloat16)
    return rec, 1+1/group

def sym_fp8_scale(x, group=32):
    orig_shape=x.shape
    x_flat=x.flatten().float()
    numel=x.numel()
    pad=(group - numel%group)%group
    if pad>0: x_flat=F.pad(x_flat,(0,pad))
    x_blocks=x_flat.view(-1,group)
    is_fin=torch.isfinite(x_blocks)
    x_clean=torch.where(is_fin,x_blocks,torch.zeros_like(x_blocks))
    block_max=x_clean.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales_fp32=(block_max/127.0).squeeze(-1)
    scales_q=scales_fp32.to(torch.float8_e4m3fn).float()
    scaled=x_clean/scales_q.unsqueeze(-1)
    q_blocks=torch.round(scaled).clamp(-128,127).to(torch.int8)
    q_flat=q_blocks.flatten()[:numel]
    q_p=F.pad(q_flat,(0,pad)) if pad>0 else q_flat
    deq=q_p.view(-1,group).float()*scales_q.unsqueeze(-1)
    rec=deq.flatten()[:numel].view(orig_shape).to(torch.bfloat16)
    return rec, 1+1/group

def eval_configs():
    configs=[]
    # flat G sweep BF16
    for g in [16,32,48,64,96,128]:
        q,s,shp=quantize_int8_g32(features,g)
        rec=dequantize_int8_g32(q,s,shp,g)
        m=metrics(gt_f32, rec.float())
        bpe=1+2/g
        configs.append((f"G{g} BF16", bpe, m))
    # log8 variants
    for g in [32,64,128]:
        rec,bpe=sym_with_log8(features,g)
        m=metrics(gt_f32, rec.float())
        configs.append((f"G{g} log8", bpe, m))
    # fp8
    for g in [32,64]:
        try:
            rec,bpe=sym_fp8_scale(features,g)
            m=metrics(gt_f32, rec.float())
            configs.append((f"G{g} FP8_E4M3", bpe, m))
        except: pass
    # mixed 7b naive (to show)
    # Print table sorted by bpe
    print("\n=== Compression vs Accuracy (lower rel better, negligible <0.1pp over baseline) ===")
    # header
    print(f"{'Config':20s} {'B/elem':>7s} {'vs BF16':>8s} {'vs FP32':>8s} {'rel%':>7s} {'delta':>7s} {'mape':>7s} {'psnr':>6s} {'cos':>8s} {'max':>7s} {'out%':>6s}")
    for name,bpe,m in sorted(configs, key=lambda x: x[1]):
        delta=m['rel']-base_m['rel']
        print(f"{name:20s} {bpe:7.4f} {2/bpe:8.3f}x {4/bpe:8.3f}x {m['rel']:7.4f} {delta:+7.4f} {m['mape']:7.3f} {m['psnr']:6.1f} {m['cos']:8.6f} {m['maxe']:7.4f} {m['out']:6.3f}")
    # highlight negligible
    print("\nNegligible threshold: delta <0.10pp relRMSE (within <20% relative degradation), cos >0.9998, psnr drop <2dB")
    for name,bpe,m in configs:
        if m['rel']-base_m['rel']<0.10 and bpe<1.0625:
            print(f"  WIN {name}: {bpe:.4f} B/elem ({2/bpe:.2f}x) save {(1-bpe/1.0625)*100:.1f}% vs base for +{m['rel']-base_m['rel']:.4f}pp")
    return configs

configs=eval_configs()

# --- Sensitivity per block/token/channel ---
print("\n=== Sensitivity Analysis ===")
# Per-block sensitivity: measure per-block MSE contribution
x_flat=features.flatten().float()
pad=(32 - len(x_flat)%32)%32
if pad>0:
    x_flat_p=F.pad(x_flat,(0,pad))
else:
    x_flat_p=x_flat
x_blocks=x_flat_p.view(-1,32)
M=x_blocks.shape[0]
# compute per-block absmax and range and err 8b vs 7b
b_min=x_blocks.amin(dim=-1)
b_max=x_blocks.amax(dim=-1)
b_range=(b_max-b_min).clamp(min=1e-8)
b_absmax=x_blocks.abs().amax(dim=-1).clamp(min=1e-8)
# per block 8b error
s8=b_absmax/127
q8=torch.clamp(torch.round(x_blocks / s8.unsqueeze(-1)),-128,127)
rec8=q8*s8.unsqueeze(-1)
err8=((x_blocks-rec8)**2).sum(dim=-1)
# 7b
s7=b_absmax/63
q7=torch.clamp(torch.round(x_blocks / s7.unsqueeze(-1)),-64,63)
rec7=q7*s7.unsqueeze(-1)
err7=((x_blocks-rec7)**2).sum(dim=-1)
ratio=err7/(err8+1e-12)
print(f"Blocks M={M} range median {b_range.median():.4f} 99th {torch.quantile(b_range,0.99):.3f} absmax median {b_absmax.median():.4f}")
print(f"err8 total {err8.sum():.2f} err7 total {err7.sum():.2f} ratio 7b/8b median {ratio.median():.2f} mean {ratio.mean():.2f} 90th {torch.quantile(ratio,0.9):.2f}")
print(f"blocks <2x 7b/8b: {(ratio<2).sum()}/{M} {(ratio<2).sum()/M*100:.2f}% <4x {(ratio<4).sum()/M*100:.1f}%")
# cumulative error distribution
sorted_err, idx=torch.sort(err8, descending=True)
cum=torch.cumsum(sorted_err,dim=0)/err8.sum()
for frac in [0.5,0.8,0.9,0.95]:
    k=(cum>=frac).nonzero()[0].item()+1
    print(f"{frac*100:.0f}% MSE from top {k}/{M} blocks ({k/M*100:.2f}%)")

# Token sensitivity if features is [N, seq, dim]
if features.ndim==3:
    N,seq,dim=features.shape
    print(f"\nToken sensitivity N={N} seq={seq} dim={dim} (tokens)")
    # per token average err8 across N and dim blocks (dim/32 blocks per token)
    # compute per token MSE averaged across dataset
    token_err=[]
    for t in range(seq):
        tok=features[:,t,:].float().reshape(-1,32) # [N*dim/32,32]
        # need per-block scale per 32 dim slice
        b_absmax_t=tok.abs().amax(dim=-1).clamp(min=1e-8)
        s8_t=b_absmax_t/127
        q8_t=torch.clamp(torch.round(tok / s8_t.unsqueeze(-1)),-128,127)
        rec8_t=q8_t*s8_t.unsqueeze(-1)
        err8_t=((tok-rec8_t)**2).mean().item() # mean per elem MSE
        # also range
        token_err.append(err8_t)
    token_err=torch.tensor(token_err)
    print(f"token MSE mean {token_err.mean():.6f} std {token_err.std():.6f} min {token_err.min():.6f} max {token_err.max():.6f} max/min {token_err.max()/token_err.min():.2f}x")
    print(f"CLS token 0 MSE {token_err[0]:.6f} vs patch median {token_err[1:].median():.6f} ratio {token_err[0]/token_err[1:].median():.2f}")
    # most sensitive tokens
    sorted_t, idx_t=torch.sort(token_err, descending=True)
    print(f"top 5 sensitive tokens: {idx_t[:5].tolist()} {sorted_t[:5].tolist()}")
    print(f"bottom 5 robust tokens: {idx_t[-5:].tolist()} {sorted_t[-5:].tolist()}")

# Channel sensitivity
if features.ndim==3:
    print(f"\nChannel sensitivity dim={dim}")
    # per channel across N*seq
    ch_err=[]
    for c in range(dim):
        ch=features[:,:,c].float().view(-1,32) if seq%32==0 else features[:,:,c].float().flatten() # seq=197 not divisible
        # flatten all channel values into blocks of 32 across seq dimension: need to handle seq=197 -> 6*32=192 +5 leftover
        flat_ch=features[:,:,c].float().flatten()
        pad_c=(32 - len(flat_ch)%32)%32
        if pad_c>0:
            flat_ch=F.pad(flat_ch,(0,pad_c))
        blocks_c=flat_ch.view(-1,32)
        b_absmax_c=blocks_c.abs().amax(dim=-1).clamp(min=1e-8)
        s8_c=b_absmax_c/127
        q8_c=torch.clamp(torch.round(blocks_c / s8_c.unsqueeze(-1)),-128,127)
        rec8_c=q8_c*s8_c.unsqueeze(-1)
        err8_c=((blocks_c-rec8_c)**2).mean().item()
        ch_err.append(err8_c)
    ch_err=torch.tensor(ch_err)
    print(f"channel MSE median {ch_err.median():.6f} 90th {torch.quantile(ch_err,0.9):.6f} 99th {torch.quantile(ch_err,0.99):.6f} max/min {ch_err.max()/ch_err.min():.2f}x")
    print(f"channels <0.5*median (robust): {(ch_err<ch_err.median()*0.5).sum()}/{dim} ({(ch_err<ch_err.median()*0.5).sum()/dim*100:.1f}%)")
    # histogram?
    # sample least sensitive channels could be compressed to 4b?

print("\n=== End Sensitivity ===")
