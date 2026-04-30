"""
QuadWaterfall: Quad-Modal Waterfall Axial Transformer
Target: MCubeS  —  RGB + AoLP + DoLP + NIR  →  20 material classes
Input: 1024 × 1224  (training crops: 512 × 512)

Architecture Influenced From
--------------------
WTPose  [WACVW 2025]  → Waterfall cascade with dilated attention blocks
FuseForm [WACVW 2025] → MMCA global fusion + Local Fusion + transformer decoder

Novel contributions
-------------------
1. Axial (row + column) attention replaces full 2-D self-attention inside
the waterfall module.  At stage-1 resolution (256 × 306 = 78 K tokens):
Full SA : (78x336)²  ≈  6.1 B  ops / head
Axial   : 256x306² + 306x256²  ≈  44 M ops / head  →  140× reduction

2. Quad-modal waterfall: all 4-modality × 4-stage features aggregated at
H/4 before the cascade — preserves fine-grained texture detail that
lower-resolution models discard.

3. Per-modality 2-D projection before tokenisation removes the memory
bottleneck of a naïve 4096-wide token concatenation.
"""

from __future__ import annotations
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

# Global constants (match MiT-B2 / MiT-B4 channel layout)
_STAGE_CH = [64, 128, 320, 512] # embed_dims for every PVT-v2-B2/B4 stage
_SR_RATIOS = [8, 4, 2, 1] # MMCA spatial-reduction per stage
_DILATIONS = (2, 4, 6, 8) # waterfall dilation schedule (WTPose Table 2)
_QWTM_CH = 128 # bottleneck channels out of waterfall module

# Shared primitives
class Mlp(nn.Module):
    # Standard two-layer GELU MLP used throughout the transformer blocks
    def __init__(self, dim, ratio = 4, p = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(p),nn.Linear(dim * ratio, dim), nn.Dropout(p),)
    def forward(self, x):
        return self.net(x)

class InputProjection(nn.Module):
    # projects a single-channel auxiliary modality (AoLP / DoLP / NIR) to
    # 3 channels so it can be fed into a standard 3-channel MiT encoder
    # a small Conv-BN-GELU stem learns a meaningful initial embedding
    def __init__(self, in_ch = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )
    def forward(self, x):
        return self.conv(x)

# Axial attention  —  memory-efficient 2-D self-attention
class RowColAttention(nn.Module):
    # The row pass attends over each row independently (each token sees all W
    # neighbours in its row); the column pass attends within each column.
    # the two passes give every token access to all other tokens in
    # two hops, approximating full 2-D self-attention.
    def __init__(self, dim, heads = 8, p = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
        self.heads = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5

        self.row_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.col_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p)

    def _self_attn(self, qkv_fn, seq):
        # seq: (B_eff, N, C)  →  attended (B_eff, N, C)
        # Uses F.scaled_dot_product_attention (Flash Attention when available)
        # to avoid materialising the full (B, heads, N, N) attention matrix.
        B, N, C = seq.shape
        qkv = qkv_fn(seq).reshape(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # each (B, heads, N, dh)
        dropout_p = self.drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return out.transpose(1, 2).reshape(B, N, C)

    def forward(self, x, H, W):
        # x: (B, HxW, C)  →  (B, HxW, C)
        B, _, C = x.shape
        grid = x.reshape(B, H, W, C)

        # Row pass: attend within each row
        rows = grid.reshape(B * H, W, C)
        rows = rows + self._self_attn(self.row_qkv, rows)
        grid = rows.reshape(B, H, W, C)

        # Column pass: attend within each column
        cols = grid.permute(0, 2, 1, 3).reshape(B * W, H, C)
        cols = cols + self._self_attn(self.col_qkv, cols)
        grid = cols.reshape(B, W, H, C).permute(0, 2, 1, 3)

        return self.proj(grid.reshape(B, H * W, C))

class DilatedRowColAttention(nn.Module):
    # Dilated variant of RowColAttention.

    # Pixels are partitioned into d² interleaved sub-grids (one per dilation
    # offset).  Each sub-grid has size (H/d) × (W/d) and is attended over
    # independently with RowColAttention.  This expands the effective receptive
    # field to cover a (2d·k+1) window (k = kernel size in the base attention)
    # while keeping complexity the same as non-dilated axial attention.

    # dilation = 1 → identical to RowColAttention (no overhead).

    def __init__(self, dim, heads = 8, dilation = 1, p = 0.0):
        super().__init__()
        self.d = dilation
        self.attn = RowColAttention(dim, heads, p)

    def forward(self, x, H, W):
        if self.d == 1:
            return self.attn(x, H, W)

        B, L, C = x.shape
        d = self.d
        grid = x.reshape(B, H, W, C)

        # Pad so H, W are divisible by d
        ph = (-H) % d
        pw = (-W) % d
        if ph or pw:
            grid = F.pad(grid.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)

        Hp, Wp = grid.shape[1], grid.shape[2]
        nh, nw = Hp // d, Wp // d

        # Interleave: (B, Hp, Wp, C) → (Bxd², nh·nw, C)
        sub = grid.reshape(B, nh, d, nw, d, C).permute(0, 2, 4, 1, 3, 5)
        sub = sub.reshape(B * d * d, nh * nw, C)

        out = self.attn(sub, nh, nw)

        # Scatter back to (B, H, W, C)
        out = out.reshape(B, d, d, nh, nw, C).permute(0, 3, 1, 4, 2, 5)
        out = out.reshape(B, Hp, Wp, C)[:, :H, :W, :]
        return out.reshape(B, H * W, C)

# Waterfall Transformer Block  (WTB)
class WaterfallTransformerBlock(nn.Module):
    # One block in the waterfall cascade

    # Structure  (D-MHSA → MLP → N-MHSA → MLP)  maps directly to WTPose:
    # D-MHSA  →  DilatedRowColAttention   (expands receptive field globally)
    # N-MHSA  →  DilatedRowColAttention with dilation=1  (local refinement)

    # Using axial attention instead of window attention means the block can
    # run at full H/4 resolution without the quadratic memory explosion.

    # Dilation rates used in QWTM: (2, 4, 6, 8), giving effective receptive
    # field sizes of 13×13, 25×25, 37×37, 49×49 tokens respectively.
    def __init__(self, dim, heads = 8, dilation = 2, mlp_ratio = 4, p = 0.0,):
        super().__init__()
        # Dilated branch (global context)
        self.n_d1 = nn.LayerNorm(dim)
        self.d_attn = DilatedRowColAttention(dim, heads, dilation, p)
        self.n_d2 = nn.LayerNorm(dim)
        self.d_mlp = Mlp(dim, mlp_ratio, p)

        # Non-dilated branch (local context)
        self.n_n1 = nn.LayerNorm(dim)
        self.nd_attn = DilatedRowColAttention(dim, heads, 1, p)
        self.n_n2 = nn.LayerNorm(dim)
        self.nd_mlp = Mlp(dim, mlp_ratio, p)

    def forward(self, x, H, W):
        # Dilated pass
        x = x + self.d_attn(self.n_d1(x), H, W)
        x = x + self.d_mlp(self.n_d2(x))
        # Non-dilated pass
        x = x + self.nd_attn(self.n_n1(x), H, W)
        x = x + self.nd_mlp(self.n_n2(x))
        return x

# Quad-Modal Waterfall Module  (QWTM)
class QuadModalWaterfallModule(nn.Module):
    # Core architectural contribution: adapts the WTPose waterfall for 4-modal input
    # 1. Upsample stages 2–4 of all modalities to stage-1 resolution (H/4).
    # 2. Per-modality 2-D projection: (B, 1024, H/4, W/4) → (B, C_wt/4, H/4, W/4)
        # reduces channel dim BEFORE tokenisation, avoiding the 4096-wide tensor that a naïve concatenation would produce (≈ 2 GB at 1024×1224).
    # 3. Concatenate 4 modality projections → (B, C_wt, H/4, W/4).
    # 4. Depth-wise pooling branch (WTPose).
    # 5. Waterfall cascade: 4 WTBs at increasing dilation rates. Each WTB outputs are collected (waterfall branches).
    # 6. All branches + DWP branch concatenated → 1×1 fusion → (B, C_wt, H/4, W/4).

    def __init__(self, stage_ch = _STAGE_CH, n_mod = 4, out_ch = _QWTM_CH, heads = 8, dilations = _DILATIONS, mlp_ratio = 4, p = 0.0, checkpoint = True,):
        super().__init__()
        self.checkpoint = checkpoint
        total_per_mod = sum(stage_ch)       # 64+128+320+512 = 1024
        per_mod_ch = out_ch // n_mod     # 128 // 4 = 32  →  4×32 = 128 out

        # Per-modality 2-D projections (run in conv space, before any tokenisation)
        self.mod_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(total_per_mod, per_mod_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(per_mod_ch),
                nn.GELU(),
            )
            for _ in range(n_mod)
        ])

        self.input_norm = nn.LayerNorm(out_ch)

        # Depth-wise pooling branch
        self.dwpool = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

        # Waterfall cascade
        self.wtbs = nn.ModuleList([
            WaterfallTransformerBlock(out_ch, heads, d, mlp_ratio, p)
            for d in dilations
        ])

        # Final fusion: (N_dil + 1_DWP) branches → out_ch
        n_branches = len(dilations) + 1
        self.final = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, feats_by_mod, H1, W1):
        # Returns (B, out_ch, H1, W1)
        B = feats_by_mod[0][0].shape[0]

        # Upsample all stages to stage-1 res, per-modality channel reduction
        mod_2d = []
        for mod_stages, proj in zip(feats_by_mod, self.mod_projs):
            stacked = torch.cat([
                F.interpolate(s, (H1, W1), mode='bilinear', align_corners=False)
                for s in mod_stages
            ], dim=1)                                     # (B, sum_C, H1, W1)
            mod_2d.append(proj(stacked))                  # (B, out_ch//M, H1, W1)

        # Concatenate modalities
        z2d = torch.cat(mod_2d, dim=1)                   # (B, out_ch, H1, W1)

        # Depth-wise pooling branch
        dwp = self.dwpool(z2d)                            # (B, out_ch, H1, W1)

        # Waterfall cascade on tokenised features
        z = self.input_norm(z2d.flatten(2).transpose(1, 2))   # (B, H1·W1, out_ch)
        wt_outs = []
        for wtb in self.wtbs:
            if self.checkpoint and self.training:
                # Recompute activations during backprop instead of storing them.
                # H1/W1 are Python ints so wrap in a closure to avoid issues.
                z = grad_ckpt(wtb, z, H1, W1, use_reentrant=False)
            else:
                z = wtb(z, H1, W1)
            wt_outs.append(z.transpose(1, 2).reshape(B, -1, H1, W1))

        # Fuse all branches
        return self.final(torch.cat(wt_outs + [dwp], dim=1))  # (B, out_ch, H1, W1)

# § 5  Multimodal Cross-Attention  (MMCA)
class MMCA(nn.Module):
    """
    FuseForm Global Fusion (§3.2.1): for modality m, the query Qm is compared
    against the keys and values of ALL other modalities.  Over training, this
    forces high-information tokens from complementary sensors (e.g., polarisation
    cues invisible in RGB) to be emphasised while low-information tokens fade.

    Spatial reduction (sr_ratio, following PVT) halves the K/V sequence length
    at fine scales, keeping memory linear in sr_ratio².
    """
    def __init__(self, dim, heads = 8,sr_ratio = 1, p = 0.0,):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5
        self.sr_ratio = sr_ratio

        self.q_proj  = nn.Linear(dim, dim)
        self.k_proj  = nn.Linear(dim, dim)
        self.v_proj  = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop    = nn.Dropout(p)

        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def _reduce(self, t, H, W):
        """Apply spatial reduction to K/V tokens when sr_ratio > 1."""
        if self.sr_ratio == 1:
            return t
        B, N, C = t.shape
        x2d = t.transpose(1, 2).reshape(B, C, H, W)
        x2d = self.sr_conv(x2d)               # (B, C, H/sr, W/sr)
        return self.sr_norm(x2d.flatten(2).transpose(1, 2))

    def forward(self, q_tok, kv_list, H, W,):
        B, N, C = q_tok.shape

        Q = self.q_proj(q_tok).reshape(B, N, self.heads, self.dh).permute(0, 2, 1, 3)

        # Reduce and project K, V from all other modalities
        kv_all = torch.cat([self._reduce(t, H, W) for t in kv_list], dim=1)
        K = self.k_proj(kv_all).reshape(B, -1, self.heads, self.dh).permute(0, 2, 1, 3)
        V = self.v_proj(kv_all).reshape(B, -1, self.heads, self.dh).permute(0, 2, 1, 3)

        dropout_p = self.drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout_p)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, C))

# Local Fusion
class LocalFusion(nn.Module):
    """
    FuseForm §3.2.2: parallel multi-scale convolutions to capture local texture
    correlations that the global MMCA mechanism may miss.

    Kernels 1×1, 3×3, 5×5, 7×7 are applied in parallel (each producing C/4
    channels), concatenated, and projected back to C.  This makes the fusion
    block sensitive to fine-grained surface textures — the key motivation for
    this paper's design: materials differ in micro-texture, not just shape.
    """
    def __init__(self, dim):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim // 4, kernel_size=k, padding=k // 2)
            for k in (1, 3, 5, 7)
        ])
        self.merge = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """x: (B, H·W, C)  →  (B, H·W, C)"""
        B, N, C = x.shape
        g = self.proj_in(x).transpose(1, 2).reshape(B, C, H, W)
        g = self.act(self.merge(torch.cat([c(g) for c in self.convs], dim=1)))
        return self.proj_out(g.flatten(2).transpose(1, 2))

# Per-stage Multimodal Fusion Block
class MultimodalFusionBlock(nn.Module):
    """
    Applied at each of the 4 encoder stages (one per spatial resolution).
    Implements FuseForm eq. (1):

        x′_m = x_m + GF(x_m) + LF(x_m)          [per modality m]
        x    = LN( Linear( Concat(x′_m | m∈M) ) ) [fuse all M modalities]

    GF = global fusion via MMCA
    LF = local fusion via parallel convolutions

    The output is a single (B, C, H, W) fused feature map, used as the
    skip connection in the corresponding decoder stage.
    """
    def __init__(self, dim, heads = 8, sr_ratio = 1, p = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mmca = MMCA(dim, heads, sr_ratio, p)
        self.lf = LocalFusion(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, feats, H, W):
        """
        feats: list[4] of (B, C, H, W) — one tensor per modality
        Returns: (B, C, H, W) fused feature map
        """
        B, C = feats[0].shape[:2]
        toks = [f.flatten(2).transpose(1, 2) for f in feats]   # list[4] of (B,N,C)

        updated = []
        for i, t in enumerate(toks):
            tn = self.norm(t)
            others = [self.norm(toks[j]) for j in range(4) if j != i]
            updated.append(t + self.mmca(tn, others, H, W) + self.lf(tn, H, W))

        out = self.proj(torch.cat(updated, dim=-1))             # (B, N, C)
        return out.transpose(1, 2).reshape(B, C, H, W)


# Decoder
class SkipFusion(nn.Module):
    """
    FuseForm §3.3.1: merge feed-forward output with stage skip connection.
    Learnable scalar weights ω₁, ω₂ balance the two streams.
    Between the two 1×1 convolutions a 2-layer projection expands and contracts
    the channel dim, giving the block expressivity beyond simple addition.
    """
    def __init__(self, ff_ch, skip_ch, out_ch):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

        self.c1 = nn.Conv2d(ff_ch + skip_ch, out_ch, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Linear(out_ch, out_ch * 2), nn.ReLU(),
            nn.Linear(out_ch * 2, out_ch), nn.ReLU(),
        )
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, xff, xsk):
        B, _, H, W = xff.shape
        z = self.c1(torch.cat([self.w1 * xff, self.w2 * xsk], dim=1))
        z = self.proj(z.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, H, W)
        return F.relu(self.bn(self.c2(z)))

class DecoderLayer(nn.Module):
    """
    One layer inside a DecoderBlock.
    FuseForm eq. (3): self-attention + local fusion + MLP, all combined.
    axial=True uses RowColAttention (needed at stage 1, H/4 resolution);
    axial=False uses standard nn.MultiheadAttention (fine for H/8 and below).
    """
    def __init__(self, dim, heads, mlp_ratio = 4, p = 0.0, axial = True,):
        super().__init__()
        self.axial = axial
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)

        if axial:
            self.attn = RowColAttention(dim, heads, p)
        else:
            self.attn = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)

        self.mlp = Mlp(dim, mlp_ratio, p)
        self.lf = LocalFusion(dim)
        self.proj = nn.Linear(dim * 2, dim)   # fuse LF + MLP outputs

    def forward(self, t, H, W):
        """t: (B, H·W, C)"""
        ln = self.n1(t)
        sa = self.attn(ln, H, W) if self.axial else self.attn(ln, ln, ln)[0]
        t = t + sa
        t = t + self.proj(torch.cat([self.lf(self.n3(t), H, W),self.mlp(self.n2(t))], dim=-1))
        return t

class DecoderBlock(nn.Module):
    """Stack of DecoderLayers. Operates on (B, C, H, W) tensors."""
    def __init__(self, dim, heads, n_layers = 2, mlp_ratio = 4, p = 0.0, axial = True,):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dim, heads, mlp_ratio, p, axial)
            for _ in range(n_layers)
        ])

    def forward(self, x, H, W):
        B, C, _, _ = x.shape
        t = x.flatten(2).transpose(1, 2)       # (B, H·W, C)
        for layer in self.layers:
            t = layer(t, H, W)
        return t.transpose(1, 2).reshape(B, C, H, W)

# Full Model  —  QuadWaterfall
class QuadWaterfall(nn.Module):
    """
    Quad-Modal Waterfall Axial Transformer for MCubeS material segmentation.

    Encoder backbone
    ----------------
      RGB   → PVT-v2-B4  (ImageNet pretrained, 3-channel input)
      AoLP  → PVT-v2-B2  (pretrained, 1-ch → InputProjection → 3-ch)
      DoLP  → PVT-v2-B2  (same)
      NIR   → PVT-v2-B2  (same)

    Each encoder produces 4 stage feature maps:
      Stage 1: (B,  64, H/4,  W/4)
      Stage 2: (B, 128, H/8,  W/8)
      Stage 3: (B, 320, H/16, W/16)
      Stage 4: (B, 512, H/32, W/32)

    Fusion pipeline
    ---------------
    1. MultimodalFusionBlock at each stage (MMCA + LocalFusion)
         → 4 skip-connection feature maps at H/4 … H/32
    2. QuadModalWaterfallModule (QWTM)
         → strong multi-scale bottleneck at H/4 (128 ch)
    3. Transformer decoder (SkipFusion + DecoderBlock × 4 stages)
         QWTM output injected as extra skip at the H/4 decoder stage
    4. ×4 bilinear up-sample → num_classes logits at input resolution

    Decoder channel schedule
    ------------------------
      Stage 4 input  : 512 ch  →  256 ch   (H/32)
      Stage 3        : 256 + skip(320)  →  128 ch  (H/16)
      Stage 2        : 128 + skip(128)  →   64 ch  (H/8)
      Stage 1        :  64 + skip(64+QWTM=192) → 64 ch  (H/4)
      Seg head       :  64 ch  →  num_classes  (H, full res)

    Args
    ----
      num_classes       : material classes  (MCubeS = 20)
      rgb_var           : PVT-v2 variant for RGB — 'b4' recommended
      aux_var           : PVT-v2 variant for AoLP/DoLP/NIR — 'b2' recommended
      pretrained        : load ImageNet pretrained MiT weights via timm
      p                 : global dropout rate
      enc_checkpoint    : gradient-checkpoint the 4 encoder forward passes
                          (saves ~2–3 GB activation memory, ~20% slower)
      qwtm_checkpoint   : gradient-checkpoint each WTB inside QWTM
                          (saves ~2 GB, already defaulted on inside QWTM)
    """

    def __init__(self, num_classes = 20, rgb_var = 'b4', aux_var = 'b2', pretrained = True, p = 0.0, enc_checkpoint = True, qwtm_checkpoint = True,):
        super().__init__()
        self.enc_checkpoint = enc_checkpoint

        C  = _STAGE_CH       # [64, 128, 320, 512]
        Cq = _QWTM_CH        # 128

        # ── Input projections  (1-ch → 3-ch for auxiliary modalities) ─────────
        self.aolp_in = InputProjection(1)
        self.dolp_in = InputProjection(1)
        self.nir_in  = InputProjection(1)

        # ── Encoders ──────────────────────────────────────────────────────────
        _enc_kw = dict(features_only=True, out_indices=(0, 1, 2, 3), pretrained=pretrained)
        self.rgb_enc  = timm.create_model(f'pvt_v2_{rgb_var}', **_enc_kw)
        self.aolp_enc = timm.create_model(f'pvt_v2_{aux_var}', **_enc_kw)
        self.dolp_enc = timm.create_model(f'pvt_v2_{aux_var}', **_enc_kw)
        self.nir_enc  = timm.create_model(f'pvt_v2_{aux_var}', **_enc_kw)

        # ── Per-stage fusion blocks ────────────────────────────────────────────
        # Number of attention heads scales with channel dim (64/1=64 dh throughout)
        _heads = [1, 2, 5, 8]
        self.fuse = nn.ModuleList([
            MultimodalFusionBlock(C[s], _heads[s], _SR_RATIOS[s], p)
            for s in range(4)
        ])

        # ── Quad-Modal Waterfall Module ────────────────────────────────────────
        self.qwtm = QuadModalWaterfallModule(
            stage_ch=C, n_mod=4, out_ch=Cq, heads=8, dilations=_DILATIONS, p=p,
            checkpoint=qwtm_checkpoint,
        )

        # ── Decoder ────────────────────────────────────────────────────────────
        #  dec4  : initial channel compression from stage-4 features
        self.dec4  = nn.Sequential(
            nn.Conv2d(C[3], 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
        )
        #  Stage 3  (H/16) — small resolution: standard SA is fine
        self.skip3 = SkipFusion(ff_ch=256, skip_ch=C[2], out_ch=128)
        self.dec3  = DecoderBlock(128, heads=4, n_layers=2, p=p, axial=False)

        #  Stage 2  (H/8) — still manageable: standard SA
        self.skip2 = SkipFusion(ff_ch=128, skip_ch=C[1], out_ch=64)
        self.dec2  = DecoderBlock(64,  heads=4, n_layers=2, p=p, axial=False)

        #  Stage 1  (H/4) — high resolution: MUST use axial SA
        #  Skip = stage-1 fused (64 ch) + QWTM output (Cq ch) concatenated
        self.skip1 = SkipFusion(ff_ch=64, skip_ch=C[0] + Cq, out_ch=64)
        self.dec1  = DecoderBlock(64,  heads=8, n_layers=2, p=p, axial=True)

        # ── Segmentation head ─────────────────────────────────────────────────
        # H/4 → × 4 bilinear → H  then per-pixel classification
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _hw(t: torch.Tensor):
        return t.shape[-2], t.shape[-1]

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        rgb:  torch.Tensor,    # (B, 3, H, W)
        aolp: torch.Tensor,    # (B, 1, H, W)
        dolp: torch.Tensor,    # (B, 1, H, W)
        nir:  torch.Tensor,    # (B, 1, H, W)
    ) -> torch.Tensor:         # (B, num_classes, H, W) logits

        # ── Encode all four modalities ─────────────────────────────────────────
        # Gradient checkpointing on encoders avoids storing all intermediate
        # block activations (~2–3 GB for MiT-B4 + 3×MiT-B2 at B=2, 512²).
        def _enc(enc, x):
            if self.enc_checkpoint and self.training:
                return grad_ckpt(enc, x, use_reentrant=False)
            return enc(x)

        r = _enc(self.rgb_enc, rgb)
        a = _enc(self.aolp_enc, self.aolp_in(aolp))
        d = _enc(self.dolp_enc, self.dolp_in(dolp))
        n = _enc(self.nir_enc,  self.nir_in(nir))
        # r, a, d, n: each a list of 4 tensors  [(B,64,H/4,W/4), ..., (B,512,H/32,W/32)]

        # ── Per-stage MMCA + LocalFusion → skip connections ───────────────────
        skips = []
        for s in range(4):
            Hs, Ws = self._hw(r[s])
            skips.append(self.fuse[s]([r[s], a[s], d[s], n[s]], Hs, Ws))
        # skips[s]: (B, C[s], H/2^(s+2), W/2^(s+2))

        # ── Quad-Modal Waterfall Module ────────────────────────────────────────
        H1, W1   = self._hw(r[0])
        qwtm_out = self.qwtm([r, a, d, n], H1, W1)           # (B, 128, H/4, W/4)

        # ── Decoder — top-down with skip connections ───────────────────────────
        x = self.dec4(skips[3])                               # (B, 256, H/32, W/32)

        H3, W3 = self._hw(skips[2])
        x = F.interpolate(x, (H3, W3), mode='bilinear', align_corners=False)
        x = self.dec3(self.skip3(x, skips[2]), H3, W3)        # (B, 128, H/16, W/16)

        H2, W2 = self._hw(skips[1])
        x = F.interpolate(x, (H2, W2), mode='bilinear', align_corners=False)
        x = self.dec2(self.skip2(x, skips[1]), H2, W2)        # (B,  64, H/8,  W/8)

        x = F.interpolate(x, (H1, W1), mode='bilinear', align_corners=False)
        skip1_aug = torch.cat([skips[0], qwtm_out], dim=1)    # (B, 64+128, H/4, W/4)
        x = self.dec1(self.skip1(x, skip1_aug), H1, W1)       # (B,  64, H/4,  W/4)

        # ── Final up-sample → input resolution ────────────────────────────────
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return self.seg_head(x)                                # (B, num_classes, H, W)


# Utilities
def count_params(model):
    """Return total and trainable parameter counts (in millions)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building QuadWaterfall (pretrained=False for quick test) …")
    model = QuadWaterfall(num_classes=20, pretrained=False).eval()
    stats = count_params(model)
    print(f"  Parameters: {stats['total_M']:.1f}M total, {stats['trainable_M']:.1f}M trainable")

    # Use 512×512 crops as done during MCubeS training
    B, H, W = 2, 512, 512
    rgb  = torch.randn(B, 3, H, W)
    aolp = torch.randn(B, 1, H, W)
    dolp = torch.randn(B, 1, H, W)
    nir  = torch.randn(B, 1, H, W)

    with torch.no_grad():
        logits = model(rgb, aolp, dolp, nir)

    assert logits.shape == (B, 20, H, W), f"Shape error: {logits.shape}"
    print(f"  Inputs:")
    print(f"    RGB  {tuple(rgb.shape)}")
    print(f"    AoLP {tuple(aolp.shape)}")
    print(f"    DoLP {tuple(dolp.shape)}")
    print(f"    NIR  {tuple(nir.shape)}")
    print(f"  Output {tuple(logits.shape)}")
    print("  ✓ Smoke test passed")
