"""
modern_tft.py
=============

A Temporal Fusion Transformer (Lim et al., 2019) rebuilt with a modern
transformer recipe. The classic TFT skeleton is preserved end to end:

    inputs -> per-variable embedding -> Variable Selection Networks (VSN)
           -> static covariate encoders -> LSTM encoder/decoder (locality)
           -> static enrichment -> temporal self-attention
           -> gated skip -> quantile head

Modern optimizations and where each one lives:

  - No bias on linear layers ........ every nn.Linear uses bias=False;
                                       continuous embeddings are a pure outer
                                       product (no offset).
  - Clean residual path ............. blocks add raw x; norms only ever sit on
                                       the sublayer output, never on the skip.
  - RMSNorm before AND after ........ "sandwich" norm:
                                       post_norm(sublayer(pre_norm(x))).
  - Gated activations ............... GeGLU FFN in the transformer block; GRNs
                                       are themselves gated (GLU/GeGLU/ReGLU).
  - Parallel blocks, summed ......... out = x + attn(...) + ffn(...).
  - RoPE ............................ rotary position embedding on Q, K.
  - d_ff = 2.5 * d_model ............ set in TFTConfig.__post_init__.
  - Aspect-ratio sizing ............. recommend_param_budget() helper.
  - dropout=0.1, wd=0.1 + cosine .... defaults + build_optimizer (decoupled wd,
                                       param groups) + cosine-with-warmup.
  - QK-norm ......................... RMSNorm on Q and K (per head) pre-dot.
  - GQA / MQA ....................... n_kv_heads < n_heads => grouped queries;
                                       n_kv_heads == 1 => multi-query.
  - Grad checkpoint + accumulation .. checkpoint on blocks; accumulation in loop.
  - Flash attention ................. F.scaled_dot_product_attention. For TFT's
                                       interpretable attention maps, flip
                                       need_weights=True to use the eager path.
  - Mixed precision ................. autocast (+ GradScaler for fp16) in loop.
  - torch.compile ................... operator fusion, memory-coalesced layouts,
                                       and automatic Triton lowering.
  - Optional Triton ................. hand-written RMSNorm forward kernel used at
                                       inference, as an illustration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Optional Triton. torch.compile already emits Triton for the whole model; this
# hand-written forward kernel is an illustration, used only at inference time
# (no custom backward -> no autograd through it).
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - triton is optional
    HAS_TRITON = False


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class TFTConfig:
    # ---- data schema -------------------------------------------------------
    static_cat_cardinalities: tuple[int, ...] = ()    # categorical static covariates
    n_static_real: int = 0                            # continuous static covariates
    known_cat_cardinalities: tuple[int, ...] = ()     # known-future categoricals
    n_known_real: int = 0                             # known-future continuous
    observed_cat_cardinalities: tuple[int, ...] = ()  # past-only categoricals
    n_observed_real: int = 1                          # past-only continuous (incl. target history)

    # ---- model dims --------------------------------------------------------
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 1            # GQA: < n_heads => grouped; == 1 => MQA
    n_blocks: int = 1             # TFT classically uses 1 attention layer
    d_ff: Optional[int] = None    # defaults to round(2.5 * d_model)
    dropout: float = 0.1

    # ---- attention ---------------------------------------------------------
    rope_base: float = 10_000.0
    qk_norm: bool = True
    use_flash: bool = True        # SDPA fast path; False => eager + weights
    parallel_blocks: bool = True
    grad_checkpoint: bool = False
    max_len: int = 4096           # RoPE table size

    # ---- gating ------------------------------------------------------------
    ffn_gate: str = "geglu"       # block FFN: "geglu" | "reglu" | "swiglu"
    grn_gate: str = "glu"         # GRN gate: "glu" | "geglu" | "reglu"

    # ---- head --------------------------------------------------------------
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = round(2.5 * self.d_model)
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ===========================================================================
# Norm (RMSNorm, with optional Triton inference kernel + QK-norm reuse)
# ===========================================================================
if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        X += row * stride
        Y += row * stride
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, x * rstd * w, mask=mask)

    def _rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        N = x.shape[-1]
        x2 = x.reshape(-1, N).contiguous()
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_fwd_kernel[(x2.shape[0],)](x2, weight, y, x2.stride(0), N, eps, BLOCK=BLOCK)
        return y.reshape(x.shape)


class RMSNorm(nn.Module):
    """Root-mean-square norm. On CUDA at inference it uses the Triton kernel;
    during training it stays in PyTorch so torch.compile can fuse it."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and x.is_cuda and not torch.is_grad_enabled():
            return _rmsnorm_triton(x, self.weight, self.eps)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float, max_len: int):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)             # (max_len, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)      # (max_len, hd)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd) ; cos/sin: (T, hd)
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


# ===========================================================================
# Gated FFN -- GeGLU / ReGLU / SwiGLU
# ===========================================================================
class GatedFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, gate: str = "geglu", dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = {"geglu": F.gelu, "reglu": F.relu, "swiglu": F.silu}[gate]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.drop(self.act(self.w_gate(x)) * self.w_up(x)))


# ===========================================================================
# Attention -- GQA + RoPE + QK-norm; Flash fast path, eager interpretable path
# ===========================================================================
class Attention(nn.Module):
    def __init__(self, cfg: TFTConfig):
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.n_heads
        self.nkv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.rep = self.nh // self.nkv

        self.q_proj = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.attn_drop = cfg.dropout

    def forward(self, x, cos, sin, need_weights: bool = False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)   # (B,nh,T,hd)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)  # (B,nkv,T,hd)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        q = self.q_norm(q)               # norm BEFORE the dot product
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # expand KV groups -> n_heads (storage/cache stays at nkv heads)
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)

        if self.cfg.use_flash and not need_weights:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
            weights = None
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            attn = scores.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)
            out = attn @ v
            weights = attn.mean(dim=1)   # (B,T,T) interpretable map (avg heads)

        out = out.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o_proj(out), weights


# ===========================================================================
# Modern transformer block -- sandwich norm, clean residual, parallel
# ===========================================================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg: TFTConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.ffn = GatedFFN(cfg.d_model, cfg.d_ff, cfg.ffn_gate, cfg.dropout)
        self.pre_attn = RMSNorm(cfg.d_model)
        self.post_attn = RMSNorm(cfg.d_model)
        self.pre_ffn = RMSNorm(cfg.d_model)
        self.post_ffn = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def _attn(self, x, cos, sin, need_weights):
        a, w = self.attn(self.pre_attn(x), cos, sin, need_weights)
        return self.drop(self.post_attn(a)), w

    def _ffn(self, x):
        return self.drop(self.post_ffn(self.ffn(self.pre_ffn(x))))

    def forward(self, x, cos, sin, need_weights: bool = False):
        use_ckpt = self.cfg.grad_checkpoint and self.training and not need_weights
        if self.cfg.parallel_blocks:
            if use_ckpt:
                a, _ = checkpoint(self._attn, x, cos, sin, False, use_reentrant=False)
                f = checkpoint(self._ffn, x, use_reentrant=False)
                w = None
            else:
                a, w = self._attn(x, cos, sin, need_weights)
                f = self._ffn(x)
            return x + a + f, w          # clean residual: raw x
        else:
            a, w = self._attn(x, cos, sin, need_weights)
            x = x + a
            x = x + self._ffn(x)
            return x, w


# ===========================================================================
# TFT building blocks: GRN, embeddings, variable selection, gated skip
# ===========================================================================
class GRN(nn.Module):
    """Gated Residual Network -- TFT's workhorse. RMSNorm, no bias, gated."""

    def __init__(self, in_dim, hidden, out_dim, ctx_dim=None, dropout=0.1, gate="glu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.ctx = nn.Linear(ctx_dim, hidden, bias=False) if ctx_dim else None
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden, out_dim, bias=False)
        self.val = nn.Linear(hidden, out_dim, bias=False)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self._gate_act = {"glu": torch.sigmoid, "geglu": F.gelu, "reglu": F.relu}[gate]

    def forward(self, a, c=None):
        h = self.fc1(a)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = F.elu(h)
        h = self.drop(self.fc2(h))
        gated = self.val(h) * self._gate_act(self.gate(h))
        return self.norm(self.skip(a) + gated)


class ContinuousEmbedding(nn.Module):
    """Per-variable scalar -> d_model. Pure outer product, no bias."""

    def __init__(self, n_vars: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_vars, d_model) * 0.02)

    def forward(self, x):                            # x: (..., n_vars)
        return x.unsqueeze(-1) * self.weight         # (..., n_vars, d_model)


class CategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities, d_model: int):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, d_model) for c in cardinalities)

    def forward(self, x):                            # x: (..., n_cat) long
        if len(self.embs) == 0:
            return x.new_zeros(*x.shape[:-1], 0, 0)
        return torch.stack([e(x[..., i]) for i, e in enumerate(self.embs)], dim=-2)


class VariableSelectionNetwork(nn.Module):
    """Soft variable selection: per-variable GRN + softmax weights."""

    def __init__(self, n_vars: int, d_model: int, ctx_dim: Optional[int], dropout: float, gate: str):
        super().__init__()
        self.n_vars = n_vars
        self.flatten_grn = GRN(n_vars * d_model, d_model, n_vars, ctx_dim, dropout, gate)
        self.var_grns = nn.ModuleList(
            GRN(d_model, d_model, d_model, None, dropout, gate) for _ in range(n_vars)
        )

    def forward(self, var_embeds: torch.Tensor, ctx=None):
        flat = var_embeds.flatten(start_dim=-2)                # (..., n_vars*d_model)
        weights = self.flatten_grn(flat, ctx).softmax(dim=-1)  # (..., n_vars)
        processed = torch.stack(
            [grn(var_embeds[..., i, :]) for i, grn in enumerate(self.var_grns)], dim=-2
        )                                                      # (..., n_vars, d_model)
        out = (weights.unsqueeze(-1) * processed).sum(dim=-2)  # (..., d_model)
        return out, weights


class GateAddNorm(nn.Module):
    """GLU gated skip connection + RMSNorm (TFT's gate-add-norm), modernized."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.val = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, residual):
        x = self.drop(x)
        x = self.val(x) * torch.sigmoid(self.gate(x))
        return self.norm(x + residual)


# ===========================================================================
# The model
# ===========================================================================
class ModernTFT(nn.Module):
    def __init__(self, cfg: TFTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ---- embeddings -----------------------------------------------------
        self.static_cat_emb = CategoricalEmbedding(cfg.static_cat_cardinalities, d)
        self.static_real_emb = ContinuousEmbedding(cfg.n_static_real, d) if cfg.n_static_real else None
        self.known_cat_emb = CategoricalEmbedding(cfg.known_cat_cardinalities, d)
        self.known_real_emb = ContinuousEmbedding(cfg.n_known_real, d) if cfg.n_known_real else None
        self.obs_cat_emb = CategoricalEmbedding(cfg.observed_cat_cardinalities, d)
        self.obs_real_emb = ContinuousEmbedding(cfg.n_observed_real, d) if cfg.n_observed_real else None

        n_static = len(cfg.static_cat_cardinalities) + cfg.n_static_real
        n_known = len(cfg.known_cat_cardinalities) + cfg.n_known_real
        n_obs = len(cfg.observed_cat_cardinalities) + cfg.n_observed_real
        assert n_static >= 1 and n_known >= 1 and (n_known + n_obs) >= 1

        # ---- variable selection --------------------------------------------
        self.vsn_static = VariableSelectionNetwork(n_static, d, None, cfg.dropout, cfg.grn_gate)
        self.vsn_encoder = VariableSelectionNetwork(n_known + n_obs, d, d, cfg.dropout, cfg.grn_gate)
        self.vsn_decoder = VariableSelectionNetwork(n_known, d, d, cfg.dropout, cfg.grn_gate)

        # ---- static covariate encoders (4 contexts) ------------------------
        self.grn_select = GRN(d, d, d, None, cfg.dropout, cfg.grn_gate)   # c_s
        self.grn_enrich = GRN(d, d, d, None, cfg.dropout, cfg.grn_gate)   # c_c
        self.grn_lstm_h = GRN(d, d, d, None, cfg.dropout, cfg.grn_gate)   # h0
        self.grn_lstm_c = GRN(d, d, d, None, cfg.dropout, cfg.grn_gate)   # c0

        # ---- LSTM enc/dec (locality enhancement) ---------------------------
        self.enc_lstm = nn.LSTM(d, d, batch_first=True)
        self.dec_lstm = nn.LSTM(d, d, batch_first=True)
        self.lstm_gate = GateAddNorm(d, cfg.dropout)

        # ---- static enrichment ---------------------------------------------
        self.enrich_grn = GRN(d, d, d, d, cfg.dropout, cfg.grn_gate)

        # ---- temporal self-attention (modern blocks) ----------------------
        self.rope = RoPE(cfg.head_dim, cfg.rope_base, cfg.max_len)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_blocks))
        self.attn_gate = GateAddNorm(d, cfg.dropout)

        # ---- quantile head --------------------------------------------------
        self.head = nn.Linear(d, len(cfg.quantiles), bias=False)

    def _gather(self, cat_emb, cat_x, real_emb, real_x):
        parts = []
        if cat_x is not None and cat_x.shape[-1] > 0:
            parts.append(cat_emb(cat_x))
        if real_x is not None and real_emb is not None and real_x.shape[-1] > 0:
            parts.append(real_emb(real_x))
        return torch.cat(parts, dim=-2)

    def forward(self, batch: dict, need_weights: bool = False):
        """
        batch keys:
            static_cat   : (B, n_static_cat)        long
            static_real  : (B, n_static_real)       float
            known_cat    : (B, T, n_known_cat)      long   (T = L + H)
            known_real   : (B, T, n_known_real)     float
            observed_cat : (B, L, n_observed_cat)   long
            observed_real: (B, L, n_observed_real)  float
        """
        known_real = batch.get("known_real")
        known_cat = batch.get("known_cat")
        T = (known_real if known_real is not None else known_cat).shape[1]
        obs = batch.get("observed_real")
        obs = obs if obs is not None else batch.get("observed_cat")
        L = obs.shape[1]
        H = T - L

        # ---- static -> selection + 4 contexts ------------------------------
        static_vars = self._gather(self.static_cat_emb, batch.get("static_cat"),
                                   self.static_real_emb, batch.get("static_real"))
        static_vec, static_w = self.vsn_static(static_vars)
        c_s = self.grn_select(static_vec)
        c_c = self.grn_enrich(static_vec)
        h0 = self.grn_lstm_h(static_vec).unsqueeze(0)
        c0 = self.grn_lstm_c(static_vec).unsqueeze(0)

        # ---- embeddings ----------------------------------------------------
        known_vars = self._gather(self.known_cat_emb, known_cat, self.known_real_emb, known_real)
        obs_vars = self._gather(self.obs_cat_emb, batch.get("observed_cat"),
                                self.obs_real_emb, batch.get("observed_real"))

        # ---- encoder selection (past) --------------------------------------
        enc_vars = torch.cat([known_vars[:, :L], obs_vars], dim=-2)
        enc_feat, enc_w = self.vsn_encoder(enc_vars, c_s.unsqueeze(1).expand(-1, L, -1))

        # ---- decoder selection (future) ------------------------------------
        dec_vars = known_vars[:, L:]
        dec_feat, dec_w = self.vsn_decoder(dec_vars, c_s.unsqueeze(1).expand(-1, H, -1))

        # ---- LSTM enc/dec + gated skip -------------------------------------
        enc_out, (hn, cn) = self.enc_lstm(enc_feat, (h0, c0))
        dec_out, _ = self.dec_lstm(dec_feat, (hn, cn))
        lstm_out = torch.cat([enc_out, dec_out], dim=1)
        vsn_feat = torch.cat([enc_feat, dec_feat], dim=1)
        temporal = self.lstm_gate(lstm_out, vsn_feat)

        # ---- static enrichment ---------------------------------------------
        enriched = self.enrich_grn(temporal, c_c.unsqueeze(1).expand(-1, T, -1))

        # ---- temporal self-attention ---------------------------------------
        cos, sin = self.rope(T)
        x = enriched
        attn_map = None
        for blk in self.blocks:
            x, w = blk(x, cos, sin, need_weights)
            if w is not None:
                attn_map = w
        attended = self.attn_gate(x, temporal)

        # ---- output (decoder steps only) -----------------------------------
        logits = self.head(attended[:, L:])          # (B, H, Q)

        out = {"prediction": logits}
        if need_weights:
            out.update(static_weights=static_w, encoder_weights=enc_w,
                       decoder_weights=dec_w, attention=attn_map)
        return out


# ===========================================================================
# Loss
# ===========================================================================
def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """Pinball / quantile loss. preds: (B,H,Q), target: (B,H)."""
    q = torch.as_tensor(quantiles, device=preds.device, dtype=preds.dtype)
    errors = target.unsqueeze(-1) - preds
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


# ===========================================================================
# Optimization helpers
# ===========================================================================
def recommend_param_budget(n_tokens: int, ratio: int = 150) -> float:
    """'Aspect ratio' sizing: ~100-200 tokens per parameter. For a numeric TFT,
    treat (timesteps x active features) as a rough token proxy; use as a sanity
    bound, not a hard target."""
    return n_tokens / ratio


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.1):
    """Decoupled weight decay (AdamW). Norm gains / embeddings excluded; decay
    acts as an optimizer complement on the matmul weights."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "emb" in name.lower()) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())


def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compile_model(model: nn.Module, **kwargs) -> nn.Module:
    """torch.compile: operator fusion, memory-coalesced layouts, Triton lowering."""
    return torch.compile(model, **kwargs)


# ===========================================================================
# Training step (grad checkpoint + accumulation, mixed precision)
# ===========================================================================
def train_epoch(model, loader, optimizer, scheduler, cfg: TFTConfig,
                device="cuda", accum_steps: int = 1, amp_dtype=torch.bfloat16):
    model.train()
    use_scaler = (amp_dtype == torch.float16) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        target = batch.pop("target")
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
            out = model(batch)
            loss = quantile_loss(out["prediction"], target, cfg.quantiles) / accum_steps

        scaler.scale(loss).backward() if use_scaler else loss.backward()

        if (i + 1) % accum_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()


# ===========================================================================
# Smoke test (CPU, eager): forward + backward + interpretability path
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TFTConfig(
        static_cat_cardinalities=(5, 3), n_static_real=2,
        known_cat_cardinalities=(7,), n_known_real=2,
        observed_cat_cardinalities=(), n_observed_real=3,
        d_model=32, n_heads=4, n_kv_heads=2, n_blocks=2,
        dropout=0.1, qk_norm=True, use_flash=True,
        parallel_blocks=True, grad_checkpoint=True,
        quantiles=(0.1, 0.5, 0.9),
    )
    B, L, H = 8, 24, 6
    T = L + H
    model = ModernTFT(cfg)
    batch = dict(
        static_cat=torch.randint(0, 3, (B, 2)),
        static_real=torch.randn(B, 2),
        known_cat=torch.randint(0, 7, (B, T, 1)),
        known_real=torch.randn(B, T, 2),
        observed_real=torch.randn(B, L, 3),
    )
    target = torch.randn(B, H)

    print(f"params: {count_params(model):,}")
    print(f"d_ff (2.5*d_model): {cfg.d_ff}")

    out = model(batch)
    loss = quantile_loss(out["prediction"], target, cfg.quantiles)
    loss.backward()
    print(f"prediction (B,H,Q): {tuple(out['prediction'].shape)} | loss {loss.item():.4f} | backward OK")

    model.eval()
    with torch.no_grad():
        interp = model(batch, need_weights=True)
    print(f"attention {tuple(interp['attention'].shape)} | "
          f"static sel {tuple(interp['static_weights'].shape)} | "
          f"encoder sel {tuple(interp['encoder_weights'].shape)}")

    opt = build_optimizer(model)
    sch = cosine_warmup_schedule(opt, 10, 100)
    print(f"optimizer groups: {len(opt.param_groups)} (decay / no-decay) | OK")
