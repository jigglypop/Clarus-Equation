"""ClarusLM: CE 3x3+1 격자 위상동형 AGI의 LLM 구현.

Transformer에 대한 CE 수정 4가지:
  LBONorm      -- LayerNorm 대체 (라플라스-벨트라미 확산)
  GaugeLattice -- FFN 대체 (SU(3) x SU(2) x U(1) + Phi)
  spectral_norm -- 유니타리 조건 |det T|^2 <= 1 (환각 구조적 억제)
  curvature loss -- lambda |Delta_g Phi|^2 정규화

Backend dispatch (auto):
  CUDA  -- fused kernels (training + inference)
  Rust  -- clarus/core via PyO3 (CPU training/inference)
  Torch -- pure PyTorch fallback

Reference: docs/6_뇌/agi.md
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from clarus.engine import get_constants, backend_info
    _ce = get_constants()
    ALPHA_S = _ce.alpha_s
    ALPHA_W = _ce.alpha_w
    ALPHA_EM = _ce.alpha_em_mz
    _BACKEND = backend_info()
except ImportError:
    ALPHA_S = 0.11789
    ALPHA_W = 0.03352
    ALPHA_EM = 0.00775
    _BACKEND = "standalone"

try:
    from clarus.ops import (
        topk_silu as _ops_topk_silu,
        lbo_fused_fwd as _ops_lbo_fwd,
        power_iter_step as _ops_power_iter,
        gauge_lattice_fwd as _ops_gauge_fwd,
        ops_backend as _ops_backend,
    )
    _HAS_OPS = True
except ImportError:
    _HAS_OPS = False


def split_ce_dims(total_dim):
    total_alpha = ALPHA_S + ALPHA_W + ALPHA_EM
    d3 = max(1, round(total_dim * ALPHA_S / total_alpha))
    d2 = max(1, round(total_dim * ALPHA_W / total_alpha))
    d1 = total_dim - d3 - d2
    if d1 < 1:
        if d3 >= d2 and d3 > 1:
            d3 -= 1
        elif d2 > 1:
            d2 -= 1
        d1 = total_dim - d3 - d2
    if d1 < 1:
        raise ValueError(f'Invalid CE split for total_dim={total_dim}')
    return d3, d2, d1


def _lbo_cpu_fwd_fn(
    x: torch.Tensor, V: torch.Tensor, h_val: torch.Tensor,
    scale: torch.Tensor, bias: torch.Tensor,
    alpha_conf_abs: torch.Tensor, dim: int,
    need_curvature: bool,
) -> Tuple[torch.Tensor, float]:
    x_n = F.layer_norm(x, [dim])
    phi_sq = x_n.detach().pow(2).mean()
    conformal = torch.exp(-alpha_conf_abs * phi_sq)
    V_eff = V * conformal
    proj = x_n @ V_eff.t()
    xW = proj @ V_eff
    curvature: float = 0.0
    if need_curvature:
        Lx = x_n - xW
        nrm = Lx.detach().norm()
        curvature = (nrm * nrm / Lx.numel()).item()
    pre = torch.lerp(x_n, xW, h_val)
    return torch.addcmul(bias, pre, scale), curvature


try:
    _lbo_cpu_fwd = torch.jit.script(_lbo_cpu_fwd_fn)
except Exception:
    _lbo_cpu_fwd = _lbo_cpu_fwd_fn


class LBONorm(nn.Module):
    """LayerNorm + Laplace-Beltrami 확산 + 등각 자기참조.

    xW = x @ V_eff^T @ V_eff  (low-rank projection, conformal-scaled)
    Lx = x - xW               (Laplacian residual)
    out = (x - h*Lx) * scale + bias

    수렴 조건 (agi.md 6.2): h < 1/lambda_max(V^T V).
    등각 인자 (agi.md 7.2): g[Phi] = e^{-2*alpha*Phi} delta
      -> V_eff = V * e^{-alpha_conf * |x|^2}
    h=0이면 LayerNorm과 동일.
    """

    def __init__(self, dim, rank=None, step_size=0.1):
        super().__init__()
        if rank is None:
            rank = max(4, dim // 8)
        self.dim = dim
        self.rank = rank
        self.V = nn.Parameter(torch.randn(rank, dim) * (1.0 / math.sqrt(rank)))
        self.h = nn.Parameter(torch.tensor(step_size))
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.alpha_conf = nn.Parameter(torch.tensor(0.01))
        self._curvature = 0.0
        self._need_curvature = True
        self._h_step = 0
        self._h_period = 16
        self._h_next_update = 1
        self.register_buffer('_spectral_v', F.normalize(torch.randn(dim), dim=0))
        self.register_buffer('_h_max', torch.tensor(1.0))

    def _h_bound(self):
        """sigma_max(V) via 1-step power iteration -> h < 1/sigma_max^2.
        Cached for _h_period forward calls. Dispatches to Rust on CPU."""
        self._h_step += 1
        if self._h_step >= self._h_next_update:
            self._h_next_update = self._h_step + self._h_period
            if _HAS_OPS and not self.V.is_cuda:
                new_v, sigma = _ops_power_iter(
                    self.V, self._spectral_v, self.dim, self.rank)
                with torch.no_grad():
                    self._spectral_v.copy_(new_v)
                    self._h_max.fill_(1.0 / (sigma ** 2 + 1e-6))
            else:
                with torch.no_grad():
                    u = F.normalize(self.V @ self._spectral_v, dim=0)
                    self._spectral_v.copy_(F.normalize(self.V.t() @ u, dim=0))
                    sigma_max = (self.V @ self._spectral_v).norm()
                    self._h_max.fill_(1.0 / (sigma_max ** 2 + 1e-6))
        return self._h_max

    def forward(self, x):
        h_max = self._h_bound()
        h_val = self.h.abs().clamp(max=h_max)

        # CUDA: dispatch to fused kernel (training + inference)
        if _HAS_OPS and x.is_cuda:
            x = F.layer_norm(x, (self.dim,))
            out, curv = _ops_lbo_fwd(
                x, self.V, float(h_val.detach()), self.scale, self.bias,
                float(self.alpha_conf.detach().abs()), self.dim, self.rank,
                need_curvature=self._need_curvature,
            )
            if self._need_curvature:
                self._curvature = curv
            return out

        # CPU: JIT-fused (layer_norm + conformal + LBO + scale/bias)
        out, curv = _lbo_cpu_fwd(
            x, self.V, h_val, self.scale, self.bias,
            self.alpha_conf.abs(), self.dim, self._need_curvature,
        )
        if self._need_curvature:
            self._curvature = curv
        return out


EPS2 = 0.0487  # CE bootstrap fixed point


class TopKSiLU(nn.Module):
    """SiLU + TopK sparsity.

    Backend dispatch:
      CUDA  -- fused element-wise kernel (threshold pre-computed).
      Rust  -- fused SiLU + quickselect per row (rayon parallel).
      Torch -- running-threshold fallback.
    """

    def __init__(self, dim, ratio=1.0, cal_period=8, cal_samples=64):
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.k = max(1, math.ceil(ratio * dim))
        self.full = (ratio >= 1.0)
        self._cal_period = cal_period
        self._cal_samples = cal_samples
        self._step = 0
        self._next_cal = 1
        if not self.full:
            self.register_buffer('_thr', torch.tensor(0.0))

    def forward(self, x):
        if self.full or self.k >= x.size(-1):
            return F.silu(x)

        # CUDA: fused kernel (faster than per-element PyTorch)
        if _HAS_OPS and x.is_cuda:
            thr = float(self._thr) if hasattr(self, '_thr') else 0.0
            return _ops_topk_silu(x, self.k, self.ratio, threshold=thr)

        # CPU: PyTorch running-threshold
        h = F.silu(x)
        if self.training:
            self._step += 1
            if self._step >= self._next_cal:
                self._next_cal = self._step + self._cal_period
                with torch.no_grad():
                    flat = h.detach().abs().reshape(-1, h.size(-1))
                    n = min(self._cal_samples, flat.size(0))
                    thr = flat[:n].kthvalue(
                        self.dim - self.k + 1, dim=-1
                    ).values.mean()
                    if self._thr.item() == 0.0:
                        self._thr.fill_(thr)
                    else:
                        self._thr.lerp_(thr, 0.2)
            return h.masked_fill(h.abs() < self._thr, 0.0)
        else:
            if self._thr.item() > 0:
                return h.masked_fill(h.abs() < self._thr, 0.0)
            abs_h = h.abs()
            thr = abs_h.kthvalue(
                x.size(-1) - self.k + 1, dim=-1, keepdim=True
            ).values
            return h.masked_fill(abs_h < thr, 0.0)


class GaugeLattice(nn.Module):
    """3x3+1 게이지 격자 FFN.

    diag(SU(3), SU(2), U(1)) + eps * mix + Phi_LBO.
    채널 비율: alpha_s:alpha_w:alpha_em = 74.1:21.1:4.9 (CE 연역).
    low-rank mixing을 0 초기화해 perturbative coupling으로 시작한다.
    sparsity: 활성 뉴런 비율. 1.0=dense, EPS2=CE 부트스트랩 희소.
    """

    def __init__(self, dim, mult=4, hidden_dim=None, mix_rank=None, sparsity=1.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = max(dim, int(round(hidden_dim if hidden_dim is not None else dim * mult)))
        self.mix_rank = max(0, dim // 8 if mix_rank is None else int(mix_rank))
        self.sparsity = sparsity
        self.d3, self.d2, self.d1 = split_ce_dims(dim)
        h3, h2, h1 = split_ce_dims(self.hidden_dim)

        self.su3_up = nn.Linear(self.d3, h3, bias=False)
        self.su3_act = TopKSiLU(h3, sparsity)
        self.su3_down = nn.utils.spectral_norm(nn.Linear(h3, self.d3, bias=False))

        self.su2_up = nn.Linear(self.d2, h2, bias=False)
        self.su2_act = TopKSiLU(h2, sparsity)
        self.su2_down = nn.utils.spectral_norm(nn.Linear(h2, self.d2, bias=False))

        self.u1_up = nn.Linear(self.d1, h1, bias=False)
        self.u1_act = TopKSiLU(h1, sparsity)
        self.u1_down = nn.utils.spectral_norm(nn.Linear(h1, self.d1, bias=False))

        if self.mix_rank > 0:
            self.mix_down = nn.Linear(dim, self.mix_rank, bias=False)
            self.mix_up = nn.Linear(self.mix_rank, dim, bias=False)
            nn.init.zeros_(self.mix_up.weight)
        else:
            self.mix_down = None
            self.mix_up = None
        self.phi = LBONorm(dim)

    @property
    def mixing_ratio(self):
        """||U_down U_up^T||_F / ||T_diag||_F -- perturbative condition (agi.md 6.3)."""
        if self.mix_up is None:
            return 0.0
        w_up = self.mix_up.weight_orig if hasattr(self.mix_up, 'weight_orig') else self.mix_up.weight
        w_down = self.mix_down.weight
        mix_norm = (w_down.t() @ w_up.t()).norm()
        su3_w = self.su3_down.weight_orig if hasattr(self.su3_down, 'weight_orig') else self.su3_down.weight
        su2_w = self.su2_down.weight_orig if hasattr(self.su2_down, 'weight_orig') else self.su2_down.weight
        u1_w = self.u1_down.weight_orig if hasattr(self.u1_down, 'weight_orig') else self.u1_down.weight
        diag_norm = (su3_w.norm() ** 2 + su2_w.norm() ** 2 + u1_w.norm() ** 2).sqrt()
        return mix_norm / (diag_norm + 1e-8)

    def forward(self, x):
        s3 = self.d3
        s32 = s3 + self.d2
        y = torch.cat([
            self.su3_down(self.su3_act(self.su3_up(x[..., :s3]))),
            self.su2_down(self.su2_act(self.su2_up(x[..., s3:s32]))),
            self.u1_down(self.u1_act(self.u1_up(x[..., s32:]))),
        ], dim=-1)
        if self.mix_up is not None:
            y = y + self.mix_up(self.mix_down(y))
        return self.phi(y)


class ClarusAttention(nn.Module):
    """Multi-head attention + spectral norm (유니타리 조건).

    F.scaled_dot_product_attention 사용 (Flash Attention 자동 디스패치).
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.utils.spectral_norm(nn.Linear(dim, dim, bias=False))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class ClarusBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult=4, ffn_hidden_dim=None,
                 mix_rank=None, sparsity=1.0):
        super().__init__()
        self.norm1 = LBONorm(dim)
        self.attn = ClarusAttention(dim, n_heads)
        self.norm2 = LBONorm(dim)
        self.ffn = GaugeLattice(
            dim,
            ffn_mult,
            hidden_dim=ffn_hidden_dim,
            mix_rank=mix_rank,
            sparsity=sparsity,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    @property
    def curvature(self):
        return (self.norm1._curvature + self.norm2._curvature
                + self.ffn.phi._curvature) / 3


class ClarusLM(nn.Module):
    """Clarus Equation Language Model.

    Args:
        vocab_size: 어휘 크기
        dim: 모델 차원
        n_layers: 블록 수
        n_heads: 어텐션 헤드 수
        max_seq_len: 최대 시퀀스 길이
        ffn_mult: FFN 확장 배수
        lambda_curv: 곡률 정규화 계수
    """

    def __init__(self, vocab_size, dim=256, n_layers=6, n_heads=8,
                 max_seq_len=512, ffn_mult=4, ffn_hidden_dim=None,
                 mix_rank=None, lambda_curv=0.01, lambda_mix=0.01,
                 sparsity=1.0, use_checkpoint=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.lambda_curv = lambda_curv
        self.lambda_mix = lambda_mix
        self._use_checkpoint = use_checkpoint
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [
                ClarusBlock(
                    dim,
                    n_heads,
                    ffn_mult=ffn_mult,
                    ffn_hidden_dim=ffn_hidden_dim,
                    mix_rank=mix_rank,
                    sparsity=sparsity,
                )
                for _ in range(n_layers)
            ])
        self.norm = LBONorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.register_buffer('_pos_idx', torch.arange(max_seq_len), persistent=False)
        self.apply(self._init)
        self._lbo_modules = [m for m in self.modules() if isinstance(m, LBONorm)]
        self._curv_enabled = None

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            w = m.weight_orig if hasattr(m, 'weight_orig') else m.weight
            nn.init.normal_(w, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _set_curvature_tracking(self, enabled: bool):
        if enabled == self._curv_enabled:
            return
        self._curv_enabled = enabled
        for m in self._lbo_modules:
            m._need_curvature = enabled

    def forward(self, idx, targets=None):
        B, T = idx.shape
        need_curv = targets is not None and self.lambda_curv > 0
        self._set_curvature_tracking(need_curv)

        x = self.tok_emb(idx) + self.pos_emb(self._pos_idx[:T])
        if self.training and self._use_checkpoint:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
        else:
            for block in self.blocks:
                x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce
            if need_curv:
                curv = sum(b.curvature for b in self.blocks) / len(self.blocks)
                loss = loss + self.lambda_curv * curv
            if self.lambda_mix > 0:
                with torch.no_grad():
                    mix_r = sum(b.ffn.mixing_ratio for b in self.blocks) / len(self.blocks)
                loss = loss + self.lambda_mix * float(mix_r)
        return logits, loss

    @torch.no_grad()
    def continuation_loss(self, prefix, full_sequence):
        if full_sequence.size(1) <= prefix.size(1):
            return 0.0
        logits = self(full_sequence[:, :-1])[0]
        targets = full_sequence[:, 1:].clone()
        if prefix.size(1) > 1:
            targets[:, :prefix.size(1) - 1] = -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return float(loss.item())

    @torch.no_grad()
    def _sample(self, idx, n, temperature=0.8, top_k=40):
        for _ in range(n):
            x = idx[:, -self.max_seq_len:]
            logits = self(x)[0][:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40,
                 c3_passes=1, c3_candidates=1):
        if c3_passes <= 1 or c3_candidates <= 1:
            return self._sample(idx, n, temperature=temperature, top_k=top_k)

        output = idx.clone()
        remaining = n
        for pass_idx in range(c3_passes):
            rounds_left = c3_passes - pass_idx
            step_tokens = max(1, math.ceil(remaining / rounds_left))
            prefix = output
            best_candidate = None
            best_score = None
            for _ in range(c3_candidates):
                candidate = self._sample(
                    prefix.clone(),
                    step_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                score = self.continuation_loss(prefix, candidate)
                if best_score is None or score < best_score:
                    best_candidate = candidate
                    best_score = score
            output = best_candidate
            remaining = n - (output.size(1) - idx.size(1))
            if remaining <= 0:
                break
        return output

    def lattice_summary(self):
        """3x3+1 격자 구조 요약."""
        b = self.blocks[0].ffn
        lines = [
            f'SU(3) binding:  {b.d3:4d} dims',
            f'SU(2) decision: {b.d2:4d} dims',
            f'U(1)  attention:{b.d1:4d} dims',
            f'FFN hidden dim: {b.hidden_dim:4d}',
            f'Mixing rank:    {b.mix_rank:4d}',
            f'Sparsity:       {b.sparsity:.4f} (k_su3={b.su3_act.k})',
            f'Phi smoothing:  LBO (rank={b.phi.V.shape[0]})',
        ]
        return '\n'.join(lines)
