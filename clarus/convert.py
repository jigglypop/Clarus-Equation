"""CE-AGI Model Converter: CE teacher transplant -> CE runtime artifact.

Primary path:
  1. Phase 1 transplant (`LayerNorm -> LBONorm`, attention `c_proj -> spectral`)
  2. Optional Phase 2 (`MLP -> GaugeLatticeV2`)
  3. Extract CE runtime matrix from the transplanted teacher
  4. Distill prompt encoder + CE lexical decoder on the actual runtime path
  5. Save a CE-only runtime artifact, with optional CE-teacher weights for eval
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from math import e, pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ce_ops import pack_sparse as ce_pack_sparse, pq_build_codebook
    from .hopfield import block_hidden, prompt_state, relax as hopfield_relax
except ImportError:
    from clarus.ce_ops import pack_sparse as ce_pack_sparse, pq_build_codebook
    from clarus.hopfield import block_hidden, prompt_state, relax as hopfield_relax

_ad = 4 / (e ** (4 / 3) * pi ** (4 / 3))
PORTAL = (_ad * (1 - _ad)) ** 2
BYPASS = 1 / (e ** (1 / 3) * pi ** (1 / 3))
T_WAKE = 1 / (3 + _ad * (1 - _ad))
R_C = pi
ACTIVE_RATIO = 0.0487
STRUCT_RATIO = 0.2623
WAKE_RATIO = 0.6891
NREM_RATIO = 0.2623
REM_RATIO = 0.0487
TARGET_W_DENSITY = 0.0316
DECODER_QUERY_BLEND = 0.7
DECODER_CANDIDATE_RATIO = ACTIVE_RATIO
CURVATURE_ALPHA = 1.5
CURVATURE_LAMBDA = 1.25
CURVATURE_STEEPNESS = 8.0
CURVATURE_EVAL_TOPK = 256
REPEAT_WINDOW = 16
REPEAT_NGRAM = 3
ALPHA_S = 0.11789
ALPHA_W = 0.03352
ALPHA_EM = 0.00775
CALIBRATION_PROMPTS = (
    "오늘 날씨가 참 좋아서 산책하기에 적합하다고",
    "인공지능의 미래는 우리가 상상하는 것보다",
    "맛있는 음식을 만드는 비결은 신선한 재료와",
    "한국의 역사에서 가장 중요한 사건은",
    "대한민국의 수도는 서울이며 인구가",
    "경제 전망은 올해 하반기부터 점차 개선될",
    "한국의 전통 문화는 세계적으로 인정받고 있으며",
    "과학 기술의 발전은 인류의 삶을 근본적으로",
    "사회 문제를 해결하려면 시민의 참여와 정부의",
    "좋은 소설의 조건은 탄탄한 서사 구조와 깊이",
    "음악이 사람에게 주는 힘은 감정을 치유하고",
    "학교 교육에서 중요한 것은 학생의 자율성과",
    "기업이 성장하기 위해서는 혁신적인 전략과",
    "새로운 아이디어를 만들려면 다양한 분야의 지식을",
    "여행을 떠나기 전에 준비해야 할 것들은",
    "건강을 유지하는 방법은 규칙적인 운동과 균형",
    "데이터 분석의 핵심은 정확한 수집과 올바른",
    "수학이 어려운 이유는 추상적 개념의 연결을",
    "환경 보호를 위해 우리가 할 수 있는 일은",
    "한국어 문법의 특징은 어순이 비교적 자유롭고",
    "디지털 기술이 교육에 미치는 영향은 학습 방식을",
    "우주 탐사의 의미는 인류의 생존 가능성을",
    "철학이 현대 사회에 필요한 이유는 가치 판단의",
    "언어를 배우는 가장 효과적인 방법은 꾸준한",
    "미래 도시의 모습은 친환경 에너지와 스마트",
    "한국 문학의 대표적인 작품으로는 이광수의",
    "좋은 리더가 갖추어야 할 덕목은 소통과",
    "기후 변화가 우리 생활에 미치는 영향은",
    "인터넷이 세상을 바꾼 방식은 정보 접근성을",
    "한국의 사계절 중 가을은 특히 단풍이",
    "효율적인 시간 관리를 위해서는 우선순위를",
    "건축 디자인에서 중요한 원칙은 기능과 미학의",
    "한글의 창제 원리는 발음 기관의 모양을",
    "예술이 사회에 기여하는 바는 창의적 사고와",
    "의료 기술의 발전으로 평균 수명이 크게",
    "한국 음식의 특징은 발효 식품이 발달하고",
    "독서가 뇌에 미치는 긍정적 효과는 집중력과",
    "지속 가능한 발전을 위한 핵심 과제는",
    "정보화 사회에서 개인정보 보호가 중요한 이유는",
    "한국의 교통 체계는 고속철도와 지하철이",
    "스포츠가 청소년 교육에 미치는 영향은",
    "전통 시장과 현대 마트의 차이점은 지역 경제와",
    "대학 교육의 목적은 전문 지식과 비판적 사고를",
    "한국의 IT 산업이 세계적으로 성장한 배경에는",
    "올바른 식습관을 기르기 위해서는 영양소의",
    "민주주의의 핵심 가치는 자유와 평등 그리고",
    "한국어에서 존댓말과 반말의 구분은 사회적",
    "자연 과학과 인문학의 융합이 필요한 이유는",
)


@dataclass
class DistillRows:
    context_first: list[torch.Tensor]
    context_prev: list[torch.Tensor]
    context_last: list[torch.Tensor]
    context_mean: list[torch.Tensor]
    context_decay: list[torch.Tensor]
    context_phi: list[torch.Tensor]
    context_len: list[float]
    context_target: list[torch.Tensor]
    state_x: list[torch.Tensor]
    prev_x: list[torch.Tensor]
    target_y: list[torch.Tensor]
    soft_y: list[torch.Tensor]
    teacher_top_ids: list[torch.Tensor]
    teacher_top_probs: list[torch.Tensor]


@dataclass
class TokenHead:
    token_ids: torch.Tensor
    state_proj: torch.Tensor
    prev_proj: torch.Tensor
    bias: torch.Tensor
    scale: float


@dataclass
class ContextParams:
    first_proj: torch.Tensor
    prev_proj: torch.Tensor
    last_proj: torch.Tensor
    mean_proj: torch.Tensor
    decay_proj: torch.Tensor
    phi_proj: torch.Tensor
    len_proj: torch.Tensor
    bias: torch.Tensor
    max_positions: int


def log(msg):
    print(msg, flush=True)


def _normalized_residual(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    norm = x.norm()
    if not torch.isfinite(norm) or norm.item() < 1e-8:
        return torch.zeros_like(x)
    return x / norm


def _recency_mean(seq: torch.Tensor) -> torch.Tensor:
    if seq.ndim != 2:
        raise ValueError("expected [seq, dim] tensor")
    steps = torch.arange(1, seq.shape[0] + 1, device=seq.device, dtype=seq.dtype).unsqueeze(1)
    return (seq * steps).sum(dim=0) / steps.sum().clamp_min(1.0)


def _context_features(
    prompt_emb: torch.Tensor,
    phi: torch.Tensor,
    *,
    max_positions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    prompt_emb = prompt_emb.float()
    first_emb = prompt_emb[0]
    prev_emb = prompt_emb[-2] if prompt_emb.shape[0] > 1 else prompt_emb[-1]
    last_emb = prompt_emb[-1]
    mean_emb = prompt_emb.mean(dim=0)
    decay_emb = _recency_mean(prompt_emb)
    len_ratio = float(min(prompt_emb.shape[0], max_positions)) / float(max(max_positions, 1))
    return first_emb, prev_emb, last_emb, mean_emb, decay_emb, phi.float(), len_ratio


def project_context_state(
    prompt_emb: torch.Tensor,
    phi: torch.Tensor,
    params: ContextParams,
) -> torch.Tensor:
    first_emb, prev_emb, last_emb, mean_emb, decay_emb, phi_vec, len_ratio = _context_features(
        prompt_emb,
        phi,
        max_positions=params.max_positions,
    )
    state = params.bias.clone()
    state = state + first_emb @ params.first_proj
    state = state + prev_emb @ params.prev_proj
    state = state + last_emb @ params.last_proj
    state = state + mean_emb @ params.mean_proj
    state = state + decay_emb @ params.decay_proj
    state = state + phi_vec @ params.phi_proj
    state = state + len_ratio * params.len_proj
    return state


def _linear_io_weight(layer: nn.Linear) -> torch.Tensor:
    return layer.weight.detach().T


class LBONorm(nn.Module):
    """LayerNorm + bounded low-rank diffusion.

    `h=0` reproduces LayerNorm exactly after copying scale/bias.
    """

    def __init__(self, dim: int, rank: int | None = None, step_size: float = 0.0):
        super().__init__()
        if rank is None:
            rank = max(4, dim // 8)
        self.dim = int(dim)
        self.rank = int(rank)
        self.V = nn.Parameter(torch.randn(self.rank, self.dim) * (1.0 / math.sqrt(self.rank)))
        self.h = nn.Parameter(torch.tensor(float(step_size)))
        self.scale = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        self.alpha_conf = nn.Parameter(torch.tensor(0.01))
        self._curvature = 0.0

    def _h_bound(self) -> torch.Tensor:
        sigma = torch.linalg.matrix_norm(self.V.float(), ord=2)
        return self.h.new_tensor(1.0 / (float(sigma.item()) ** 2 + 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.layer_norm(x, (self.dim,))
        h_val = self.h.abs().clamp(max=self._h_bound())
        if float(h_val.item()) <= 1e-8:
            self._curvature = 0.0
            return torch.addcmul(self.bias, x_n, self.scale)
        conf = torch.exp(-self.alpha_conf.abs() * x_n.pow(2).mean(dim=-1, keepdim=True))
        proj = x_n @ self.V.T
        x_w = (proj * conf) @ self.V
        lx = x_n - x_w
        self._curvature = float(lx.detach().pow(2).mean().item())
        mixed = torch.lerp(x_n, x_w, h_val)
        return torch.addcmul(self.bias, mixed, self.scale)


class SpectralProj(nn.Module):
    """Spectral-direction projection with output-preserving gain.

    The unit-direction weight is stored separately from the scalar gain so the
    initial forward pass matches the original projection exactly.
    """

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.weight_unit = nn.Parameter(torch.empty(self.out_dim, self.in_dim))
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(self.out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_unit)
        with torch.no_grad():
            self.gain.fill_(1.0)
            if self.bias is not None:
                self.bias.zero_()

    @classmethod
    def from_io_weight(
        cls,
        weight_io: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> "SpectralProj":
        module = cls(weight_io.shape[0], weight_io.shape[1], bias=bias is not None).to(device)
        weight_oi = weight_io.T.float().to(device)
        sigma = torch.linalg.matrix_norm(weight_oi, ord=2).clamp_min(1e-6)
        with torch.no_grad():
            module.weight_unit.copy_(weight_oi / sigma)
            module.gain.fill_(float(sigma.item()))
            if bias is not None and module.bias is not None:
                module.bias.copy_(bias.float().to(device))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_unit * self.gain, self.bias)

    def export_weight(self) -> torch.Tensor:
        return (self.weight_unit.detach() * self.gain.detach()).T


class GaugeLatticeV2(nn.Module):
    """Gauge-lattice FFN with cross-frequency coupling between gauge channels."""

    def __init__(self, dim: int, mult: int = 4, mix_rank: int | None = None):
        super().__init__()
        total = ALPHA_S + ALPHA_W + ALPHA_EM
        self.dim = int(dim)
        self.d3 = max(1, round(dim * ALPHA_S / total))
        self.d2 = max(1, round(dim * ALPHA_W / total))
        self.d1 = dim - self.d3 - self.d2

        h = dim * mult
        h3 = max(1, round(h * ALPHA_S / total))
        h2 = max(1, round(h * ALPHA_W / total))
        h1 = max(1, h - h3 - h2)
        if mix_rank is None:
            mix_rank = max(4, dim // 8)

        self.su3 = nn.Sequential(
            nn.Linear(self.d3, h3, bias=False),
            nn.GELU(),
            nn.Linear(h3, self.d3, bias=False),
        )
        self.su2 = nn.Sequential(
            nn.Linear(self.d2, h2, bias=False),
            nn.GELU(),
            nn.Linear(h2, self.d2, bias=False),
        )
        self.u1 = nn.Sequential(
            nn.Linear(self.d1, h1, bias=False),
            nn.GELU(),
            nn.Linear(h1, self.d1, bias=False),
        )
        self.mix_down = nn.Linear(dim, mix_rank, bias=False)
        self.mix_up = nn.Linear(mix_rank, dim, bias=False)
        nn.init.zeros_(self.mix_up.weight)

        # Cross-frequency coupling: pairwise low-rank bridges between channels
        # Coupling strengths proportional to gauge mixing angles
        cf_rank = max(2, mix_rank // 4)
        self.cf_32_down = nn.Linear(self.d3, cf_rank, bias=False)
        self.cf_32_up = nn.Linear(cf_rank, self.d2, bias=False)
        self.cf_21_down = nn.Linear(self.d2, cf_rank, bias=False)
        self.cf_21_up = nn.Linear(cf_rank, self.d1, bias=False)
        self.cf_31_down = nn.Linear(self.d3, cf_rank, bias=False)
        self.cf_31_up = nn.Linear(cf_rank, self.d1, bias=False)
        nn.init.zeros_(self.cf_32_up.weight)
        nn.init.zeros_(self.cf_21_up.weight)
        nn.init.zeros_(self.cf_31_up.weight)

        self.phi = LBONorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.d3
        x3, x2, x1 = x[..., :s], x[..., s : s + self.d2], x[..., s + self.d2 :]

        y3 = self.su3(x3)
        y2 = self.su2(x2)
        y1 = self.u1(x1)

        # Cross-frequency coupling: gauge channel bridges
        y2 = y2 + self.cf_32_up(self.cf_32_down(y3))
        y1 = y1 + self.cf_21_up(self.cf_21_down(y2))
        y1 = y1 + self.cf_31_up(self.cf_31_down(y3))

        y = torch.cat([y3, y2, y1], dim=-1)
        y = y + self.mix_up(self.mix_down(y))
        return self.phi(y)

    def export_weight(self) -> torch.Tensor:
        def seq_weight(branch: nn.Sequential) -> torch.Tensor:
            w1 = _linear_io_weight(branch[0])
            w2 = _linear_io_weight(branch[2])
            return 0.5 * (w1 @ w2)

        out = torch.zeros(self.dim, self.dim, device=self.mix_down.weight.device)
        s = self.d3
        out[:s, :s] = seq_weight(self.su3)
        out[s : s + self.d2, s : s + self.d2] = seq_weight(self.su2)
        out[s + self.d2 :, s + self.d2 :] = seq_weight(self.u1)
        out = out + (_linear_io_weight(self.mix_down) @ _linear_io_weight(self.mix_up))

        cf_32 = _linear_io_weight(self.cf_32_down) @ _linear_io_weight(self.cf_32_up)
        cf_21 = _linear_io_weight(self.cf_21_down) @ _linear_io_weight(self.cf_21_up)
        cf_31 = _linear_io_weight(self.cf_31_down) @ _linear_io_weight(self.cf_31_up)
        out[:s, s : s + self.d2] += cf_32
        out[s : s + self.d2, s + self.d2 :] += cf_21
        out[:s, s + self.d2 :] += cf_31
        return out


def init_lbo_from_ln(lbo: LBONorm, ln) -> None:
    with torch.no_grad():
        lbo.scale.copy_(ln.weight.detach().float())
        lbo.bias.copy_(ln.bias.detach().float())
        lbo.h.zero_()


def transplant_phase1(model, device: str | torch.device = "cpu"):
    """Phase 1: LayerNorm -> LBONorm, c_proj -> SpectralProj."""

    device = torch.device(device)
    dim = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
    old_ln_f = model.transformer.ln_f
    model.transformer.ln_f = LBONorm(dim).to(device)
    init_lbo_from_ln(model.transformer.ln_f, old_ln_f)

    for block in model.transformer.h:
        new_ln1 = LBONorm(dim).to(device)
        new_ln2 = LBONorm(dim).to(device)
        init_lbo_from_ln(new_ln1, block.ln_1)
        init_lbo_from_ln(new_ln2, block.ln_2)
        block.ln_1 = new_ln1
        block.ln_2 = new_ln2

        old_proj = block.attn.c_proj
        old_w = old_proj.weight.detach().float()
        old_b = None if getattr(old_proj, "bias", None) is None else old_proj.bias.detach().float()
        block.attn.c_proj = SpectralProj.from_io_weight(old_w, old_b, device=device)

    return model


def transplant_phase2(
    model,
    device: str | torch.device = "cpu",
    *,
    distill_steps: int = 500,
    batch_size: int = 8,
    seq_len: int = 32,
    mix_rank: int | None = None,
):
    """Phase 2: MLP -> GaugeLatticeV2 with block-wise distillation."""

    device = torch.device(device)
    dim = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size

    for block_idx, block in enumerate(model.transformer.h):
        old_mlp = block.mlp
        new_lattice = GaugeLatticeV2(dim, mult=4, mix_rank=mix_rank).to(device)
        if distill_steps > 0:
            opt = torch.optim.Adam(new_lattice.parameters(), lr=5e-4)
            old_mlp.eval()
            for param in old_mlp.parameters():
                param.requires_grad = False
            loss_value = float("nan")
            for _ in range(distill_steps):
                x = torch.randn(batch_size, seq_len, dim, device=device) * 0.3
                with torch.no_grad():
                    target = old_mlp(x)
                pred = new_lattice(x)
                loss = F.mse_loss(pred, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_value = float(loss.item())
            log(f"  block {block_idx:2d} phase2_distill={loss_value:.4f}")
        block.mlp = new_lattice
        del old_mlp

    return model


def _module_io_weight(module) -> torch.Tensor:
    if hasattr(module, "export_weight"):
        return module.export_weight().detach()
    if isinstance(module, nn.Linear):
        return _linear_io_weight(module)
    if hasattr(module, "weight"):
        return module.weight.detach()
    raise TypeError(f"unsupported module weight export: {type(module)!r}")


def _mlp_io_weight(module) -> torch.Tensor:
    if hasattr(module, "export_weight"):
        return module.export_weight().detach()
    if hasattr(module, "c_fc") and hasattr(module, "c_proj"):
        return module.c_fc.weight.detach() @ _module_io_weight(module.c_proj)
    raise TypeError(f"unsupported MLP export: {type(module)!r}")


def extract_hopfield(mdl, d: int, n_layer: int, d_head: int):
    del n_layer
    device = next(mdl.parameters()).device
    w_total = torch.zeros(d, d, device=device)
    for layer in mdl.transformer.h:
        w_attn = layer.attn.c_attn.weight.detach()
        wq, wk, wv = w_attn[:, :d], w_attn[:, d : 2 * d], w_attn[:, 2 * d :]
        wo = _module_io_weight(layer.attn.c_proj).to(device)
        qk = wq @ wk.T
        w_total += (qk + qk.T) / (2 * d_head ** 0.5)
        vo = wv @ wo
        w_total += (vo + vo.T) / 2
        wf = _mlp_io_weight(layer.mlp).to(device)
        w_total += (wf + wf.T) / 4
    return 0.5 * (w_total + w_total.T)


def make_negative_definite(w: torch.Tensor):
    eigvals, eigvecs = torch.linalg.eigh(w)
    lam_min, lam_max = eigvals[0].item(), eigvals[-1].item()
    log(f"  raw eig: [{lam_min:.4f}, {lam_max:.4f}]")
    margin = 0.01 * abs(lam_min) if abs(lam_min) > 1e-6 else 0.01
    if lam_max <= -margin:
        log("  already negative-definite")
        return w, eigvals
    shift = lam_max + margin
    eigvals_shifted = eigvals - shift
    w_safe = eigvecs @ torch.diag(eigvals_shifted) @ eigvecs.T
    w_safe = 0.5 * (w_safe + w_safe.T)
    log(f"  shift={shift:.4f}  eig: [{eigvals_shifted[0]:.4f}, {eigvals_shifted[-1]:.4f}]")
    return w_safe, eigvals_shifted


def build_lattice(n: int):
    side = int(np.ceil(n ** (1 / 3)))
    coords = np.zeros((n, 3), dtype=np.float32)
    for idx in range(n):
        coords[idx] = [idx // (side * side), (idx // side) % side, idx % side]
    return coords, side


def sparsify_3d(w: torch.Tensor, n: int, r_c: float = R_C):
    coords, side = build_lattice(n)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)
    mask = torch.from_numpy((dist_sq < r_c ** 2).astype(np.float32)).to(w.device)
    mask.fill_diagonal_(0)
    w_sp = w * mask
    nnz = int(mask.sum().item())
    log(f"  lattice {side}^3, r_c=pi")
    log(f"  K_avg={nnz / n:.1f}  density={nnz / (n * (n - 1)) * 100:.2f}%")
    return w_sp


def ridge_solve(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    x = x.float()
    y = y.float()
    xtx = x.T @ x
    xty = x.T @ y
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
    return torch.linalg.solve(xtx + float(ridge) * eye, xty)


def fit_linear_with_bias(x: torch.Tensor, y: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x.float(), ones], dim=1)
    weight_bias = ridge_solve(x_aug, y.float(), ridge=ridge)
    return weight_bias[:-1], weight_bias[-1]


def fit_context_projections(
    rows: DistillRows,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_x = torch.stack(rows.context_first, dim=0).float()
    prev_x = torch.stack(rows.context_prev, dim=0).float()
    last_x = torch.stack(rows.context_last, dim=0).float()
    mean_x = torch.stack(rows.context_mean, dim=0).float()
    decay_x = torch.stack(rows.context_decay, dim=0).float()
    phi_x = torch.stack(rows.context_phi, dim=0).float()
    len_x = torch.tensor(rows.context_len, dtype=torch.float32, device=first_x.device).unsqueeze(1)
    target = torch.stack(rows.context_target, dim=0).float()
    feat = torch.cat([first_x, prev_x, last_x, mean_x, decay_x, phi_x, len_x], dim=1)
    proj, bias = fit_linear_with_bias(feat, target, ridge=ridge)
    d = target.shape[1]
    return (
        proj[:d].cpu(),
        proj[d : 2 * d].cpu(),
        proj[2 * d : 3 * d].cpu(),
        proj[3 * d : 4 * d].cpu(),
        proj[4 * d : 5 * d].cpu(),
        proj[5 * d : 6 * d].cpu(),
        proj[6 * d].cpu(),
        bias.cpu(),
    )


def fit_decoder_from_rows(
    rows: DistillRows,
    *,
    prev_scale: float,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_x = torch.stack(rows.state_x, dim=0).float()
    prev_x = torch.stack(rows.prev_x, dim=0).float()
    target_y = torch.stack(rows.target_y, dim=0).float()
    feat = torch.cat([state_x, float(prev_scale) * prev_x], dim=1)
    proj, bias = fit_linear_with_bias(feat, target_y, ridge=ridge)
    d = target_y.shape[1]
    state_proj = proj[:d]
    prev_proj = proj[d : 2 * d]
    pred = state_x @ state_proj + float(prev_scale) * (prev_x @ prev_proj) + bias
    denom = float(pred.pow(2).sum().item())
    if denom > 1e-8:
        scale = float((pred * target_y).sum().item() / denom)
        state_proj = state_proj * scale
        prev_proj = prev_proj * scale
        bias = bias * scale
    return state_proj.cpu(), prev_proj.cpu(), bias.cpu()


def fit_token_head_from_rows(
    rows: DistillRows,
    *,
    prev_scale: float,
    ridge: float,
    max_vocab: int,
    scale: float,
) -> TokenHead | None:
    top_ids = torch.stack(rows.teacher_top_ids, dim=0).long()
    top_probs = torch.stack(rows.teacher_top_probs, dim=0).float()
    flat_ids = top_ids.reshape(-1)
    flat_probs = top_probs.reshape(-1)
    uniq_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    mass = torch.zeros(uniq_ids.shape[0], dtype=flat_probs.dtype, device=flat_probs.device)
    mass.scatter_add_(0, inverse, flat_probs)
    if max_vocab > 0 and uniq_ids.numel() > max_vocab:
        keep = torch.topk(mass, max_vocab).indices
        uniq_ids = uniq_ids.index_select(0, keep)
        uniq_ids, _ = torch.sort(uniq_ids)
    if uniq_ids.numel() == 0:
        return None

    token_map = {int(token_id): col for col, token_id in enumerate(uniq_ids.tolist())}
    state_x = torch.stack(rows.state_x, dim=0).float()
    prev_x = torch.stack(rows.prev_x, dim=0).float()
    y = torch.zeros((top_ids.shape[0], uniq_ids.shape[0]), dtype=torch.float32, device=state_x.device)
    for row_idx, (row_ids, row_probs) in enumerate(zip(top_ids.tolist(), top_probs.tolist(), strict=False)):
        for token_id, prob in zip(row_ids, row_probs, strict=False):
            col_idx = token_map.get(int(token_id))
            if col_idx is not None:
                y[row_idx, col_idx] += float(prob)

    feat = torch.cat([state_x, float(prev_scale) * prev_x], dim=1)
    proj, bias = fit_linear_with_bias(feat, y, ridge=ridge)
    d = state_x.shape[1]
    state_proj = proj[:d]
    prev_proj = proj[d : 2 * d]
    return TokenHead(
        token_ids=uniq_ids.long().cpu(),
        state_proj=state_proj.cpu(),
        prev_proj=prev_proj.cpu(),
        bias=bias.cpu(),
        scale=float(scale),
    )


def build_runtime_codebook(emb_weight: torch.Tensor, query: torch.Tensor, top_k: int) -> torch.Tensor:
    scores = emb_weight @ query
    top_ids = torch.topk(scores, min(int(top_k), scores.numel())).indices
    return emb_weight.index_select(0, top_ids)


def teacher_prefix_state(model, prompt_ids: torch.Tensor):
    with torch.no_grad():
        emb, phi = prompt_state(model, prompt_ids)
        h = emb
        for block in model.transformer.h:
            h = block_hidden(block, h)
        m0 = h[:, -1, :].squeeze(0)
        hidden = model.transformer.ln_f(h)[:, -1, :].squeeze(0)
        logits = model.lm_head(hidden.unsqueeze(0)).squeeze(0)
    return {
        "prompt_emb": emb.squeeze(0).detach(),
        "phi": phi.detach(),
        "m0": m0.detach(),
        "hidden": hidden.detach(),
        "logits": logits.detach(),
    }


def collect_distill_rows(
    model,
    tok,
    *,
    emb_weight: torch.Tensor,
    w_pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ln_f_w: torch.Tensor,
    ln_f_b: torch.Tensor,
    tau: float,
    device: torch.device,
    max_new_tokens: int,
    cb_topk: int,
    teacher_topk: int,
    relax_steps: int,
    beta: float,
    metric_rank: int,
    w_eigvecs: torch.Tensor | None,
    dense_w: torch.Tensor | None,
    context_params: ContextParams | None = None,
) -> DistillRows:
    from tqdm import tqdm as _tqdm

    rows = DistillRows([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    max_pos = max(int(model.transformer.wpe.weight.shape[0]), 1)
    total_iters = len(CALIBRATION_PROMPTS) * max_new_tokens
    iter_count = 0
    pbar = _tqdm(total=total_iters, desc="  distill", unit="row", ncols=80)
    for p_idx, prompt in enumerate(CALIBRATION_PROMPTS):
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        for tok_idx in range(max_new_tokens):
            iter_count += 1
            pbar.set_postfix_str(f"p{p_idx+1}/{len(CALIBRATION_PROMPTS)}")
            pbar.update(1)
            prefix = teacher_prefix_state(model, ids)
            prompt_emb = prefix["prompt_emb"]
            first_emb = prompt_emb[0]
            prev_emb = prompt_emb[-2] if prompt_emb.shape[0] > 1 else prompt_emb[-1]
            decay_emb = _recency_mean(prompt_emb)
            rows.context_last.append(prompt_emb[-1].detach().cpu())
            rows.context_first.append(first_emb.detach().cpu())
            rows.context_prev.append(prev_emb.detach().cpu())
            rows.context_mean.append(prompt_emb.mean(dim=0).detach().cpu())
            rows.context_decay.append(decay_emb.detach().cpu())
            rows.context_phi.append(prefix["phi"].detach().cpu())
            rows.context_len.append(float(min(prompt_emb.shape[0], max_pos)) / float(max_pos))
            rows.context_target.append(prefix["m0"].detach().cpu())

            state_seed = prefix["m0"]
            if context_params is not None:
                state_seed = project_context_state(
                    prompt_emb.to(device),
                    prefix["phi"].to(device),
                    context_params,
                )
            codebook = build_runtime_codebook(
                emb_weight,
                state_seed.to(emb_weight.device),
                top_k=cb_topk,
            ).to(device)
            m_star, _, _ = hopfield_relax(
                w_pack,
                state_seed,
                prefix["phi"],
                state_seed,
                codebook,
                beta,
                PORTAL,
                tau=tau,
                dt=min(0.01, 0.9 * tau),
                max_steps=relax_steps,
                backend="torch",
                metric_rank=metric_rank,
                lambda0=1.0,
                lambda_phi=0.5,
                lambda_var=0.25,
                noise_scale=0.0,
                seed=0,
                w_eigvecs=w_eigvecs,
                dense_w=dense_w,
            )
            ce_hidden = F.layer_norm(m_star, (m_star.numel(),), ln_f_w.to(device), ln_f_b.to(device))
            top_vals, top_idx = torch.topk(prefix["logits"], min(int(teacher_topk), prefix["logits"].numel()))
            probs = F.softmax(top_vals, dim=0)
            target_id = int(top_idx[0].item())
            prev_id = int(ids[0, -1].item())

            rows.state_x.append(ce_hidden.detach().cpu())
            rows.prev_x.append(emb_weight[prev_id].detach().cpu())
            rows.target_y.append(prefix["hidden"].detach().cpu())
            rows.soft_y.append(prefix["hidden"].detach().cpu())
            rows.teacher_top_ids.append(top_idx.detach().cpu())
            rows.teacher_top_probs.append(probs.detach().cpu())

            ids = torch.cat([ids, torch.tensor([[target_id]], device=device)], dim=1)

    pbar.close()
    return rows


def extract_clone_state(model):
    clone_state = {}
    for name, param in model.named_parameters():
        if torch.is_tensor(param) and torch.is_floating_point(param):
            clone_state[name] = param.detach().cpu().to(dtype=torch.float16)
        else:
            clone_state[name] = param.detach().cpu()
    return clone_state


def convert(
    model_name: str,
    out_path: str,
    device_name: str = "cpu",
    sparse: bool = False,
    *,
    phase: int = 1,
    phase2_steps: int = 500,
    save_pq: bool = False,
    pq_only: bool = False,
    pq_subdim: int = 64,
    pq_bits: int = 8,
    pq_iters: int = 16,
    pq_batch_size: int = 4096,
    pq_sample_size: int = 16384,
    decoder_prev_scale: float = 0.35,
    distill_tokens: int = 48,
    distill_cb_topk: int = 1024,
    distill_teacher_topk: int = 32,
    distill_ridge: float = 1e-3,
    relax_steps: int = 64,
    metric_rank: int = 16,
    token_head_max_vocab: int = 4096,
    token_head_scale: float = 1.0,
    decoder_query_blend: float = DECODER_QUERY_BLEND,
    decoder_candidate_ratio: float = DECODER_CANDIDATE_RATIO,
    curvature_alpha: float = CURVATURE_ALPHA,
    curvature_lambda: float = CURVATURE_LAMBDA,
    curvature_steepness: float = CURVATURE_STEEPNESS,
    curvature_eval_topk: int = CURVATURE_EVAL_TOPK,
    repeat_window: int = REPEAT_WINDOW,
    repeat_ngram: int = REPEAT_NGRAM,
    save_clone: bool = False,
):
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    device = torch.device(device_name)
    if pq_only and not save_pq:
        raise ValueError("pq_only requires --save-pq")
    phase = 2 if int(phase) >= 2 else 1
    log(f"Loading {model_name} ...")
    tok = PreTrainedTokenizerFast.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    teacher.to(device).eval()

    cfg = teacher.config
    d = cfg.n_embd if hasattr(cfg, "n_embd") else cfg.hidden_size
    n_layer = cfg.n_layer if hasattr(cfg, "n_layer") else cfg.num_hidden_layers
    n_head = cfg.n_head if hasattr(cfg, "n_head") else cfg.num_attention_heads
    d_head = d // n_head
    vocab = cfg.vocab_size
    log(f"  d={d}  layers={n_layer}  heads={n_head}  vocab={vocab}")

    log("\n[1/6] Phase 1 transplant ...")
    transplant_phase1(teacher, device=device)
    if phase >= 2:
        log("\n[2/6] Phase 2 GaugeLattice distill ...")
        transplant_phase2(
            teacher,
            device=device,
            distill_steps=int(phase2_steps),
        )
    else:
        log("\n[2/6] Phase 2 skipped (phase=1)")

    log("\n[3/6] Extracting CE runtime matrix ...")
    t0 = time.time()
    w_hop = extract_hopfield(teacher, d, n_layer, d_head)
    if sparse:
        w_hop = sparsify_3d(w_hop, d)
    w_cond, eigvals = make_negative_definite(w_hop)
    tau = 1.0 / abs(eigvals[-1].item()) if abs(eigvals[-1].item()) > 1e-8 else 1.0
    log(f"  ||W||={w_cond.norm():.2f}  tau={tau:.6f}  time={time.time()-t0:.1f}s")

    log("\n[4/6] Packing runtime tensors ...")
    w_values, w_col_idx, w_row_ptr = ce_pack_sparse(w_cond.cpu(), backend="torch")
    _, eigvecs_runtime = torch.linalg.eigh(w_cond.cpu())
    hess_rank = min(max(metric_rank // 2, 0), 8)
    w_eigvecs = None if hess_rank <= 0 else eigvecs_runtime[:, :hess_rank].T.contiguous()
    dense_w = w_cond.to(device) if w_values.numel() == w_cond.numel() else None
    packed_kb = (
        w_values.numel() * w_values.element_size()
        + w_col_idx.numel() * w_col_idx.element_size()
        + w_row_ptr.numel() * w_row_ptr.element_size()
    ) / 1024
    log(f"  nnz={w_values.numel()}  packed={packed_kb:.1f} KB")

    emb_weight = teacher.transformer.wte.weight.detach().cpu().float()
    pos_weight = teacher.transformer.wpe.weight.detach().cpu().float()
    ln_f_w = teacher.transformer.ln_f.scale.detach().cpu().float()
    ln_f_b = teacher.transformer.ln_f.bias.detach().cpu().float()
    prefix_ref = teacher_prefix_state(
        teacher,
        tok.encode("오늘 날씨가", return_tensors="pt").to(device),
    )
    hidden_norm_ref = float(prefix_ref["hidden"].norm().item())

    pq_payload = None
    if save_pq:
        log("\n[4.5/6] Building PQ lexical memory ...")
        t_pq = time.time()
        pq_payload = pq_build_codebook(
            emb_weight,
            subdim=pq_subdim,
            bits=pq_bits,
            iters=pq_iters,
            batch_size=pq_batch_size,
            sample_size=pq_sample_size,
        )
        pq_bytes = (
            pq_payload["centroids"].numel() * pq_payload["centroids"].element_size()
            + pq_payload["codes"].numel() * pq_payload["codes"].element_size()
        )
        log(f"  pq size={pq_bytes/1024/1024:.2f} MB  time={time.time()-t_pq:.1f}s")

    log("\n[5/6] Distilling runtime prompt/decoder heads ...")
    log(f"  pass 1/2: context projection ({len(CALIBRATION_PROMPTS)} prompts x {distill_tokens} tokens) ...")
    t_distill = time.time()
    context_rows = collect_distill_rows(
        teacher,
        tok,
        emb_weight=emb_weight.to(device),
        w_pack=(w_values.to(device), w_col_idx.to(device), w_row_ptr.to(device)),
        ln_f_w=ln_f_w,
        ln_f_b=ln_f_b,
        tau=tau,
        device=device,
        max_new_tokens=int(distill_tokens),
        cb_topk=int(distill_cb_topk),
        teacher_topk=int(distill_teacher_topk),
        relax_steps=int(relax_steps),
        beta=1.0,
        metric_rank=int(metric_rank),
        w_eigvecs=None if w_eigvecs is None else w_eigvecs.to(device),
        dense_w=dense_w,
        context_params=None,
    )
    (
        ctx_first_proj,
        ctx_prev_proj,
        ctx_last_proj,
        ctx_mean_proj,
        ctx_decay_proj,
        ctx_phi_proj,
        ctx_len_proj,
        ctx_bias,
    ) = fit_context_projections(context_rows, ridge=float(distill_ridge))
    context_params = ContextParams(
        first_proj=ctx_first_proj.to(device),
        prev_proj=ctx_prev_proj.to(device),
        last_proj=ctx_last_proj.to(device),
        mean_proj=ctx_mean_proj.to(device),
        decay_proj=ctx_decay_proj.to(device),
        phi_proj=ctx_phi_proj.to(device),
        len_proj=ctx_len_proj.to(device),
        bias=ctx_bias.to(device),
        max_positions=int(pos_weight.shape[0]),
    )
    log(f"  pass 1/2 done in {time.time()-t_distill:.1f}s")
    log(f"  pass 2/2: decoder projection ({len(CALIBRATION_PROMPTS)} prompts x {distill_tokens} tokens) ...")
    t_distill2 = time.time()
    rows = collect_distill_rows(
        teacher,
        tok,
        emb_weight=emb_weight.to(device),
        w_pack=(w_values.to(device), w_col_idx.to(device), w_row_ptr.to(device)),
        ln_f_w=ln_f_w,
        ln_f_b=ln_f_b,
        tau=tau,
        device=device,
        max_new_tokens=int(distill_tokens),
        cb_topk=int(distill_cb_topk),
        teacher_topk=int(distill_teacher_topk),
        relax_steps=int(relax_steps),
        beta=1.0,
        metric_rank=int(metric_rank),
        w_eigvecs=None if w_eigvecs is None else w_eigvecs.to(device),
        dense_w=dense_w,
        context_params=context_params,
    )
    log(f"  pass 2/2 done in {time.time()-t_distill2:.1f}s")
    state_proj, prev_proj, decoder_query_bias = fit_decoder_from_rows(
        rows,
        prev_scale=decoder_prev_scale,
        ridge=float(distill_ridge),
    )
    token_head = fit_token_head_from_rows(
        rows,
        prev_scale=decoder_prev_scale,
        ridge=float(distill_ridge),
        max_vocab=int(token_head_max_vocab),
        scale=float(token_head_scale),
    )
    log(
        f"  rows={len(rows.state_x)}  context_rows={len(context_rows.context_target)} "
        f"context={tuple(ctx_last_proj.shape)}  "
        f"state={tuple(state_proj.shape)}  prev={tuple(prev_proj.shape)}"
    )
    if token_head is not None:
        log(f"  token_head={token_head.token_ids.numel()} tokens")

    log("\n[6/6] Saving artifact ...")
    tokenizer_json = tok._tokenizer.to_str()
    tokenizer_specials = {
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "unk_token": tok.unk_token,
        "sep_token": tok.sep_token,
        "pad_token": tok.pad_token,
        "cls_token": tok.cls_token,
        "mask_token": tok.mask_token,
    }
    cloned_vocab_weight = None if pq_only or emb_weight is None else emb_weight.clone()
    cloned_vocab_bias = None if cloned_vocab_weight is None else torch.zeros(vocab, dtype=cloned_vocab_weight.dtype)
    engine = {
        "artifact_version": 4,
        "model_name": model_name,
        "phase": phase,
        "d": d,
        "vocab": vocab,
        "n_layer": n_layer,
        "tau": tau,
        "portal": PORTAL,
        "bypass": BYPASS,
        "t_wake": T_WAKE,
        "r_c": R_C,
        "active_ratio": ACTIVE_RATIO,
        "struct_ratio": STRUCT_RATIO,
        "wake_ratio": WAKE_RATIO,
        "nrem_ratio": NREM_RATIO,
        "rem_ratio": REM_RATIO,
        "target_w_density": TARGET_W_DENSITY,
        "allow_pretrained_fallback": False,
        "teacher_phase": phase,
        "W": w_cond.cpu(),
        "W_values": w_values.cpu(),
        "W_col_idx": w_col_idx.cpu(),
        "W_row_ptr": w_row_ptr.cpu(),
        "W_eigvecs": None if w_eigvecs is None else w_eigvecs.cpu(),
        "W_layers": [],
        "sparse": sparse,
        "emb_weight": None if pq_only else emb_weight,
        "pos_weight": pos_weight,
        "ln_f_weight": ln_f_w,
        "ln_f_bias": ln_f_b,
        "hidden_norm_ref": hidden_norm_ref,
        "decoder_prev_scale": float(decoder_prev_scale),
        "decoder_prev_proj": prev_proj,
        "decoder_state_proj": state_proj,
        "decoder_query_bias": decoder_query_bias,
        "decoder_vocab_weight": cloned_vocab_weight,
        "decoder_vocab_bias": cloned_vocab_bias,
        "decoder_vocab_scale": 1.0,
        "decoder_query_blend": float(decoder_query_blend),
        "decoder_candidate_ratio": float(decoder_candidate_ratio),
        "decoder_token_ids": None if token_head is None else token_head.token_ids,
        "decoder_token_state_proj": None if token_head is None else token_head.state_proj,
        "decoder_token_prev_proj": None if token_head is None else token_head.prev_proj,
        "decoder_token_bias": None if token_head is None else token_head.bias,
        "decoder_token_scale": 1.0 if token_head is None else float(token_head.scale),
        "curvature_alpha": float(curvature_alpha),
        "curvature_lambda": float(curvature_lambda),
        "curvature_steepness": float(curvature_steepness),
        "curvature_eval_topk": int(curvature_eval_topk),
        "repeat_window": int(repeat_window),
        "repeat_ngram": int(repeat_ngram),
        "context_first_proj": ctx_first_proj,
        "context_prev_proj": ctx_prev_proj,
        "context_last_proj": ctx_last_proj,
        "context_mean_proj": ctx_mean_proj,
        "context_decay_proj": ctx_decay_proj,
        "context_phi_proj": ctx_phi_proj,
        "context_len_proj": ctx_len_proj,
        "context_bias": ctx_bias,
        "default_init_layer": n_layer - 1,
        "tokenizer_json": tokenizer_json,
        "tokenizer_specials": tokenizer_specials,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "distill_rows": len(rows.state_x),
    }
    if pq_payload is not None:
        engine["pq_centroids"] = pq_payload["centroids"]
        engine["pq_codes"] = pq_payload["codes"]
        engine["pq_subdim"] = int(pq_payload["subdim"])
        engine["pq_bits"] = int(pq_payload["bits"])
        if pq_only:
            engine["emb_weight"] = None
    if save_clone:
        clone_state = extract_clone_state(teacher)
        engine["clone_kind"] = "ce_teacher"
        engine["clone_config"] = teacher.config.to_dict()
        engine["clone_state"] = clone_state
        clone_bytes = sum(v.numel() * v.element_size() for v in clone_state.values())
        log(f"  saved_ce_teacher={clone_bytes/1024/1024:.2f} MB")

    torch.save(engine, out_path)
    file_size = os.path.getsize(out_path)
    teacher_size = sum(p.numel() * p.element_size() for p in teacher.parameters())
    emb_bytes = 0 if engine["emb_weight"] is None else engine["emb_weight"].numel() * engine["emb_weight"].element_size()
    pos_bytes = pos_weight.numel() * pos_weight.element_size()
    pq_bytes = 0
    if pq_payload is not None:
        pq_bytes = (
            pq_payload["centroids"].numel() * pq_payload["centroids"].element_size()
            + pq_payload["codes"].numel() * pq_payload["codes"].element_size()
        )
    ctx_bytes = (
        ctx_first_proj.numel() * ctx_first_proj.element_size()
        + ctx_prev_proj.numel() * ctx_prev_proj.element_size()
        + ctx_last_proj.numel() * ctx_last_proj.element_size()
        + ctx_mean_proj.numel() * ctx_mean_proj.element_size()
        + ctx_decay_proj.numel() * ctx_decay_proj.element_size()
        + ctx_phi_proj.numel() * ctx_phi_proj.element_size()
        + ctx_len_proj.numel() * ctx_len_proj.element_size()
        + ctx_bias.numel() * ctx_bias.element_size()
    )
    token_bytes = 0
    if token_head is not None:
        token_bytes = (
            token_head.token_ids.numel() * token_head.token_ids.element_size()
            + token_head.state_proj.numel() * token_head.state_proj.element_size()
            + token_head.prev_proj.numel() * token_head.prev_proj.element_size()
            + token_head.bias.numel() * token_head.bias.element_size()
        )
    vocab_head_bytes = 0
    if cloned_vocab_weight is not None:
        vocab_head_bytes += cloned_vocab_weight.numel() * cloned_vocab_weight.element_size()
    if cloned_vocab_bias is not None:
        vocab_head_bytes += cloned_vocab_bias.numel() * cloned_vocab_bias.element_size()
    runtime_kb = (
        w_values.numel() * w_values.element_size()
        + w_col_idx.numel() * w_col_idx.element_size()
        + w_row_ptr.numel() * w_row_ptr.element_size()
        + pos_bytes
        + emb_bytes
        + ctx_bytes
        + state_proj.numel() * state_proj.element_size()
        + prev_proj.numel() * prev_proj.element_size()
        + decoder_query_bias.numel() * decoder_query_bias.element_size()
        + vocab_head_bytes
        + token_bytes
        + pq_bytes
    ) / 1024

    log("\n=== Conversion Complete ===")
    log(f"  Output:   {out_path}")
    log(f"  File:     {file_size / 1024 / 1024:.2f} MB")
    log(f"  Teacher:  {teacher_size / 1024 / 1024:.2f} MB")
    log(f"  Ratio:    {file_size / max(teacher_size, 1) * 100:.1f}%")
    log(f"  Runtime:  {runtime_kb / 1024:.2f} MB")
    log(f"  Distill:  rows={len(rows.state_x)} phase={phase}")


def main():
    ap = argparse.ArgumentParser(description="Convert transformer to CE runtime artifact")
    ap.add_argument("--model", default="skt/kogpt2-base-v2")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2])
    ap.add_argument("--phase2-steps", type=int, default=500)
    ap.add_argument("--sparse", action="store_true", help="Apply 3D sparsification")
    ap.add_argument("--save-pq", action="store_true", help="Save product-quantized lexical memory")
    ap.add_argument("--pq-only", action="store_true", help="Drop full embedding table when PQ is saved")
    ap.add_argument("--pq-subdim", type=int, default=64)
    ap.add_argument("--pq-bits", type=int, default=8)
    ap.add_argument("--pq-iters", type=int, default=16)
    ap.add_argument("--pq-batch-size", type=int, default=4096)
    ap.add_argument("--pq-sample-size", type=int, default=16384)
    ap.add_argument("--decoder-prev-scale", type=float, default=0.35)
    ap.add_argument("--distill-tokens", type=int, default=48)
    ap.add_argument("--distill-cb-topk", type=int, default=1024)
    ap.add_argument("--distill-teacher-topk", type=int, default=32)
    ap.add_argument("--distill-ridge", type=float, default=1e-3)
    ap.add_argument("--relax-steps", type=int, default=64)
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--token-head-max-vocab", type=int, default=4096)
    ap.add_argument("--token-head-scale", type=float, default=1.0)
    ap.add_argument("--decoder-query-blend", type=float, default=DECODER_QUERY_BLEND)
    ap.add_argument("--decoder-candidate-ratio", type=float, default=DECODER_CANDIDATE_RATIO)
    ap.add_argument("--curvature-alpha", type=float, default=CURVATURE_ALPHA)
    ap.add_argument("--curvature-lambda", type=float, default=CURVATURE_LAMBDA)
    ap.add_argument("--curvature-steepness", type=float, default=CURVATURE_STEEPNESS)
    ap.add_argument("--curvature-eval-topk", type=int, default=CURVATURE_EVAL_TOPK)
    ap.add_argument("--repeat-window", type=int, default=REPEAT_WINDOW)
    ap.add_argument("--repeat-ngram", type=int, default=REPEAT_NGRAM)
    ap.add_argument("--save-clone", action="store_true", help="Save CE-transplanted teacher weights into the artifact")
    args = ap.parse_args()

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        args.output = os.path.join(os.path.dirname(__file__), f"{safe_name}.ce.pt")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    convert(
        args.model,
        args.output,
        args.device,
        sparse=args.sparse,
        phase=args.phase,
        phase2_steps=args.phase2_steps,
        save_pq=args.save_pq,
        pq_only=args.pq_only,
        pq_subdim=args.pq_subdim,
        pq_bits=args.pq_bits,
        pq_iters=args.pq_iters,
        pq_batch_size=args.pq_batch_size,
        pq_sample_size=args.pq_sample_size,
        decoder_prev_scale=args.decoder_prev_scale,
        distill_tokens=args.distill_tokens,
        distill_cb_topk=args.distill_cb_topk,
        distill_teacher_topk=args.distill_teacher_topk,
        distill_ridge=args.distill_ridge,
        relax_steps=args.relax_steps,
        metric_rank=args.metric_rank,
        token_head_max_vocab=args.token_head_max_vocab,
        token_head_scale=args.token_head_scale,
        decoder_query_blend=args.decoder_query_blend,
        decoder_candidate_ratio=args.decoder_candidate_ratio,
        curvature_alpha=args.curvature_alpha,
        curvature_lambda=args.curvature_lambda,
        curvature_steepness=args.curvature_steepness,
        curvature_eval_topk=args.curvature_eval_topk,
        repeat_window=args.repeat_window,
        repeat_ngram=args.repeat_ngram,
        save_clone=args.save_clone,
    )


if __name__ == "__main__":
    main()
