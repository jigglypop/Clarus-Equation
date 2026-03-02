"""ClarusLM: CE 3x3+1 격자 위상동형 AGI의 LLM 구현.

Transformer에 대한 CE 수정 4가지:
  LBONorm      -- LayerNorm 대체 (라플라스-벨트라미 확산)
  GaugeLattice -- FFN 대체 (SU(3) x SU(2) x U(1) + Phi)
  spectral_norm -- 유니타리 조건 |det T|^2 <= 1 (환각 구조적 억제)
  curvature loss -- lambda |Delta_g Phi|^2 정규화

Reference: docs/6_뇌/agi.md
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# CE coupling constants (d=3에서 연역, 자유 파라미터 0)
ALPHA_S = 0.11789   # SU(3) strong
ALPHA_W = 0.03352   # SU(2) weak
ALPHA_EM = 0.00775  # U(1) electromagnetic


class LBONorm(nn.Module):
    """LayerNorm + Laplace-Beltrami 확산.

    1) F.layer_norm으로 정규화 (안정성 보장, LayerNorm과 동일)
    2) 저랭크 확산: xW = x V^T V, Lx = x - xW
    3) x_out = (x_norm - h * Lx) * scale + bias

    h=0이면 LayerNorm과 완전 동일. h>0이면 기하학적 확산 추가.
    곡률 ||Lx||^2 를 저장하여 정규화 손실에 사용.
    """

    def __init__(self, dim, rank=None, step_size=0.1):
        super().__init__()
        if rank is None:
            rank = max(4, dim // 8)
        self.dim = dim
        self.V = nn.Parameter(torch.randn(rank, dim) * (1.0 / math.sqrt(rank)))
        self.h = nn.Parameter(torch.tensor(step_size))
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self._curvature = 0.0

    def forward(self, x):
        x = F.layer_norm(x, (self.dim,))
        xW = F.linear(F.linear(x, self.V), self.V.T)
        Lx = x - xW
        h = self.h.abs().clamp(max=0.5)
        self._curvature = (Lx * Lx).mean()
        return (x - h * Lx) * self.scale + self.bias


class GaugeLattice(nn.Module):
    """3x3+1 게이지 격자 FFN.

    diag(SU(3), SU(2), U(1)) + Phi_LBO.
    채널 비율: alpha_s:alpha_w:alpha_em = 74.1:21.1:4.9 (CE 연역).
    대각 전이 행렬 -- 채널 간 혼합 없음.
    """

    def __init__(self, dim, mult=4):
        super().__init__()
        total = ALPHA_S + ALPHA_W + ALPHA_EM
        self.d3 = max(1, round(dim * ALPHA_S / total))
        self.d2 = max(1, round(dim * ALPHA_W / total))
        self.d1 = dim - self.d3 - self.d2
        assert self.d1 >= 1

        h = dim * mult
        h3 = max(1, round(h * ALPHA_S / total))
        h2 = max(1, round(h * ALPHA_W / total))
        h1 = max(1, h - h3 - h2)

        self.su3 = nn.Sequential(
            nn.Linear(self.d3, h3, bias=False), nn.SiLU(),
            nn.Linear(h3, self.d3, bias=False))
        self.su2 = nn.Sequential(
            nn.Linear(self.d2, h2, bias=False), nn.SiLU(),
            nn.Linear(h2, self.d2, bias=False))
        self.u1 = nn.Sequential(
            nn.Linear(self.d1, h1, bias=False), nn.SiLU(),
            nn.Linear(h1, self.d1, bias=False))
        self.phi = LBONorm(dim)

    def forward(self, x):
        s = self.d3
        y = torch.cat([
            self.su3(x[..., :s]),
            self.su2(x[..., s:s + self.d2]),
            self.u1(x[..., s + self.d2:]),
        ], dim=-1)
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
    def __init__(self, dim, n_heads, ffn_mult=4):
        super().__init__()
        self.norm1 = LBONorm(dim)
        self.attn = ClarusAttention(dim, n_heads)
        self.norm2 = LBONorm(dim)
        self.ffn = GaugeLattice(dim, ffn_mult)

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
                 max_seq_len=512, ffn_mult=4, lambda_curv=0.01):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.lambda_curv = lambda_curv
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [ClarusBlock(dim, n_heads, ffn_mult) for _ in range(n_layers)])
        self.norm = LBONorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            w = m.weight_orig if hasattr(m, 'weight_orig') else m.weight
            nn.init.normal_(w, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            curv = sum(b.curvature for b in self.blocks) / len(self.blocks)
            loss = ce + self.lambda_curv * curv
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40):
        for _ in range(n):
            x = idx[:, -self.max_seq_len:]
            logits = self(x)[0][:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx

    def lattice_summary(self):
        """3x3+1 격자 구조 요약."""
        b = self.blocks[0].ffn
        total = ALPHA_S + ALPHA_W + ALPHA_EM
        lines = [
            f'SU(3) binding:  {b.d3:4d} dims ({ALPHA_S/total*100:.1f}%)',
            f'SU(2) decision: {b.d2:4d} dims ({ALPHA_W/total*100:.1f}%)',
            f'U(1)  attention:{b.d1:4d} dims ({ALPHA_EM/total*100:.1f}%)',
            f'Phi   smoothing: LBO (rank={b.phi.V.shape[0]})',
        ]
        return '\n'.join(lines)
