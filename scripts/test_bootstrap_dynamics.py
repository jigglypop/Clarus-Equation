"""CE relax 동역학의 부트스트랩 수렴 직접 측정.

핵심 검증 (8_Roadmap.md 0.5절 옵션 1):
  clarus/engine.py의 relax는 Hopfield식 W 행렬 위에서 부트스트랩 방정식을
  실제로 푸는 dynamical solver다. 사양이 transformer가 아닌 이 substrate에
  대해서는 옳을 수 있다.

측정:
  1) 수축률 rho_observed = ||m_{n+1} - m*|| / ||m_n - m*||
     - 사양 예측: rho = D_eff * eps^2 = 3.178 * 0.0487 = 0.155
  2) 평형 m*의 emergent sparsity:
     - threshold-based active fraction
     - top-eps^2 분포 분리도
  3) 에너지 단조 감소 (Hopfield 보장)

성공 기준:
  - rho_observed가 0.155 +/- 0.05 안에 들어오면 -> 사양 valid in this substrate
  - active fraction이 ~4.87% 근방에 emergent하면 -> spec confirmed
  - 둘 다 실패하면 -> 수식 자체가 W 행렬 동역학에서도 안 맞음

3가지 W 변형 비교:
  - random:    무작위 W (대조군, 평형 없음 가능)
  - sym_psd:   대칭 PSD W (Hopfield 표준)
  - sym_sparse: 대칭 + sparse (CE 사양 r_c=pi 반경 적용)

Usage:
  .venv/Scripts/python.exe scripts/test_bootstrap_dynamics.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from clarus.constants import (
    AD, PORTAL, BYPASS, T_WAKE,
    ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
    BOOTSTRAP_CONTRACTION,
)
from clarus.ce_ops import pack_sparse, relax_packed


def safe_print(*a, **k):
    print(*a, **k, flush=True)


# ---------------------------------------------------------------------------
# W 변형 생성
# ---------------------------------------------------------------------------
def make_W_random(dim: int, seed: int) -> torch.Tensor:
    """랜덤 W (베이스라인). 일반적으로 부트스트랩 평형 없음."""
    g = torch.Generator(); g.manual_seed(seed)
    W = torch.randn(dim, dim, generator=g) / math.sqrt(dim)
    return W


def make_W_sym_psd(dim: int, seed: int) -> torch.Tensor:
    """대칭 PSD W (Hopfield 표준). 에너지 단조 감소 보장."""
    g = torch.Generator(); g.manual_seed(seed)
    A = torch.randn(dim, dim, generator=g) / math.sqrt(dim)
    W = A @ A.T  # PSD
    # diag 제거 후 정규화 (Hopfield 관습).
    W.fill_diagonal_(0.0)
    W = W / W.norm()
    # eigenvalue를 음수로 만들기 (energy = -1/2 m^T W m이 minimum 가지도록).
    eigvals = torch.linalg.eigvalsh(W)
    lam_max = float(eigvals[-1].item())
    if lam_max >= 0:
        W = W - (lam_max + 1e-3) * torch.eye(dim)
    return W


def make_W_sym_sparse(dim: int, seed: int, radius: float = math.pi) -> torch.Tensor:
    """3D 격자 좌표 위에서 r_c=pi 반경 안의 노드만 연결 (CE 사양 5_Sparsity.md)."""
    side = int(math.ceil(dim ** (1.0 / 3.0)))
    idx = torch.arange(dim)
    x = idx // (side * side)
    y = (idx // side) % side
    z = idx % side
    coords = torch.stack([x, y, z], dim=1).float()
    dist = torch.cdist(coords, coords)
    mask = (dist > 0) & (dist <= radius)
    g = torch.Generator(); g.manual_seed(seed)
    W = torch.randn(dim, dim, generator=g) / math.sqrt(dim)
    W = W * mask.float()
    W = 0.5 * (W + W.T)  # 대칭화
    eigvals = torch.linalg.eigvalsh(W)
    lam_max = float(eigvals[-1].item())
    if lam_max >= 0:
        W = W - (lam_max + 1e-3) * torch.eye(dim)
    return W


# ---------------------------------------------------------------------------
# 동역학 측정
# ---------------------------------------------------------------------------
def relax_one_step(values, col_idx, row_ptr, b, phi, m, *, dense_w=None,
                   tau=2.0, dt=0.01, lambda0=1.0):
    """1 step만 진행하는 relax (정직한 trajectory 측정용)."""
    m_next, _, _ = relax_packed(
        values, col_idx, row_ptr, b, phi, m,
        portal=PORTAL, bypass=BYPASS, t_wake=T_WAKE,
        beta=1.0, cb_w=PORTAL,
        lambda0=lambda0, lambda_phi=0.5, lambda_var=0.25,
        tau=tau, dt=dt,
        max_steps=1, tol=0.0,
        anneal_ratio=1.0, noise_scale=0.0,  # 결정적 측정
        metric_rank=0, backend="torch",
        seed=0, dense_w=dense_w,
    )
    return m_next


def measure_active_fraction(m: torch.Tensor, threshold_factor: float = 1.0) -> tuple:
    """threshold = factor * std(|m|) 위의 비율 + top-percentile 분포."""
    abs_m = m.abs()
    n = abs_m.numel()
    std = abs_m.std().clamp_min(1e-8)
    thr = threshold_factor * float(std.item())
    active_frac = float((abs_m > thr).float().mean().item())
    # 분위수 기반: top 4.87% / 다음 26.2% / 나머지
    sorted_abs, _ = abs_m.sort(descending=True)
    k_active = max(1, int(round(ACTIVE_RATIO * n)))
    k_struct = max(1, int(round((ACTIVE_RATIO + STRUCT_RATIO) * n)))
    e_active = float((sorted_abs[:k_active] ** 2).sum().item())
    e_struct = float((sorted_abs[k_active:k_struct] ** 2).sum().item())
    e_bg = float((sorted_abs[k_struct:] ** 2).sum().item())
    e_total = e_active + e_struct + e_bg
    if e_total < 1e-12:
        return active_frac, 0.0, 0.0, 0.0
    return active_frac, e_active / e_total, e_struct / e_total, e_bg / e_total


def run_dynamics_test(W: torch.Tensor, name: str, args) -> dict:
    """단일 W에 대해 relax trajectory 기록."""
    safe_print(f"\n{'='*60}\n  W variant: {name}\n{'='*60}")
    dim = W.shape[0]
    values, col_idx, row_ptr = pack_sparse(W, backend="torch")
    dense_w = W if args.use_dense else None
    g = torch.Generator(); g.manual_seed(args.seed)
    m = torch.randn(dim, generator=g)
    m = m / m.norm() * float(dim)
    b = torch.zeros(dim)
    phi = torch.zeros(dim)
    # 평형 m_star: 충분히 많은 step 돌려서 수렴값 확보.
    m_star_iter, _, _ = relax_packed(
        values, col_idx, row_ptr, b, phi, m,
        portal=PORTAL, bypass=BYPASS, t_wake=T_WAKE,
        beta=1.0, cb_w=PORTAL,
        lambda0=1.0, lambda_phi=0.5, lambda_var=0.25,
        tau=2.0, dt=0.01,
        max_steps=args.max_steps_converge, tol=1e-6,
        anneal_ratio=1.0, noise_scale=0.0,
        metric_rank=0, backend="torch",
        seed=0, dense_w=dense_w,
    )
    safe_print(f"  m_star norm: {m_star_iter.norm().item():.4f}  "
               f"|m_star_max|: {m_star_iter.abs().max().item():.4f}")
    # Trajectory 측정: m_0에서 시작해 step마다 거리 + 활성도.
    history = []
    m_curr = m.clone()
    for k in range(args.max_steps):
        dist = float((m_curr - m_star_iter).norm().item())
        active_frac, e_act, e_str, e_bg = measure_active_fraction(m_curr,
                                                                   args.thr_factor)
        history.append({
            "step": k,
            "dist_to_mstar": dist,
            "m_norm": float(m_curr.norm().item()),
            "active_frac": active_frac,
            "energy_active": e_act,
            "energy_struct": e_str,
            "energy_bg": e_bg,
        })
        m_curr = relax_one_step(values, col_idx, row_ptr, b, phi, m_curr,
                                dense_w=dense_w)
    # 마지막 측정 (after final step).
    dist = float((m_curr - m_star_iter).norm().item())
    active_frac, e_act, e_str, e_bg = measure_active_fraction(m_curr, args.thr_factor)
    history.append({
        "step": args.max_steps,
        "dist_to_mstar": dist,
        "m_norm": float(m_curr.norm().item()),
        "active_frac": active_frac,
        "energy_active": e_act,
        "energy_struct": e_str,
        "energy_bg": e_bg,
    })
    # 수축률 추정: log(dist) 직선 fit.
    valid = [(h["step"], h["dist_to_mstar"]) for h in history
             if h["dist_to_mstar"] > 1e-12]
    if len(valid) >= 5:
        # 중간 절반 사용 (transient + stagnation 회피).
        mid_start = len(valid) // 4
        mid_end = 3 * len(valid) // 4
        steps_t = torch.tensor([float(s) for s, _ in valid[mid_start:mid_end]])
        log_d = torch.tensor([math.log(d) for _, d in valid[mid_start:mid_end]])
        if len(steps_t) >= 2:
            slope = float(((steps_t - steps_t.mean()) * (log_d - log_d.mean())).sum() /
                          ((steps_t - steps_t.mean()) ** 2).sum().clamp_min(1e-12))
            rho_observed = math.exp(slope)
        else:
            rho_observed = float("nan")
    else:
        rho_observed = float("nan")
    last = history[-1]
    safe_print(f"  observed rho (per relax step): {rho_observed:.4f}  "
               f"(spec: {BOOTSTRAP_CONTRACTION:.4f})")
    safe_print(f"  final dist to m_star  : {last['dist_to_mstar']:.6f}")
    safe_print(f"  final active fraction : {last['active_frac']*100:.2f}%  "
               f"(spec: {ACTIVE_RATIO*100:.2f}%)")
    safe_print(f"  energy distribution top {ACTIVE_RATIO*100:.1f}% / "
               f"next {STRUCT_RATIO*100:.1f}% / rest:")
    safe_print(f"    {last['energy_active']*100:.2f}% / "
               f"{last['energy_struct']*100:.2f}% / {last['energy_bg']*100:.2f}%")
    safe_print(f"    spec target: {ACTIVE_RATIO*100:.2f}% / "
               f"{STRUCT_RATIO*100:.2f}% / {BACKGROUND_RATIO*100:.2f}%")
    return {
        "name": name,
        "dim": dim,
        "rho_observed": rho_observed,
        "rho_spec": BOOTSTRAP_CONTRACTION,
        "final_active_frac": last["active_frac"],
        "active_frac_spec": ACTIVE_RATIO,
        "final_energy_distribution": [
            last["energy_active"], last["energy_struct"], last["energy_bg"]
        ],
        "spec_distribution": [ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO],
        "history": history,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=80,
                    help="Trajectory measurement steps.")
    ap.add_argument("--max-steps-converge", type=int, default=400,
                    help="Steps to converge to m_star.")
    ap.add_argument("--thr-factor", type=float, default=1.0,
                    help="Active threshold = thr_factor * std(|m|).")
    ap.add_argument("--use-dense", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default="random,sym_psd,sym_sparse")
    ap.add_argument("--output",
                    default="examples/ai/results/bootstrap_dynamics.json")
    return ap.parse_args()


def main():
    args = parse_args()
    safe_print("=" * 60)
    safe_print(" CE bootstrap dynamics direct test")
    safe_print(f"  dim={args.dim}  max_steps={args.max_steps}  seed={args.seed}")
    safe_print(f"  spec: rho={BOOTSTRAP_CONTRACTION:.4f}, "
               f"p*=({ACTIVE_RATIO:.4f}, {STRUCT_RATIO:.4f}, {BACKGROUND_RATIO:.4f})")
    safe_print("=" * 60)

    builders = {
        "random": make_W_random,
        "sym_psd": make_W_sym_psd,
        "sym_sparse": make_W_sym_sparse,
    }
    results = []
    for variant in args.variants.split(","):
        if variant not in builders:
            safe_print(f"  unknown variant '{variant}', skipping")
            continue
        W = builders[variant](args.dim, args.seed)
        r = run_dynamics_test(W, variant, args)
        results.append(r)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    safe_print(f"\n[SAVED] {args.output}")

    safe_print("\n" + "=" * 60)
    safe_print(" Summary: spec validation in CE relax substrate")
    safe_print("=" * 60)
    safe_print(f"{'variant':<14s}  {'rho_obs':>9s}  {'rho_spec':>9s}  "
               f"{'active_obs':>11s}  {'active_spec':>12s}")
    for r in results:
        safe_print(
            f"{r['name']:<14s}  {r['rho_observed']:>9.4f}  {r['rho_spec']:>9.4f}  "
            f"{r['final_active_frac']*100:>10.2f}%  {r['active_frac_spec']*100:>11.2f}%"
        )
    safe_print("\n해석:")
    safe_print("  - rho_observed가 0.155 +/- 0.05 안에 들어오는 변형이 있으면")
    safe_print("    -> 사양은 그 substrate에서 valid")
    safe_print("  - active_obs가 ~4.87% 근방이면 -> emergent sparsity 확인")
    safe_print("  - 둘 다 멀면 -> 수식 자체가 이 W 동역학에서도 안 맞음")


if __name__ == "__main__":
    main()
