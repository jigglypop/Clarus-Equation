"""3분배 가중치 분류기 (docs/7_AGI/5_Sparsity.md 4절).

CE 부트스트랩 고정점 p* = (eps^2, Omega_DM, Omega_Lambda) = (0.0487, 0.2623, 0.6891)
을 가중치 단위 분류로 구현한다:

  ACTIVE  (4.87%): 매 사이클 학습. 추론 시 항상 사용.
  STRUCT  (26.2%): NREM 위상에서만 학습 (구조적 가중치).
  BG      (68.9%): 동결. fine-tune 중 변경 없음.

매 NREM 사이클 후 사용 빈도(decayed grad magnitude EMA) 기반으로 재분류한다.
이 모듈은 모델에 비침습적으로 grad mask를 적용한다 (forward 변경 없음, gradient만 차단).

Usage:
    from clarus.sparsity import TernaryClassifier
    from clarus.constants import ACTIVE_RATIO, STRUCT_RATIO

    cls = TernaryClassifier(model,
        active_ratio=ACTIVE_RATIO, struct_ratio=STRUCT_RATIO)

    # 학습 루프:
    for cycle in range(N):
        # WAKE: gradient 누적
        wake_phase(...)
        cls.update_freq()           # |grad| EMA 업데이트
        cls.apply_grad_mask(allow_struct=False)  # ACTIVE만 통과

        # NREM: 평탄화 + step (이 위상에서만 STRUCT도 학습)
        cls.apply_grad_mask(allow_struct=True)   # ACTIVE + STRUCT 통과
        nrem_phase(...)

        # 사이클마다 재분류 (사양에 따라)
        if cycle % reclassify_every == 0:
            cls.reclassify()

참고:
- 메모리: tracked weight 당 (freq float32 + active bool + struct bool) ~6 bytes/elem.
  KoGPT2 Linear weights (~50M) 기준 ~300MB.
- target_filter로 추적 범위 제한 가능 (기본: 모든 nn.Linear, embed/head 제외).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn

try:
    from .constants import ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO
except ImportError:
    from clarus.constants import ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO


def _default_filter(name: str, module: nn.Module) -> bool:
    """기본 필터: nn.Linear만, embed/head/lm_head 류 제외."""
    if not isinstance(module, nn.Linear):
        return False
    lname = name.lower()
    if any(skip in lname for skip in ("embed", "head", "norm", "ln_")):
        return False
    return True


@dataclass
class _Tracked:
    """단일 추적 weight의 상태."""
    name: str
    weight: nn.Parameter
    freq: torch.Tensor          # |grad| EMA, weight와 같은 shape
    mask_active: torch.Tensor   # bool, weight와 같은 shape
    mask_struct: torch.Tensor   # bool (active + struct = top struct_ratio + active_ratio)


class TernaryClassifier:
    """3분배 가중치 분류기.

    Attributes:
        active_ratio: ACTIVE 비율 (기본 ACTIVE_RATIO = 0.0487).
        struct_ratio: STRUCT 비율 (기본 STRUCT_RATIO = 0.2623).
                      ACTIVE는 STRUCT의 부분집합 (top-k 안에 top-k_act가 들어감).
        freq_decay: |grad| EMA 감쇠 (기본 0.99).
        tracked: 추적 중인 weight 목록.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        active_ratio: float = ACTIVE_RATIO,
        struct_ratio: float = STRUCT_RATIO,
        freq_decay: float = 0.99,
        target_filter: Callable[[str, nn.Module], bool] = _default_filter,
        init_by_magnitude: bool = True,
    ) -> None:
        if not 0.0 < active_ratio < 1.0:
            raise ValueError(f"active_ratio must be in (0, 1), got {active_ratio}")
        if not 0.0 < struct_ratio < 1.0:
            raise ValueError(f"struct_ratio must be in (0, 1), got {struct_ratio}")
        if active_ratio + struct_ratio >= 1.0:
            raise ValueError(
                f"active_ratio + struct_ratio must be < 1, "
                f"got {active_ratio} + {struct_ratio} = {active_ratio + struct_ratio}"
            )
        self.active_ratio = float(active_ratio)
        self.struct_ratio = float(struct_ratio)
        self.freq_decay = float(freq_decay)
        self.tracked: list[_Tracked] = []
        for name, module in model.named_modules():
            if not target_filter(name, module):
                continue
            # spectral_norm wrap된 경우 weight_orig를 사용 (weight는 매 forward 재계산).
            w = module.weight_orig if hasattr(module, 'weight_orig') else module.weight
            if not isinstance(w, nn.Parameter):
                continue
            freq = torch.zeros_like(w.data)
            self.tracked.append(_Tracked(
                name=name, weight=w, freq=freq,
                mask_active=torch.ones_like(w.data, dtype=torch.bool),
                mask_struct=torch.ones_like(w.data, dtype=torch.bool),
            ))
        if not self.tracked:
            raise RuntimeError("TernaryClassifier: no parameters matched target_filter")
        if init_by_magnitude:
            self._reclassify_from(lambda t: t.weight.data.abs())

    @torch.no_grad()
    def update_freq(self) -> None:
        """|grad| EMA 업데이트. backward 직후, optim.step() 전에 호출."""
        for t in self.tracked:
            if t.weight.grad is None:
                continue
            # freq <- decay * freq + (1 - decay) * |grad|
            t.freq.mul_(self.freq_decay).add_(t.weight.grad.abs(), alpha=1.0 - self.freq_decay)

    @torch.no_grad()
    def reclassify(self) -> None:
        """현재 freq 기준으로 ACTIVE / STRUCT 마스크 재계산 (5_Sparsity.md 4.4)."""
        self._reclassify_from(lambda t: t.freq)

    @torch.no_grad()
    def _reclassify_from(self, score_fn: Callable[[_Tracked], torch.Tensor]) -> None:
        """공통 재분류 핵심: score 텐서로부터 active/struct 마스크 재구성."""
        for t in self.tracked:
            score = score_fn(t).view(-1)
            n = score.numel()
            k_act = max(1, int(self.active_ratio * n))
            k_str_total = max(k_act + 1, int((self.active_ratio + self.struct_ratio) * n))
            k_str_total = min(k_str_total, n)
            # top k_str_total 인덱스 (active + struct 합집합).
            _, idx_str = torch.topk(score, k_str_total)
            mask_struct = torch.zeros(n, dtype=torch.bool, device=score.device)
            mask_struct[idx_str] = True
            # active = top k_act 인덱스 (struct의 부분집합).
            _, idx_act = torch.topk(score, k_act)
            mask_active = torch.zeros(n, dtype=torch.bool, device=score.device)
            mask_active[idx_act] = True
            t.mask_active = mask_active.view(t.weight.shape)
            t.mask_struct = mask_struct.view(t.weight.shape)

    @torch.no_grad()
    def apply_grad_mask(self, *, allow_struct: bool) -> None:
        """gradient를 분류에 따라 차단.

        allow_struct=False  : ACTIVE만 통과 (WAKE 위상의 quick-step 모드).
        allow_struct=True   : ACTIVE + STRUCT 통과 (NREM 위상 본 학습).
        BG는 항상 차단.
        """
        for t in self.tracked:
            if t.weight.grad is None:
                continue
            mask = t.mask_struct if allow_struct else t.mask_active
            t.weight.grad.mul_(mask.to(t.weight.grad.dtype))

    def stats(self) -> dict:
        """현재 분류 분포 요약."""
        total = sum(t.weight.numel() for t in self.tracked)
        active = sum(int(t.mask_active.sum().item()) for t in self.tracked)
        struct_only = sum(
            int((t.mask_struct & ~t.mask_active).sum().item()) for t in self.tracked
        )
        bg = total - active - struct_only
        return {
            "tracked_params": len(self.tracked),
            "total_weights": total,
            "active": active,
            "active_pct": 100.0 * active / total if total else 0.0,
            "struct": struct_only,
            "struct_pct": 100.0 * struct_only / total if total else 0.0,
            "bg": bg,
            "bg_pct": 100.0 * bg / total if total else 0.0,
            "target_active_pct": 100.0 * self.active_ratio,
            "target_struct_pct": 100.0 * self.struct_ratio,
            "target_bg_pct": 100.0 * (1.0 - self.active_ratio - self.struct_ratio),
        }

    def memory_bytes(self) -> int:
        """추가 메모리 사용량 (freq float + 두 bool 마스크)."""
        total = 0
        for t in self.tracked:
            total += t.freq.numel() * t.freq.element_size()
            total += t.mask_active.numel() * t.mask_active.element_size()
            total += t.mask_struct.numel() * t.mask_struct.element_size()
        return total
