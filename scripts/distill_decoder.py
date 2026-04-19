"""Distill the CE runtime decoder projections from a teacher language model.

Offline only. Loads an existing CE runtime artifact (built via
`scripts/build_artifact.py`) plus a teacher Hugging Face causal LM, samples
(prompt -> teacher hidden) pairs across a Korean corpus, fits the linear
decoder projections by closed-form ridge regression, and writes them back
into the artifact. The teacher is released before saving; the resulting
artifact remains runtime-isolated (model is None at inference, no
clone_state, no allow_pretrained_fallback).

Why a single linear layer fits at all:
- The CE runtime decoder query is, by construction:
      query = state_hidden @ state_proj
              + prev_scale * prev_emb @ prev_proj
              + query_bias
- For each prompt we treat (state_hidden, prev_emb) as features and the
  teacher's final-layer hidden (after ln_f) at the last position as the
  target. Ridge regression yields the optimal linear (state_proj,
  prev_proj, bias) given those features.
- The next-token logits are then computed at runtime as query @ emb.T
  (or via PQ scores), so closing the gap to teacher_h is exactly the
  distillation objective for the next-token distribution.

Usage:
    python scripts/distill_decoder.py \
        --artifact clarus/skt_kogpt2-base-v2.ce.pt \
        --teacher skt/kogpt2-base-v2 \
        --device cpu
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from clarus.engine import CEEngine


CORPUS = [
    "인공지능의 미래는 우리가 생각하는 것보다 훨씬 빠르게 다가오고 있다.",
    "오늘 날씨가 좋아서 친구들과 함께 한강 공원에 나가 자전거를 탔다.",
    "한국어를 배우는 가장 좋은 방법은 매일 한국 드라마를 자막 없이 보는 것이다.",
    "좋은 모델의 조건은 정확성과 효율성의 균형을 잘 맞추는 데 있다.",
    "대한민국의 교육 제도는 입시 위주의 한계를 극복하려는 시도가 계속되고 있다.",
    "건강한 식단을 유지하려면 과일과 채소를 충분히 섭취하는 습관이 중요하다.",
    "세계 경제의 흐름은 최근 인공지능 기술의 발전으로 큰 변화를 맞이하고 있다.",
    "과학 기술의 발전은 인류의 삶을 편리하게 만들었지만 새로운 윤리적 문제도 낳았다.",
    "한국의 전통 음식 중 대표적인 것은 김치와 비빔밥, 그리고 불고기이다.",
    "효율적인 학습을 위해서는 반복과 복습이 필수적이며 충분한 휴식도 중요하다.",
    "한글의 우수성은 과학적 원리에 기반한 자모 결합 체계에서 잘 드러난다.",
    "도시와 농촌의 격차를 줄이기 위해서는 공공 인프라 투자가 우선되어야 한다.",
    "서울의 봄은 짧지만 벚꽃이 만개할 때면 거리 전체가 분홍빛으로 물든다.",
    "가장 좋아하는 책은 어릴 때 처음 읽었던 어린 왕자이며 지금도 자주 펼쳐 본다.",
    "한국의 사계절은 뚜렷한 변화를 보이며 각 계절마다 고유한 매력을 지닌다.",
    "기후 변화 문제를 해결하기 위해서는 국제적인 협력과 개인의 실천이 모두 필요하다.",
    "독서는 새로운 지식을 얻는 가장 좋은 방법이며 사고의 폭을 넓혀 준다.",
    "대학교 입학을 준비하는 학생들은 다양한 분야에 대한 호기심을 잃지 말아야 한다.",
    "한국의 영화 산업은 최근 세계적인 인정을 받으며 국제 영화제에서 좋은 성과를 거두고 있다.",
    "음악은 언어를 초월하여 사람들의 감정에 깊이 영향을 미치는 예술 형식이다.",
    "기술의 발전 속에서도 인간의 따뜻한 마음과 공감 능력은 변치 않는 가치이다.",
    "환경 보호를 위해서는 일회용품 사용을 줄이고 재활용을 생활화하는 것이 중요하다.",
    "한국의 역사는 수많은 외침과 시련을 이겨낸 강인한 민족 정신을 보여 준다.",
    "여행은 새로운 문화를 직접 체험하고 자신을 돌아볼 수 있는 좋은 기회가 된다.",
    "스포츠는 신체 건강을 증진할 뿐 아니라 협동심과 인내심을 길러 준다.",
    "한국의 전통 가옥인 한옥은 자연과 조화를 이루는 건축 양식으로 유명하다.",
    "꾸준한 운동은 신체뿐만 아니라 정신 건강에도 매우 긍정적인 영향을 준다.",
    "디지털 시대에는 정보를 비판적으로 받아들이는 능력이 점점 더 중요해지고 있다.",
    "가족과 함께 보내는 시간은 인생에서 가장 소중한 자산 중 하나이다.",
    "예술은 우리에게 새로운 시각을 제공하며 일상의 아름다움을 발견하게 해 준다.",
    "한국의 명절인 설날과 추석에는 가족이 모여 차례를 지내고 음식을 나눈다.",
    "꿈을 이루기 위해서는 명확한 목표 설정과 꾸준한 노력이 함께 필요하다.",
    "친구와의 깊은 대화는 어려운 시기를 견디는 데 큰 힘이 되어 준다.",
    "기술 혁신은 새로운 기회를 만들어 내지만 동시에 일자리 변화를 가져오기도 한다.",
    "한국의 자연 경관은 산과 바다가 가까이 있어 다양한 풍경을 즐길 수 있다.",
    "외국어를 배우면 새로운 사고 방식과 다양한 문화를 이해하는 데 도움이 된다.",
    "성공의 정의는 사람마다 다르지만 자신만의 기준을 갖는 것이 중요하다.",
    "한국의 음식 문화는 발효 음식이 발달하여 건강에 좋은 식단으로 평가받는다.",
    "책을 읽는 습관은 어린 시절부터 자연스럽게 길러 주는 것이 좋다.",
    "지속 가능한 발전은 현 세대의 필요를 충족하면서도 미래 세대를 배려하는 것이다.",
    "한국 전통 음악인 국악은 독특한 가락과 리듬으로 깊은 감동을 전한다.",
    "운전을 처음 배울 때는 안전을 최우선으로 고려하는 자세가 필요하다.",
    "사랑은 받는 것보다 주는 것이 더 큰 행복을 가져다 주는 감정이다.",
    "수학은 모든 과학의 기초가 되며 논리적 사고력을 길러 주는 학문이다.",
    "한국의 도시들은 빠른 속도로 발전하면서도 전통과 현대가 공존하는 모습을 보인다.",
    "어려움을 극복한 경험은 사람을 한층 더 성숙하게 만들어 준다.",
    "환경 문제는 한 국가만의 노력으로는 해결할 수 없는 글로벌 과제이다.",
    "한국의 차 문화는 녹차를 중심으로 발달했으며 다도라는 독특한 형식이 있다.",
    "교육의 본질은 단순한 지식 전달이 아닌 비판적 사고력을 기르는 데 있다.",
    "건강은 잃기 전에 지켜야 하며 작은 습관이 큰 차이를 만든다.",
    "예술가는 자신의 작품을 통해 시대의 정신과 감정을 표현한다.",
    "한국의 전통 의상인 한복은 우아한 곡선과 아름다운 색감으로 사랑받는다.",
    "여행지에서 만난 낯선 사람과의 대화는 잊지 못할 추억이 되곤 한다.",
    "꾸준한 독서는 어휘력과 표현력을 풍부하게 만드는 가장 좋은 방법이다.",
    "한국의 첨단 기술 산업은 세계 시장에서 중요한 위치를 차지하고 있다.",
    "자연 속에서 보내는 시간은 도시 생활의 피로를 풀어 주는 좋은 휴식이다.",
    "어린이들이 마음껏 뛰놀 수 있는 공간이 점점 줄어드는 것이 안타깝다.",
    "한국의 전통 차례 음식에는 깊은 의미와 가족의 정성이 담겨 있다.",
    "올바른 식습관과 규칙적인 운동은 평생 건강을 위한 가장 확실한 투자이다.",
    "도서관은 누구에게나 열려 있는 평등한 지식의 보고이다.",
]


def collect_pairs(eng: CEEngine, teacher, tokenizer, device: str, max_window: int = 24):
    """Collect (state, prev_emb, h_teacher) triples by sliding windows.

    For each sentence, build prompts of length k = 2 .. min(len, max_window),
    then take the teacher's final-layer hidden state at position k-1 (after
    ln_f) as the regression target for the runtime decoder query at the
    same position.
    """
    states: list[torch.Tensor] = []
    prevs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    ln_f = teacher.transformer.ln_f
    teacher.eval()

    with torch.no_grad():
        for sent in CORPUS:
            ids_full = tokenizer.encode(sent, return_tensors="pt").to(device)
            seq_len = ids_full.shape[1]
            if seq_len < 3:
                continue
            stop = min(seq_len, max_window)
            out = teacher(ids_full[:, :stop], output_hidden_states=True)
            h_seq = out.hidden_states[-1][0]
            h_seq = ln_f(h_seq).float().cpu()
            for k in range(2, stop):
                prompt_ids = ids_full[:, :k].clone()
                state, _ = eng.runtime_prompt_state(prompt_ids)
                emb_seq = eng.prompt_embeddings(prompt_ids)
                states.append(state.detach().float().cpu().view(-1))
                prevs.append(emb_seq[-1].detach().float().cpu().view(-1))
                targets.append(h_seq[k - 1].view(-1))
    if not states:
        raise RuntimeError("no training pairs collected; corpus too short")
    return torch.stack(states), torch.stack(prevs), torch.stack(targets)


def fit_ridge(
    state: torch.Tensor,
    prev: torch.Tensor,
    target: torch.Tensor,
    *,
    prev_scale: float,
    ridge: float,
):
    """Solve target = state @ A + prev_scale * prev @ B + c by ridge LS."""
    n, d = state.shape
    X = torch.cat([state, prev_scale * prev, torch.ones(n, 1)], dim=1)
    XtX = X.T @ X + ridge * torch.eye(X.shape[1])
    XtY = X.T @ target
    theta = torch.linalg.solve(XtX, XtY)
    A = theta[:d, :].contiguous()
    B = theta[d : 2 * d, :].contiguous()
    c = theta[2 * d, :].contiguous()
    pred = X @ theta
    rss = float(((pred - target) ** 2).sum().item())
    tss = float(((target - target.mean(dim=0, keepdim=True)) ** 2).sum().item())
    r2 = 1.0 - rss / max(tss, 1e-12)
    return A, B, c, r2


def main():
    ap = argparse.ArgumentParser(description="Decoder distillation (offline, teacher-free at runtime)")
    ap.add_argument("--artifact", default="clarus/skt_kogpt2-base-v2.ce.pt")
    ap.add_argument("--teacher", default="skt/kogpt2-base-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--prev-scale", type=float, default=0.35)
    ap.add_argument("--max-window", type=int, default=24)
    ap.add_argument("--blend", type=float, default=0.5,
                    help="decoder_query_blend: blend*projected + (1-blend)*state_hidden.")
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  CE Decoder Distillation (offline, teacher discarded)")
    print("=" * 60)
    print(f"  artifact: {args.artifact}")
    print(f"  teacher : {args.teacher}")

    print("\n[1/4] Loading runtime artifact (CEEngine, isolated)...")
    eng = CEEngine(args.artifact, device=args.device)
    if eng.model is not None or eng.model_source != "runtime":
        raise RuntimeError("Loaded artifact is not in runtime-only state; aborting.")
    if "clone_state" in eng.data or "clone_config" in eng.data:
        raise RuntimeError("Artifact contains clone_state/clone_config; rebuild as runtime-only first.")
    print(f"  model is None: True  source={eng.model_source}  d={eng.d}  vocab={eng.vocab}")

    print("\n[2/4] Loading teacher (build-time only)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).to(args.device)

    print(f"\n[3/4] Collecting (state, prev_emb, h_teacher) pairs from {len(CORPUS)} sentences...")
    t0 = time.perf_counter()
    state, prev, target = collect_pairs(eng, teacher, tok, args.device, max_window=args.max_window)
    print(f"  pairs: {state.shape[0]}  state={tuple(state.shape)}  target={tuple(target.shape)}  took {time.perf_counter()-t0:.1f}s")

    print(f"\n[4/4] Ridge regression  prev_scale={args.prev_scale}  ridge={args.ridge}")
    A, B, c, r2 = fit_ridge(state, prev, target, prev_scale=args.prev_scale, ridge=args.ridge)
    print(f"  R^2 (target reconstruction) = {r2:.4f}")
    print(f"  state_proj norm: {A.norm().item():.3f}  prev_proj norm: {B.norm().item():.3f}  bias norm: {c.norm().item():.3f}")

    print("\n[release] freeing teacher before save")
    del teacher, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[save] writing distilled projections back into artifact")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    artifact["decoder_state_proj"] = A
    artifact["decoder_prev_proj"] = B
    artifact["decoder_query_bias"] = c
    artifact["decoder_prev_scale"] = float(args.prev_scale)
    artifact["decoder_query_blend"] = float(args.blend)
    artifact["distill_meta"] = {
        "teacher": args.teacher,
        "ridge": float(args.ridge),
        "n_pairs": int(state.shape[0]),
        "r2": float(r2),
    }
    torch.save(artifact, args.artifact)
    size_mb = os.path.getsize(args.artifact) / 1024 / 1024
    print(f"  saved: {args.artifact}  size={size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
