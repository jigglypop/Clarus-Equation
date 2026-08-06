# 7. Euler-Riemann Attention: CE 이론 상수의 Transformer 이식

## 7.1 동기

`docs/경로적분.md` §51-67은 `{e, π, i, 1, 0}`을 CE의 **최소 생성 문법**으로 규정한다.

| 상수 | 코어 역할 | 대표 출력 | 등급 |
|---|---|---|---|
| $e$ | 접힘 생존 함수 | $S(D) = e^{-D}$ | Exact / Selection |
| $\pi$ | 주기 정규화 | $\alpha_{\text{total}} = 1/(2\pi)$ | Selection |
| $i$ | 경로적분 위상 | $Z = \int \mathcal D\phi\,e^{iS/\hbar}$ | Exact |
| $1$ | 정규화된 완전 상태 | $e^0 = 1$ | Selection |
| $0$ | 영점·분기 선택 | $d(d-3) = 0$ | Selection |

본 장은 이 문법을 Transformer의 inductive bias로 시험한다. 추가로 유한하게
검증된 zeta-zero ordinate \(\gamma_n\)를 deterministic frequency bank로
사용한다. 이는 RH, GUE 또는 Hilbert–Pólya의 구현이 아니다.

---

## 7.2 설계 공간

### 7.2.1 Euler-CE Attention (core)

**Position encoding**:
$$
\theta_{h,k}(n) = n \cdot \omega_h \cdot \pi^{1-k/(d_{\text{head}}/2)},\quad
\omega_h = \text{sigmoid}(\text{pi\_gate}_h)
$$
즉 $\pi$를 RoPE의 10000 대신 기저 주파수로. Head별 gate로 소프트 선택.

**Attention decay**:
$$
A_{ij}^{\text{bias}} = -\frac{|i-j|}{\xi_h}\cdot\text{sigmoid}(\text{e\_gate}_h)
$$
즉 $e^{-|i-j|/\xi}$ 이 log-space bias로 추가 (ALiBi-like).

**합성**:
$$
A_{ij} = \text{softmax}_j\!\Bigl(\underbrace{Q_i\cdot R_\pi(i{-}j)\cdot K_j / \sqrt d}_{\pi\text{ phase}} + \underbrace{A_{ij}^{\text{bias}}}_{e\text{ decay}}\Bigr)\cdot\text{mask}_{\text{causal}}
$$

구현: `reality_stone/python/reality_stone/clarus/ce_euler.py::EulerCEAttention`

### 7.2.2 Recursive ClarusCell

CE bootstrap 방정식
$$
\varepsilon^2 = \exp\!\bigl[-(1-\varepsilon^2)\,D_{\text{eff}}\bigr]
$$
은 $x^* = F(x^*)$ 형태의 **고정점 방정식**. Transformer 블록에 이를 이식:
$$
h^{t+1} = F(h^t),\qquad t = 0, 1, \dots, k-1
$$

- **Fixed depth**: $k \in \{1, 2, 3, 6\}$
- **While-loop**: halt when $\|h^{t+1} - h^t\| / \|h^t\| < \text{tol}$
- **Fixed-point regularizer**: $\mathcal L_{\text{FP}} = \lambda\,\|F(F(h)) - F(h)\|^2$

구현: `reality_stone/python/reality_stone/clarus/ce_euler.py::RecursiveEulerCEBlock`

### 7.2.3 Verified-zero Rotary

유한 table에서 검증된 positive-zero ordinate를 주파수 후보로 쓴다.
유한 목록을 사용하는 데 전체 RH는 필요하지 않다.

$$
\theta_{n,k} = n \cdot (\gamma_k / \gamma_1)^{-1} \cdot s_h
$$
첫 100개 \(\gamma_n\)은 provenance가 있는 table로 고정한다. 그 밖의 항이
필요하면 zero-counting equation의 역해를 approximate frequency로 별도
표시하며 \(2\pi n/\log n\) 값을 실제 zero라고 부르지 않는다.

정규화된 고영점 간격의 GUE 거동은 Montgomery–Dyson conjecture와 수치
증거의 대상이지 RH가 보장하는 정리가 아니다.

구현: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::RiemannRotaryAttention`

### 7.2.4 FFN 변형

| 이름 | 식 |
|---|---|
| std | $\text{GELU}(W_1 x)\,W_2$ |
| swiglu | $\text{SiLU}(W_g x) \odot (W_u x)\,W_d$ |
| euler_decay | $\text{GELU}(h)\cdot e^{-\lvert h\rvert/\xi}$ |
| euler_phase | $\text{GELU}(h)\cdot (1 + \eta\cos(\pi h/\tau))$ |
| euler_full | $\text{GELU}(h)\cdot (1 + \eta\cos(\pi h/\tau))\cdot e^{-\lvert h\rvert/\xi}$ |
| zeta | $x\cdot\sigma(x)\cdot(1 + \lambda\cdot z(x))$, $z = \lvert\zeta(1/2 + ix)\rvert^2$ |

구현: `reality_stone/python/reality_stone/clarus/ce_ffn.py`, `reality_stone/python/reality_stone/clarus/ce_zeta.py`

### 7.2.5 Riemann FFN init

\(W_{\text{up}}\)의 한 축을 선택한 \(\gamma_n\) 간격으로 스케일한다.
유한 deterministic initialization이며 “키 좌표가 GUE 분포”라고 단정하지
않는다. 구현: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::riemann_zero_init`

---

## 7.3 벤치 결과 (모두 char-level LM, 400K-char docs corpus, 2-layer, $d_{\text{model}}=96$)

### 7.3.1 Position Encoding (5 seeds × 500 steps, PR #1 commit 3bbc558)

| 변형 | PPL | RoPE 대비 |
|---|---|---|
| std_abs (absolute PE) | 25.29 ± 0.22 | +64σ worse |
| std_rope | 14.94 ± 0.28 | baseline |
| euler_hard (fixed bits) | 14.53 ± 0.30 | **−2.20σ WIN** |
| euler_soft (learnable) | 14.13 ± 0.22 | **−5.12σ WIN** |
| **euler_ce (π-phase + e-decay)** | **14.16 ± 0.18** | **−5.17σ WIN** |

**결론**: Euler rotary가 RoPE를 통계적으로 유의하게 이김 (5σ). 이론-정합(euler_ce)이 분산 최저.

### 7.3.2 Recursive ClarusCell (3 seeds × 300 steps, commit f7f17f4)

| 변형 | PPL | RoPE 대비 | k1 대비 | 시간 |
|---|---|---|---|---|
| std_rope | 22.99 ± 0.67 | baseline | — | 1.0x |
| euler_ce_k1 | 21.36 ± 0.54 | −2.2σ | — | 1.15x |
| euler_ce_k2 | 19.06 ± 0.31 | −9.3σ | −6.4σ | 1.83x |
| **euler_ce_k3** | **18.35 ± 0.22** | **−11.5σ** | **−9.0σ** | 2.54x |
| euler_ce_halt ($k{=}6$) | 18.33 ± 0.03 | −12.1σ | −9.8σ | 5.00x |
| euler_ce_k2_fp | 18.35 ± 0.38 | −10.5σ | −7.9σ | 3.19x |

**결론**: $k=3$에서 사실상 수렴. $k=6$ halt는 분산 0.03으로 극도 안정하지만 +2x 비용. **$k=3$ 권장**. FP 정규화는 $k=2$로 $k=3$ 성능 달성 (동일 비용 아님, 3.19x > 2.54x).

### 7.3.3 FFN + Position 조합 (3 seeds × 300 steps, commit 9ede7f3)

모두 k=3 recursive 위:

| 변형 | PPL | euler_ce_std 대비 |
|---|---|---|
| euler_ce_std (baseline) | 18.42 ± 0.47 | — |
| **euler_ce_swiglu** | **17.78 ± 0.78** | **−1.21σ WIN** |
| euler_ce_euler_full | 18.98 ± 0.59 | +1.30σ LOSS |
| riemann_std | 18.72 ± 0.39 | +0.85σ TIE |
| riemann_euler_full | 19.11 ± 0.54 | +1.67σ LOSS |

### 7.3.4 Design 2 ζ activation (LOSS 기록)

| FFN | PPL | time |
|---|---|---|
| swiglu | 17.78 ± 0.78 | 30.9s |
| **zeta** | 19.57 ± 0.63 | **521s (17x)** |

**결론**: |ζ(1/2+ix)|² modulation은 학습에 해로움 + 비현실적 비용.

### 7.3.5 Design 4 Riemann FFN init

| Init | PPL | std |
|---|---|---|
| kaiming | 17.78 | 0.78 |
| **riemann** | 17.51 | **0.24 (3x 감소)** |

평균 TIE (z=−0.57), **분산 현저히 감소**. Seed 안정성 개선.

---

## 7.4 이론적 해석

### 7.4.1 Euler와 zero-frequency variant의 실험상 동률

Design 1과 Euler-CE가 이 소규모 benchmark에서 TIE였다. 이 결과만으로 두
kernel의 수학적 등가, GUE 통계 또는 “최대 질서 + 최대 혼돈”을 추론할 수
없다. 공학적 결론은 이 데이터셋·모델 크기에서 verified-zero frequency의
추가 복잡도가 평균 성능 이득을 보이지 않았다는 것까지다.

### 7.4.2 자기재귀 = Bootstrap 고정점

\(k=3\) 결과는 해당 학습 실험의 관측이다. scalar bootstrap의 내부
고정점 선형률은 약 0.155지만 이는 임의 neural map의 Banach 상수가 아니다.
3-step 충분성은 별도 Jacobian/Lipschitz 측정 없이 이 식에서 따라오지 않는다.

### 7.4.3 Borbély $T_{\text{WAKE}}$ 경험 검증 (이전 commit 7f7fa18)

Dual-graph attention의 gate를 학습 자유도로 풀었을 때, Borbély 2-process에서 유도한 $T_{\text{WAKE}} = 0.315$ 초기값에서 평균 +0.002만 이동. **이론 상수의 경험적 준최적성 확인**.

---

## 7.5 최종 Winning Stack

| 층 | 채택 | 공헌 |
|---|---|---|
| Position encoding | **EulerCEAttention** (π-phase + e-decay) | RoPE 대비 −5.17σ |
| Depth structure | **Recursive k=3** (bootstrap fixed-point) | single-pass 대비 −9.0σ |
| FFN | **SwiGLU + Riemann init** | −1.21σ + 분산 3x 감소 |

**최종 PPL**: 22.99 (RoPE 기준) → 17.51 (= **−23.8% reduction**)

---

## 7.6 버림 / 실패 목록

| 설계 | 결과 | 원인 |
|---|---|---|
| Dual-Laplacian attention (ce_dual) | −3.4σ LOSS | Convex softmax 혼합이 sharpness 희석 |
| Parallel std+dual (ce_par) | +52% params로 TIE | 비효율 |
| Euler-FFN full modulation | +1.30σ LOSS | Pointwise MLP에 π-phase modulation 과함 |
| ζ(1/2+ix) activation | +1.8 PPL, 17x 느림 | 진동 과다 + 비현실적 비용 |
| Berry-Keating $xp$ attention (Design 3) | 미구현 | 연구급 난이도 |

---

## 7.7 실험 재현

```bash
# Position encoding 벤치
python3 examples/ai/bench_euler_rotary.py --steps 500 --seeds 5

# Recursive depth 벤치
python3 examples/ai/bench_recursive_euler.py --steps 300 --seeds 3

# Position × FFN 조합
python3 examples/ai/bench_riemann_combo.py --steps 300 --seeds 3

# Design 4 FFN init 분산
python3 examples/ai/bench_riemann_ffn_init.py --steps 300 --seeds 3
```

---

## 7.8 이론 상수 출처

- Riemann 영점 첫 100개: `reality_stone/python/reality_stone/clarus/ce_riemann_attn.py::RIEMANN_ZEROS_IM` (Titchmarsh 표, Odlyzko 검증)
- Euler 상수 bitfield: `reality_stone/python/reality_stone/clarus/ce_euler.py::EULER_BASIS`
- Borbély $T_{\text{WAKE}} = 1/(3 + \text{AD}(1-\text{AD}))$: `reality_stone/python/reality_stone/clarus/constants.py`
- Euler 계수는 CE 문서에서 가져온 design prior이고, zero frequency는
  수론 table에서 가져온 별도 engineering choice다. 두 출처의 결합은
  연역 정리가 아니라 ablation 대상이다.
