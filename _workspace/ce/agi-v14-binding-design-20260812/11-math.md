# 11-math — agi-v14-binding-design-20260812

Status: COMPLETE

## 대상·정의역·전제

- 계약: `00-contract.md` (Status: COMPLETE 확인). PREDECESSOR: `_workspace/ce/agi-v13b-convex-spectral-20260812` — balanced split의 이상 선형 학습기 식별 가능성(heldout 1.0)은 선행 run에서 확정된 결론을 인용하고 재유도하지 않는다 (정본: `reality_stone/python/reality_stone/clarus/local_cloud_v13_benchmark.py` 모듈 docstring 19–28행).
- 과제(동결 생성기 기준, 독립 경로로 재확인): tick 0에 local 4×4 관측의 행 $i$ 전체에 $b_i$ 주입(비트당 4회 반복), tick 1에 shared 4채널에 one-hot $c$ 주입, 그 외 틱은 순수 노이즈 $\mathcal N(0,\sigma^2)$. 라벨 $y=\operatorname{sign}(c^\top W b)$,
$$W=\begin{pmatrix}1&1&1\ 1&-1&-1\ -1&1&-1\ -1&-1&1\end{pmatrix}.$$
- 정확 검산(상대오차 ≤ 1e−9, artifacts 스크립트): $\sum_k w_k=0$; $\|Wb\|^2=12$ (8개 $b$ 전부, $W^\top W=4I_3$의 귀결); 마진 $|w_c^\top b|\in\{1,3\}$ (홀수 내적, $b=\pm w_c$일 때만 3, 즉 셀의 3/4이 마진 1); $\sum_{b\in\{\pm1\}^3}\operatorname{sign}(w_k^\top b)\,b=4w_k$ (3변수 다수결 항등식의 귀결).

## C1 — 무손실 슬롯 + bilinear readout의 충분성 (정리)

**구조.** 상태 $h=(s_b,s_c)\in\mathbb R^4\times\mathbb R^4$, 갱신
$$s_x(t{+}1)=(1-g_x(t))\,s_x(t)+g_x(t)\,\varphi_x(o_t),\qquad x\in\{b,c\}.$$
닫힌 게이트($g=0$)의 Jacobian은 항등행렬 — 고유값 정확히 1 (marginal stability), 구현상 무연산이므로 유지 오차 0. 쓰기 함수는 $\varphi_b(o)=\gamma\cdot\mathrm{rowmean}(o^{loc})$ (4반복 평균, 노이즈 표준편차 $\sigma/2$로 축소), $\varphi_c(o)=\gamma\cdot o^{sh}$, input gain $\gamma>0$ (동결값 0.5). 판독은 $\hat y=\operatorname{sign}\langle \hat W,\ s_c\otimes s_b\rangle$, $\hat W=[W\ |\ 0]\in\mathbb R^{4\times4}$ (distractor 열 0).

**정리 (C1).** 게이트가 오라클($g_b=1\Leftrightarrow t{=}0$, $g_c=1\Leftrightarrow t{=}1$)이면 임의의 $T\ge2$에서
$$s_b(T)=\gamma(b^{(4)}+\eta_b),\quad s_c(T)=\gamma(c+\eta_c),\qquad \eta_b\sim\mathcal N(0,\tfrac{\sigma^2}{4}I_4),\ \eta_c\sim\mathcal N(0,\sigma^2 I_4)$$
($b^{(4)}$는 distractor 포함 4비트, $\eta$는 해당 신호 틱의 노이즈만). 판정 통계는
$$\frac{S}{\gamma^2}=c^\top W b+\underbrace{w_c^\top\eta_{b,1:3}+\eta_c^\top Wb}_{G\,\sim\,\mathcal N(0,\sigma_N^2)}+\underbrace{\eta_c^\top W\eta_{b,1:3}}_{Q},\qquad \sigma_N^2=\tfrac{3}{4}\sigma^2+12\sigma^2=\tfrac{51}{4}\sigma^2,$$
$G$는 $\eta_b\perp\eta_c$로 정확히 가우시안이고 $\mathbb E[Q^2]=\tfrac{\sigma^4}{4}\operatorname{tr}(WW^\top)=3\sigma^4$. 마진 $|c^\top Wb|\ge1$이므로 임의의 $t\in(0,1)$에 대해
$$\Pr(\hat y\neq y)\ \le\ \Phi\Big({-}\frac{1-t}{\sigma_N}\Big)+\frac{3\sigma^4}{t^2},\qquad \sigma_N=\frac{\sqrt{51}}{2}\sigma\approx 3.571\,\sigma,$$
그리고 마진 분포(1이 3/4, 3이 1/4)를 반영한 1차 정밀 근사로
$$\Pr(\hat y\neq y)=\tfrac34\,\Phi(-1/\sigma_N)+\tfrac14\,\Phi(-3/\sigma_N)+O(\sigma^2).$$
$\gamma$는 $\gamma^2>0$으로 부호에서 소거된다 (게인 축이 원인이 아니라는 v13 실험 배제와 정합).

**T-불변성.** 닫힌 구간이 문자 그대로 항등사상이므로 $s(T)$는 $T\ge2$에 의존하지 않는다 — horizon/combined 패널의 저하 원인(누설 유지 $r<1$ 또는 학습된 게이트의 재쓰기)이 구조적으로 제거된다. 위 오차 상계는 $T$에 무관.

**Heldout-불변성.** 판정 규칙은 셀 정체성을 쓰지 않으므로 상계가 train/heldout 구분 없이 성립. 식별 가능성: 판독은 특징 $z=s_c\otimes s_b\in\mathbb R^{16}$ 위 $\hat W$의 **선형** 파라미터화이고 참 라벨이 $\hat W^*=[W|0]$로 마진 $\gamma^2(1-O(\sigma))$로 실현되므로, balanced split의 24 train 셀에서 마진 일관 선형 학습기가 8 heldout 셀을 정확히 라벨링(선행 run 확정 결론 인용) — 수치로도 독립 확인(아래 표, fitted $\hat W$와 참 $W$의 코사인 0.998).

**불완전 게이트(1차 전파).** salience 게이트 $g=\mathbf 1[\|x\|_\infty>\theta]$, 틱당 오탐률 $\delta$, 신호 미탐률 $p_{miss}$. last-write-wins에서 슬롯 오염은 신호 틱 이후 노이즈 틱(슬롯당 $T-1$개)에서의 오탐 발생과 동치이고, 오염된 슬롯의 조건부 오류는 노이즈 전용 쌍선형 형식의 부호 대칭에 의해 정확히 $\tfrac12$:
$$\Pr(\hat y\neq y)\ \le\ P_{clean}+(T-1)\,\tfrac{\delta_b+\delta_c}{2}\cdot 2+p_{miss}+O(\delta^2)\ =\ P_{clean}+(T-1)(\delta_b+\delta_c)+p_{miss}+O(\delta^2).$$
$\theta=\tfrac12$일 때 $\delta_b\le 32\,\Phi(-\theta/\sigma)$, $\delta_c\le 8\,\Phi(-\theta/\sigma)$, $p_{miss}\le\Phi(-(1-\theta)/\sigma)^4$: $\sigma=0.08$에서 $\Phi(-6.25)\approx 2\times 10^{-10}$이므로 $T=16$에서도 총 기여 $\lesssim 10^{-8}$ — 오라클 게이트 가정은 salience 구현으로 사실상 대체 가능(신호 진폭 1 대 $\sigma\le 0.08$의 분리).

### C1 수치 검산 (독립 경로: 동결 생성기 + 오라클 슬롯 직접 구현)

경계 검증 — $\sigma=0.08$: MC 오류율(2×10⁶ 표본) $1.815\times 10^{-4}$ vs 상계 $\Phi(-1/\sigma_N)=2.320\times 10^{-4}$, 정밀 근사 $\tfrac34\Phi(-3.5)=1.75\times 10^{-4}$ (MC 표준오차 $9.5\times 10^{-6}$ 이내 일치). $\sigma=0.04$: MC 0 vs $1.27\times 10^{-12}$.

| 패널 (seeds 9000–9007, eval 256/seed) | acc(참 $W$) | acc(fitted $\hat W$, logistic) |
|---|---|---|
| T=4, σ=0.04, iid / balanced-heldout | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| T=4, σ=0.08, iid / balanced-heldout | 1.0000 / 1.0000 | 0.9961 / 0.9912 |
| T=8, σ=0.04, iid / balanced-heldout | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| T=8, σ=0.08, iid / balanced-heldout | 1.0000 / 0.9995 | 0.9995 / 0.9961 |
| T=16, σ=0.04, iid / balanced-heldout | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| T=16, σ=0.08, iid / balanced-heldout | 0.9995 / 0.9995 | 1.0000 / 0.9951 |

- 참-$W$ 판독의 잔차 오류(≤ 5×10⁻⁴)는 위 오차 상계와 일치하는 노이즈 유한 오차. $T$ 의존성 없음(T-불변성 확인), heldout=iid 수준(heldout-불변성 확인).
- fitted $\hat W$의 잔차는 유한 표본 추정 오차: train 96→480→1920 에피소드에서 heldout 평균 0.9961→0.9990→0.9995 (T=8, σ=0.08 최악점), 수렴 최적화 시 개별 seed에서 heldout 1.0 도달, 참 $W$와의 코사인 0.998. 구조적 천장 아님.

**판정: C1 CONFIRMED (P0/P1 없음).** 계약 목표(G1 ≥ 0.95×gru20=0.845, G3 heldout ≥ 0.90)를 오라클 구조가 전 패널에서 초과 달성.

## C2 — 필요성 방향: 연결 저장 + 선형 판독의 표현 한계 (정리)

**정리 (C2-선형).** 상태가 연결 $[\hat b;\hat c;\hat d]$이고 판독이 선형 $\operatorname{sign}(v^\top\hat b+u^\top\hat c+a\hat d+\beta)$이면, 32셀 전부에서 $y=\operatorname{sign}(c^\top Wb)$와 일치할 수 없다.

*증명.* $c$가 one-hot이므로 $u^\top c+\beta=\beta_k$ (컨텍스트별 편향). distractor는 $d=\pm 1$ 양쪽에서 성립해야 하므로 두 부등식을 더하면 $a$ 항이 소거된다. 정답 가정 $y_k(b)(v^\top b+\beta_k)>0$을 컨텍스트 $k$의 8개 $b$에 대해 합하면, $\sum_b y_k(b)=0$ (라벨 4/4 균형)과 항등식 $\sum_b y_k(b)\,b=4w_k$에 의해 $v^\top w_k>0$ ($k=1,\dots,4$). 그런데 $\sum_k w_k=0$이므로 $\sum_k v^\top w_k=0$ — 네 양수의 합이 0이 되어 모순. (증명 끝)

**정리 (C2-가법, 일반화).** 곱 항 없는 임의의 가법 판독 $\operatorname{sign}(\phi(\hat b)+\psi(\hat c))$ ($\phi,\psi$ 임의 함수)도 32셀 전부를 실현할 수 없다.

*증명.* 컨텍스트별 양성 집합은 동일 함수 $\phi$의 초과수준집합 $P_k=\{b:\phi(b)>\theta_k\}$이므로 전순서로 중첩된다. 각 $|P_k|=4$ (라벨 균형)이므로 중첩+등기수 ⇒ 네 집합이 전부 동일. 그러나 참 양성 집합은 컨텍스트마다 다르다(임의의 $j\neq k$에서 $|P_j\cap P_k|=2$) — 모순. (증명 끝)

**정량 천장 (수치, artifacts).** 노이즈 없는 32셀에서:
- 가법 판독 최대 정확도 = **26/32 = 0.8125** (8! 순서 전수 열거, 정확).
- 선형 판독($\phi=v^\top b$) 최대 = **21/32 = 0.6563** (20만+정수격자 방향의 조밀 표집, 컨텍스트별 최적 임계 정확 계산).
- 노이즈가 있어도 상계 그대로 유효: 각 노이즈 실현마다 유도되는 분류기는 여전히 셀 위의 가법 분류기이므로 기대 정확도 ≤ 26/32 (실현별 상계의 기대값).

**balanced heldout 귀결 (seeds 9000–9015 전수).** train 24셀 최대 선형 정확도 17–19/24 (train조차 완전 적합 불가), train-최적 방향들의 heldout 정확도는 **모든 seed에서 ≤ 0.50** (동률 중 최선 기준, 대부분 0.25–0.375, 최소 0.125). 즉 선형-연결 족의 heldout 천장은 원리적으로 0.5 이하.

**GRU 0.55와의 정합 (관측 수준, 증명 아님).** gru20의 heldout 0.55는 선형-연결 천장(≤0.50) 바로 위이고 가법 천장(0.8125)과 완전 해(1.0)에는 크게 미달. GRU의 비선형 재귀는 원리상 곱 항을 근사할 수 있으므로 C2는 GRU에 대한 표현 하한이 아니다 — 0.55는 학습 편향이 조합(곱) 해를 찾지 못한 최적화 관측이며, C2가 증명하는 것은 "곱 항 없는 저장·판독으로는 어떤 학습기도 원리적으로 불가"라는 표현 필요성이다.

**판정: C2 CONFIRMED — 쌍선형(곱) 항은 표현상 필요 (반례 없음).**

## 숨은 공리·자유도

1. **게이트의 신호 틱 정렬** — C1의 핵심 공리. 오라클 대신 salience 임계로 대체 가능함을 1차 전파로 정량화했으나(기여 ~10⁻⁸), 이 게이트를 **학습으로** 획득하는 문제는 C1의 범위 밖(경로 레인 C3 소관).
2. **last-write-wins 덮어쓰기** — 오염 분석이 이 규약에 의존. 볼록 결합 갱신이면 오염이 부분적이라 상계는 더 좋아진다(보수적).
3. **쓰기 함수의 채널 구조 지식** — $\varphi_b$의 4반복 평균은 분산 최적 선형 사상일 뿐이며, 임의의 full-rank 선형 읽기도 $\sigma_N$ 상수만 바꾸고 결론 불변.
4. **C2의 상태 가정** — "연결만 저장"은 $\hat b,\hat c$의 임의 노이즈 추정을 허용하지만, 상태가 $b,c$의 비선형 결합(예: 곱셈적 binding)을 저장하는 경우는 C2 범위 밖 — 그것이 바로 C1이 요구하는 구조.
5. 오차 상계의 $t$는 자유 모수(임의 $t\in(0,1)$에서 성립); 수치 비교는 마진 분포 반영 정밀 근사식으로 수행.

## 경계·반례·교차 예측

- 반례 탐색: C2에 대해 32셀 전수 + 방향 조밀 표집으로 반례 부재 확인(21/32가 실현 상한). C1에 대해 $T=16$, $\sigma=0.08$ 극단에서도 상계 내.
- 교차 예측 1: C1 구조 구현체는 horizon 패널에서 id 패널과 통계적으로 동일해야 한다(누설 없음) — 구현 단계(30/31)의 검증 항목.
- 교차 예측 2: 곱 항이 있어도 유지가 누설($r<1$)이면 마진이 $r^T$ 비율로 붕괴 — v13의 T=8 붕괴 패턴과 정합.
- 교차 예측 3: 선형-연결 판독을 강제한 어떤 모델도 balanced heldout ≤ 0.5 (seeds 9000–9015) — 음성 대조로 사용 가능.

## P0 / P1 / P2

- P0: 없음.
- P1: 없음 (C1·C2 모두 증명 + 독립 수치로 닫힘). 단 "학습된 게이트가 오라클 게이트로 수렴하는가"는 본 계약의 주장이 아니며 C3/구현 단계의 열린 문제로 이관.
- P2: (i) 계약의 $s_b\in\mathbb R^4$는 distractor 포함 4차원(참 비트는 3) — $\hat W$의 4번째 열을 0으로 두는 표기를 명기할 것. (ii) 선형 최대 21/32는 방향 표집 기반 수치 인증(가법 상계 26/32는 전수 열거로 정확) — 보고 시 구분 유지.

## 재현

```
cd C:/Users/dongh/OneDrive/Desktop/Clarus-Equation
./.venv/Scripts/python.exe _workspace/ce/agi-v14-binding-design-20260812/artifacts/verify_c1_oracle.py
./.venv/Scripts/python.exe _workspace/ce/agi-v14-binding-design-20260812/artifacts/verify_c1_samplesize.py
./.venv/Scripts/python.exe _workspace/ce/agi-v14-binding-design-20260812/artifacts/verify_c2_linear.py
./.venv/Scripts/python.exe _workspace/ce/agi-v14-binding-design-20260812/artifacts/verify_c2_additive.py
```

로그: `artifacts/verify_c1_oracle.log`, `artifacts/verify_c1_samplesize.log`, `artifacts/verify_c2_linear.log`, `artifacts/verify_c2_additive.log`.

Status: COMPLETE
