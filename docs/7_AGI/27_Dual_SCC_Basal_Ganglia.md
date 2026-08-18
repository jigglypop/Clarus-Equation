# 이중 recurrent-layer 기저핵 제어 코어

이 문서는 이중 SCC graph와 controller state를 이용한 제어 코어의 형식 객체·수축·진단 범위를 정의한다. 독자는 directed graph·SCC·simplex·validation 기본을 아는 독자를 전제로 하며, 기저핵 비유는 구현 controller의 기능 지도이지 생물학적 회로 기제의 증명은 아니다.

형식 객체와 graph assumptions 뒤에 수축·잔차·Action/HOLD·timebase·무차원 계약·생물학 해석·폐쇄·재현 순으로 읽는다. dataset·seed·baseline·metric·threshold는 등록된 fixture에서만 유효하며, 반례·OOD·ablation 실패는 승격이 아니라 rollback 경계다.


> 날짜: 2026-08-11
>
> 정본 범위: 형식적으로 닫힌 두 블록 수축 코어, 명시적 action/HOLD simplex, 인과적 controller protocol
>
> 행동 승격: **HOLD — DSCC-6/7 미시험**
>
> 생물학적 지위: **basal-ganglia-inspired engineering abstraction**

이 문서는 기저핵에 해부학적으로 정확히 두 개의 SCC가 있다고 주장하지 않는다.
하나의 정적 유향 그래프에서 두 블록 사이에 양방향 경로가 있으면 둘은 하나의 더 큰
maximal SCC에 속한다. 여기서 말하는 “둘”은 서로 다른 색과 시간척도를 가진
layer-restricted recurrent subgraph 두 개다. 색을 지운 결합 그래프는 하나의
macro-SCC다.

## 1. 형식 객체

형식 객체는 두 recurrent layer의 node·edge·state·controller output shape와 정의역을 고정한다. SCC는 graph-theoretic component이며, 연결 구조가 실제 기저핵의 해부학·기능과 동일하다는 주장은 이 정의에 포함되지 않는다.

**[정의 DSCC-1]** 느린 상태와 빠른 상태를

$$
b\in[-1,1]^m,\qquad g\in[-1,1]^n
$$

으로 둔다. 현재 구현은 축소된 `m=2`, `n=3` 코어이며, 빠른 상태의 세 번째 좌표는
별도 HOLD reference channel이다. 고정된 무차원 입력 `u=(u_b,u_g)`에서 동시
Jacobi 갱신은

$$
\begin{aligned}
b^+ &= \tanh\!\left(u_b+A_b b+C_{bg}g\right),\\
g^+ &= \tanh\!\left(u_g+A_g g+C_{gb}b\right)
\end{aligned}
$$

이다. 두 우변은 반드시 같은 이전 반복 상태 `(b,g)`를 읽는다.

**[정의]** 선언된 block norm에서 전역 Lipschitz 상계를

$$
M=
\begin{pmatrix}
\rho_b & \kappa_{bg}\\
\kappa_{gb} & \rho_g
\end{pmatrix}\ge0
$$

으로 기록한다. 평균 Jacobian이나 관측 trajectory의 표본 최대값은 이 전역 상계를
대체하지 못한다.

## 2. 수축과 잔차 인증

수축·잔차 인증은 정의한 norm·step·input boundary에서 controller state가 어떤 충분조건을 만족하는지 검사한다. 인증 pass는 지정 fixture와 수식 가정의 일치이며, 비정상 입력·다른 graph·finite precision 반례는 별도 failure 조건이다.

**[정리 DSCC-2]** `T_u`가 완비된 product cube의 self-map이고
`rho(M)<1`이면, 양의 가중치 `w`에 대한 weighted max norm에서 `T_u`는 수축이다.
따라서 고정 입력마다 유일한 고정점이 존재한다.

2×2 비음수 행렬에서는 다음 조건이 동치다.

$$
\rho_b<1,\quad \rho_g<1,\quad
\kappa_{bg}\kappa_{gb}<(1-\rho_b)(1-\rho_g).
$$

구성적으로

$$
w=(I-M)^{-1}\mathbf 1>0,\qquad
q=\max_i\frac{(Mw)_i}{w_i}<1
$$

을 사용할 수 있다.

**[반례 DSCC-3]** 각 layer의 self-gain이 `0.5`이고 두 cross-gain이 `0.75`인
대칭 `tanh` map은 각 layer만 보면 수축처럼 보이지만 `rho(M)=1.25`다. 이 map은
`0` 외에 `±0.7104117834878703` 고정점을 가지므로 유일성 주장이 완전히 깨진다.
따라서 “각 SCC가 따로 수렴한다”만으로 결합 수렴을 주장하는 부모 명제는 활성
정본에 두지 않는다.

**[정리 DSCC-4]** 근사 고정점의 layer residual을
`e=(e_b,e_g)^T`라 하면 실제 고정점 오차 `d`는 componentwise로

$$
d\le(I-M)^{-1}e
$$

를 만족한다. weighted residual `r_w`에는

$$
\|x-x^*\|_w\le\frac{r_w}{1-q}
$$

가 성립한다. 구현은 유효한 small-gain certificate, 유한 상태, layer topology,
잔차 경계 중 하나라도 실패하거나 반복 예산을 소진하면 결과를 내지 않고 fail closed한다.

이 정리는 고정 입력 Jacobi solve만 다룬다. Gauss–Seidel, 비동기, multi-rate,
online learning 또는 시간에 따라 바뀌는 입력에는 실제 composed map의 새 gain
certificate가 필요하다.

## 3. Action/HOLD simplex

simplex는 controller가 action과 hold probability 또는 weight를 output으로 내는 제약된 codomain이다. 확률 정규화·tie·termination 규칙이 가정이며, policy 우위는 행동 baseline·seed·OOD task 비교가 없으면 주장하지 않는다.

**[정의 DSCC-5]** 빠른 action 좌표에서 조건부 행동 확률을

$$
q_a=\frac{\exp(g_a/\tau_a)}{\sum_j\exp(g_j/\tau_a)}
$$

로 만들고, 별도 reference logit으로 HOLD 확률 `p_H`를 만든다. 최종 질량은

$$
p_a=(1-p_H)q_a,\qquad p_H+\sum_a p_a=1
$$

이다.

**[정리]** HOLD reference에만 공통 항을 더하면 `p_H`는 바뀔 수 있지만

$$
\frac{q_a}{q_j}=\exp\!\left((g_a-g_j)/\tau_a\right)
$$

이므로 조건부 action odds와 순위는 바뀌지 않는다. 이 정리는 “STN 활성이 항상
HOLD를 증가시킨다”는 생물학적 명제가 아니다. 2026년 projection-specific 결과는
STN 자극이 조건에 따라 행동을 가속하고 deferral을 없앨 수도 있음을 보였다
([Zhou et al. 2026](https://doi.org/10.1523/ENEURO.0065-26.2026)).

## 4. 시간축 controller protocol

protocol은 graph update, controller decision, action handoff가 어느 tick 순서로 producer와 consumer를 잇는지 정한다. timebase·latency·serialization이 바뀌면 parity가 깨질 수 있으며, loop execution은 생물학 시간의 재현이 아니다.

**[산출]** `DualSCCController`는 다음 순서를 강제한다.

1. `begin_trial`
2. 정규화 drive를 받는 `observe`
3. core policy만 읽는 `decide`
4. `commit_probe` 또는 `commit_action`
5. due tick 이후 single-use token으로만 `commit_feedback`

완전 수렴한 고정점은 초기조건을 잊으므로 initial state를 “기억”이라고 부르면 안 된다.
그래서 이전 slow/fast anchor는 다음 관측의 **지연된 frozen drive**로 명시적으로
주입된다. 이 설계는 같은 현재 관측에서도 과거 피드백이 다음 상태와 action readout을
바꾸게 한다. 다만 across-trial closed-loop 안정성은 위 고정입력 수축 정리의 자손이
아니다. 별도 ISS/fading-memory 또는 joint spectral certificate가 필요하다.

## 5. 무차원 계약

무차원 계약은 graph state·gain·residual·probability가 어떤 reference scale로 정규화되는지 명시한다. audit 통과는 단위 일관성의 기계 조건이며, 성능·안정성·인과 기제의 충분조건은 아니다.

**[공리]** `tanh`, `exp`, `log`, probability kernel에 들어가는 모든 값은 무차원이다.

| 원시량 | 코어 입력 |
|---|---|
| 시간 `Delta t`, 시간상수 `tau` | `Delta t/tau` 또는 정수 tick |
| reward·utility·probe cost | 고정 기준 `r0` 또는 `u0`로 나눈 비 |
| firing rate | `nu/nu0` |
| 에너지 | `E/E0` |
| 관측 | 좌표별 고정 scale로 정규화한 `o_bar` |
| gain, residual, tolerance, entropy | 처음부터 무차원 |

차원을 가진 원시 시간·reward를 직접 넣는 경로는 dimensionless gate가 거부한다.
이 검사는 수학적 typing만 보장하며 생물학적 대응이나 행동 효능을 보장하지 않는다.

## 6. 생물학에서 허용되는 해석

생물학 해석은 Action/HOLD·recurrent control의 기능 비유에 한정한다. 종·영역·관측 조건·외부 data 없이 controller tensor를 기저핵 회로·도파민·행동 기제로 승격하지 않으며, 비유의 실패는 형식 코어의 반례와 구분한다.

**[경험: 출처가 허용하는 동기]** 다음 정도만 허용된다.

- direct/indirect pathway가 같은 행동 주변에서 함께 관여할 수 있다는 관측은
  proposal과 suppression의 동시 상태를 허용한다
  ([Cui et al. 2013](https://doi.org/10.1038/nature11846),
  [Tecuapetla et al. 2016](https://doi.org/10.1016/j.cell.2016.06.032)).
- STN–GPe reciprocal edge는 빠른 recurrent competition의 동기다
  ([Loucif et al. 2005](https://doi.org/10.1113/jphysiol.2005.093807)).
- arkypallidal 및 thalamostriatal return path는 typed delayed feedback의 동기다
  ([Mallet et al. 2016](https://doi.org/10.1016/j.neuron.2015.12.017),
  [Mandelbaum et al. 2019](https://doi.org/10.1016/j.neuron.2019.02.035)).
- 여러 회로와 시간척도가 공존하므로 two-timescale multiplex를 시험할 수 있다
  ([Foster et al. 2021](https://doi.org/10.1038/s41586-021-03993-3),
  [Mohebi et al. 2024](https://doi.org/10.1038/s41593-023-01566-3)).

**[미완성/금지]** direct/indirect를 slow/fast SCC와 일대일 대응시키거나, 정확히 두
해부학적 SCC가 발견됐다고 쓰거나, reciprocal anatomy가 수축·고정점·AGI를
증명한다고 쓰지 않는다.

## 7. 잠금 진단 결과와 폐쇄 경계

진단 결과는 version·dataset·seed·baseline·metric·threshold가 고정된 좁은 test evidence다. closure는 해당 graph fixture의 상태이며, counterexample·OOD·ablation failure·새 controller feature는 재검증과 rollback을 요구한다.

**[산출: 형식·수치]** 잠긴 구현에서

| 항목 | 값 |
|---|---:|
| gain matrix | `[[0.34, 0.08], [1.00, 0.30]]` |
| spectral radius | `0.6035489375751566` |
| determinant margin | `0.382` |
| weighted contraction `q` | `0.7698795180722892` |
| 최대 residual error bound | `9.999823191719769e-11` |
| 최대 simplex error | `2.220446049250313e-16` |

**[경험: 축소 진단]** 12 paired validation seeds × 240 episodes에서 등록된 legacy
diagnostic gate는 37/38이었다. 유일한 false 항목은 의도적으로 false로 고정한
`causal_integrity_instrumented`다. factorial interaction은 ID
`0.128125 [0.106555, 0.149695]`, OOD
`0.109757 [0.082200, 0.137314]`였고, high-minus-low-conflict HOLD gap은
ID `0.334722`, OOD `0.262500`이었다.

**[미완성 DSCC-6/7]** 위 수치는 behavioral promotion 증거가 아니다. 진단에는
외부 Bayesian context filter가 남아 있고, monolithic 및 일부 null arm은 실제
capacity-matched controller가 아니라 algebraic alias이며, 내부 cross-summary
shuffle/time control과 attempted-future-read 계측이 없다. 따라서 전체 verdict는
사전등록대로 **HOLD**, 지위는 `UNTESTED_INVALID_DESIGN`이다. 이는 DSCC-6/7의
반증도 아니고 확인도 아니다.

다음 승격은 새 preregistration과 닫힌 새 시드에서만 가능하다. 필요한 부모 작업은
`16+16` learned controller, distinct matched monolithic/single-layer arms, 실제
cross-summary lesion/shuffle/sign/time controls, 독립 integrity instrumentation이다.
현재 미개봉 test seed `2026082300..2026082363`은 열지 않는다.

## 8. 구현과 재현 자원

재현 자원은 entry point·artifact·환경·seed를 고정해 형식·진단 결과를 다시 계산하는 계약이다. 코드·test pass는 과학적 참 또는 생물학적 입증이 아니며, serialization·dependency·baseline mismatch는 known gap으로 남는다.

- core: `reality_stone/python/reality_stone/clarus/dual_scc_basal_ganglia.py`
- causal protocol: `reality_stone/python/reality_stone/clarus/dual_scc_controller.py`
- reduced diagnostic: `reality_stone/python/reality_stone/clarus/dual_scc_probe_benchmark.py`
- runner: `examples/agi/dual_scc_probe_bench.py`
- preregistration: `artifacts/agi/dual_scc_basal_ganglia_preregistration_v1.json`
- implementation lock: `artifacts/agi/dual_scc_basal_ganglia_implementation_lock_v1.json`
- validation artifact SHA-256:
  `0F9605D01F35732C9563489D7FDDF3205D1627CC632777359A2CA297B3262A78`

이 모듈은 기존 runtime에 연결되지 않은 opt-in research component다. 따라서 기존
agent/runtime의 default 행동은 변경하지 않았다. 형식 코어는 완결됐지만 runtime
integration, DSCC-6/7 행동 효능, 생물학적 동일성, 의식 및 AGI는 모두 열린 상태다.
