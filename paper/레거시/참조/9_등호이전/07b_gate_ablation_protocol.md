# 07b. Gate Ablation Protocol

이 문서는 residual gate의 효과를 no-injection·injection·shuffle control로 분리하는 toy 실험 protocol이다. 대상은 runtime mechanism의 계산적 효과이며, AGI 능력·의식·생물학적 대응의 증명은 아니다.

독자는 07의 state definition과 07a의 fixture를 먼저 읽는다. 고정식·비교군·seed/split·metric·threshold, task·통계·반증 판정 순서로 읽는다.

## 0. 목표

목표는 residual injection이 단순 추가 파라미터·seed 운보다 독립적인 효과를 갖는지 시험하는 것이다. null 결과와 control 우세는 설계 가설의 반증 또는 미완성 결과다.

07a는 toy runtime gate를 알고리즘으로 썼다. 이 문서는 그 주장을 실험으로 보낼 때의 사전등록 기준을 닫는다.

핵심 구분:

> 후보분포 재가중과 잔류 압축은 유한수학으로 닫힌다.  
> $\phi$ 재주입이 성능을 높이는지는 정리가 아니라 ablation 문제다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| $\alpha_\phi=0$ vs $\alpha_\phi>0$ 비교 설계 | `[정의]` | 통제군/실험군 정의 |
| 후보분포/잔류 지표 계산 | `[산출]` | 유한합 |
| 성능 개선 주장 | `[예측]` | 데이터와 통계 검정 필요 |
| hallucination 억제 주장 | `[예측]` | claim-level evidence gate 필요 |

## 1. 고정할 런타임 식

고정식은 state shape·timebase·optimizer·precision을 treatments 사이에서 동일하게 둔다. 변경된 implementation은 confounder가 되어 비교를 무효화한다.

한 step의 score는

$$
s_t(a)
=
l_t(a)
-\beta_tE^{\mathrm{base}}_t(a)
+\beta_t\alpha_\phi\langle h(a),\phi_t\rangle
$$

이고

$$
\mu_{\beta,t}
=
\operatorname{softmax}(s_t)
$$

이다.

선택 후 raw residual은

$$
\nu_{\mathrm{ns},t}(a)
=
\mathbf 1_{a\ne a_t^*}\mu_{\beta,t}(a)
$$

이고

$$
\phi_{t+1}
=
\lambda_\phi\phi_t
+\eta_\phi
\sum_{a\ne a_t^*}\nu_{\mathrm{ns},t}(a)Ph(a)
$$

로 둔다.

## 2. 비교군

비교군은 residual의 존재와 정보 내용을 분리하도록 설계한다. 모든 군은 동일 dataset split·seed grid·budget을 사용해야 한다.

### Control A: no residual injection

Control A는 residual pathway 자체를 제거한 baseline이다. 성능 차이는 이 기준 대비 분모와 불확실성으로 보고한다.

$$
\alpha_\phi=0.
$$

이 경우 $\phi_t$는 기록되더라도 다음 step score에 들어가지 않는다.

### Treatment B: residual injection

Treatment B는 명시한 residual tensor를 같은 consumer에 주입한다. 효과는 threshold와 confidence interval을 넘지 못하면 확증되지 않는다.

$$
\alpha_\phi>0.
$$

단, $\beta_t,\lambda_\phi,\eta_\phi,P$, 후보 top-k, random seed 정책은 control과 동일하게 둔다.

### Optional Control C: shuffled residual

shuffled control은 residual의 양이나 shape가 아니라 내용 정렬이 효과의 원인인지 검사한다. shuffle procedure와 seed는 preregister해야 한다.

$\phi_t$의 norm은 유지하되 후보와의 대응을 shuffle한다.

목적:

> 개선이 단순한 norm/gain 효과인지, 실제 비선택 후보 구조 때문인지 분리한다.

## 3. 고정해야 할 것

고정 변수에는 data provenance, split, model init, decoding, compute budget, logging이 포함된다. 하나라도 군별로 달라지면 leakage 또는 confounding 가능성을 보고한다.

실험 전에 아래 항목을 고정한다.

| 항목 | 고정값 예시 |
|---|---|
| 후보 집합 | top-k token/action 또는 전체 finite set |
| readout | argmax 또는 sampling |
| $\beta_t$ | 상수 또는 schedule |
| $\alpha_\phi$ | grid 또는 단일값 |
| $\lambda_\phi$ | residual decay |
| $\eta_\phi$ | residual write gain |
| $P$ | embedding-to-residual projection |
| review threshold | entropy, residual mass, risk 기준 |
| seed 수 | 최소 5개 이상 |
| 평가 데이터 | 실험 전 freeze |

이 표가 비어 있으면 실험은 사후 튜닝으로 내려간다.

## 4. 지표

지표는 label·표본단위·분모·metric direction·baseline·OOD set을 명시한다. proxy metric만으로 hallucination 또는 intelligence를 정의하지 않는다.

각 step마다 유한 후보분포에서 다음 지표를 기록한다.

| 지표 | 식 | 판정 |
|---|---|---|
| entropy | $H_t=-\sum_a\mu_t(a)\log\mu_t(a)$ | `[정의]` |
| margin | $\mu_t(a_1)-\mu_t(a_2)$ | `[정의]` |
| residual mass | $q_t=1-\mu_t(a_t^*)$ | `[정의]` |
| selected risk | $E_{\mathrm{risk}}(a_t^*)$ | `[경험식]` |
| contradiction rate | evidence-conflict rate | `[정의]` |
| NLL or loss | task likelihood | `[정의]` |
| latency | runtime cost | `Engineering` |

## 5. 효과 판별 기준

효과 기준은 threshold, uncertainty, multiple seeds, expected failure를 사전에 고정한다. 사후 최적 threshold 선택은 예측이 아니라 탐색 결과다.

성능 개선 주장은 아래 중 최소 하나를 만족해야 한다.

| 주장 | 요구 조건 |
|---|---|
| prediction 도움 | Treatment가 Control A보다 NLL/accuracy에서 seed 평균 개선, bootstrap CI가 0을 넘지 않음 |
| hallucination 억제 | contradiction/evidence-conflict rate가 감소하고 accuracy가 통계적으로 악화되지 않음 |
| review 품질 향상 | high residual mass 구간에서 review precision이 상승 |
| OOD 안정성 | OOD split에서 entropy 폭주 또는 risk 선택률 감소 |

보류 조건:

| 관측 결과 | 해석 |
|---|---|
| 평균 개선 없음 | $\phi$ 재주입 bridge는 보류 |
| 성능 개선 + hallucination 증가 | 안전성 근거가 충족되지 않음 |
| shuffled residual도 동일 개선 | 구조적 residual 효과 아님 |
| latency가 목표 초과 | 공학 비용 기준이 충족되지 않음 |

## 6. 최소 toy task

toy task는 residual이 시간 지연·모순·noise 상황에서 필요한지 드러내는 controlled fixture다. 성공이 일반 언어/세계 모델 OOD 성능을 보장하지 않는다.

### Task 1: delayed disambiguation

이 task는 과거 대안 후보를 보존하는지 검사한다. label leakage가 없는 temporal split이 필요하다.

초기 step에서 두 후보가 비슷하고, 뒤 step에서 이전 비선택 후보가 정답 단서가 된다.

기대:

> $\phi$가 비선택 후보를 보존했다면 Treatment가 더 빨리 복구한다.

### Task 2: contradiction recall

이 task는 상충 정보의 회수와 gate 판정을 분리한다. 정답 label과 contradiction annotation provenance를 기록한다.

한 후보 claim이 선택되지 않았지만 contradiction evidence로 남아야 한다.

기대:

> Treatment가 다음 claim 생성에서 같은 오류를 덜 반복한다.

### Task 3: noisy top-2 switch

이 task는 근접 후보와 noise 아래의 switching 안정성을 검사한다. noise distribution을 바꾸면 metric 해석도 달라진다.

정답 후보가 top-2 사이에서 흔들리는 sequence.

기대:

> Treatment가 margin과 residual mass를 이용해 premature manifest를 줄인다.

## 7. 통계 규칙

통계 규칙은 seed를 독립 반복 단위로, task example을 내부 표본으로 구분한다. 작은 toy fixture에서 유의성만으로 큰 효과를 주장하지 않는다.

최소 규칙:

1. seed별 metric을 먼저 계산한다.
2. seed 평균 차이에 대해 bootstrap confidence interval을 계산한다.
3. primary metric과 safety metric을 분리한다.
4. primary metric만 좋아지고 safety가 나빠지면 효과 확인으로 세지 않는다.

권장:

$$
\Delta M
=
M_{\alpha_\phi>0}-M_{\alpha_\phi=0}
$$

를 모든 seed에서 계산하고, 95% bootstrap CI가 원하는 방향으로 0을 넘는지 본다.

## 8. 이론에 주는 판정

ablation은 residual mechanism의 구현 가설을 지지하거나 기각할 수 있다. 코드를 통과하거나 toy metric이 개선돼도 AGI·의식·CE 물리 해석은 미완성으로 남는다.

| 실험 결과 | 이론 판정 |
|---|---|
| Treatment만 우위 | embedding residual bridge의 경험 근거 강화 |
| Treatment와 shuffled가 모두 우위 | residual norm/gain 효과 가능성, 커널 재검토 |
| Treatment 우위 없음 | $\phi$ 재주입 효과는 `[미완성]` 유지 |
| safety 악화 | hallucination gate bridge 하향 |

중요:

> 이 실험은 PreEq 수학 코어를 검증하는 것이 아니다.  
> 검증하는 것은 AGI runtime에서 어떤 $K_\phi$, 어떤 재주입 규약이 쓸모 있는가다.

## 9. 결론

결론적으로 protocol은 반증 가능한 runtime experiment의 최소 contract다. preregistration·control·OOD·rollback이 없으면 관찰된 차이를 residual의 인과 효과로 읽을 수 없다.

07b가 닫는 것은 성능 주장이 아니라 판정 절차다.

$$
\boxed{
\alpha_\phi=0
\quad\text{vs}\quad
\alpha_\phi>0
}
$$

를 같은 후보분포, 같은 에너지, 같은 seed 정책 아래 비교해야 한다. 그래야 $\phi$ 잔류장이 실제 runtime 도구인지, 아니면 문서상 예쁜 해석인지 구분할 수 있다.

최소 synthetic 구현은 `examples/pre_eq/toy_gate_ablation.py`에 있다.

```powershell
python examples\pre_eq\toy_gate_ablation.py
python -m pytest tests\test_pre_eq_toy_gate.py -q
```
