# 07b. Gate Ablation Protocol

## 0. 목표

07a는 toy runtime gate를 알고리즘으로 썼다. 이 문서는 그 주장을 실험으로 보낼 때의 사전등록 기준을 닫는다.

핵심 구분:

> 후보분포 재가중과 잔류 압축은 유한수학으로 닫힌다.  
> \(\phi\) 재주입이 성능을 높이는지는 정리가 아니라 ablation 문제다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| \(\alpha_\phi=0\) vs \(\alpha_\phi>0\) 비교 설계 | `Tooling` | 통제군/실험군 정의 |
| 후보분포/잔류 지표 계산 | `Exact` | 유한합 |
| 성능 개선 주장 | `Open/Experiment` | 데이터와 통계 검정 필요 |
| hallucination 억제 주장 | `Open/Experiment` | claim-level evidence gate 필요 |

## 1. 고정할 런타임 식

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

### Control A: no residual injection

$$
\alpha_\phi=0.
$$

이 경우 \(\phi_t\)는 기록되더라도 다음 step score에 들어가지 않는다.

### Treatment B: residual injection

$$
\alpha_\phi>0.
$$

단, \(\beta_t,\lambda_\phi,\eta_\phi,P\), 후보 top-k, random seed 정책은 control과 동일하게 둔다.

### Optional Control C: shuffled residual

\(\phi_t\)의 norm은 유지하되 후보와의 대응을 shuffle한다.

목적:

> 개선이 단순한 norm/gain 효과인지, 실제 비선택 후보 구조 때문인지 분리한다.

## 3. 고정해야 할 것

실험 전에 아래 항목을 고정한다.

| 항목 | 고정값 예시 |
|---|---|
| 후보 집합 | top-k token/action 또는 전체 finite set |
| readout | argmax 또는 sampling |
| \(\beta_t\) | 상수 또는 schedule |
| \(\alpha_\phi\) | grid 또는 단일값 |
| \(\lambda_\phi\) | residual decay |
| \(\eta_\phi\) | residual write gain |
| \(P\) | embedding-to-residual projection |
| review threshold | entropy, residual mass, risk 기준 |
| seed 수 | 최소 5개 이상 |
| 평가 데이터 | 실험 전 freeze |

이 표가 비어 있으면 실험은 사후 튜닝으로 내려간다.

## 4. 지표

각 step마다 유한 후보분포에서 다음 지표를 기록한다.

| 지표 | 식 | 판정 |
|---|---|---|
| entropy | \(H_t=-\sum_a\mu_t(a)\log\mu_t(a)\) | `Exact` |
| margin | \(\mu_t(a_1)-\mu_t(a_2)\) | `Exact` |
| residual mass | \(q_t=1-\mu_t(a_t^*)\) | `Exact` |
| selected risk | \(E_{\mathrm{risk}}(a_t^*)\) | `Bridge/Task` |
| contradiction rate | rejected/evidence conflict rate | `Task metric` |
| NLL or loss | task likelihood | `Task metric` |
| latency | runtime cost | `Engineering` |

## 5. 성공 기준

성능 개선 주장은 아래 중 최소 하나를 만족해야 한다.

| 주장 | 성공 조건 |
|---|---|
| prediction 도움 | Treatment가 Control A보다 NLL/accuracy에서 seed 평균 개선, bootstrap CI가 0을 넘지 않음 |
| hallucination 억제 | contradiction/evidence-fail rate가 감소하고 accuracy가 통계적으로 악화되지 않음 |
| review 품질 향상 | high residual mass 구간에서 review precision이 상승 |
| OOD 안정성 | OOD split에서 entropy 폭주 또는 risk 선택률 감소 |

실패 기준:

| 결과 | 판정 |
|---|---|
| 평균 개선 없음 | \(\phi\) 재주입 bridge는 보류 |
| 성능 개선 + hallucination 증가 | 안전 gate 실패 |
| shuffled residual도 동일 개선 | 구조적 residual 효과 아님 |
| latency가 목표 초과 | 공학적 실패 |

## 6. 최소 toy task

### Task 1: delayed disambiguation

초기 step에서 두 후보가 비슷하고, 뒤 step에서 이전 비선택 후보가 정답 단서가 된다.

기대:

> \(\phi\)가 비선택 후보를 보존했다면 Treatment가 더 빨리 복구한다.

### Task 2: contradiction recall

한 후보 claim이 선택되지 않았지만 contradiction evidence로 남아야 한다.

기대:

> Treatment가 다음 claim 생성에서 같은 오류를 덜 반복한다.

### Task 3: noisy top-2 switch

정답 후보가 top-2 사이에서 흔들리는 sequence.

기대:

> Treatment가 margin과 residual mass를 이용해 premature manifest를 줄인다.

## 7. 통계 규칙

최소 규칙:

1. seed별 metric을 먼저 계산한다.
2. seed 평균 차이에 대해 bootstrap confidence interval을 계산한다.
3. primary metric과 safety metric을 분리한다.
4. primary metric만 좋아지고 safety가 나빠지면 성공으로 세지 않는다.

권장:

$$
\Delta M
=
M_{\alpha_\phi>0}-M_{\alpha_\phi=0}
$$

를 모든 seed에서 계산하고, 95% bootstrap CI가 원하는 방향으로 0을 넘는지 본다.

## 8. 이론에 주는 판정

| 실험 결과 | 이론 판정 |
|---|---|
| Treatment 성공, shuffled 실패 | embedding residual bridge 강화 |
| Treatment와 shuffled 둘 다 성공 | residual norm/gain 효과, 커널 재검토 |
| Treatment 실패 | \(\phi\) 재주입은 open 유지 |
| safety 악화 | hallucination gate bridge 하향 |

중요:

> 이 실험은 PreEq 수학 코어를 검증하는 것이 아니다.  
> 검증하는 것은 AGI runtime에서 어떤 \(K_\phi\), 어떤 재주입 규약이 쓸모 있는가다.

## 9. 결론

07b가 닫는 것은 성능 주장이 아니라 판정 절차다.

$$
\boxed{
\alpha_\phi=0
\quad\text{vs}\quad
\alpha_\phi>0
}
$$

를 같은 후보분포, 같은 에너지, 같은 seed 정책 아래 비교해야 한다. 그래야 \(\phi\) 잔류장이 실제 runtime 도구인지, 아니면 문서상 예쁜 해석인지 구분할 수 있다.

최소 synthetic 구현은 `examples/pre_eq/toy_gate_ablation.py`에 있다.

```powershell
python examples\pre_eq\toy_gate_ablation.py
python -m pytest tests\test_pre_eq_toy_gate.py -q
```
