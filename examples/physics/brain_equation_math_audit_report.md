# 뇌 방정식 부분 게이트 수학 점검

이 문서는 event-level readiness 결과를 수학 부등식 관점에서 다시 정리한 것이다.
여기서 통과는 신경활성 검증이 아니라, 특수 연산자 분해가 데이터 구조상 붕괴하지 않는다는 뜻이다.

## 핵심 부등식

$$
\bar W_d
=
\frac{1}{|C_d|}\sum_{i\in C_d}\|s_i-\mu_d\|^2
$$

$$
B_{d,d'}=\|\mu_d-\mu_{d'}\|^2
$$

필요한 최소 조건은 같은 domain 내부 분산이 가장 가까운 다른 domain prototype 거리보다 작고, holdout에서 matched 손실이 wrong/generic보다 작아야 한다.

$$
\bar W < \min_{d\ne d'} B_{d,d'},\qquad
\mathcal L_{\mathrm{matched}}
<
\min(
\mathcal L_{\mathrm{wrong}},
\mathcal L_{\mathrm{generic}}
)
$$

## 결과

| gate | cases | domains | mean within | nearest between | within/between | matched/wrong | matched/generic | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ds000116_modality | 12 | 2 | 0.00011709 | 0.07416692 | 0.001579 | 0.001890 | 0.007522 | pass |
| ds000201_task_domain | 12 | 3 | 0.00053635 | 0.04336198 | 0.012369 | 0.022158 | 0.055802 | pass |
| ds000201_cognitive_arousal | 22 | 3 | 0.00031234 | 0.00542348 | 0.057591 | 0.031815 | 0.054405 | pass |

## 추가 식별성 점검

| gate | domain rank | domain cond | domain R2 | domain+subject rank | domain+subject cond | permutation p |
|---|---:|---:|---:|---:|---:|---:|
| ds000116_modality | 2/2 | 2.618 | 0.993725 | 3/3 | 3.186 | 0.001499 |
| ds000201_task_domain | 3/3 | 3.732 | 0.971109 | 4/4 | 4.217 | 0.000500 |
| ds000201_cognitive_arousal | 3/3 | 3.620 | 0.963233 | 6/6 | 6.264 | 0.000500 |

## 해석

- 세 게이트 모두 prototype 사이 거리가 domain 내부 분산보다 크다.
- 세 게이트 모두 holdout에서 matched operator가 wrong/generic보다 낮은 손실을 낸다.
- domain 설계행렬이 full rank이면 event-level 특수 연산자 계수는 적어도 이 표본 안에서 선형적으로 구분된다.
- domain+subject 설계행렬이 full rank이면 subject offset을 넣어도 domain 항이 완전히 붕괴하지 않는다.
- permutation p 값은 domain label을 섞었을 때 현재보다 강한 분리가 얼마나 자주 나오는지의 경험적 점검이다.
- 따라서 현재 수학적으로 닫힌 것은 특수 입력항 분해의 event-level 필요조건이다.
- 아직 닫히지 않은 것은 BOLD/EEG에서 같은 부등식이 region-resolved 상태 \(p_r\) 위에서도 유지되는지다.
