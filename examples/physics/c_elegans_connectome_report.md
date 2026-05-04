# C. elegans 원시 신경계 그래프 게이트

이 보고서는 인간 뇌 데이터가 아니라 OpenWorm/ConnectomeToolbox의 Witvliet adult C. elegans connectome으로 전역 뇌 방정식의 진화적 최소 문법을 점검한 것이다.

## 검증 질문

$$
\mathcal L_{\mathrm{L1/L2/L3\ block}}
<
\mathcal L_{\mathrm{flat/random\ layer}}
$$

여기서 L1은 감각/input, L2는 중간/relay, L3는 premotor/integrative 층으로 본다.

## 결과

| 항목 | 값 |
|---|---:|
| module count | 12 |
| used edges | 2102 |
| used synapses | 7222.0 |
| flat loss | 1810764.638889 |
| L1/L2/L3 block loss | 1302631.125000 |
| block / flat | 0.719382 |
| block / random mean | 0.805983 |
| permutation p | 0.026795 |
| passed | True |

## 방향성

| 항목 | 값 |
|---|---:|
| forward fraction | 0.370396 |
| backward fraction | 0.096511 |
| lateral fraction | 0.533093 |
| forward/backward | 3.837877 |

C. elegans 회로는 단순 feedforward 사슬이 아니라 recurrent/residual 구조를 강하게 갖는다. 따라서 성공 기준은 forward dominance가 아니라 layer-block 설명력이다.

## Robustness

| matrix | block/flat | block/random mean | permutation p | module-family/flat | pass |
|---|---:|---:|---:|---:|---|
| all_weighted | 0.719382 | 0.805983 | 0.026795 | 0.704610 | True |
| chemical_weighted | 0.717370 | 0.803143 | 0.026595 | 0.710934 | True |
| electrical_weighted | 0.830206 | 0.921385 | 0.089982 | 0.715055 | False |
| all_binary | 0.914439 | 1.004132 | 0.491102 | 0.609626 | False |
| chemical_binary | 0.902778 | 0.989400 | 0.365927 | 0.644033 | False |
| electrical_binary | 0.819574 | 0.911768 | 0.129774 | 0.664681 | False |

weighted all/chemical/binary 조건에서 L1/L2/L3 층화가 유지되는지 본다. electrical-only는 gap junction 성격상 층화보다 lateral coupling이 강할 수 있으므로 별도 해석한다.

## 반례 점검

| model | labels | params | block/flat | p | BIC-like | saturated |
|---|---:|---:|---:|---:|---:|---|
| all_one | 1 | 1 | 1.000000 | 1.000000 | 1364.250 | False |
| layer | 3 | 9 | 0.719382 | 0.026795 | 1356.580 | False |
| module_family | 5 | 25 | 0.704610 | 0.266347 | 1433.110 | False |
| lateral_vs_other | 2 | 4 | 0.960934 | 0.435713 | 1373.421 | False |
| avoidance_vs_other | 2 | 4 | 0.933393 | 0.248550 | 1369.234 | False |
| taxis_vs_other | 2 | 4 | 0.915743 | 0.180764 | 1366.485 | False |
| module | 12 | 144 | 0.000000 | 1.000000 | -3978.867 | True |

포화모델을 제외한 BIC-like 최저 모델은 `layer`이다.
포화모델을 제외한 순수 손실 최저 모델은 `module_family`이다.
따라서 L1/L2/L3 층화가 유일한 설명인지, 아니면 module family가 더 경제적인지 함께 봐야 한다.

## 안정성

| 항목 | 값 |
|---|---:|
| lambda max | 1242.163296 |
| gamma upper bound at rho=0.20 | 0.00096606 |
| modal radius at half bound | 0.400000 |

## 함의

- 인간 이전의 원시 신경계에도 감각-중간-운동 층화 그래프가 통계적으로 보인다.
- 전역 뇌 방정식의 그래프 항 \(\Delta_G\)는 고등피질 전용 항이 아니라 원시 회로에도 적용 가능한 최소 문법일 수 있다.
- 다만 이 결과는 구조 connectome gate이며, 동역학 전이 \(P_n\to P_{n+1}\) 검증은 아니다.
