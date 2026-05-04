# Drosophila larva: 원시 신경계 다음 단계

C. elegans의 weighted chemical L1/L2/L3 구조 다음에 무엇이 추가되는지 보기 위해 Drosophila larva connectome을 분석했다.

## 핵심 가설

다음 단계에서 새로 뚜렷해지는 것은 단순한 양 증가가 아니라, mushroom body 기반의 학습/기억/action-selection loop다.

$$
\mathrm{primitive\ control}
\rightarrow
\mathrm{primitive\ control + memory/action\ selection\ loop}
$$

## 데이터

| 항목 | 값 |
|---|---:|
| nodes | 2956 |
| cell types | 18 |
| memory nodes | 382 |
| memory node fraction | 0.129229 |
| used edges | 116922 |
| used synapses | 352611.0 |

## 기능 class

| class | nodes |
|---|---:|
| descending_action | 924 |
| projection_relay | 514 |
| sensory_input | 434 |
| mushroom_body_memory | 382 |
| lateral_integration | 356 |
| unknown | 346 |

## 모델 비교

| model | block/flat | block/random mean | permutation p |
|---|---:|---:|---:|
| primitive 3-class | 0.921003 | 0.965091 | 0.111296 |
| extended 5-class with mushroom body | 0.824371 | 0.980333 | 0.301233 |

extended 5-class는 primitive 3-class보다 손실을 낮추지만, permutation p 기준은 통과하지 못한다. 따라서 이것은 최종 게이트 통과가 아니라 다음 진화 단계 후보의 정량적 신호로 둔다.

## 반례 점검: competing models

| model | labels | params | block/flat | p | BIC-like | saturated |
|---|---:|---:|---:|---:|---:|---|
| all_one | 1 | 1 | 1.000000 | 1.000000 | 5034.912 | False |
| primitive | 3 | 9 | 0.921003 | 0.111296 | 5054.495 | False |
| extended_memory | 6 | 36 | 0.824371 | 0.301233 | 5174.662 | False |
| action_split | 5 | 25 | 0.815008 | 0.135288 | 5107.373 | False |
| sensory_modality | 6 | 36 | 0.824371 | 0.301233 | 5174.662 | False |
| cell_type | 18 | 324 | 0.000000 | 1.000000 | -8952.451 | True |

포화모델을 제외한 BIC-like 최저 모델은 `all_one`이다.
포화모델을 제외한 순수 손실 최저 모델은 `action_split`이다.
이 값들은 생물학적 최종 증명이 아니라, 우리 예상식이 competing explanation보다 얼마나 경제적인지 보는 반례 점검이다.

## 새 loop 지표

| 지표 | 값 |
|---|---:|
| memory internal fraction | 0.184328 |
| projection -> memory fraction | 0.038711 |
| memory -> action fraction | 0.026834 |
| memory -> lateral fraction | 0.027333 |
| lateral -> memory fraction | 0.024287 |
| descending -> memory fraction | 0.023961 |
| total memory-loop touched fraction | 0.325455 |
| sensory -> projection / sensory -> action | 6.404666 |
| memory internal / boundary | 1.016627 |
| matrix memory touched fraction | 0.365641 |

## memory 계열 cell type strength

| cell type | in | out | total | self |
|---|---:|---:|---:|---:|
| KC | 25764.0 | 41411.0 | 67175.0 | 14854.0 |
| MB-FBN | 29492.0 | 22509.0 | 52001.0 | 6518.0 |
| MBON | 25812.0 | 13608.0 | 39420.0 | 1784.0 |
| MBIN | 12206.0 | 7691.0 | 19897.0 | 274.0 |
| MB-FFN | 7343.0 | 8089.0 | 15432.0 | 400.0 |

## 해석

- C. elegans 이후 단계에서 단순 감각-중간-운동 구조 위에 mushroom body 계열 memory/action-selection 회로가 후보로 나타난다.
- 그러나 현재 cell-type block permutation은 통과하지 못했으므로, 이 단계는 '검증 완료'가 아니라 '후보 발견'이다.
- 가장 강한 정량 신호는 mushroom-body 내부 synapse가 전체 synapse의 큰 비중을 차지하고, projection/lateral/action class와 반복적으로 연결된다는 점이다.
- 따라서 지능으로 가는 다음 스텝 후보는 양 증가 자체가 아니라 학습 가능한 내부 상태 loop의 출현이다.
