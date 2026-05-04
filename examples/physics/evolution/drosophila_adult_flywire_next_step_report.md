# Drosophila adult FlyWire: celltype/action/memory gate

C. elegans의 weighted chemical routing 다음 단계가 무엇인지 adult FlyWire connectome으로 다시 점검했다.

## 데이터

| 항목 | 값 |
|---|---:|
| source neurons | 138639 |
| hemibrain-typed units | 4929 |
| W.TOT nonzero edges | 15091983 |
| W.TOT synapses | 54492922.0 |
| kept unit-level synapses | 19500884.0 |
| kept fraction | 0.357861 |

자료는 FlyWire/Codex 파생 `connectome.mat`의 `W.TOT`와 같은 record의 `annotations.mat`를 사용했다.

## refined functional class

| class | units |
|---|---:|
| central_integration | 3714 |
| descending_motor | 480 |
| visual_system | 416 |
| projection_relay | 170 |
| central_complex_action | 125 |
| sensory_input | 12 |
| mushroom_body_memory | 5 |
| ascending_body_input | 4 |
| homeostatic_modulatory | 3 |

## 모델 비교

| model | labels | block/flat | block/random mean | p | BIC-like |
|---|---:|---:|---:|---:|---:|
| all_one | 1 | 1.000000 | 1.000000 | 1.000000 | -56775104.182 |
| primitive | 3 | 0.981755 | 0.981764 | 0.000999 | -57222313.771 |
| adult_refined_celltype_action_memory | 9 | 0.955373 | 0.955410 | 0.000999 | -57882900.453 |
| super_class | 9 | 0.932781 | 0.932817 | 0.000999 | -58464318.351 |
| class_or_refined | 19 | 0.953105 | 0.953201 | 0.000999 | -57935876.589 |

포화모델 제외 순수 손실 최저는 `super_class`이다.
포화모델 제외 BIC-like 최저는 `super_class`이다.

## memory/action loop

| 지표 | 값 |
|---|---:|
| memory_internal_fraction | 0.011084 |
| projection_to_memory_fraction | 0.007466 |
| memory_to_action_fraction | 0.000010 |
| central_complex_to_descending_fraction | 0.000115 |
| memory_action_loop_fraction | 0.018769 |
| random mean memory_action_loop_fraction | 0.005020 |
| observed/random mean | 3.738545 |
| random >= observed p | 0.012987 |

## top memory/action units

### mushroom body memory

| unit | in | out | total | self |
|---|---:|---:|---:|---:|
| KCab-m | 148719.0 | 86521.0 | 235240.0 | 65680.0 |
| KCab-s | 101138.0 | 63200.0 | 164338.0 | 51413.0 |
| KCab-c | 61207.0 | 32102.0 | 93309.0 | 23859.0 |
| KCab-ap2 | 27343.0 | 20669.0 | 48012.0 | 14908.0 |
| KCab-ap1 | 23499.0 | 15469.0 | 38968.0 | 13228.0 |

### central complex/action

| unit | in | out | total | self |
|---|---:|---:|---:|---:|
| vDeltaA | 38617.0 | 45276.0 | 83893.0 | 5437.0 |
| ER2 | 23968.0 | 26110.0 | 50078.0 | 17695.0 |
| hDeltaB | 9133.0 | 35411.0 | 44544.0 | 944.0 |
| ER3d | 21901.0 | 21602.0 | 43503.0 | 19113.0 |
| ER4d | 18100.0 | 18418.0 | 36518.0 | 15217.0 |
| vDeltaF | 15824.0 | 19473.0 | 35297.0 | 2356.0 |
| FC2B | 26392.0 | 6809.0 | 33201.0 | 142.0 |
| hDeltaI | 21933.0 | 10579.0 | 32512.0 | 525.0 |
| hDeltaA | 20005.0 | 10012.0 | 30017.0 | 167.0 |
| EL | 21955.0 | 5883.0 | 27838.0 | 308.0 |
| FB5A | 8841.0 | 18808.0 | 27649.0 | 119.0 |
| hDeltaH | 15915.0 | 11592.0 | 27507.0 | 176.0 |

## 판정

- adult refined model gate: `True`
- memory/action loop gate: `True`
- closed: `True`

Drosophila larva에서는 memory/action/celltype 항이 후보였지만 strict block gate가 약했다.
adult FlyWire에서는 refined functional block이 random label보다 유의하게 좋고, mushroom body와 central-complex/descending action loop가 random보다 강하다.
따라서 이 단계는 memory 단독 추가가 아니라 celltype/action/memory 공동 분화 항으로 닫는다.

$$
P_{n+1}
=
\Pi[
\rho P_n
+\gamma\mathcal L(W_{\mathrm{chem}})P_n
+\sum_d a_{d,n}U_d
+D_{\mathrm{celltype}}c_n
+A_{\mathrm{action}}b_n
+M_{\mathrm{MB/CX}}m_n
+H(q_n-q_*)
]
$$
