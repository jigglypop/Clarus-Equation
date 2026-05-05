## 단계 2: Drosophila larva에서 adult FlyWire까지

C. elegans 다음 단계에서는 단순 3층 routing 위에 cell type, action, memory가 같이 분화한다.

초기 후보식:

$$
P_{n+1}
=
\Pi
\left[
\rho P_n
+\gamma\mathcal L(W_{\mathrm{chem}})P_n
+\sum_d a_{d,n}U_d
+D_{\mathrm{celltype}}c_n
+A_{\mathrm{descending}}b_n
+M_{\mathrm{MB}}m_n
+H(q_n-q_*)
\right]
$$

기억 상태:

$$
m_{n+1}
=
\lambda m_n
+\Psi_{\mathrm{MB}}(P_n,b_n,r_n)
$$

### Drosophila larva candidate gate

초기 가설:

$$
\mathrm{primitive\ control}
\rightarrow
\mathrm{primitive\ control + memory/action\ selection\ loop}
$$

데이터:

| 항목 | 값 |
|---|---:|
| nodes | 2956 |
| cell types | 18 |
| memory nodes | 382 |
| memory node fraction | 0.129229 |
| used edges | 116922 |
| used synapses | 352611.0 |

기능 class:

| class | nodes |
|---|---:|
| descending_action | 924 |
| projection_relay | 514 |
| sensory_input | 434 |
| mushroom_body_memory | 382 |
| lateral_integration | 356 |
| unknown | 346 |

모델 비교:

| model | block/flat | block/random mean | permutation p |
|---|---:|---:|---:|
| primitive 3-class | 0.921003 | 0.964755 | 0.106579 |
| extended 5-class with mushroom body | 0.824371 | 0.980404 | 0.299340 |

반례 점검:

| model | labels | params | block/flat | p | BIC-like | saturated |
|---|---:|---:|---:|---:|---:|---|
| all_one | 1 | 1 | 1.000000 | 1.000000 | 5034.912 | False |
| primitive | 3 | 9 | 0.921003 | 0.106579 | 5054.495 | False |
| extended_memory | 6 | 36 | 0.824371 | 0.299340 | 5174.662 | False |
| action_split | 5 | 25 | 0.815008 | 0.145571 | 5107.373 | False |
| sensory_modality | 6 | 36 | 0.824371 | 0.299340 | 5174.662 | False |
| cell_type | 18 | 324 | 0.000000 | 1.000000 | -8952.451 | True |

새 loop 지표:

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

memory 계열 cell type strength:

| cell type | in | out | total | self |
|---|---:|---:|---:|---:|
| KC | 25764.0 | 41411.0 | 67175.0 | 14854.0 |
| MB-FBN | 29492.0 | 22509.0 | 52001.0 | 6518.0 |
| MBON | 25812.0 | 13608.0 | 39420.0 | 1784.0 |
| MBIN | 12206.0 | 7691.0 | 19897.0 | 274.0 |
| MB-FFN | 7343.0 | 8089.0 | 15432.0 | 400.0 |

larva 판정:

$$
\boxed{
\text{Drosophila larva에서는 memory/action/celltype 항이 후보로 나타나지만 strict gate는 통과하지 못한다.}
}
$$

의의:

1. C. elegans의 weighted routing만으로는 larva fly brain의 구조를 충분히 설명하지 못한다.
2. mushroom body 내부 및 projection/action/lateral과의 loop가 강한 후보로 보인다.
3. 그러나 action split도 순수 손실 기준으로 강하므로 "memory 단독 추가"라고 말하면 과하다.
4. 따라서 다음 항은 \(M_{\mathrm{memory}}\) 하나가 아니라 \(D_{\mathrm{celltype}}+A_{\mathrm{action}}+M_{\mathrm{memory}}\)의 공동 분화로 수정해야 한다.

### Drosophila adult FlyWire closure

larva 후보를 adult FlyWire/Codex 파생 connectome으로 다시 열었다. 사용 자료는 `connectome.mat`의 `W.TOT`와 같은 record의 `annotations.mat`다.

데이터:

| 항목 | 값 |
|---|---:|
| source neurons | 138639 |
| hemibrain-typed units | 4929 |
| W.TOT nonzero edges | 15091983 |
| W.TOT synapses | 54492922.0 |
| kept unit-level synapses | 19500884.0 |
| kept fraction | 0.357861 |

refined functional class:

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

모델 비교:

| model | labels | block/flat | block/random mean | p | BIC-like |
|---|---:|---:|---:|---:|---:|
| all_one | 1 | 1.000000 | 1.000000 | 1.000000 | -56775104.182 |
| primitive | 3 | 0.981755 | 0.981764 | 0.000999 | -57222313.771 |
| adult_refined_celltype_action_memory | 9 | 0.955373 | 0.955410 | 0.000999 | -57882900.453 |
| super_class | 9 | 0.932781 | 0.932817 | 0.000999 | -58464318.351 |
| class_or_refined | 19 | 0.953105 | 0.953201 | 0.000999 | -57935876.589 |

memory/action loop:

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

top mushroom body memory units:

| unit | in | out | total | self |
|---|---:|---:|---:|---:|
| KCab-m | 148719.0 | 86521.0 | 235240.0 | 65680.0 |
| KCab-s | 101138.0 | 63200.0 | 164338.0 | 51413.0 |
| KCab-c | 61207.0 | 32102.0 | 93309.0 | 23859.0 |
| KCab-ap2 | 27343.0 | 20669.0 | 48012.0 | 14908.0 |
| KCab-ap1 | 23499.0 | 15469.0 | 38968.0 | 13228.0 |

top central complex/action units:

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

판정:

| adult gate | 결과 |
|---|---|
| adult refined model gate | True |
| memory/action loop gate | True |
| closed | True |

따라서 Drosophila 단계는 다음 항으로 닫는다.

$$
\boxed{
\mathrm{weighted\ chemical\ routing}
\rightarrow
\mathrm{celltype/action/memory\ co\ differentiation}
}
$$

최종 Drosophila 축약식:

$$
P_{n+1}
=
\Pi
\left[
\rho P_n
+\gamma\mathcal L(W_{\mathrm{chem}})P_n
+\sum_d a_{d,n}U_d
+D_{\mathrm{celltype}}c_n
+A_{\mathrm{action}}b_n
+M_{\mathrm{MB/CX}}m_n
+H(q_n-q_*)
\right]
$$

의의:

1. 두 번째 큰 도약은 memory 단독이 아니다.
2. 감각, relay, descending/action, mushroom body, central complex, lateral integration 역할이 함께 분화한다.
3. 그 분화된 역할들이 닫힌 loop를 만들 때 C. elegans의 primitive routing을 넘어선다.
4. 따라서 "신경계 양이 늘면 지능이 생긴다"보다 "weighted routing 위에 역할 분화와 내부 loop가 붙는다"가 더 정확하다.

