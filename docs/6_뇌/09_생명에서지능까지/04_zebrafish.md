## 단계 3: Zebrafish larva activity

Zebrafish부터는 connectome proxy를 넘어 실제 calcium activity state를 본다. 목표는 두 가지다.

1. 실제 activity state가 다음 activity state를 예측하는가.
2. activity 또는 perturbation이 행동과 연결되는가.

축약식:

$$
P_{t+1}
=
\Pi
\left[
\rho P_t
+L_{\mathrm{lowrank}}(P_t)
+C_{\mathrm{assembly}}(P_t)
+U_{\mathrm{laser},t}
+H(q_t-q_*)
\right]
$$

행동 관측식:

$$
y_t
=
\Omega(P_{t-\ell},q_t)
+\xi_t,
\qquad
y_t\in
\{
\mathrm{bout},
\mathrm{left/right},
\mathrm{speed},
\mathrm{turn},
\mathrm{heading}
\}
$$

\(C_{\mathrm{assembly}}\)는 두 뉴런이 같은 assembly를 공유하는지에 따른 correlation block 항이고, \(L_{\mathrm{lowrank}}\)는 저차원 recurrent state 예측 항이다.

### Zebrafish optic tectum spontaneous activity

이 자료에는 행동이 없으므로 whole-brain behavior gate가 아니라 activity-only pilot이다.

| fish | cells | time | assemblies | covered | coassembly/flat | coassembly p | recurrent/baseline | R2 | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| zf_20151104-f1 | 114 | 4245 | 5 | 77 | 0.681772 | 0.000500 | 0.447369 | 0.552631 | True |
| zf_20170215-f3 | 75 | 5660 | 4 | 64 | 0.572443 | 0.000500 | 0.330413 | 0.669587 | True |

의의:

1. connectome-only 단계를 넘어 실제 calcium activity에서 assembly/recurrent state 구조가 보인다.
2. assembly CSV는 단일 라벨이 아니라 assembly별 뉴런 인덱스 목록이다. 따라서 검증 단위는 단일 라벨 block이 아니라 두 뉴런이 assembly를 공유하는지 여부다.
3. 행동 자료가 없으므로 stimulus-action 방정식은 아직 아니지만, 척추동물 국소 회로의 폐쇄 동역학 후보는 통과한다.

### Zebrafish freely swimming activity

Figshare freely-swimming zebrafish figure5/S8 chunk로 자유수영 상태의 region activity를 확인했다. 이 chunk에는 정렬된 tail-behavior label이 없으므로 최종 activity-to-behavior gate는 아니다.

| 항목 | 값 |
|---|---:|
| region count | 18 |
| green free/imm mean similarity | 0.476526 |
| red free/imm mean similarity | 0.011651 |
| free green recurrent/baseline | 0.154988 |
| free red recurrent/baseline | 0.157001 |
| pass | True |

의의:

1. 자유수영 조건에서도 region activity는 평균 baseline보다 저차원 recurrent state로 훨씬 잘 닫힌다.
2. free/imm similarity는 조건이 달라도 일부 region-level activity 구조가 보존되는지 보는 보조 지표다.
3. 다음 병목은 같은 자료에서 neural trace와 tail/stage tracking의 시간 정렬이다.

### Zebrafish laser perturbation to behavior

figure8/c/LR chunk의 `boutInfo.mat`를 사용해 left/right laser 조건이 회전 행동 방향을 바꾸는지 검증했다. 이것은 neural trace -> behavior decoding이 아니라 perturbation -> behavior closure다.

| group | left n | left mean angle | right n | right mean angle | left-right | p |
|---|---:|---:|---:|---:|---:|---:|
| control | 264 | -0.753407 | 219 | -1.333287 | 0.579880 | 0.871826 |
| experimental | 308 | 48.403138 | 306 | -42.963969 | 91.367107 | 0.000200 |

| 항목 | 값 |
|---|---:|
| experimental/control effect ratio | 157.562119 |
| pass | True |

의의:

1. control에서는 left/right laser에 따른 회전 방향 차이가 거의 없다.
2. experimental fish에서는 left laser와 right laser가 반대 부호의 큰 회전각을 만든다.
3. 따라서 척추동물 단계의 motor output 항은 임의 잡음이 아니라 방향성 perturbation에 의해 조절되는 닫힌 행동 출력이다.

### Zebrafish activity to behavior-frame gate

figure8/g chunk의 `e2` neural activity와 `FrameBout` 행동 bout frame을 사용했다. 연속 tail/stage movement decoding은 아니고, neural activity가 행동 bout frame과 baseline frame을 구분하는지 보는 association gate다.

| 항목 | 값 |
|---|---:|
| activity shape region x frame | 3987 x 3360 |
| bout frames | 147 |
| baseline pool frames | 2270 |
| mean AUC | 0.887293 |
| AUC sd | 0.033238 |
| mean balanced accuracy | 0.812500 |
| balanced accuracy sd | 0.036968 |
| permutation AUC mean | 0.500352 |
| p | 0.000167 |
| pass | True |

의의:

1. 저차원 neural activity만으로 행동 bout frame과 baseline frame을 holdout에서 구분한다.
2. 이 결과는 perturbation-to-behavior보다 자연 activity 쪽에 더 가깝다.
3. 그러나 아직 speed, heading, turn angle의 연속 decoding은 아니다.

### Zebrafish activity to direction gate

figure8/f chunk의 `e2`, `LeftLS`, `RightLS`를 사용해 neural activity window가 left/right 조건을 구분하는지 봤다.

| 항목 | 값 |
|---|---:|
| activity shape region x frame | 3987 x 3360 |
| left trials | 11 |
| right trials | 10 |
| window frames | 5 |
| pre-baseline frames | 5 |
| AUC | 1.000000 |
| balanced accuracy | 1.000000 |
| permutation AUC mean | 0.355527 |
| p | 0.001996 |
| pass | True |

의의:

1. leave-one-trial-out에서 left/right laser trial을 neural activity만으로 구분한다.
2. 이미 left/right laser가 반대 방향 회전 행동을 만든다는 gate가 통과했으므로, 이 결과는 activity-direction-output 결합을 보강한다.
3. 표본 수는 21 trial로 작으므로 최종 결론은 continuous movement decoding에서 다시 확인해야 한다.

### Zebrafish continuous alignment audit

최종 목표는 neural activity frame으로 speed, heading, turn angle을 직접 예측하는 continuous decoding gate다.

필요한 관측식:

$$
\hat y_t
=
f_\theta(P_{t-\ell},q_t),
\qquad
y_t\in\{\mathrm{speed},\mathrm{heading},\mathrm{turn}\}
$$

현재 partial chunk가 그 목표를 지원하는지 점검했다.

가능한 것:

| 항목 | 값 |
|---|---|
| e2 neural matrix 있음 | True |
| behavior bout frame label 있음 | True |
| left/right laser frame label 있음 | True |
| stage/head/yolk tracking txt 있음 | True |
| LR 일부 폴더 timestamp.mat 있음 | True |

빠진 것:

| 항목 | 값 |
|---|---|
| neural mat 안에 stage/head/tail 좌표 있음 | False |
| neural mat 안에 e2 column별 absolute timestamp 있음 | False |

판정:

| 항목 | 값 |
|---|---|
| activity -> behavior-frame gate 가능 | True |
| activity -> direction gate 가능 | True |
| 현재 partial만으로 continuous movement decoding 가능 | False |

의의:

1. 현재 partial 자료는 neural activity와 discrete behavior labels는 연결한다.
2. 하지만 `e2`의 각 column이 stage tracking의 어느 시간/프레임에 해당하는지 알려주는 per-frame alignment가 없다.
3. 이 상태에서 speed/heading/turn angle을 직접 예측하면 임의 정렬이 되어 검증이 무효다.

### Zebrafish e2-LR alignment probe

`e2[:, t]` neural frame을 같은 시각의 stage/head/yolk movement에 붙일 수 있는지 확인했다.

e2 event sequence:

| 항목 | 값 |
|---|---|
| e2 shape | 3987 x 3360 |
| event count | 21 |
| sorted events | 80, 212, 331, 441, 589, 732, 863, 1016, 1192, 1324, 1614, 1873, 2100, 2293, 2475, 2588, 2695, 2850, 3004, 3151, 3262 |

LR session match summary:

| session | raw frames | raw laser onsets | stage/raw | timestamp.mat | best RMSE e2 frames | laser match | candidate |
|---|---:|---:|---:|---|---:|---|---|
| control/20221018_1027_g8s-lssm-none_10dpf | 14238 | 60 | 5.000 | True | 90.050 | False | False |
| control/20221116_1027_g8s-lssm-huc-none_8dpf | 14244 | 60 | 5.000 | False | 89.998 | False | False |
| control/20221116_1108_g8s-lssm-huc-none_8dpf | 13340 | 60 | 5.001 | False | 89.957 | False | False |
| control/20221116_1543_g8s-lssm-huc-none_8dpf | 13943 | 60 | 5.001 | False | 89.998 | False | False |
| control/20221116_1627_g8s-lssm-huc-none_8dpf | 14078 | 63 | 5.000 | False | 89.998 | False | False |
| exp/20221016_1556_g8s-lssm-chriR_8dpf | 18923 | 100 | 5.000 | False | 0.000 | True | True |
| exp/20221017_1453_g8s-lssm-chriR_9dpf | 20715 | 85 | 0.000 | True | 89.868 | False | False |
| exp/20221018_1626_g8s-lssm-chriR_10dpf | 14192 | 60 | 5.000 | False | 90.050 | False | False |
| exp/20221019_1053_g8s-lssm-chriR_11dpf | 17959 | 60 | 4.868 | False | 44.800 | False | False |
| exp/20221019_1609_g8s-lssm-chriR_11dpf | 17380 | 60 | 5.000 | False | 48.381 | False | False |

best attempted match:

| 항목 | 값 |
|---|---|
| session | exp/20221016_1556_g8s-lssm-chriR_8dpf |
| RMSE e2 frames | 0.000 |
| MAE e2 frames | 0.000 |
| max abs error e2 frames | 0.000 |
| slope raw->e2 | 1.000000 |
| laser schedule match | True |
| stage/raw frame ratio | 5.000211 |
| timestamp certified | False |
| candidate inferred alignment | True |

verdict:

| 항목 | 값 |
|---|---:|
| laser-schedule matches | 1 / 10 |
| timestamp-certified alignments | 0 / 10 |
| candidate inferred alignments | 1 / 10 |
| certified continuous decoding ready | False |
| candidate inferred decoding ready | True |

의의:

1. `exp/20221016_1556...`은 `e2` event sequence와 raw laser schedule이 정확히 맞고 stage/raw frame ratio도 약 5배다.
2. 하지만 그 session에는 `timestamp.mat`가 없다.
3. 따라서 이것은 candidate inferred alignment이지 final timestamp-certified alignment가 아니다.

### Zebrafish candidate continuous decoding

이 gate는 timestamp-certified 최종 검증이 아니라 inferred alignment 위에서 신호가 있는지 보는 정찰 검증이다.

alignment:

| 항목 | 값 |
|---|---:|
| raw offset e2 -> raw | 9339 |
| raw frames | 18923 |
| stage rows | 94619 |
| stage/raw ratio | 5.000211 |
| status | candidate_inferred_not_timestamp_certified |

best decoding:

| target | best lag e2 frames | R2 | mse/base | shift p | candidate |
|---|---:|---:|---:|---:|---|
| speed | 10 | 0.123460 | 0.876540 | 0.066667 | True |
| turn | 150 | 0.010998 | 0.989002 | 0.066667 | False |

판정:

| 항목 | 값 |
|---|---|
| final continuous gate pass | False |

의의:

1. speed는 inferred alignment에서 약한 후보 신호가 있다.
2. turn/heading은 닫히지 않았다.
3. p가 0.05 아래로 내려가지 않았고 timestamp-certified도 아니므로 최종 continuous movement gate를 통과했다고 보지 않는다.

### Zebrafish supplementary audit

`Others_Supplementary.7z`까지 받은 뒤 공개 supplementary 안에 explicit e2 timestamp 또는 e2-resampled behavior trace가 있는지 감사했다.

archive:

| item | value |
|---|---:|
| md5 ok | True |
| files | 109 |
| mat files | 19 |
| txt files | 35 |

bridge checks:

| check | value |
|---|---:|
| e2-named files in supplementary | 0 |
| timestamp-named files in supplementary | 0 |
| matched e2 session files | 0 |
| has e2 timestamp variable | False |
| has e2-resampled behavior | False |
| Z-tracking mats | 15 |
| Z-tracking mats expZ only | True |
| CalTrace mats | 1 |
| interpolation mats | 2 |

verdict:

| 항목 | 값 |
|---|---|
| timestamp-certified continuous ready | False |
| verdict | blocked_missing_e2_behavior_bridge |

Others_Supplementary supplies Z-position/stage alignment and calcium trace/interpolation QA, but not an explicit e2 timestamp or e2-resampled speed/turn trace.

따라서 zebrafish continuous movement는 다음처럼 닫는다.

$$
\boxed{
\mathrm{zebrafish\ activity\ closure}
\quad\text{is supported, but}\quad
\mathrm{continuous\ movement\ decoding}
\quad\text{is blocked by missing alignment, not by a tested dynamics failure.}
}
$$

### Zebrafish 종합 의의

Zebrafish 단계에서 닫힌 것은 네 가지다.

1. assembly 공유 쌍은 random membership보다 상관구조를 훨씬 잘 설명했다.
2. 저차원 recurrent state는 평균 baseline보다 다음 시점을 잘 예측했다.
3. 방향성 perturbation은 방향성 회전 행동으로 강하게 닫혔다.
4. neural activity는 행동 bout frame과 baseline frame을 구분했고, left/right 방향 조건도 leave-one-trial-out에서 구분했다.

닫히지 않은 것은 하나다.

1. neural activity frame을 speed/heading/turn angle의 continuous tracking frame에 timestamp-certified로 붙이는 최종 bridge.

따라서 zebrafish 판정은 "activity state closure는 통과, continuous movement는 data-boundary"다.

