# 저층부에서 고등생물로 올라가는 방정식 로드맵

## 핵심 방향

최초 생명 자체를 직접 찾기는 어렵다. 그래서 현재 가능한 길은 공개 자료가 있는 층을 따라 올라가며, 각 단계에서 방정식에 어떤 항이 새로 생기는지 확인하는 것이다.

```text
생명 최소 동역학
-> 원시 신경계의 weighted chemical stimulus-output routing
-> 세포형/action/memory 분화
-> 척추동물의 실제 activity state/recurrent dynamics
-> 포유류/인간의 영역별 작업공간과 장기 예측
```

공통 형태는 다음처럼 둔다.

$$
S_{n+1}
=
\Pi_{\mathcal C}
\left[
A_G S_n
+U_n
+H_n
+F_n
+M_n
\right]
$$

여기서 \(S_n\)은 단계별 상태, \(A_GS_n\)은 그래프 결합, \(U_n\)은 외부 입력, \(H_n\)은 항상성/각성/대사 항, \(F_n\)은 가소성/복제/느린 변화, \(M_n\)은 memory/internal model, \(\Pi_{\mathcal C}\)는 가능한 상태공간으로의 제약 투영이다.

## 단계 0: 생명 최소 동역학

생명은 “선택 가능한 열린 자기유지 동역학”으로 정의한다.

$$
X_{n+1}
=
\Pi
\left[
X_n
+R_{\mathrm{autocat}}(X_n,E_n)
-D_{\mathrm{decay}}(X_n)
-L_{\mathrm{leak}}(X_n)
+B_{\mathrm{boundary}}(X_n)
+C_{\mathrm{copy}}(X_n)
\right]
$$

아직 직접 검증은 어렵다. LUCA 비교유전체학, RNA world/ribozyme 실험, autocatalytic set 모델, protocell/lipid vesicle 실험을 통해 하위 gate를 따로 검증해야 한다.

## 단계 1: C. elegans 원시 신경계

상태:

$$
P_n=(p_{L1},p_{L2},p_{L3})
$$

방정식:

$$
P_{n+1}
=
\Pi
\left[
\rho P_n
+\gamma\Delta_{G_{\mathrm{chem}}}P_n
+U_{\mathrm{stimulus}}
+H_{\mathrm{body}}
\right]
$$

검증 요약:

| gate | 결과 |
|---|---|
| weighted chemical L1/L2/L3 | pass |
| developmental stage 1-8 | 7/8 pass |
| binary graph | fail |
| stimulus-domain to same output-domain flow | pass |
| developmental stimulus-output channel | 8/8 pass |

의의:

$$
\boxed{
\Delta_G는 binary adjacency가 아니라 weighted chemical graph여야 한다.
}
$$

또한 가장 이른 층에서 먼저 보이는 것은 추상 계층 자체보다 weighted chemical stimulus-output routing이다. 즉 원시 신경계는 “많은 뉴런”보다 “가중 경로가 감각-domain을 적절한 output-domain으로 보내는 구조”가 먼저다.

## 단계 2: Drosophila larva

초기 예상:

$$
P_n=(P_{\mathrm{sensory}},P_{\mathrm{relay}},P_{\mathrm{action}},P_{\mathrm{memory}})
$$

검증 후 수정:

$$
P_{n+1}
=
\Pi[
\cdots
+D_{\mathrm{celltype}}
+A_{\mathrm{descending/action}}
+M_{\mathrm{memory}}
]
$$

결과 요약:

| 검증 | 결과 |
|---|---|
| primitive 3-class | 약함 |
| extended memory | primitive보다 10.5% 개선, strict p 실패 |
| action split | 순수 손실 기준으로 memory 모델보다 좋음 |
| memory-loop touched fraction | 36.6% |

의의:

$$
\boxed{
\text{두 번째 큰 도약은 memory 단독이 아니라 cell type/action/memory 공동 분화다.}
}
$$

즉 무작정 신경계 양만 늘린다고 지능이 생기는 게 아니다. 감각, relay, descending/action, memory, lateral integration처럼 역할이 나뉘고, 그 역할들이 닫힌 순환을 만들 때 다음 단계로 간다.

## 단계 3: Zebrafish larva activity

이번에는 connectome-only가 아니라 실제 calcium activity 자료를 썼다. 두 갈래로 확인했다.

1. optic tectum spontaneous activity: 국소 회로의 assembly/recurrent structure.
2. freely swimming figure5/S8 chunk: 자유수영 조건에서도 region activity가 저차원 recurrent state로 닫히는지.
3. freely swimming figure8/c/LR chunk: 방향성 optogenetic perturbation이 회전 행동으로 닫히는지.
4. freely swimming figure8/g chunk: neural activity가 행동 bout frame과 baseline frame을 구분하는지.
5. freely swimming figure8/f chunk: neural activity window가 left/right 방향 조건을 구분하는지.
6. continuous alignment audit: 현재 partial chunk만으로 speed/heading/turn angle 직접 decoding이 가능한지.

다만 아직 연속적인 tail speed/heading/turn angle을 직접 예측하는 최종 continuous decoding gate는 아니다.

검증한 식:

$$
P_{n+1}
=
\Pi[
\rho P_n
+L_{\mathrm{lowrank}}(P_n)
+C_{\mathrm{assembly}}(P_n)
]
$$

여기서 \(C_{\mathrm{assembly}}\)는 두 뉴런이 같은 assembly를 공유하는지에 따른 correlation block 항이고, \(L_{\mathrm{lowrank}}\)는 저차원 recurrent state 예측 항이다.

결과:

| fish | cells | time | assemblies | coassembly/flat | p | recurrent/baseline | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zf_20151104-f1 | 114 | 4245 | 5 | 0.681772 | 0.000500 | 0.447369 | 0.552631 |
| zf_20170215-f3 | 75 | 5660 | 4 | 0.572443 | 0.000500 | 0.330413 | 0.669587 |

자유수영 activity robustness:

| chunk | regions | green free/imm similarity | red free/imm similarity | green recurrent/baseline | red recurrent/baseline |
|---|---:|---:|---:|---:|---:|
| figure5/S8 | 18 | 0.476526 | 0.011651 | 0.154988 | 0.157001 |

방향성 perturbation -> behavior:

| group | left mean angle | right mean angle | left-right | p |
|---|---:|---:|---:|---:|
| control | -0.753407 | -1.333287 | 0.579880 | 0.871826 |
| experimental | 48.403138 | -42.963969 | 91.367107 | 0.000200 |

activity -> behavior-frame association:

| activity shape | bout frames | mean AUC | balanced accuracy | p |
|---|---:|---:|---:|---:|
| 3987 x 3360 | 147 | 0.887293 | 0.812500 | 0.000167 |

activity -> direction association:

| trials L/R | window | AUC | balanced accuracy | p |
|---:|---:|---:|---:|---:|
| 11 / 10 | 5 frames | 1.000000 | 1.000000 | 0.001996 |

continuous alignment audit:

| question | result |
|---|---|
| activity -> behavior-frame 가능 | True |
| activity -> direction 가능 | True |
| 현재 partial만으로 continuous movement decoding 가능 | False |

의의:

$$
\boxed{
\text{척추동물 단계에서는 구조적 경로뿐 아니라 실제 활동 상태공간의 닫힘이 자유행동 조건에서도 보이기 시작한다.}
}
$$

이 단계에서 밝혀진 것은 아직 “연속 움직임 전체를 읽는 방정식”은 아니지만, 네 가지는 닫혔다. 첫째, assembly 공유 쌍은 random membership보다 상관구조를 훨씬 잘 설명했고, 저차원 recurrent state는 평균 baseline보다 다음 시점을 잘 예측했다. 둘째, 자유수영 chunk에서 activity state 자체는 강하게 닫혔다. 셋째, 방향성 perturbation은 방향성 회전 행동으로 강하게 닫혔다. 넷째, neural activity는 행동 bout frame과 baseline frame을 구분했고, left/right 방향 조건도 leave-one-trial-out에서 구분했다.

중요한 한계도 닫혔다. 현재 partial chunk에는 stage/head/yolk tracking txt와 neural `e2` matrix가 따로 존재하지만, `e2`의 각 column을 tracking frame에 직접 붙이는 per-frame alignment가 없다. 따라서 지금 가진 partial만으로 speed, heading, turn angle을 직접 예측하면 임의 정렬이 되므로 검증으로 인정하면 안 된다.

## 현재 구조의 전체 의의

지금까지의 구조는 다음 명제를 만든다.

$$
\boxed{
\text{지능은 신경계 양의 단순 증가가 아니라, 보존되는 weighted routing 위에 역할 분화와 닫힌 activity state가 쌓인 결과다.}
}
$$

따라서 방정식의 하위 항은 이렇게 닫힌다.

```text
생명: autocatalysis + boundary + copying
원시 신경계: weighted chemical routing
단순 동물: stimulus-output channel
곤충: cell type/action/memory differentiation
척추동물: assembly + recurrent activity state
고등동물: whole-brain behavior + internal model + workspace
```

## 다음 단계

다음으로 가장 적당한 검증은 whole-brain zebrafish 또는 adult fly/FlyWire다.

| 후보 | 목적 | 부담 |
|---|---|---|
| zebrafish whole-brain + behavior | activity가 실제 swimming/turning을 예측하는지 | 데이터 큼 |
| adult fly hemibrain/FlyWire | mushroom body, central complex, action selection 검증 | annotation 처리 필요 |
| mouse Neuropixels/IBL | 포유류 영역 루프 검증 | 인간 전 단계, 계산 부담 큼 |

현재 흐름상 우선순위는 zebrafish continuous behavior decoding이다. 다만 다음 실행에는 추가 자료가 필요하다.

필요한 파일:

| 필요한 것 | 이유 |
|---|---|
| e2 column별 timestamp | neural frame을 tracking frame에 붙이기 위해 |
| 또는 e2 frame으로 resample된 speed/heading/turn angle | 바로 regression gate 가능 |
| 또는 raw light-field + synchronized tracking chunk | 직접 정렬/추출 가능 |

이유는 C. elegans/Drosophila에서 구조를 봤고, 이번 zebrafish에서 자유수영 activity state, perturbation-to-behavior, activity-to-bout-frame, activity-to-direction association까지 봤으므로, 마지막으로 neural trace와 tail/stage tracking을 시간 정렬해 speed, heading, turn angle 자체를 예측해야 하기 때문이다.
