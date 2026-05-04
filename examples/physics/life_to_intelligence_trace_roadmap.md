# 저층부에서 고등생물로 올라가는 방정식 로드맵

## 핵심 방향

최초 생명 자체를 직접 찾기는 어렵다. 대신 더 현실적인 경로는 다음이다.

```text
생명 최소 동역학
-> 원시 신경계
-> 곤충 학습/행동선택 회로
-> 척추동물 whole-brain dynamics
-> 포유류/인간 뇌
```

이 경로는 자료를 타고 올라갈 수 있다. 각 단계마다 공개 자료가 있고, 각 단계의 방정식 항이 하나씩 추가되는지 볼 수 있다.

## 공통 형식

모든 단계는 다음 압축식의 특수형으로 볼 수 있다.

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

여기서 \(S_n\)은 단계마다 다르다.

| 단계 | 상태 \(S_n\) |
|---|---|
| prebiotic/life | 내부 화학 농도와 경계 상태 |
| C. elegans | neuron/module 상태 |
| Drosophila | cell type / memory-action loop 상태 |
| zebrafish | whole-brain region/activity state |
| mouse/human | region-resolved \(p_r=(x_a,x_s,x_b)\) |

항의 의미:

| 항 | 의미 |
|---|---|
| \(A_GS_n\) | 현재 상태 관성 + graph coupling |
| \(U_n\) | 외부 입력, 자극, 과제 |
| \(H_n\) | 항상성, 에너지, 몸 상태 |
| \(F_n\) | 가소성, 복제, 장기 변화 |
| \(M_n\) | memory/workspace/internal model |
| \(\Pi_{\mathcal C}\) | 가능한 상태공간으로 투영 |

## 단계 0: 생명 최소식

상태:

$$
X_n=(\mathrm{chemical\ concentrations},\mathrm{boundary},\mathrm{template})
$$

최소식:

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

닫아야 할 gate:

| gate | 부등식 |
|---|---|
| 자기유지 | \(\|X_{n+1}-X^*\|<\|X_n-X^*\|\) |
| 열린 flux | \(R_{\mathrm{autocat}}(X,E)>D_{\mathrm{decay}}(X)\) |
| 경계 | \(L_{\mathrm{leak}}<R_{\mathrm{production}}\) |
| 복제 | \(C(X)\to X'\approx X\) |
| 변이/선택 | \(\mathrm{growth}(X_i)\ne\mathrm{growth}(X_j)\) |

이 단계에서 생명은 다음처럼 정의한다.

$$
\boxed{
\text{생명 후보}
=
\text{선택 가능한 열린 자기유지 동역학}
}
$$

자료 후보:

| 자료/문헌 | 쓸 수 있는 것 |
|---|---|
| LUCA comparative genomics | LUCA 이후 공통 core 추정 |
| RNA world / ribozyme 실험 | 복제/촉매 후보 |
| autocatalytic set 모델 | 자기촉매 네트워크 |
| protocell/lipid vesicle 실험 | 경계와 leakage |

## 단계 1: 원시 신경계 C. elegans

상태:

$$
P_n=(p_{L1},p_{L2},p_{L3})
$$

식:

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

현재 닫힌 gate:

| 검증 | 결과 |
|---|---|
| weighted chemical L1/L2/L3 | pass |
| developmental stage 1-8 | 7/8 pass |
| binary graph | fail |
| stimulus-domain to same output-domain flow | pass |
| developmental stimulus-output channel | 8/8 pass |

핵심 함의:

$$
\boxed{
\Delta_G는 binary adjacency가 아니라 weighted chemical graph여야 한다.
}
$$

그리고 최초 신경계에서는:

$$
\boxed{
\text{감각-domain이 같은 output-domain으로 보존되는 구조 경로가 있다.}
}
$$

발달 단계 1-8 전체에서도 이 channel은 유지된다.

| matrix | 결과 |
|---|---|
| chemical weighted | 8/8 pass, mean matched/wrong 3.213504 |
| binary | 0/8 pass |

이것은 중요한 순서를 시사한다.

$$
\boxed{
\text{행동 domain channel은 L1/L2/L3 전체 block 구조보다 더 이른 단계부터 안정적으로 보존된다.}
}
$$

즉 최초 신경계에서 먼저 보이는 것은 “큰 뇌 구조”가 아니라 **weighted chemical stimulus-output routing**일 수 있다.

## 단계 2: 곤충 Drosophila larva

상태:

$$
P_n=(P_{\mathrm{sensory}},P_{\mathrm{relay}},P_{\mathrm{action}},P_{\mathrm{memory}})
$$

초기 예상식:

$$
P_{n+1}
=
\Pi[
\cdots
+M_{\mathrm{memory/action}}
]
$$

반례 점검 후 수정식:

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

현재 결과:

| 검증 | 결과 |
|---|---|
| primitive 3-class | 약함 |
| extended memory | primitive보다 10.5% 개선, p 실패 |
| action split | 순수 손실에서 extended memory보다 좋음 |
| memory-loop touched fraction | 36.6% |

함의:

$$
\boxed{
\text{2스텝은 memory 단독이 아니라 cell type/action/memory 공동 분화다.}
}
$$

자료 후보:

| 자료 | 목적 |
|---|---|
| Drosophila larva connectome | 현재 사용 |
| hemibrain / FlyWire adult | mushroom body, central complex, action selection 재검증 |
| neuPrint / Codex | cell type annotation 기반 세밀 검증 |

## 단계 3: Zebrafish larva

여기가 다음으로 중요하다. 이유는 whole-brain activity와 행동이 함께 있는 자료가 많기 때문이다.

상태:

$$
P_n=(P_{\mathrm{forebrain}},P_{\mathrm{midbrain}},P_{\mathrm{hindbrain}},P_{\mathrm{motor}})
$$

추가될 항:

$$
W_{\mathrm{wholebrain\ dynamics}}
$$

후보식:

$$
P_{n+1}
=
\Pi[
\rho P_n
+\gamma\Delta_GP_n
+U_{\mathrm{visual/odor/touch}}
+H_{\mathrm{arousal}}
+A_{\mathrm{motor}}
]
$$

검증할 gate:

| gate | 의미 |
|---|---|
| stimulus matched | 시각/냄새/접촉 자극이 서로 다른 circuit로 들어가는가 |
| behavior output | tail movement/turning/swimming이 motor state로 예측되는가 |
| whole-brain recurrence | 현재 brain state가 다음 행동을 baseline보다 잘 예측하는가 |
| arousal modulation | 각성 상태가 stimulus-action gain을 바꾸는가 |

자료 후보:

| 자료 | 목적 |
|---|---|
| zebrafish whole-brain calcium imaging | \(P_n\to P_{n+1}\) 전이 |
| larval zebrafish behavior datasets | stimulus-action mapping |
| Fish1 connectome | graph \(\Delta_G\) 후보 |

## 단계 4: Mouse

여기서 포유류 특유의 loop가 명확해진다.

추가 후보:

$$
C_{\mathrm{cortex-thalamus}}
+B_{\mathrm{basal\ ganglia}}
+M_{\mathrm{hippocampus}}
$$

식:

$$
P_{n+1}
=
\Pi[
\rho P_n
+\gamma\Delta_GP_n
+\sum_dU^{(d)}
+H(Q-Q^*)
+F_{\mathrm{syn}}
+M_{\mathrm{hippocampal}}
]
$$

자료 후보:

| 자료 | 목적 |
|---|---|
| Allen Mouse Brain Connectivity | weighted structural graph |
| Allen Cell Types | cell type map |
| Neuropixels Visual Coding | region activity + stimulus |
| IBL behavior/ephys | decision/action loop |

검증:

$$
\mathcal L_{\mathrm{cortex/thalamus/basal}}
<
\mathcal L_{\mathrm{primitive\ graph}}
$$

## 단계 5: Human

인간은 마지막이다. 지금은 바로 닫기 어렵다.

현재 닫은 것:

| 자료 | 닫은 범위 |
|---|---|
| ds000116 | visual/auditory event-level modality |
| ds000201 | pain/face/control, vigilance/working memory/arousal |

아직 필요한 것:

| 필요 | 이유 |
|---|---|
| BOLD/EEG region state | 진짜 \(p_r\) 필요 |
| weighted connectome | \(\Delta_G\) 검증 |
| 반복 측정 | \(P_n\to P_{n+1}\) 전이 |
| subject/session/task split | 과적합 방지 |

최종 gate:

$$
\mathcal L_{\mathrm{full}}(P_{n+1}^{\mathrm{obs}})
<
\mathcal L_{\mathrm{best\ ablation}}
$$

## 전체 진화식

현재 가장 안전한 계층은 다음이다.

```text
0. open autocatalytic life dynamics
1. weighted chemical stimulus-action control
2. cell type/action/memory differentiation
3. whole-brain recurrent state dynamics
4. cortex-thalamus-basal-ganglia-hippocampus loop
5. human workspace/abstract domain
```

방정식 항으로는:

$$
\begin{aligned}
S_{n+1}
=
\Pi[
&\rho S_n
+\gamma\Delta_{G_{\mathrm{weighted}}}S_n
+U_{\mathrm{stimulus/domain}} \\
&+H_{\mathrm{homeostasis}}
+F_{\mathrm{plasticity}}
+D_{\mathrm{celltype}}
+A_{\mathrm{action}}
+M_{\mathrm{memory/workspace}}
]
\end{aligned}
$$

## 다음 실행 순서

1. `C. elegans` stimulus-behavior proxy를 developmental stage 1-8로 확장한다.
2. Drosophila adult/FlyWire에서 memory vs action split을 다시 검증한다.
3. Zebrafish whole-brain activity 자료를 찾아 실제 \(P_n\to P_{n+1}\) 전이를 만든다.
4. Mouse Allen/IBL로 포유류 loop를 검증한다.
5. 마지막에 인간 BOLD/EEG로 돌아온다.

현재 가장 견고한 결론:

$$
\boxed{
\text{저층부부터 자료를 타고 올라가는 방식이 인간 뇌 방정식을 직접 맞추는 것보다 훨씬 적절하다.}
}
$$
