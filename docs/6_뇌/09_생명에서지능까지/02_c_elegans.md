## 단계 1: C. elegans 원시 신경계

C. elegans는 "최소 원시 신경계"의 첫 실자료 gate다. 상태는 감각/input, 중간/relay, premotor/integrative output의 coarse state로 접는다.

$$
P_n
=
\left(
P_{\mathrm{L1}},
P_{\mathrm{L2}},
P_{\mathrm{L3}}
\right)
$$

축약식:

$$
P_{n+1}
=
\Pi
\left[
\rho P_n
+(1-\rho)P_*
+\gamma\mathcal L(W_{\mathrm{chem}})P_n
+\sum_{d\in\mathcal D_{\mathrm{worm}}}a_{d,n}U_d
+H(q_n-q_*)
\right]
$$

여기서 \(W_{\mathrm{chem}}\)은 Witvliet/OpenWorm 계열 C. elegans chemical weighted connectome이다.

### C. elegans graph gate

질문:

$$
\mathcal L_{\mathrm{L1/L2/L3\ block}}
<
\mathcal L_{\mathrm{flat/random\ layer}}
$$

결과:

| 항목 | 값 |
|---|---:|
| module count | 12 |
| used edges | 2102 |
| used synapses | 7222.0 |
| flat loss | 1810764.638889 |
| L1/L2/L3 block loss | 1302631.125000 |
| block / flat | 0.719382 |
| block / random mean | 0.804904 |
| permutation p | 0.027986 |
| passed | True |

방향성:

| 항목 | 값 |
|---|---:|
| forward fraction | 0.370396 |
| backward fraction | 0.096511 |
| lateral fraction | 0.533093 |
| forward/backward | 3.837877 |

C. elegans 회로는 단순 feedforward 사슬이 아니다. lateral/recurrent 성분이 크다. 따라서 성공 기준은 forward dominance가 아니라 layer-block reconstruction이다.

Robustness:

| matrix | block/flat | block/random mean | permutation p | module-family/flat | pass |
|---|---:|---:|---:|---:|---|
| all_weighted | 0.719382 | 0.804904 | 0.027986 | 0.704610 | True |
| chemical_weighted | 0.717370 | 0.802082 | 0.028986 | 0.710934 | True |
| electrical_weighted | 0.830206 | 0.920392 | 0.090955 | 0.715055 | False |
| all_binary | 0.914439 | 1.004515 | 0.498251 | 0.609626 | False |
| chemical_binary | 0.902778 | 0.989918 | 0.370815 | 0.644033 | False |
| electrical_binary | 0.819574 | 0.911938 | 0.136932 | 0.664681 | False |

반례 점검:

| model | labels | params | block/flat | p | BIC-like | saturated |
|---|---:|---:|---:|---:|---:|---|
| all_one | 1 | 1 | 1.000000 | 1.000000 | 1364.250 | False |
| layer | 3 | 9 | 0.719382 | 0.027986 | 1356.580 | False |
| module_family | 5 | 25 | 0.704610 | 0.265867 | 1433.110 | False |
| lateral_vs_other | 2 | 4 | 0.960934 | 0.435782 | 1373.421 | False |
| avoidance_vs_other | 2 | 4 | 0.933393 | 0.255372 | 1369.234 | False |
| taxis_vs_other | 2 | 4 | 0.915743 | 0.179910 | 1366.485 | False |
| module | 12 | 144 | 0.000000 | 1.000000 | -3978.867 | True |

포화모델을 제외한 BIC-like 최저 모델은 `layer`다. 순수 손실 최저는 `module_family`지만 p와 BIC-like 기준에서는 layer가 더 안전하다.

안정성:

| 항목 | 값 |
|---|---:|
| lambda max | 1242.163296 |
| gamma upper bound at rho=0.20 | 0.00096606 |
| modal radius at half bound | 0.400000 |

판정:

$$
\boxed{
\mathcal L_{\mathrm{weighted\ chemical\ L1/L2/L3}}
<
\mathcal L_{\mathrm{flat}}
\quad\text{and}\quad
p<0.05
}
$$

의의:

1. 원시 신경계에도 감각-input, 중간-relay, premotor/integrative output의 coarse layer 문법이 보인다.
2. 이 문법은 binary adjacency가 아니라 weighted chemical synapse에 실려 있다.
3. 전역 뇌 방정식의 \(\Delta_G\) 또는 \(\mathcal L(W)\) 항은 고등피질 전용이 아니라 원시 회로에도 적용 가능한 후보다.

### C. elegans developmental graph gate

질문:

$$
\mathcal L_{\mathrm{chemical\ weighted\ L1/L2/L3}}^{(\mathrm{stage})}
<
\mathcal L_{\mathrm{flat/random}}^{(\mathrm{stage})}
$$

단계별 결과:

| stage | synapses | chem block/flat | chem p | all block/flat | electrical block/flat | binary block/flat | lambda max | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1235.0 | 0.756479 | 0.057971 | 0.758016 | 0.891754 | 0.766957 | 210.529187 | False |
| 2 | 1791.0 | 0.737261 | 0.043978 | 0.734655 | 0.826484 | 0.737934 | 318.376645 | True |
| 3 | 1957.0 | 0.734236 | 0.041979 | 0.726537 | 0.778895 | 0.769426 | 346.999413 | True |
| 4 | 2697.0 | 0.714496 | 0.029485 | 0.712483 | 0.784445 | 0.800157 | 513.036590 | True |
| 5 | 3958.0 | 0.728187 | 0.030485 | 0.723991 | 0.759796 | 0.793939 | 666.960449 | True |
| 6 | 4113.0 | 0.717524 | 0.029985 | 0.718669 | 0.820706 | 0.846896 | 722.188214 | True |
| 7 | 6624.0 | 0.717431 | 0.028986 | 0.720519 | 0.816848 | 0.823661 | 1141.809166 | True |
| 8 | 7222.0 | 0.717370 | 0.028986 | 0.719382 | 0.830206 | 0.914439 | 1242.163296 | True |

요약:

| 항목 | 값 |
|---|---:|
| stages | 8 |
| passed weighted chemical stages | 7 |
| mean chemical block/flat | 0.727873 |
| min chemical block/flat | 0.714496 |
| max chemical block/flat | 0.756479 |
| mean chemical permutation p | 0.036482 |
| Spearman stage vs chemical block/flat | -0.761905 |
| Spearman stage vs synapses | 1.000000 |
| Spearman stage vs lambda max | 1.000000 |

판정:

$$
\boxed{
\text{C. elegans의 weighted chemical layer structure는 성체에서 갑자기 생긴 것이 아니라 발달 초기에 이미 나타난다.}
}
$$

의의:

1. 원시 routing은 성체의 우연한 산물이 아니라 발달 전반에서 유지되는 구조다.
2. synapse 수와 \(\lambda_{\max}\)는 stage와 함께 증가하지만, block/flat은 오히려 안정적으로 낮아진다.
3. 따라서 "처음에는 무작위 연결이고 나중에만 기능층이 생긴다"는 반례가 약해진다.

### C. elegans stimulus-output gate

stimulus-output routing은 row-normalized chemical transition \(T(W_{\mathrm{chem}})\)로 둔다.

$$
R_d
=
\frac{
\mathrm{Flow}(L1_d\rightarrow L3_d)
}{
\mathrm{mean}_{d'\ne d}\mathrm{Flow}(L1_d\rightarrow L3_{d'})
}
$$

질문:

$$
\mathrm{Flow}(L1_d\to L3_d)
>
\mathrm{Flow}(L1_d\to L3_{d'\ne d})
$$

결과:

| 항목 | 값 |
|---|---:|
| matched mean | 0.37828533 |
| wrong mean | 0.11022711 |
| matched / wrong | 3.431872 |
| permutation p | 0.034393 |
| effect threshold | 1.500000 |
| passed | True |

domain flow:

| route | combined flow | direct | two-step |
|---|---:|---:|---:|
| Anterior->Anterior | 0.40329661 | 0.42442357 | 0.38216966 |
| Anterior->Lateral | 0.24763359 | 0.16310845 | 0.33215873 |
| Anterior->Avoidance | 0.01528595 | 0.01280956 | 0.01776234 |
| Anterior->Taxis | 0.04885711 | 0.04269855 | 0.05501568 |
| Lateral->Anterior | 0.10947095 | 0.02564103 | 0.19330087 |
| Lateral->Lateral | 0.58360729 | 0.65811966 | 0.50909491 |
| Lateral->Avoidance | 0.05052657 | 0.04273504 | 0.05831809 |
| Lateral->Taxis | 0.04143545 | 0.00000000 | 0.08287091 |
| Avoidance->Anterior | 0.11056432 | 0.08068460 | 0.14044405 |
| Avoidance->Lateral | 0.24579962 | 0.20782396 | 0.28377527 |
| Avoidance->Avoidance | 0.30458385 | 0.38141809 | 0.22774962 |
| Avoidance->Taxis | 0.07716321 | 0.04645477 | 0.10787165 |
| Taxis->Anterior | 0.14174111 | 0.12020033 | 0.16328189 |
| Taxis->Lateral | 0.17264602 | 0.10434057 | 0.24095147 |
| Taxis->Avoidance | 0.06160138 | 0.05175292 | 0.07144985 |
| Taxis->Taxis | 0.22165357 | 0.23622705 | 0.20708010 |

developmental stimulus-output:

| stage | synapses | chem matched/wrong | chem p | chem pass | binary matched/wrong | binary pass |
|---:|---:|---:|---:|---|---:|---|
| 1 | 1235.0 | 3.186427 | 0.035482 | True | 1.446780 | False |
| 2 | 1791.0 | 2.600517 | 0.035482 | True | 1.264549 | False |
| 3 | 1957.0 | 3.302136 | 0.035482 | True | 1.380026 | False |
| 4 | 2697.0 | 3.464723 | 0.035482 | True | 1.328685 | False |
| 5 | 3958.0 | 3.297920 | 0.035482 | True | 1.208942 | False |
| 6 | 4113.0 | 2.995919 | 0.035482 | True | 1.193745 | False |
| 7 | 6624.0 | 3.428513 | 0.035482 | True | 1.040436 | False |
| 8 | 7222.0 | 3.431872 | 0.035482 | True | 1.113045 | False |

요약:

| 항목 | 값 |
|---|---:|
| stages | 8 |
| passed chemical weighted stages | 8 |
| mean chemical matched/wrong | 3.213504 |
| min chemical matched/wrong | 2.600517 |
| max chemical matched/wrong | 3.464723 |
| mean p value | 0.035482 |
| Spearman stage vs matched/wrong | 0.476190 |
| Spearman stage vs synapses | 1.000000 |

판정:

$$
\boxed{
\text{C. elegans의 stimulus-output domain channel은 발달 초기부터 안정적이며, binary가 아니라 weighted chemical graph에 실려 있다.}
}
$$

의의:

1. 원시 신경계의 첫 기능은 "많은 뉴런"이 아니라 자극 domain을 적절한 output domain으로 보내는 weighted routing이다.
2. 실제 행동 trial은 아니지만 connectome proxy에서 domain-preserving channel이 닫혔다.
3. 이 결과로 최소 primitive neural control 식의 \(\mathcal L(W_{\mathrm{chem}})\)와 \(U_d\rightarrow b_d\) 항은 다음 단계로 넘길 수 있다.

정리:

$$
\boxed{
\mathrm{primitive\ neural\ control}
=
\mathrm{weighted\ chemical\ routing}
+\mathrm{domain\ preserving\ stimulus\ output}
+\mathrm{homeostatic\ projection}
}
$$

