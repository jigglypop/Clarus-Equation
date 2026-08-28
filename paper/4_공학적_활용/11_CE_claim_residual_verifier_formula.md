# 11. CE Claim Residual Verifier Formula

이 문서는 claim residual verifier의 수식 정본이며, score가 아닌 evidence-bounded 복원 규칙을 정의한다. 독자는 확률분포, 공분산, graph regularization을 안다고 가정한다. 입력은 후보·근거·scale·source graph이고 출력은 posterior·mask·복원 답변이며, 성공은 adversarial/OOD holdout에서 false-allow를 낮추는 것이고 실패는 scale drift·source 상관·오탐을 무시하는 것이다.

## 0. 판정

판정은 현재 수식의 적용 범위와 미완성 bridge를 요약하며, verifier 통과가 claim의 형식적 증명은 아니라는 경계를 둔다.

이 문서는 LLM verifier를 다시 세우기 위한 수식 정본이다.

현재 `10_PreEq_LLM_manifest_verifier.md`의 count 기반 defect는 smoke-test용 저차 근사다. `paper/4_공학적_활용/09_무차원_잔차장_환각억제.md`와 `paper/9_등호이전/01_공리와증명.md` 기준으로는, 실제 verifier의 기본 단위는 답변 문자열이 아니라 **claim residual field**여야 한다.

따라서 v2 verifier는 다음 순서로 닫는다.

1. 답변 후보를 claim 후보 묶음으로 분해한다.
2. 각 claim에 대해 evidence 기준장 $B$, 불확실성 $C$, 무차원 잔차 $e$를 만든다.
3. source reliability와 independent source quorum을 계산한다.
4. signed claim graph curvature를 계산한다.
5. claim action $\phi_i$와 answer action $\Delta(y)$를 만든다.
6. PreEq Gibbs posterior로 후보 답변을 재가중한다.
7. accepted claim mask만 answer composer로 복원한다.

문서군 전체 기준으로 verifier는 두 stack을 결합해야 한다.

| stack | 역할 | 지위 |
|---|---|---|
| claim residual gate | evidence 기준장 $B$, scale $C$, source/graph action으로 claim을 accept/reject | `Tooling/Open test` |
| PreEq manifest verifier | 답변 후보 $y_k$ 위 posterior $p_\beta$와 abstain gate | posterior 수학은 `Exact`, 적용은 `Tooling/Open test` |

claim residual gate가 외부 사실성 방어선이고, PreEq manifest verifier가 후보 답변 선택 방어선이다. 둘 중 하나만 쓰면 문서 정본에 못 미친다.

## 1. 후보공간

후보공간은 유한집합으로 고정하며, 생성기가 빠뜨린 참 후보나 무한 탐색의 선택 효과는 이 정의 밖에 있다.

질문 $x$와 evidence bundle $E$가 주어졌다고 하자. LLM 또는 후보 생성기는 답변 후보 유한집합을 만든다.

$$
A_x=\{y_1,\dots,y_N\}.
$$

후보 $y_k$는 claim들의 유한집합으로 분해된다.

$$
y_k=\{c_{k1},\dots,c_{kn_k}\}.
$$

기본 prior는 모델 logprob, reranker score, self-consistency frequency 등을 normalize한 것이다.

$$
\mu_0(y_k)\ge0,\qquad \sum_k\mu_0(y_k)=1.
$$

이 단계는 `PreEq`의 유한 후보공간이다. 후보 posterior 자체는 `Exact`이고, claim/evidence feature를 어떻게 만들지는 `Bridge/Open test`다.

## 2. Claim-Evidence 기준장

각 축의 score는 extractor와 evidence snapshot에 의존하므로, parser 오류와 label 불일치를 residual의 물리적 신호로 읽지 않는다.

claim $c_i$는 여러 검증 축을 가진 벡터로 읽는다.

$$
z_i=
\begin{bmatrix}
z_{\rm entail}\\
z_{\rm numeric}\\
z_{\rm temporal}\\
z_{\rm unit}\\
z_{\rm citation}\\
z_{\rm causal}
\end{bmatrix}.
$$

source $m$, axis $a$가 제공하는 기준값과 불확실성을

$$
b_{ima},\qquad \sigma_{ima}
$$

로 둔다. source reliability $q_m\in[\epsilon^2,1]$, source-axis weight $w_{ima}\ge0$가 있을 때 precision은

$$
p_{ima}
:=
\frac{w_{ima}q_m}{\sigma_{ima}^2+\epsilon_\sigma^2}.
$$

axis별 기준장은 precision-weighted mean이다.

$$
B_{ia}
:=
\frac{\sum_m p_{ima}b_{ima}}
{\sum_m p_{ima}}.
$$

source disagreement는

$$
d_{ia}^2
:=
\frac{\sum_m p_{ima}(b_{ima}-B_{ia})^2}
{\sum_m p_{ima}}.
$$

공분산은 최소 구현에서는 diagonal로 둔다.

$$
C_i
:=
\operatorname{diag}_a
\left(
\frac{1}{\sum_m p_{ima}}
+d_{ia}^2
+\epsilon_\sigma^2
\right).
$$

근거가 없는 축은 $\sum_m p_{ima}=0$이므로 $B_{ia}$를 만들지 않고 missing mask에 넣는다.

## 3. 무차원 잔차

잔차의 분자는 scale과 같은 차원을 가져야 하며 0 scale·강한 상관에서는 공분산 정규화가 필요하다.

claim residual은 반드시 무차원이어야 한다.

$$
e_i
:=
C_i^{-1/2}(z_i-B_i).
$$

local Mahalanobis energy는

$$
E_i
:=
e_i^\top e_i
=
(z_i-B_i)^\top C_i^{-1}(z_i-B_i).
$$

이는 count defect보다 우선한다. `supported/unsupported/contradicted` 카운트는 이 residual field를 만들 수 없을 때의 저차 proxy일 뿐이다.

CE 문서군의 공통 규칙에 따라 $\exp$, $\log$, fixed-point, probability core에 들어가는 모든 값은 무차원이어야 한다. 따라서 $\Delta$, $\phi_i$, $E_i$, $\beta\Delta$, $D_{\rm eff}\phi_i$는 모두 무차원량이어야 한다. latency, token count, embedding norm, physical scale 같은 값은 직접 exponent에 넣지 않고 별도 normalization 또는 Buckingham-Pi식 무차원화 뒤에만 쓴다.

## 4. Source 신뢰도와 독립성

source 신뢰도 갱신은 과거 검증의 요약이며 출처 간 복제·상관이 있으면 독립 표본처럼 세지 않는 경계가 필요하다.

source $m$의 누적 신뢰도는 최근 검증된 claim residual 평균으로 갱신한다.

$$
q_m
:=
\exp\left[
-D_{\rm eff}
\left(
\overline E_m
+\lambda_\tau\overline\tau_m^2
\right)
\right].
$$

같은 원문을 복사한 mirror source는 독립 근거로 세지 않는다. source family $g$별 precision mass는

$$
M_{ig}
:=
\sum_{m\in g}\sum_a
\frac{w_{ima}q_m}{\sigma_{ima}^2+\epsilon_\sigma^2}.
$$

Kish effective source count:

$$
N_{{\rm eff},i}
:=
\frac{\left(\sum_gM_{ig}\right)^2}
{\sum_gM_{ig}^2}.
$$

독립성 penalty:

$$
P_{{\rm ind},i}
:=
\left[
\max\left(
0,\frac{N_\star-N_{{\rm eff},i}}{N_\star}
\right)
\right]^2,
\qquad N_\star=2.
$$

## 5. Signed Claim Graph

signed edge는 지지와 모순의 모델 선택이며 잘못된 edge의 전파는 edge ablation으로 반증해야 한다.

claim graph를

$$
G_C=(V_C,E_C,A,R)
$$

로 둔다. $A_{ij}\ge0$는 연결 강도, $R_{ij}\in\{+1,-1\}$는 지지/모순 방향이다.

claim local reliability는

$$
q_i
:=
\exp\left[
-D_{\rm eff}
\left(
\alpha E_i
+(1-\alpha)\tau_i^2
+P_{{\rm missing},i}
\right)
\right].
$$

source reliability까지 결합한 graph curvature:

$$
\Delta_Ge_i
:=
\sum_j A_{ij}q_jq_{{\rm src}(j)}
\left(R_{ij}e_j-e_i\right).
$$

이 항은 잘못된 claim이 이웃 claim을 오염시키는 것을 막기 위해 $q_jq_{{\rm src}(j)}$로 약화된다.

## 6. Claim Action

local action은 선택한 축·가중치의 penalty이며 낮은 값이 근거 자체의 완전성을 보증하지 않는다.

claim $i$의 local action은

$$
\phi_i
:=
\lambda_E E_i
+\lambda_G\|\Delta_Ge_i\|^2
+\lambda_\tau\tau_i^2
+\lambda_q(1-\bar q_{{\rm src},i})^2
+\lambda_{\rm ind}P_{{\rm ind},i}
+P_{{\rm missing},i}
+P_{{\rm instruction},i}
+P_{{\rm schema},i}.
$$

문서 `09_무차원_잔차장_환각억제.md`의 현재 local action에 맞춘 초기값은

$$
\lambda_E=0.10,\quad
\lambda_G=1.20,\quad
\lambda_\tau=0.10,\quad
\lambda_q=1.50,\quad
\lambda_{\rm ind}=0.80.
$$

단, 이 숫자는 `Selection/Open test`다. 실제 값은 calibration set에서 ECE/Brier/false accept rate로 조정해야 한다.

claim accept score:

$$
P_{{\rm accept},i}
:=
\exp(-D_{\rm eff}\phi_i).
$$

hard gate:

$$
N_{{\rm eff},i}\ge1.60.
$$

기본 판정:

| 조건 | 동작 |
|---|---|
| $P_{{\rm accept},i}\ge S_{\rm accept}$, $\|e_i\|\le e_{\max}$, $\|\Delta_Ge_i\|\le g_{\max}$, $N_{{\rm eff},i}\ge1.60$ | claim accept |
| missing evidence | retrieve / abstain |
| $P_{{\rm accept},i}\le S_{\rm reject}$ 또는 residual 과대 | reject |
| 중간 영역 | 추가 검색, 계산기 호출, human review |

## 7. Answer Action

answer action은 claim action을 묶는 집계 규칙이므로 긴 답변과 짧은 답변의 길이 보정·coverage를 함께 검사해야 한다.

답변 후보 $y_k$의 action은 accepted/rejected claim action을 묶은 값이다.

$$
\Delta(y_k,E)
:=
\frac{1}{n_k}
\sum_{i\in y_k}
\phi_i
+\lambda_{\rm coh}
\sum_{(i,j)\in y_k}
A_{ij}\|R_{ij}e_j-e_i\|^2
+\lambda_{\rm cov}P_{\rm coverage}(y_k)
+\lambda_{\rm abst}P_{\rm unsupported}(y_k).
$$

coverage penalty는 질문이 요구한 핵심 슬롯 중 evidence-backed accepted claim으로 덮이지 않은 비율이다.

$$
P_{\rm coverage}(y_k)
:=
1-\frac{\#\{\text{required slots covered by accepted claims}\}}
{\#\{\text{required slots}\}}.
$$

unsupported penalty는 답변에 남은 rejected/review claim 질량이다.

$$
P_{\rm unsupported}(y_k)
:=
\frac{\#\{\text{claims not accepted}\}}{n_k}.
$$

## 8. PreEq Posterior

Gibbs 재가중은 유한 후보와 온도 파라미터라는 근사 아래의 posterior proxy이며 calibrated probability는 별도 검증 대상이다.

답변 후보 posterior는 유한 후보공간 Gibbs 재가중이다.

$$
p_\beta(y_k\mid E)
:=
\frac{
\mu_0(y_k)\exp[-\beta\Delta(y_k,E)]
}{
\sum_{\ell}
\mu_0(y_\ell)\exp[-\beta\Delta(y_\ell,E)]
}.
$$

$\beta=0$이면 prior-only, $\beta\to\infty$이면 최소 answer action 후보로 농축한다. finite $\beta$에서는 posterior MAP를 선택한다.

이 posterior의 정규화와 유한 후보공간 농축은 `Exact`다. 그러나 defect $\Delta(y,E)$의 구체 선택, $\beta$, $p_{\min}$, $g_{\min}$, $\Delta_{\max}$, $r_{\rm accept}$는 `Selection`이며 benchmark로 calibration해야 한다. manifest는 "참"이 아니라 "정의된 defect에서 최소인 후보"다.

manifest 후보:

$$
y_\star
:=
\operatorname*{argmax}_{y_k\in A_x}
p_\beta(y_k\mid E).
$$

단, 다음 조건을 모두 통과해야 출력한다.

$$
\max_k p_\beta(y_k\mid E)\ge p_{\min},
$$

$$
\Delta_{(2)}-\Delta_{(1)}\ge g_{\min},
$$

$$
\Delta(y_\star,E)\le \Delta_{\max},
$$

$$
\frac{1}{n_\star}
\sum_{i\in y_\star}
\mathbf 1[\text{claim }i\text{ accepted}]
\ge r_{\rm accept}.
$$

실패하면 답을 만들지 않고 retrieve / abstain / ask-clarification으로 간다.

## 9. 출력 복원

mask 복원은 자유 생성 억제를 목표로 하지만, evidence 누락으로 인한 유용한 claim의 오탐도 human review로 측정해야 한다.

최종 답변은 자유 생성물이 아니다. accepted claim mask를 통과한 claim만 복원한다.

$$
\widehat y_\star
:=
\operatorname{Compose}
\left(
\{c_i\in y_\star:\operatorname{Accept}(c_i)=1\}
\right).
$$

원래 문서의 잔차장 복원식으로 쓰면

$$
\widehat Y
=
B+\Sigma\widehat e
$$

인데, verifier에서는 $\widehat e$가 gate를 통과한 claim에 대해서만 문장화된다.

## 10. 구현 우선순위

현재 kernel은 수식 전체의 v0 proxy이므로 구현 순서는 검증 가능한 축부터 확장해야 한다. 각 단계는 정확도뿐 아니라 오탐·미탐·계산비용의 holdout 평가를 요구한다.

현재 Rust/Python `llm_pre_eq` count kernel은 다음의 v0 proxy다.

$$
C_{\rm supported}, C_{\rm unsupported}, C_{\rm contradicted}
\quad\leadsto\quad
\Delta_{\rm count}.
$$

v2 구현은 아래 순서로 바꾼다.

1. `ClaimAxisEvidence`: axis별 $b_{ima},\sigma_{ima},w_{ima},q_m,g(m)$.
2. `ClaimResidual`: $B_i,C_i,e_i,E_i,\tau_i,N_{{\rm eff},i}$.
3. `ClaimGraph`: $A_{ij},R_{ij}$, $\Delta_Ge_i$.
4. Rust kernel: batched $E_i,\Delta_Ge_i,\phi_i,\Delta(y_k),p_\beta$.
5. Python policy: retrieve/review/abstain/compose.
6. Eval: negative controls, shuffled graph, source holdout, topic holdout, time holdout, ECE, Brier, false accept.

## 11. 주장 등급

아래 등급은 수학 정리, 구현 산출, 경험 성능과 미완성을 분리한다. 등급표는 claim의 사실성 자체가 아니라 이 문서가 제공하는 근거의 층위를 기록한다.

| 항목 | 등급 |
|---|---|
| 유한 후보 posterior $p_\beta$ | `Exact` |
| residual/metric defect 인코딩 | `Exact` |
| LLM claim vector $z_i$ 추출 | `Bridge/Open test` |
| source reliability update | `Selection/Open test` |
| graph curvature가 false accept를 줄인다는 주장 | `Open test` |
| SOTA 초과 | 금지. 공개 benchmark와 동일 base model, 동일 retrieval budget, confidence interval 전까지 주장 금지 |

금지 문장:

| 금지 | 이유 |
|---|---|
| synthetic sweep 통과 = production hallucination 감소 | synthetic은 regression일 뿐이다 |
| curvature risk 감소 = factual hallucination hard bound | curvature는 내부 안정 proxy다 |
| PreEq posterior = 진리 판정 | posterior는 defect 기반 선택이다 |
| count defect = 최종 CE verifier | count는 v0 proxy다 |
| $\phi=\Phi$ | residual readout field와 CE physical fold field는 다르다 |
| SOTA 초과 | 공개 benchmark, 강한 baseline, 비용/latency, abstain split, multi-seed 전까지 금지 |

강한 성능 주장을 하려면 TruthfulQA, FEVER, HotpotQA 또는 실제 RAG 로그에서 다음을 모두 보고해야 한다.

1. 같은 base LLM, 같은 retrieval budget, 같은 candidate count.
2. greedy, self-consistency, RAG reranker, verifier reranker baseline.
3. answer rate와 accuracy-on-answered 분리.
4. hallucination/false accept, abstention precision, ECE, Brier.
5. source holdout, topic holdout, time holdout.
6. shuffled graph, axis ablation, negative control.
7. latency, token/call cost.
8. 최소 3 seed 또는 bootstrap confidence interval.
9. 논문/leaderboard의 label policy, 평가 단위(response/span/claim/sample), split, prompt를 그대로 명시.

### 11.1 SOTA 문헌 앵커

외부 문헌은 label policy와 평가 단위가 고정된 기준점으로만 비교한다. 서로 다른 task·dataset의 숫자를 섞으면 성능 갭의 의미가 사라진다.

현재 외부 비교는 아래 문헌을 기준점으로만 둔다. 숫자는 서로 다른 label policy와 평가 단위를 섞으면 안 된다.

| benchmark / paper | 평가 단위 | 핵심 관찰 | CE 해석 |
|---|---:|---|---|
| RAGTruth (Niu et al., ACL 2024, <https://aclanthology.org/2024.acl-long.585/>) | response-level + span-level | 약 17,790개 RAG 응답, word/span annotation. GPT-4-turbo prompt response-level F1 `63.4`, fine-tuned Llama-2-13B response-level F1 `78.7`; span-level은 훨씬 낮다. | `~0.63` baseline과 `~0.79` strong detector의 근거. CE 수치는 RAGTruth response-level과만 직접 비교한다. |
| FaithBench (Bao et al., NAACL 2025, <https://aclanthology.org/2025.naacl-short.38/>) | challenging summarization sample/span | detector들이 서로 disagree한 750개 challenging summary. best sample-level BA `62.31`, F1-macro `57.06`; gray label(`questionable`, `benign`) 때문에 policy 의존성이 크다. | FaithBench F1만으로 SOTA/비SOTA를 말하지 않는다. BA, AUROC, label policy를 함께 본다. |
| FaithJudge / evolving leaderboard (Tamber et al., 2025, <https://arxiv.org/abs/2505.04847>) | claim-wise / summary-wise | FaithBench, AggreFact, RAGTruth-Summ, TofuEval-MB를 묶어 비교. zero-shot GPT-4o/o3-mini와 fine-tuned detector 모두 평균 F1-macro가 대략 `0.7`대에 머문다. FaithJudge prompting은 FaithBench에서 더 높은 agreement를 보인다. | CE가 SOTA를 주장하려면 response-level hash detector가 아니라 claim-wise/example-conditioned judge와도 비교해야 한다. |
| FEVER / TruthfulQA / HaluEval / FactScore | fact verification / truthfulness / factuality | 서로 다른 문제 정의를 가진 benchmark anchor다. | CE verifier의 일반화 검증용이지, RAGTruth/FaithBench 숫자와 직접 합산하지 않는다. |

따라서 `SOTA ~0.79-0.84 F1` 같은 문장은 반드시 “RAGTruth response-level 또는 특정 leaderboard setting”으로 제한한다. FaithBench는 deliberately hard subset이므로, 낮은 BA가 곧 CE만의 실패가 아니라 detector field 전체의 어려움도 반영한다.

## 12. 필수 불변조건

v2는 점수 향상보다 먼저 아래 불변조건을 회귀 테스트로 보존해야 한다. 한 조건의 실패는 높은 평균 benchmark보다 우선하는 반증 신호다.

v2 구현은 아래 불변조건을 반드시 테스트해야 한다.

| invariant | 의미 |
|---|---|
| dimensionless exponent | $\exp(-D_{\rm eff}\phi)$, $\exp(-\beta\Delta)$ 인자는 무차원 |
| posterior normalization | $\sum_k p_\beta(y_k\mid E)=1$ |
| $\beta=0$ prior limit | posterior가 $\mu_0$로 복귀 |
| large-$\beta$ concentration | 충분한 gap에서 최소 $\Delta$ 후보로 농축 |
| claim order invariance | claim 순서가 바뀌어도 action 불변 |
| evidence/source order invariance | source 순서가 바뀌어도 $B,C,e$ 불변 |
| duplicate source no quorum | 같은 source family 복제는 $N_{\rm eff}$를 올리지 못함 |
| missing evidence rejection | 근거 없는 claim은 accept 불가 |
| low-reliability poisoning blocked | 낮은 $q_m$ source가 거짓 claim을 quorum으로 밀 수 없음 |
| trusted contradiction blocked | 신뢰 source와 모순되는 claim은 curvature/action으로 억제 |
| Rust/numpy parity | native kernel과 fallback 결과 일치 |

## 13. CE 문서 claim 전용 축

CE 문서는 형식 지위와 의존 관계가 있으므로 일반 entailment 외의 전용 축이 필요하다. 이 축은 원장을 수정하지 않고 문서 claim의 태그·출처 불일치를 검출하는 verifier 입력이다.

`paper/1_강의`, `paper/2_경로적분과_응용`, `paper/3_상수`, `paper/7_AGI`, `paper/8_리만`까지 포함하면 verifier는 일반 claim만이 아니라 **CE 문서 내부 claim tier**도 검사해야 한다.

CE claim을 검증할 때는 claim vector $z_i$에 아래 축을 추가한다.

| axis | 검사 내용 | 실패 시 |
|---|---|---|
| dimensionless | exponent/log/fixed-point/probability core에 차원량이 직접 들어갔는가 | hard reject 또는 schema penalty |
| tier | `Exact`, `Selection`, `Bridge`, `Phenomenology`, `Open`, `Open test`가 명시됐는가 | missing-tier penalty |
| branch | $W_0/W_{-1}$, $d=0/d=3$, readout branch가 명시됐는가 | branch ambiguity penalty |
| bridge | A3b, $\Phi\leftrightarrow R$, $P_{\rm survive}\leftrightarrow\Omega_b$, portal/readout 식별을 Exact처럼 말하는가 | bridge-overclaim penalty |
| transition | $\tau_*<1$, freeze-time, C-grade readout이 필요한 관측량을 endpoint 값으로 말하는가 | transition-readout penalty |
| source manifest | covariance/Fisher/source-role/channel manifest 없이 $H_0$, source role, branch selector를 주장하는가 | provenance penalty |
| substrate | transformer에서 falsified된 $\varepsilon^2$ sparsity나 MRA OOD 주장을 일반 법칙처럼 말하는가 | substrate-overclaim penalty |
| symbol firewall | residual $\phi_i$와 CE physical fold field $\Phi$를 동일시하는가 | hard reject |

특히 다음 문장은 reject 또는 review로 내려야 한다.

| claim pattern | 이유 |
|---|---|
| "CE는 전체 우주론을 Exact로 증명했다" | $\Omega_\Lambda,\Omega_{\rm DM},H_0,A_s$ 등은 Bridge/Phenomenology/Open test 층을 포함 |
| "CE는 자유 파라미터가 0개다" | 정본은 단일 결합 입력, branch/selection, bridge 식별 이후 추가 fit knob가 없다는 의미 |
| "A3b가 Exact다" | fixed-point math는 Selection, $\Omega_b$ 식별은 Bridge |
| "고정점 endpoint로 $A_s,\eta,T_{\rm CMB}$가 바로 나온다" | transition interval $\tau_*$ 또는 residual-drive readout이 필요 |
| "MRA/Riemann PE가 OOD를 해결한다" | 문서상 MRA는 32x length OOD에서 Tier 2, ALiBi/Euler e-decay 쪽이 Tier 1 |
| "곡률 위험 감소가 factual hallucination hard bound다" | curvature는 risk proxy, factual truth gate가 아님 |

CE 문서 claim의 answer action에는 tier 위반 penalty를 더한다.

$$
\Delta_{\rm CE}(y_k,E)
:=
\Delta(y_k,E)
+\lambda_{\rm tier}P_{\rm tier}
+\lambda_{\rm bridge}P_{\rm bridge}
+\lambda_{\rm branch}P_{\rm branch}
+\lambda_{\rm transition}P_{\rm transition}
+\lambda_{\rm provenance}P_{\rm provenance}.
$$

여기서 $P_{\rm bridge}$는 Bridge/Phenomenology/Open test claim을 Exact 문장으로 올릴 때 커지고, $P_{\rm provenance}$는 source manifest, covariance role map, benchmark holdout이 없을 때 커진다.

## 14. Closed-loop / OOD 제약

verifier score가 다음 생성·검색 상태에 실제로 들어가지 않으면 closed loop라 부를 수 없다. OOD 제약은 distribution shift에서 false-allow가 증가하는지 독립 negative control로 확인해야 한다.

verifier는 점수판으로 끝나면 안 된다. `paper/7_AGI` 기준으로 critique와 residual은 다음 step의 energy 또는 state에 실제로 들어가야 한다.

$$
\frac{\partial S_{t+1}}{\partial c_{t+1}}\ne0,
\qquad
\frac{\partial E_{t+1}}{\partial c_{t+1}}\ne0.
$$

closed-loop 안정성은

$$
\widehat\rho_t
:=
\frac{\|S_{t+1}-S_t\|}
{\|S_t-S_{t-1}\|+\epsilon}
$$

와 OOD 평균 로그 수축률

$$
\mathbb E_{e_t\sim\mathcal D_{\rm out}}
\left[
\log
\frac{\|S_{t+1}-S_t\|}
{\|S_t-S_{t-1}\|+\epsilon}
\right]
<0
$$

로 측정한다. 즉 verifier가 reject/retrieve/abstain을 냈다면 그 정보는 다음 retrieval, candidate generation, graph repair, source reliability update에 반영되어야 한다.

token/latent curvature gate는 보조 risk proxy다.

$$
\kappa_{\rm combined}
=
w_1\kappa_1+w_2\kappa_2+w_3\kappa_{\rm LBO},
\qquad
\operatorname{CurvatureRiskScore}
=
\frac1T\sum_t\mathbf1[\kappa_{\rm combined}(t)>\kappa_{\rm th}].
$$

이 값은 factual truth가 아니라 internal instability signal이다. claim residual gate가 외부 사실성, curvature gate가 내부 안정성을 맡는다.

## 15. 구현 기준점

기준점은 현재 코드와 데이터 snapshot에서 재현 가능한 최소 동작을 뜻한다. 구현이 바뀌면 checksum·seed·label policy와 함께 다시 측정해야 한다.

현재 v2 구현 기준점은 다음이다.

- Python API: `reality_stone/python/reality_stone/clarus/llm_pre_eq.py`
  - `ClaimAxisEvidence`
  - `ResidualClaim`
  - `ResidualAnswerCandidate`
  - `ClaimResidualVerifier`
- Rust kernel: `reality_stone/python/reality_stone/clarus/core/src/engine/llm_pre_eq.rs`
  - `claim_answer_actions`
  - `gibbs_posterior`
- PyO3 binding: `nn_llm_claim_pre_eq_fwd`
- 실행 예제: `python examples/pre_eq/claim_residual_verifier.py`
- 외부 JSONL benchmark adapter: `python examples/pre_eq/claim_residual_benchmark.py <benchmark.jsonl>`
- 회귀 검증: `pytest tests/test_llm_pre_eq_verifier.py`

이 구현은 이전 count-only proxy를 제거한 것이 아니라 호환층으로 남기고, v2 검증 경로는 claim residual action을 정본으로 둔다.

## 16. 성능 갭 체크

성능 갭은 알려진 부족과 조치의 결과를 같은 metric에서 비교하는 표다. 조치 뒤의 개선은 독립 holdout과 paired ablation이 없으면 인과적 효과로 단정할 수 없다.

현재 확인된 부족 지점과 조치 결과는 다음이다.

- prior dominance:
  - 기존 기본값 `beta=2.0`에서는 높은 prior의 약한 후보가 posterior를 빼앗는 케이스가 있었다.
  - 기본값을 `beta=4.0`으로 올려 verifier action이 posterior에 더 강하게 반영되도록 조정했다.
- residual gate:
  - 기존 `max_residual_norm=2.5`는 partial/weak claim을 너무 쉽게 accepted claim으로 통과시켰다.
  - 기본값을 `max_residual_norm=0.5`로 낮춰 부분 정합 claim을 정답 claim과 분리한다.
- manifest action gate:
  - 기존 `max_action=3.0`은 고결함 후보를 후보군에 오래 남겼다.
  - 기본값을 `max_action=1.0`으로 낮춰 reject/abstain gate를 강화했다.
- graph escape:
  - graph 검증에서는 single-claim escape가 question-required slot을 회피하면 graph conflict를 우회할 수 있다.
  - sweep harness에서 모든 후보가 같은 required-slot 기준을 받도록 고정했다.
- source quorum:
  - single-source 후보는 `N_eff`와 independence penalty로 걸러진다.

내부 synthetic failure-mode sweep 기준:

```text
mode         exact_accuracy  answer_rate  hallucination_on_answered
adversarial 1.000000        1.000000     0.000000
noisy       1.000000        1.000000     0.000000
partial     1.000000        1.000000     0.000000
source      1.000000        1.000000     0.000000
graph       1.000000        1.000000     0.000000
missing     1.000000        1.000000     0.000000
```

이 숫자는 SOTA 주장이 아니다. 이는 `prior`, `residual`, `partial`, `source`, `graph`, `missing evidence`로 만든 내부 실패 모드가 현재 기본 설정에서 닫혔다는 회귀 검증이다.

외부 SOTA와의 정확한 차이는 아직 `RAGTruth`, `RAGTruth++`, `FaithBench`, `FEVER`, `TruthfulQA` 같은 실제 benchmark 파일을 넣어야 확정된다. 현재 adapter는 다음 JSONL 필드를 자동 인식한다.

- answer: `answer`, `response`, `output`, `summary`, `generation`, `model_output`
- context: `context`, `contexts`, `evidence`, `reference`, `documents`, `sources`
- label: `is_hallucinated`, `hallucinated`, `label`, `factuality`, `faithfulness`

실행:

```bash
python examples/pre_eq/claim_residual_benchmark.py benchmark.jsonl
```

validation split에서 threshold를 맞출 때는:

```bash
python examples/pre_eq/claim_residual_benchmark.py benchmark.validation.jsonl --calibrate
```

오답 분석 CSV를 남길 때는:

```bash
python examples/pre_eq/claim_residual_benchmark.py benchmark.jsonl --calibrate --export-errors errors.csv
```

로컬 synthetic failure-mode 전체와 JSONL 폴더 전체를 한 번에 돌릴 때는:

```bash
python examples/pre_eq/claim_residual_run_all.py --cases 1000 --jsonl-dir benchmarks/
```

`--jsonl-dir`가 없으면 내부 synthetic 6모드만 측정한다.

현재 로컬 synthetic run-all 결과:

```text
internal_strength strong-internal
external_strength unmeasured

synthetic_mode exact_accuracy answer_rate hallucination_rate
adversarial    1.000000       1.000000    0.000000
noisy          1.000000       1.000000    0.000000
partial        1.000000       1.000000    0.000000
source         1.000000       1.000000    0.000000
graph          1.000000       1.000000    0.000000
missing        1.000000       1.000000    0.000000
```

해석:

- `strong-internal`: 우리가 명시적으로 만든 prior trap, noisy residual, partial answer, source quorum, graph conflict, missing evidence failure-mode는 현재 기본값에서 닫혀 있다.
- `external_strength unmeasured`: 실제 RAGTruth/FaithBench/FEVER류 JSONL을 아직 넣지 않았으므로 외부 SOTA 대비 강도는 확정하지 않는다.
- 작은 fixture JSONL에서 `sota-competitive`가 떠도 이는 실행 경로 검증일 뿐, 외부 benchmark 성능 주장이 아니다.

실제 다운로드 benchmark run-all 결과:

```text
internal_strength strong-internal
external_strength baseline-plus

benchmark                  n     F1       BA       AUROC    AUPRC
RAGTruth test ctx2000      2675  0.6135   0.6881   0.7420   0.5822
FaithBench all             750   0.8217   0.5442   0.5986   0.7400
```

RAGTruth train subset으로 model/task prior를 추가한 supervised residual detector 결과:

```text
train_examples 1000
test_examples 2675
RAGTruth supervised F1 0.6573
RAGTruth supervised BA 0.7265
precision 0.5509
recall 0.8144
```

전체 RAGTruth train split과 model-task interaction prior를 쓴 최종 supervised residual detector 결과:

```text
train_examples 14942
test_examples 2675
RAGTruth supervised F1 0.6750
RAGTruth supervised BA 0.7448
precision 0.5840
recall 0.7996
```

이는 기존 lexical response-level보다 F1 `+6.15%p`, BA `+5.67%p` 개선이다. 그래도 RAGTruth 원 논문의 response-level fine-tuned Llama-2-13B 기준 `0.787 F1`, 그리고 이후 faithfulness leaderboard의 강한 detector/judge 계열과 비교하면 약 `11-16%p` 부족하다.

RAGTruth train split으로 sparse hashed detector를 추가한 결과:

```text
fit_examples 11953
validation_examples 2989
test_examples 2675
RAGTruth hashed F1 0.6968
RAGTruth hashed BA 0.7646
precision 0.5980
recall 0.8346
```

이는 기존 lexical response-level보다 F1 `+8.33%p`, BA `+7.65%p` 개선이다. RAGTruth response-level strong baseline `0.787 F1`까지는 약 `9%p` 남았고, FaithJudge류 example-conditioned judge의 setting까지 포함하면 gap은 label policy에 따라 더 커지거나 작아진다.

RAGTruth span annotation을 보존한 뒤 claim/span-supervised detector와 hash+claim ensemble도 검증했다.

```text
RAGTruth claim/span detector F1 0.6829
RAGTruth claim/span detector BA 0.7520
RAGTruth hash+claim ensemble F1 0.6909
RAGTruth hash+claim ensemble BA 0.7589
```

두 경로 모두 현재 최고점인 sparse hashed detector `F1 0.6968`, `BA 0.7646`보다 낮다. 따라서 현 단계에서 span annotation 보존과 claim-level 학습 경로는 구현됐지만, SOTA gap을 닫는 직접 개선은 아직 아니다.

Transformers 기반 external NLI scorer도 추가하고 `cross-encoder/nli-deberta-v3-xsmall`로 RAGTruth test를 재평가했다.

```text
RAGTruth xsmall NLI top-1 evidence F1 0.5544
RAGTruth xsmall NLI top-1 evidence BA 0.5836
RAGTruth xsmall NLI top-5 evidence F1 0.5576
RAGTruth xsmall NLI top-5 evidence BA 0.5825
```

이는 NLI 모델 자체가 무의미하다는 뜻이 아니라, response 전체를 하나의 hypothesis로 두고 CE action에 직접 주입하는 방식이 약하다는 뜻이다. 다음 개선은 NLI 점수를 단독 decision으로 쓰지 말고 hash detector의 learned feature로 넣거나, sentence/claim 단위로 쪼갠 뒤 token/span classifier와 결합해야 한다.

해석:

- RAGTruth response-level 기준으로 GPT-4-turbo prompt baseline `~0.63 F1` 근처까지는 왔다.
- RAGTruth 원 논문의 GPT-4-turbo prompt baseline은 response-level F1 `0.634`, fine-tuned Llama-2-13B는 response-level F1 `0.787`이다. hashed detector 기준 아직 약 `9%p` 부족하다.
- FaithBench는 binary F1은 높지만 challenging subset, gray label policy, positive class 비율, high-recall bias 영향이 크다. balanced accuracy `0.5442`, AUROC `0.5986`이 실제 변별력을 더 잘 보여준다.
- 현재 외부 결과는 `baseline-plus`이며, SOTA 주장은 금지한다.
- 상세 오답 분석 보고서: `benchmarks/reports/claim_residual_error_analysis.md`
- supervised lightweight detector: `examples/pre_eq/train_ragtruth_supervised_detector.py`
- saved supervised model: `benchmarks/converted/ragtruth/supervised_residual_model.json`
- sparse hashed detector: `examples/pre_eq/train_ragtruth_hash_detector.py`
- saved hashed model: `benchmarks/converted/ragtruth/hashed_detector_model.json`
- claim/span supervised detector: `examples/pre_eq/train_ragtruth_claim_detector.py`
- hash+claim ensemble scorer: `examples/pre_eq/score_ragtruth_ensemble_detector.py`
- FaithBench `unwanted-only` policy도 실험했지만 F1 `0.7494`, BA `0.5340`으로 기존 `unwanted-or-questionable` policy보다 외부 변별력이 좋아지지 않았다.

상세 원인:

- RAGTruth false negative는 lexical-near hallucination이다. 응답 대부분은 원문과 겹치지만 날짜, 수치, 부정, 범위, 인과, 작은 unsupported clause가 바뀐다.
- RAGTruth false positive는 faithful paraphrase다. 특히 `Summary`, `Data2txt`에서 정상 요약이 원문 단어와 충분히 겹치지 않아 action이 커진다.
- FaithBench false positive가 216건으로 지배적이다. positive-heavy label 구성 때문에 F1은 높지만 balanced accuracy가 낮다.
- FaithBench false negative 4건은 `budget` vs `production budget`, entity conflation처럼 lexical overlap으로 거의 구분되지 않는 의미 차이다.
- 따라서 현재 SOTA gap의 직접 원인은 CE posterior가 아니라 evidence axis다. lexical support axis를 NLI/contradiction/retrieval/span-localization axis로 바꿔야 한다.

수식상 재정의 대상은 posterior가 아니라 claim evidence vector다.

$$
z_i
:=
\big[
z_i^{\rm lex},
z_i^{\rm entail},
z_i^{\rm contra},
z_i^{\rm span},
z_i^{\rm attrib},
z_i^{\rm temporal},
z_i^{\rm src}
\big],
\qquad
e_i=C_i^{-1/2}(z_i-B_i).
$$

여기서 $z_i^{\rm lex}$는 lexical overlap, $z_i^{\rm entail}$과 $z_i^{\rm contra}$는 claim별 NLI/entailment score, $z_i^{\rm span}$은 RAGTruth/FaithBench류 span-localization 신호, $z_i^{\rm attrib}$은 근거 passage/claim attribution, $z_i^{\rm temporal}$은 날짜·순서·업데이트 충돌, $z_i^{\rm src}$는 source independence와 reliability를 뜻한다. RAGTruth의 subtle conflict와 FaithBench의 unwanted/questionable case는 주로 $z_i^{\rm entail}$, $z_i^{\rm contra}$, $z_i^{\rm span}$, $z_i^{\rm temporal}$ 축이 없을 때 false negative가 된다.

claim action은 그대로 유지하되, 축별 covariance와 benchmark policy를 분리한다.

$$
C_i
=
\operatorname{diag}
\left(
\sigma_{\rm lex}^2,
\sigma_{\rm entail}^2,
\sigma_{\rm contra}^2,
\sigma_{\rm span}^2,
\sigma_{\rm attrib}^2,
\sigma_{\rm temporal}^2,
\sigma_{\rm src}^2
\right)
+C_i^{\rm corr}.
$$

즉 다음 개선은 $\Delta(y_k,E)$를 새로 만드는 것이 아니라, 같은 posterior 안에 들어가는 $z_i,B_i,C_i$를 benchmark label policy에 맞게 학습/보정하는 것이다.

실행 명령:

```bash
python examples/pre_eq/convert_ragtruth.py \
  --response benchmarks/raw/ragtruth/response.jsonl \
  --source-info benchmarks/raw/ragtruth/source_info.jsonl \
  --output-dir benchmarks/converted/ragtruth

python examples/pre_eq/convert_faithbench.py \
  --input-dir benchmarks/raw/FaithBench-main/data_for_release \
  --output benchmarks/converted/faithbench/faithbench_all.jsonl

python examples/pre_eq/claim_residual_run_all.py \
  --cases 1000 \
  --jsonl-dir benchmarks/eval \
  --fast-lexical \
  --response-level \
  --accepted-fraction-threshold 0.0 \
  --max-context-chars 2000

python examples/pre_eq/train_ragtruth_supervised_detector.py \
  --train benchmarks/converted/ragtruth/ragtruth_train_ctx2000.jsonl \
  --test benchmarks/converted/ragtruth/ragtruth_test_ctx2000.jsonl \
  --trials 5000 \
  --validation-fraction 0.2 \
  --output-model benchmarks/converted/ragtruth/supervised_residual_model.json

python examples/pre_eq/train_ragtruth_hash_detector.py \
  --train benchmarks/converted/ragtruth/ragtruth_train_ctx2000.jsonl \
  --test benchmarks/converted/ragtruth/ragtruth_test_ctx2000.jsonl \
  --epochs 8 \
  --lr 0.35 \
  --output-model benchmarks/converted/ragtruth/hashed_detector_model.json

python examples/pre_eq/train_ragtruth_claim_detector.py \
  --train benchmarks/converted/ragtruth/ragtruth_train_ctx2000.jsonl \
  --test benchmarks/converted/ragtruth/ragtruth_test_ctx2000.jsonl \
  --epochs 6 \
  --lr 0.25 \
  --output-model benchmarks/converted/ragtruth/claim_span_detector_model.json

python examples/pre_eq/score_ragtruth_ensemble_detector.py \
  --train benchmarks/converted/ragtruth/ragtruth_train_ctx2000.jsonl \
  --test benchmarks/converted/ragtruth/ragtruth_test_ctx2000.jsonl \
  --hash-model benchmarks/converted/ragtruth/hashed_detector_model.json \
  --claim-model benchmarks/converted/ragtruth/claim_span_detector_model.json

python examples/pre_eq/claim_residual_benchmark.py \
  benchmarks/eval/ragtruth_test_ctx2000.jsonl \
  --response-level \
  --max-context-chars 2000 \
  --nli-pair-top-k 5 \
  --export-nli-pairs benchmarks/reports/ragtruth_nli_pairs.jsonl

# Transformers scorer or external NLI runner should fill:
# {"record_id": "...", "claim_index": 0, "entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}
python examples/pre_eq/score_nli_pairs_transformers.py \
  --pairs benchmarks/reports/ragtruth_nli_pairs.jsonl \
  --output benchmarks/reports/ragtruth_nli_scores.jsonl \
  --model cross-encoder/nli-deberta-v3-xsmall \
  --batch-size 16

python examples/pre_eq/claim_residual_benchmark.py \
  benchmarks/eval/ragtruth_test_ctx2000.jsonl \
  --response-level \
  --nli-evidence \
  --nli-scores-jsonl benchmarks/reports/ragtruth_nli_scores.jsonl \
  --calibrate \
  --accepted-fraction-threshold 0.0 \
  --max-context-chars 2000
```

현재 출력 metric:

- `accuracy`
- `balanced_accuracy`
- `precision`
- `recall`
- `f1`
- `auroc` / `auprc` when `--calibrate` is used
- `tp`, `fp`, `tn`, `fn`

현재 adapter는 투명한 lexical/heuristic evidence mapper다. 엔티티 스왑, unsupported claim, claim-level diagnostics export, RAGTruth span label 보존, claim/span-supervised detector smoke test, external NLI pair export/score injection test는 통과하지만, SOTA와 붙으려면 다음이 남아 있다.

- learned retrieval/reranking으로 claim별 evidence source를 구성
- 실제 NLI 또는 entailment model로 `ragtruth_nli_scores.jsonl`을 채워 contradiction axis를 교체
- true span supervision을 단순 claim-overlap이 아니라 token/span classifier로 학습
- benchmark별 validation split에서 `action_threshold`, `accepted_fraction_threshold`, `sigma` calibration
- RAGTruth++처럼 재주석된 label 기준으로 false-negative annotation 문제 보정

## 17. 한 줄 결론

결론은 count penalty 대신 evidence-bounded residual action을 쓰자는 조건부 설계 제안이다. 실제 안전성은 구현·manifest 품질·OOD holdout에서 반증 가능하게 검증되어야 한다.

LLM hallucination verifier의 CE 수식은 count penalty가 아니라

$$
\boxed{
\text{dimensionless claim residual}
\;\to\;
\text{source-weighted graph action}
\;\to\;
\text{PreEq Gibbs posterior}
\;\to\;
\text{accepted-claim-only composition}
}
$$

이다.
