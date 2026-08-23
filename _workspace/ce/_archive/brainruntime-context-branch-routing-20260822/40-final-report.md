# 두 동시 입력 가운데 하나를 고르는 문맥 회랑

Status: COMPLETE

이번 실험은 TR2에서 남은 가장 큰 반례를 직접 제거했다. TR2에서는
`WRONG_CONTEXT`도 `16/16`으로 성공했기 때문에, 좋은 결과가 문맥에 맞는
route에서 나온 것인지 단순한 block sparsity에서 나온 것인지 구분할 수
없었다. 이번에는 서로 다른 두 payload를 동시에 입력하고, 문맥은
뉴런 상태나 decoder가 아니라 entry mask에만 주었다. 따라서 올바른
branch를 열지 않으면 다른 payload가 실제로 출력된다.

## 식과 모델 지위

[정의] 상태 공간은 source 두 개, hidden 두 개, 공통 output 하나의 다섯
block으로 나뉜다.

$$
\mathbb R^{5m}=S_0\oplus S_1\oplus H_0\oplus H_1\oplus Y,
\qquad m=4.
$$

[모델 선택] 허용한 학습 support는 두 entry branch와 두 공통 output
map뿐이다.

$$
U=(H_0\times S_0)\cup(H_1\times S_1)
\cup(Y\times H_0)\cup(Y\times H_1).
$$

경험 중에는 각 문맥과 payload에 대해 다음 pulse 순서를 제공했다.

$$
S_c(k)\xrightarrow{L}H_c(k)\xrightarrow{L}Y(k),
\qquad L=2.
$$

[경험식] recurrent write는 정확히 $L$ tick 전의 presynaptic activity와
현재 postsynaptic activity의 local product를 누적한다.

$$
E_{t+1}=0.99E_t+a_{t+1}a_{t+1-L}^{\mathsf T}
-0.20a_{t+1-L}a_{t+1}^{\mathsf T}.
$$

이 식은 target recurrent matrix를 직접 계산하지 않는다. 다만 경험
pulse에 정답 $Y(k)$가 들어가므로 answer-blind plasticity가 아니라
`experience-supervised local eligibility`다.

[정의] recall에서 문맥별 mask는 entry branch 하나와 두 공통 output
map을 연다.

$$
M_c=Q_{H_cS_c}\cup Q_{YH_0}\cup Q_{YH_1},
\qquad \lVert M_c\rVert_0=3m=12.
$$

$M_0$와 $M_1$에서 $Y$에 닿는 edge는 완전히 같다. 달라지는 것은
$S_c\to H_c$ 네 edge뿐이다. recall 입력은 서로 다른 두 payload를
동시에 담고, 이후 외부 입력은 0이다.

$$
u_0=S_0(k_0)+S_1(k_1),\qquad k_0\ne k_1,
\qquad u_{t>0}=0.
$$

## 무엇이 실제로 확인됐는가

[산출] 네 block map은 모두 edge 네 개와 rank 네 개를 가졌고, 두 경로
곱의 최소 특이값은 모든 seed에서 적어도 `0.7071068287`이었다. Correct와
wrong mask는 edge 12개, delay, 이질적 neuron threshold, STP, decoder,
평균 activity, runtime-energy proxy가 같았다.

[수치 관찰] 고정한 개발 seed 16개에서 correct route는 모두 성공했다.
Wrong route와 cue 뒤 mask-swap은 정답을 침묵시킨 것이 아니라, 매 trial
반대 payload를 전달했다. 문맥을 보지 않는 static branch는 각각 정확히
절반만 맞았고, 두 branch를 모두 연 union은 두 payload가 섞여 frozen
unique-output decoder를 통과하지 못했다.

이 결과로 지지되는 문장은 다음 하나다.

> 이 synthetic delayed runtime에서는 context가 entry mask만 선택해도,
> 공통 output trunk와 공통 decoder를 유지한 채 두 동시 입력 중 하나를
> 고를 수 있었다.

## 실패도 식의 일부로 남겼다

Revision 0은 중간에 공통 relay $R$을 하나 더 두었다. Support와 rank는
모두 맞았고 선택한 hidden activity도 생겼지만, 행 정규화 뒤 relay가
frozen emission threshold를 넘지 못해 $Y$ activity가 0이었다. Threshold나
decoder를 낮추지 않고 불필요한 hop을 제거해 Revision 1을 만들었다.
Revision 2는 독립 감사에서 요구한 parity 영수증만 추가했고 모든 scored
endpoint는 Revision 1과 같았다.

## 아직 증명하지 않은 것

[미완성] 지금 context-to-mask 대응은 외부에서 주어진다. 뇌가 context를
읽어 이 mask를 스스로 학습했다는 결과가 아니다. 또한 두 entry branch를
고르는 실험이므로 cycle, bridge, clique 같은 일반 graph morphology
dictionary를 입증하지 않는다. 실제 뇌, 질환, cortical folding, 물리적
곡률, joule 단위 에너지, AGI 성능으로 확대할 근거도 없다.

[예측] 다음 falsifier는 같은 task에서 router 자체를 학습시키는 것이다.
Context cue와 local gate eligibility만으로 $M_0/M_1$ 선택을 배워야 하며,
oracle context mask, context-shuffle, gate-free static mixture와 비교해야
한다. 이 단계가 실패하면 이번 결과는 “외부에서 올바른 mask를 주면
작동한다”는 sufficiency 결과에 머문다.

## 재현

Revision 2 개발 결과는
`artifacts/development-results-r2.json`에 있고 SHA-256은
`bc8dd6f1f884e500f691fa4cddd125a11f442201bfd422ccd576d231e0abe2c5`다.
Focused test는 `3 passed`였으며 confirmation seed 32개는 열지 않았다.

