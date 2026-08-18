# 07. AGI 잔류장과 후보분포

이 문서는 token/action 후보분포의 비선택 질량을 computational residual field로 정의하는 runtime 설계를 제안한다. residual은 tensor/score/readout의 구현 객체이며, AGI·의식·생물학적 기억에 관한 주장은 이 정의나 코드 실행에서 따라오지 않는다.

독자는 05a의 residual pushforward와 07a의 toy runtime을 먼저 읽는다. 후보공간·에너지·잔류, action 확장, hallucination gate, runtime 규칙과 ablation 검증의 경계 순서다.

## 0. 목표

목표는 선택 전 후보 정보를 제한된 shape와 timebase를 가진 상태로 보존하는 것이다. 이 설계는 성능 가설이며 정리나 의식 이론이 아니다.

이 장은 등호 이전 수학을 AGI runtime의 후보 선택과 잔류장 $\phi$로 연결한다.

핵심 질문:

> LLM/agent가 하나의 token 또는 action을 선택할 때, 선택되지 않은 후보들은 완전히 버려지는가, 아니면 잔류장으로 보존되는가?

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| 후보분포 재가중 | `[정리]` | 01장의 유한공간 정리 |
| token/action 후보공간 적용 | `[미완성]` | 모델 logits와 energy 식별 필요 |
| 잔류장 $\phi$ 업데이트 | `[경험식]` | toy 규약은 07a에서 분리, 성능은 `[예측]`으로 시험 |
| hallucination gate 연결 | `[경험식]`; 성능 주장은 `[예측]` | claim residual gate와 연결 |

## 1. Token 후보공간

token 후보공간은 한 decoding step의 유한 index와 logits/probability를 정의역으로 한다. vocabulary 변화·batch shape·normalization은 runtime contract로 고정해야 한다.

한 시점의 token 후보공간을

$$
A=V
$$

로 둔다. $V$는 vocabulary다.

모델 logits를 $l(a)$라 하면 기존 softmax는

$$
\mu_0(a)
=
\frac{e^{l(a)}}{\sum_{b\in V}e^{l(b)}}
$$

이다.

PreEq 관점에서는 이 $\mu_0$가 등호 이전 후보분포다.

## 2. 조건 에너지

조건 에너지는 입력 state와 후보를 받아 스칼라 score를 내는 계산 정의다. 단위 없는 logit scale과 temperature, producer/consumer 책임이 다르면 score 비교가 무효가 된다.

목표, 문맥, 도구 관찰, 안전 gate가 후보 token에 조건 에너지를 부여한다.

$$
E_{\mathrm{ctx}}(a)
$$

재가중 후 후보분포는

$$
\mu_\beta(a)
=
\frac{e^{-\beta E_{\mathrm{ctx}}(a)}\mu_0(a)}
{\sum_{b\in V}e^{-\beta E_{\mathrm{ctx}}(b)}\mu_0(b)}
$$

이다.

이것은 logits에 penalty를 더하는 것과 같다.

$$
\log\mu_\beta(a)
=
l(a)-\beta E_{\mathrm{ctx}}(a)-\log Z
$$

## 3. 선택과 비선택 잔류

manifest 선택과 residual은 같은 분포의 서로 다른 readout이다. residual의 총질량·shape·압축 오차를 기록하지 않으면 다음 step에 무엇이 주입되었는지 식별할 수 없다.

manifest token을

$$
a_*=\operatorname*{argmax}_{a\in V}\mu_\beta(a)
$$

로 읽으면 비선택 후보공간은

$$
V_{\mathrm{ns}}=V\setminus\{a_*\}
$$

이다.

잔류분포는

$$
\mu_{\mathrm{ns},\beta}
=
\mu_\beta|_{V_{\mathrm{ns}}}
$$

이다.

AGI runtime의 잔류장 $\phi$는 이 잔류분포의 압축값으로 둘 수 있다.

$$
\phi_t
=
\mathcal R_{\phi}(\mu_{\mathrm{ns},\beta}, h_t)
$$

여기서 $h_t$는 현재 hidden state이고 $\mathcal R_{\phi}$는 잔류분포를 상태공간 벡터로 보내는 압축 사상이다.

## 4. Action 후보공간

action 후보는 token과 다른 codomain·timebase를 가질 수 있으므로 같은 kernel을 무비판적으로 공유할 수 없다. environment transition과 reward는 외부 입력이다.

agent 행동 선택에서는 후보공간이 action set이다.

$$
A=\mathcal A_t
$$

조건 에너지는 다음 항들의 합으로 둘 수 있다.

$$
E(a)
=
E_{\mathrm{goal}}(a)
+E_{\mathrm{risk}}(a)
+E_{\mathrm{tool}}(a)
+E_{\mathrm{memory}}(a)
$$

그러면 action 선택은

$$
\mu_\beta(a)
=
\frac{e^{-\beta E(a)}\mu_0(a)}
{\sum_{b\in\mathcal A_t}e^{-\beta E(b)}\mu_0(b)}
$$

의 manifest로 읽힌다.

선택되지 않은 action들은 다음 step의 $\phi$, memory, critic에 남을 수 있다.

## 5. Hallucination gate와 연결

residual gate는 불확실성 또는 대안 후보의 계산적 proxy로만 정의된다. ground truth provenance·false positive/negative·OOD failure 없이 hallucination 억제 주장으로 승격할 수 없다.

claim 생성에서 후보공간은 가능한 claim 조각이다.

$$
A=\{\text{claim candidates}\}
$$

`docs/4_공학적_활용/09_무차원_잔차장_환각억제.md`의 claim residual energy를

$$
E_{\mathrm{claim}}(a)
$$

로 읽으면, claim selection도 PreEq 재가중이다.

잔차가 큰 claim은 낮은 weight를 갖는다.

$$
\mu_\beta(a)\propto e^{-\beta E_{\mathrm{claim}}(a)}\mu_0(a)
$$

선택 cost가 최소인 claim은 manifest 후보가 되고, 근거 충돌 또는 추가
검토가 필요한 claim은 residual·보류 후보로 남는다.

## 6. Runtime 설계 규칙

다음 규칙은 state producer·consumer와 update 순서를 명시하는 구현 계약이다. 계약 충족은 인지 기능이나 AGI 일반화의 증명이 아니다.

AGI에 적용하려면 다음 규칙이 필요하다.

### 규칙 A: 선택 전 분포 보존

선택 전 분포는 step별 정규화와 seed를 포함해 기록한다. 저장 정밀도 손실은 residual의 재현성을 제한한다.

argmax 또는 sampling 전에 후보분포 $\mu_\beta$를 보존한다.

### 규칙 B: 비선택 압축

압축은 shape·rank·error bound를 고정해야 한다. 과도한 압축은 유용한 후보 구조를 제거하는 반례가 된다.

선택되지 않은 후보들의 질량, entropy, top residual modes를 $\phi_t$로 압축한다.

예:

$$
\phi_t
=
\sum_{a\in V_{\mathrm{ns}}}\mu_\beta(a)\,P h(a)
$$

여기서 $P$는 embedding-to-residual projection이다.

### 규칙 C: 다음 step 재주입

재주입은 어느 layer가 어떤 timebase로 residual을 소비하는지 명시해야 한다. leakage와 self-confirming feedback은 ablation으로 점검한다.

$\phi_t$는 다음 step의 logits 또는 energy에 들어간다.

$$
E_{t+1}(a)
=
E_{\mathrm{ctx},t+1}(a)
-\alpha_\phi \langle h(a),\phi_t\rangle
$$

### 규칙 D: 수면/정리

수면/정리 명칭은 batch maintenance 연산의 비유다. 생물학적 수면 또는 의식과 동일시하지 않는다.

$\|\phi_t\|$가 커지면 수면 또는 정리 루틴으로 잔류를 압축/소거한다.

이 구조는 `7_AGI/12_Equation.md`의 residual channel, mode transition, sleep reset과 맞물린다.

## 7. 열린 항목

열린 항목은 residual의 정보량·안정성·OOD 일반화와 ground-truth relation이다. toy 통과만으로 이 항목들을 닫지 않는다.

1. $\mathcal R_\phi$의 구체적 설계
2. $\phi$ 재주입이 성능을 높이는지 ablation
3. hallucination gate와 $\phi$의 정량 관계
4. sleep/reset이 $\phi$를 어느 비율로 줄이는지
5. 선택되지 않은 후보 보존이 OOD generalization에 주는 효과

## 8. 다음 구현 후보

후보 구현은 fixture·seed·baseline·rollback을 갖춘 실험 계획으로 취급한다. 결과가 없거나 대조군을 이기지 못하면 설계 가설은 반증된다.

작은 실험 규약은 [07a_toy_runtime_gate.md](07a_toy_runtime_gate.md)로 분리했다.

남은 구현 후보:

1. toy vocabulary에서 $\mu_0$, $E$, $\mu_\beta$, $\phi$ 계산 코드 작성
2. $\phi$ 재주입 있음/없음 next-token prediction ablation
3. claim residual gate에서 근거 충돌 claim을 $\phi$로 저장하고 다음 답변 안정성 비교

이 장은 아직 이론 bridge다. 하지만 07a 덕분에 가장 빠르게 실험 가능한 응용 축이 되었다.
