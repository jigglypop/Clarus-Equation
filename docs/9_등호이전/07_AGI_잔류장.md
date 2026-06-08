# 07. AGI 잔류장과 후보분포

## 0. 목표

이 장은 등호 이전 수학을 AGI runtime의 후보 선택과 잔류장 \(\phi\)로 연결한다.

핵심 질문:

> LLM/agent가 하나의 token 또는 action을 선택할 때, 선택되지 않은 후보들은 완전히 버려지는가, 아니면 잔류장으로 보존되는가?

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| 후보분포 재가중 | `Exact` | 01장의 유한공간 정리 |
| token/action 후보공간 적용 | `Bridge` | 모델 logits와 energy 식별 필요 |
| 잔류장 \(\phi\) 업데이트 | `Bridge/Open` | runtime 구현 규약 필요 |
| hallucination gate 연결 | `Bridge` | claim residual gate와 연결 가능 |

## 1. Token 후보공간

한 시점의 token 후보공간을

$$
A=V
$$

로 둔다. \(V\)는 vocabulary다.

모델 logits를 \(l(a)\)라 하면 기존 softmax는

$$
\mu_0(a)
=
\frac{e^{l(a)}}{\sum_{b\in V}e^{l(b)}}
$$

이다.

PreEq 관점에서는 이 \(\mu_0\)가 등호 이전 후보분포다.

## 2. 조건 에너지

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

AGI runtime의 잔류장 \(\phi\)는 이 잔류분포의 압축값으로 둘 수 있다.

$$
\phi_t
=
\mathcal R_{\phi}(\mu_{\mathrm{ns},\beta}, h_t)
$$

여기서 \(h_t\)는 현재 hidden state이고 \(\mathcal R_{\phi}\)는 잔류분포를 상태공간 벡터로 보내는 압축 사상이다.

## 4. Action 후보공간

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

선택되지 않은 action들은 다음 step의 \(\phi\), memory, critic에 남을 수 있다.

## 5. Hallucination gate와 연결

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

accepted claim은 manifest 되고 rejected/review claim은 잔류 또는 보류 상태로 남는다.

## 6. Runtime 설계 규칙

AGI에 적용하려면 다음 규칙이 필요하다.

### 규칙 A: 선택 전 분포 보존

argmax 또는 sampling 전에 후보분포 \(\mu_\beta\)를 보존한다.

### 규칙 B: 비선택 압축

선택되지 않은 후보들의 질량, entropy, top residual modes를 \(\phi_t\)로 압축한다.

예:

$$
\phi_t
=
\sum_{a\in V_{\mathrm{ns}}}\mu_\beta(a)\,P h(a)
$$

여기서 \(P\)는 embedding-to-residual projection이다.

### 규칙 C: 다음 step 재주입

\(\phi_t\)는 다음 step의 logits 또는 energy에 들어간다.

$$
E_{t+1}(a)
=
E_{\mathrm{ctx},t+1}(a)
-\alpha_\phi \langle h(a),\phi_t\rangle
$$

### 규칙 D: 수면/정리

\(\|\phi_t\|\)가 커지면 수면 또는 정리 루틴으로 잔류를 압축/소거한다.

이 구조는 `7_AGI/12_Equation.md`의 residual channel, mode transition, sleep reset과 맞물린다.

## 7. 열린 항목

1. \(\mathcal R_\phi\)의 구체적 설계
2. \(\phi\) 재주입이 성능을 높이는지 ablation
3. hallucination gate와 \(\phi\)의 정량 관계
4. sleep/reset이 \(\phi\)를 어느 비율로 줄이는지
5. 선택되지 않은 후보 보존이 OOD generalization에 주는 효과

## 8. 다음 구현 후보

작은 실험:

1. Toy vocabulary에서 \(\mu_0\), \(E\), \(\mu_\beta\), \(\phi\) 계산
2. \(\phi\) 재주입 있음/없음 next-token prediction 비교
3. claim residual gate에서 rejected claim을 \(\phi\)로 저장하고 다음 답변 안정성 비교

이 장은 아직 이론 bridge다. 하지만 도구화하면 가장 빠르게 실험 가능한 응용 축이다.
