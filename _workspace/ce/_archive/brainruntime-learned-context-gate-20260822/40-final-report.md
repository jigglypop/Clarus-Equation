# 경험으로 문맥별 회로를 고르는 최소 gate

Status: COMPLETE

BA-TR3는 맞는 회로 mask를 외부에서 주면 두 동시 입력 가운데 문맥에 맞는 payload만 전달할 수 있음을 보였다. 그러나 그 결과만으로는 뇌가 mask를 배웠다고 말할 수 없었다. 이번 BA-TR4는 recurrent 회로와 decoder를 그대로 동결한 채, 경험 중 함께 나타난 문맥 cue와 실제 branch 사용량만으로 두 선택지 가운데 하나를 고르는 별도 gate를 학습했다. 개발 seed 16개에서 learned gate는 oracle과 같은 정확도 `1.0`을 냈고, cue나 학습 문맥을 뒤집으면 살아 있는 반대 payload를 정확히 전달했다. 따라서 이번 결과는 고정된 두 후보 회로 위에서의 경험 기반 selector를 지지한다. 후보 회로 자체의 발견, 새로운 문맥 조합, 실제 뇌, 곡률 기억, 질환, 물리 에너지, AGI는 아직 검증하지 않았다.

## 무엇을 새로 학습했는가

[정의] 문맥 cue는 서로 직교하는 네 차원 무차원 벡터 $q_0,q_1$이다. 두 gate actuator의 가중치를 모은 행렬은 $\Theta\in\mathbb R^{2\times4}$이다. recurrent 회로에는 두 source-to-hidden entry branch $Q_0,Q_1$과 두 branch가 공유하는 output trunk가 이미 존재한다.

[경험식] 한 경험에서 exact-delay eligibility $E^{(n)}$가 만들어지면, 각 branch가 실제로 얼마나 사용됐는지를 다음과 같이 읽는다.

$$
u_b^{(n)}=\frac{1}{4}\sum_{(i,j)\in Q_b}
\left[E_{ij}^{(n)}\right]_+.
$$

여기서 gate는 어느 branch가 정답인지 나타내는 label을 받지 않는다. 물리적으로 활성화된 entry edge의 양수 eligibility만 branch-use 신호가 된다. 출력 $Y$, target vector, decoder score, reward, endpoint loss도 읽지 않는다.

[경험식] gate는 문맥 afferent와 branch-use actuator의 국소 결합을 누적한다.

$$
\Theta_{n+1}=\operatorname{clip}_{[-4,4]}
\left(\Theta_n+u^{(n)}q_c^{\mathsf T}\right).
$$

모든 항은 무차원이다. 학습이 끝나면 $\Theta$를 동결하고 회상 전에 다음 선택을 한 번 수행한다.

$$
\widehat b(c)=\arg\max_{b\in\{0,1\}}(\Theta q_c)_b,
$$

$$
\widehat M_c=Q_{\widehat b(c)}\cup Q_{YH_0}\cup Q_{YH_1}.
$$

두 mask는 모두 recurrent edge 12개를 사용하고, 공통 trunk 8개는 완전히 같다. 달라지는 것은 entry branch 네 개뿐이다. logit tie는 임의로 해소하지 않고 apparatus failure로 처리했다.

## 왜 단순한 oracle 재포장이 아닌가

[산출] mask compiler의 입력은 frozen gate, context cue, recurrent weight, block partition뿐이다. seed, 문맥-branch 대응표, schedule, payload, target, decoder, route 이름을 받지 않고 closure도 없다. serialized $\Theta,q$에서 별도 구현으로 다시 계산한 $\arg\max$와 compiler 결과가 16/16 seed에서 일치했다.

[산출] $q_0,q_1$만 교환하면 mask가 교환됐고, $q$를 고정한 채 $\Theta$의 두 행만 교환해도 독립 참조가 예측한 대로 mask가 교환됐다. 반대로 $\Theta,q$와 support를 고정한 채 seed, $\sigma$, schedule metadata를 바꿔도 mask는 변하지 않았다. 이 세 검사는 각각 cue, learned matrix, 외부 metadata 가운데 실제 계산 원인이 무엇인지 분리한다.

[산출] 각 seed의 문맥-branch 대응은 identity와 swap을 번갈아 사용해 전체가 8 대 8로 균형을 이뤘다. 따라서 고정된 `context 0→branch 0` 규칙은 평균 정확도 `0.5`에 머물렀다. 학습된 gate와 recurrent snapshot은 endpoint 전체를 평가한 뒤에도 byte-level digest가 같았다.

## 수치 결과

[경험식] 개발 seed `97601..97616`에서 모든 pre-endpoint gate와 모든 seed 판정이 통과했다.

| 조건 | 정확도 | 반대 payload 전달 | runtime-energy proxy | active fraction |
|---|---:|---:|---:|---:|
| learned gate | 1.000000 | 0.000000 | 0.200115 | 0.096429 |
| oracle | 1.000000 | 0.000000 | 0.200115 | 0.096429 |
| context-shuffled learning | 0.000000 | 1.000000 | 0.200115 | 0.096429 |
| wrong cue | 0.000000 | 1.000000 | 0.200115 | 0.096429 |
| cue 뒤 mask swap | 0.000000 | 1.000000 | 0.200115 | 0.096429 |
| gate lesion/static branch 0 | 0.500000 | 0.500000 | 0.200115 | 0.096429 |
| static branch 1 | 0.500000 | 0.500000 | 0.200115 | 0.096429 |
| random matched mask | 0.283854 | 0.283854 | 0.227694 | 0.111607 |
| full two-branch mask | 0.000000 | 0.000000 | 0.253479 | 0.125000 |

Wrong-control 회로는 activity가 꺼져 실패한 것이 아니다. correct와 같은 edge 수, threshold, delay, STP, 평균 activity, runtime-energy proxy를 유지하면서 다른 동시 payload를 전달했다. full 회로는 더 많은 edge와 activity를 사용했지만 두 payload 간섭 때문에 실패했다. 여기서 energy는 simulator의 무차원 proxy이며 joule이 아니다.

## 현재 결론과 남은 반례

[경험식] 이번 결과가 허용하는 문장은 다음과 같다.

> 고정된 두 후보 entry branch가 있는 synthetic delayed runtime에서, 문맥 cue와 국소 branch-use eligibility의 반복 경험만으로 어떤 branch를 열지 학습할 수 있다.

이 결과는 `mask를 외부에서 공급해야만 한다`는 BA-TR3의 남은 한계를 제거했다. 그러나 두 cue를 모두 학습했으며 선택지도 두 개뿐이다. 그러므로 지금의 $\Theta$는 두 항목을 저장한 key-value memory로도 충분히 설명된다. 또한 $Q_0,Q_1$이라는 후보 support는 설계자가 미리 정했다. 따라서 graph morphology를 발견하거나 처음 보는 문맥 조합에 맞춰 회로를 합성했다고 말할 수 없다.

[미완성] confirmation seed `99601..99632`는 봉인했다. 실제 뇌 데이터, 뉴런별 gate population의 생물학적 대응, cortical folding 또는 리만 곡률과 기억의 연결, 물리 에너지, 질환 개입, AGI 성능은 입력도 검정도 없다.

## 결과에서 바로 나오는 다음 식

[예측] 다음 BA-TR5는 단순 lookup과 회로 합성을 구분해야 한다. 두 독립 문맥 인자 $a,b\in\{0,1\}$를 사용하고, 세 조합만 경험한 뒤 $(1,1)$을 완전히 보류한다. 문맥 표현과 mask를 factor별 direct sum으로 둔다.

$$
q_{ab}=q^A_a\oplus q^B_b,
$$

$$
\widehat M_{ab}
=T^A\cup Q^A_{\arg\max(\Theta^Aq^A_a)}
\cup T^B\cup Q^B_{\arg\max(\Theta^Bq^B_b)}.
$$

각 factor gate는 그 factor의 국소 branch-use만 읽고, 학습 조합은 $(0,0),(0,1),(1,0)$로 고정한다. 보류된 $(1,1)$에서 두 factor를 동시에 맞춰야만 pass다. `joint context lookup`, `factor shuffle`, `single-factor lesion`, `static`, `oracle`을 같은 mask budget으로 비교한다. 이 검사가 통과해야 다음 주장을 단순 문맥 기억에서 처음 보는 문맥의 회로 조합으로 올릴 수 있다.

## 재현

구현은 `reality_stone/python/reality_stone/clarus/runtime_context_learned_gate.py`, runner는 `runtime_context_learned_gate_benchmark.py`, 집중 테스트는 `tests/test_runtime_context_learned_gate.py`에 있다. 개발 결과는 `artifacts/development-results.json`이며 SHA-256은 `afc9d0aba4606f9dcc7a0370894c5b5682ceabfd9128b083fed5f46522d5f064`이다. 집중 테스트는 `3 passed`였고 JUnit SHA-256은 `42088cc1e0873735da39f5c749f5916e1fb798c02b040d9b0ecb969daddd530c`이다.
