# 무차원 자기측정 깊이와 자기비동일성 흐름 형식 감사

Status: COMPLETE

Gate: PASS

## Claim ledger

| Claim ID | 지위 | 판정 |
|---|---|---|
| `C-SM-THETA-01` | [정리] | 단일 고정 $\mathcal D_P$에서 $\theta=-\ln(1-\eta)$는 additive이고 모든 유한 분할은 같은 unconditional channel을 준다. |
| `C-SM-ADAPTED-01` | [정의/조건부 구성] | object/record 또는 시간절편을 분리하고 $m_n$이 과거 filtration에 adapted일 때 causal self-monitoring recursion이 성립한다. |
| `C-SM-COST-01` | [정리] | finite $n$에서 $0\leq C_{\rm self}\leq(1-e^{-\theta_*})\ln n\leq\ln n$이다. |
| `C-SM-FLOW-01` | [정리] | $A=(I-\mathcal D_P)\rho_0\ne0$이면 모든 finite $\theta,h>0$에서 $\rho_{\theta+h}\ne\rho_\theta$이며 trace-distance speed와 length가 닫힌식으로 주어진다. |
| `C-SM-CLOCK-01` | [조건부 정리] | fixed reference, calibrated initial state와 $A\ne0$에서 residual의 logarithmic ratio로 $\theta$를 복원한다. 물리시간은 별도 rate bridge가 필요하다. |
| `C-SM-COST-LENGTH-01` | [조건부 산출] | 같은 fixed path와 $A\ne0$에서 $dC_{\rm self}=2\overline C_I dL/\|A\|_1$이다. 기회비용과 상태운동의 보편 동일시는 아니다. |
| `C-SM-DEPOSIT-01` | [정의/미완성 다리] | $d\mu_{\rm self}$는 양의 유한 정보 measure이나 spatial pushforward, retention, action과 energy scale은 별도이다. |
| `C-SM-LIMIT-01` | [완전 반례] | stationary $A=0$, periodic unitary return, noncommuting order dependence와 non-Markov recoherence가 더 강한 부모 주장을 각각 기각한다. |

## Resolved P1 revision

첫 감사에서 단일 fixed dephasing에 대해 증명한 결과를 여러 commuting
generator까지 이미 증명한 것처럼 읽을 수 있는 claim ceiling이 P1이었다.
계약을 단일 fixed $\mathcal D_P$로 좁혔고, commuting 확장은 가법성, 공통
domain과 경로 독립성을 별도로 입증해야 하는 route로 격리했다.

또한 유한 분할의 동치가 unconditional state channel에만 적용되며 outcome
history, conditional trajectory와 feedback law의 동치를 뜻하지 않는다고
명시했다. 재감사에서 두 P1 모두 해소되었다.

## Complete counterexamples

1. $A=0$인 이미 dephased된 상태는 $L=0$이지만 probability-based
   $\overline C_I$가 양수일 수 있다. 기회비용과 실제 상태운동은 동일하지 않다.
2. periodic unitary orbit는 모든 국소 구간에서 움직여도 한 주기 뒤
   원상복귀한다. local self-nonidentity만으로 시간의 화살을 얻지 못한다.
3. $z$와 $(x+z)/\sqrt2$ dephasing의 순서를 바꾸면 Frobenius 차이가
   $0.0494974746830583$이다. changing/noncommuting measurement를 단일
   path-independent $\theta$로 압축하지 못한다.
4. $\lambda(t)=\cos^2(gt/2)$인 recoherence family에서 $\theta$는
   $0\to\infty\to0$으로 되돌아간다. non-Markov 과정에 전역 monotone
   scalar clock은 없다.

## Remaining incomplete bridges

1. 측정깊이와 물리시간을 잇는 독립 rate $\gamma(t)$의 동역학적 기원.
2. 실제 continuous outcome history와 feedback를 생성하는 구체 instrument/dilation.
3. record $R$에서 spatial point carrier와 persistent $\mu_F$로 가는 retention map.
4. $\mu_F$와 환경장 $\chi$의 backreaction을 포함할 때 필요한 non-Markov 구조.
5. 정보 functional을 energy, stress tensor, dark matter clustering 또는
   dark energy pressure로 보내는 action, scale과 conservation law.

## Priority verdict

- P0: 없음.
- P1: 없음.
- Gate: PASS for the fixed-dephasing channel theorem, past-adapted operational
  self-measurement, bounded opportunity functional and conditional
  distinguishability-flow theorem.

