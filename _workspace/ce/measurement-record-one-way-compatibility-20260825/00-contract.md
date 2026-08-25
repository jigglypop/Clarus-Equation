# 외부 0차원 일방향성과 측정 기록 0차원성의 호환성 계약

Status: COMPLETE

PREDECESSOR:

- `_workspace/ce/zero-dimensional-overlap-bootstrap-20260825`
- `_workspace/ce/measurement-wall-dimensionality-20260825`

Mode: light

## 1. 질문

strict external physical sector $Z_{\rm phys}\to M$만 허용하고 $M\not\to Z_{\rm phys}$를 요구하면서, 측정의 완료 기록을 같은 0차원 sector와 동일시할 수 있는지 검사한다. 결과공간의 topological 0D, strict singleton $Z=\{\star\}$, one-dimensional Hilbert space $\mathcal H_Z\cong\mathbb C$, finite-duration measurement process를 서로 구분한다.

## 2. 고정 정의

### R1. strict physical 0D input

$$
Z_{\rm phys}=\{\star\},
\qquad
\mathcal H_Z\cong\mathbb C,
\qquad
Z_{\rm phys}\xrightarrow{\mathcal E_{Z\to M}}M,
\qquad M\not\to Z_{\rm phys}.
$$

strict singleton 또는 one-dimensional quantum input은 distinguishable internal record를 갖지 않는다.

### R2. discrete outcome record

finite measurement outcome alphabet은

$$
\mathcal O=\{r_1,\ldots,r_n\}
$$

로 둔다. $\mathcal O$는 $n>1$이어도 topological dimension 0이지만 singleton은 아니다. quantum instrument $\{\mathcal I_r\}$의 classical record channel은

$$
\mathcal C_{M\to R}(\rho)
=\sum_r\operatorname{tr}[\mathcal I_r(\rho)]
|r\rangle\langle r|_R.
$$

### R3. informative measurement

instrument effect를 $E_r=\mathcal I_r^*(I)$라 두어

$$
p_r(\rho)=\operatorname{tr}(E_r\rho)
$$

로 쓴다. 어떤 $r,\rho,\sigma$에 대해 $p_r(\rho)\ne p_r(\sigma)$이면 informative라고 정의한다.

### R4. no-signalling into the physical 0D sector

$M\not\to Z_{\rm phys}$는 $M$의 입력 상태를 바꾸어도 $Z_{\rm phys}$의 output state 또는 classical distribution이 변하지 않는다는 operational 조건으로 고정한다.

## 3. 사전 고정 검사

1. $R=Z_{\rm phys}$로 동일시할 때 informative record channel과 no-$M\to Z$가 양립하는가.
2. state-independent record distribution이 모든 effect에 대해 $E_r=c_rI$를 강제하는가.
3. strict singleton $Z$가 $n>1$ outcome을 저장할 수 있는가.
4. finite-duration process $[t_0,t_1]$가 strict 0D sector 내부의 clock 없이 어디에 놓여야 하는가.
5. one-way 조건을 유지하는 최소 분리 또는 재해석은 무엇인가.

## 4. 반증 조건

- informative $M\to R$ channel이면서도 $R=Z_{\rm phys}$이고 operational no-$M\to Z$를 만족하는 counterexample.
- $\mathcal H_Z\cong\mathbb C$ 안에 둘 이상의 perfectly distinguishable record states가 존재하는 구성.
- 외부 time parameter나 $M$ support 없이 strict singleton 내부에서 nontrivial finite-duration unitary measurement를 생성하는 구성.

## 5. 주장 상한

이 run은 channel compatibility와 type distinction만 판정한다. physical $Z$의 실재, Born rule의 기원, objective collapse, retention map, energy/stress 또는 dark-sector identity를 주장하지 않는다. 통과 가능한 최강 결론은 “record 0D와 external physical 0D를 분리하면 one-way 조건을 보존할 수 있다” 또는 명시적 추가 공리 아래의 좁은 대안이다.
