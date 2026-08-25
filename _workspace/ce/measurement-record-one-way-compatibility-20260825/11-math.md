# 측정 기록과 외부 0차원 일방향성의 수학 감사

Status: COMPLETE

## 1. instrument가 만드는 record channel

instrument $\{\mathcal I_r\}_{r=1}^n$의 effect를

$$
E_r=\mathcal I_r^*(I),
\qquad E_r\ge0,
\qquad \sum_rE_r=I
$$

로 둔다. 상태 $\rho$에서 outcome 확률과 classical record channel은

$$
p_r(\rho)=\operatorname{tr}(E_r\rho),
$$

$$
\mathcal C_{M\to R}(\rho)
=\sum_rp_r(\rho)|r\rangle\langle r|_R
$$

이다. 어떤 두 허용 상태 $\rho,\sigma$와 outcome $r$에 대해 $p_r(\rho)\ne p_r(\sigma)$이면 이 record는 input에 관해 informative하다.

## 2. 정리: no-$M\to R$이면 record는 uninformative다

**정리.** 모든 density operator $\rho$에 대해 $\mathcal C_{M\to R}(\rho)=\omega_R$인 것과, 각 effect가 $E_r=c_rI$이고 $c_r\ge0$, $\sum_rc_r=1$인 것은 동치다.

**증명.** channel이 상수이면 각 $r$에 대해 어떤 $c_r$가 존재하여

$$
\operatorname{tr}(E_r\rho)=c_r
$$

가 모든 density operator에서 성립한다. 따라서

$$
\operatorname{tr}[(E_r-c_rI)\rho]=0
\qquad\text{for all }\rho.
$$

$E_r-c_rI$는 Hermitian이다. 모든 pure state $|\psi\rangle\langle\psi|$를 대입하면 그 quadratic form이 0이고, polarization identity에 의해 operator 자체가 0이다. 그러므로 $E_r=c_rI$다. 반대로 이 조건이면 $p_r(\rho)=c_r\operatorname{tr}\rho=c_r$라서 channel은 입력과 독립이다. completeness에서 $\sum_rc_r=1$이 따른다. $\square$

따라서 strict no-$M\to Z_{\rm phys}$를 record channel에 적용하고 $R=Z_{\rm phys}$로 동일시하면 informative measurement는 불가능하다. classical encoding $X\mapsto\rho_X$를 어떻게 택해도 $I(X:R)=0$이다. conditional operation이 상태를 무작위로 바꿀 수는 있지만, label 분포가 input에 관한 정보를 싣지는 못한다.

## 3. 완전 반례

qubit projective instrument

$$
E_0=|0\rangle\langle0|,
\qquad
E_1=|1\rangle\langle1|
$$

에 $\rho_0=|0\rangle\langle0|$, $\rho_1=|1\rangle\langle1|$를 넣으면

$$
(p_0,p_1)(\rho_0)=(1,0),
\qquad
(p_0,p_1)(\rho_1)=(0,1).
$$

record distribution이 orthogonal하게 달라지므로 완전한 $M\to R$ signal이다. 따라서 “일반 informative measurement의 record가 strict external $Z$에 남지만 $M\not\to Z$다”라는 부모 명제는 계약 정의역에서 모순이다.

반대로

$$
E_0=0.3I,
\qquad E_1=0.7I
$$

이면 모든 input에서 record distribution은 $(0.3,0.7)$로 같고 no-signalling을 만족하지만, 이것은 input을 측정하지 않는 uninformative control이다.

## 4. topological 0D, singleton과 Hilbert dimension

finite discrete alphabet $\mathcal O=\{r_1,\ldots,r_n\}$은 모든 $n$에 대해 covering/topological dimension 0이다. 그러나 $n>1$이면 singleton이 아니다. $n$개의 perfectly distinguishable classical records에는 algebra $\mathbb C^n$이 필요하고, orthogonal quantum encoding에는 최소 $n$차원 Hilbert register가 필요하다.

반면 strict $Z_{\rm phys}=\{\star\}$ 또는 $\mathcal H_Z\cong\mathbb C$에는 normalized state가 하나뿐이다. 이 공간의 unitary는 global phase만 만들므로 둘 이상의 distinguishable records 또는 nontrivial internal memory dynamics를 제공하지 못한다. 외부에 branch label을 저장하면 그 저장소가 별도 record system $R$이며 $R\ne Z_{\rm phys}$다.

따라서 다음 두 문장은 양립한다.

$$
\dim_{\rm top}\mathcal O=0,
\qquad
\dim_{\mathbb C}\mathcal H_R=n>1.
$$

“위상적으로 0차원”은 “물리 상태가 하나뿐”이라는 뜻이 아니다.

## 5. 유한시간 measurement witness

system과 pointer apparatus를 qubit로 두고 $P_j=|j\rangle\langle j|$,
$X=|0\rangle\langle1|+|1\rangle\langle0|$라 하자. 구간
$[t_0,t_0+\tau]$에서

$$
H_{SA}=\frac{\pi\hbar}{2\tau}P_1\otimes X
$$

를 적용하면

$$
U=e^{-iH_{SA}\tau/\hbar}
=P_0\otimes I-iP_1\otimes X.
$$

pointer 초기상태 $|0\rangle_A$와 computational-basis readout에서 Kraus operators는

$$
M_0={}_A\langle0|U|0\rangle_A=P_0,
\qquad
M_1={}_A\langle1|U|0\rangle_A=-iP_1.
$$

따라서

$$
\mathcal I_0(\rho)=P_0\rho P_0,
\qquad
\mathcal I_1(\rho)=P_1\rho P_1.
$$

이 informative measurement의 duration $\tau$, apparatus와 time-parameterized interaction/control schedule은 $S+A$가 점유한 물리 process의 기술이다. strict $\mathcal H_Z\cong\mathbb C$ 내부의 nontrivial record dynamics가 아니다. 별도의 physical clock이 반드시 존재한다는 결론은 이 식만으로 나오지 않는다. 다만 $Z$가 intrinsic time/control을 제공하지 않는다는 기존 공리를 유지하면 시간 순서와 duration은 $M$ 또는 별도 channel schedule에 놓여야 한다.

## 6. 결론

P0: $R=Z_{\rm phys}$, informative measurement, strict singleton 및 strict no-$M\to Z_{\rm phys}$의 conjunction은 동시에 유지할 수 없다. singleton, no-signalling 또는 허용 상태집합을 바꾸는 완화 경로까지 배제하는 보편적 no-go는 아니다.

P1 resolved by scope: finite-duration measurement에는 apparatus와 time-dependent interaction/control schedule이 필요하다. strict singleton $Z$가 nontrivial record dynamics를 스스로 제공한다고 할 수 없지만, 별도 physical clock의 존재를 정리로 강제하지 않는다.

살아 있는 좁은 결론은 record outcome space $R$을 external physical singleton $Z_{\rm phys}$와 분리하는 것이다. 그러면 $Z_{\rm phys}\to M$과 no-$M\to Z_{\rm phys}$를 유지하면서, $M\to R$인 내부/환경 record formation을 허용할 수 있다.
