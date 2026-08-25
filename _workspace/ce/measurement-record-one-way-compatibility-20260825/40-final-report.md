# 외부 0차원 일방향성과 측정 기록의 0차원성

Status: COMPLETE

## 초록

이 연구는 strict external 0차원 sector $Z_{\rm phys}\to M$과 completed measurement record의 0차원성을 같은 대상으로 볼 수 있는지 검사했다. 핵심 결과는 strict singleton $Z_{\rm phys}$, informative record, strict no-$M\to Z_{\rm phys}$의 세 조건이 동시에 성립할 수 없다는 조건부 no-go다. no-signalling record channel의 모든 POVM effect는 $E_r=c_rI$여야 하므로 input 정보를 담지 못한다. finite discrete outcome alphabet은 위상적으로 0차원이지만 $n>1$이면 singleton 또는 one-dimensional Hilbert space가 아니다. 유한시간 측정은 system--apparatus interaction과 time-dependent control schedule로 구현되며 strict singleton 내부의 nontrivial record dynamics가 아니다. 가장 작은 일관된 구조는 external physical singleton $Z_{\rm phys}$와 apparatus/environment record $R$을 분리하는 것이다.

## 1. 세 개의 서로 다른 0차원

먼저 다음을 분리해야 한다.

1. $Z_{\rm phys}=\{\star\}$: 점 하나뿐인 strict physical input sector.
2. $\mathcal H_Z\cong\mathbb C$: normalized quantum state가 하나뿐인 one-dimensional Hilbert input.
3. $\mathcal O=\{r_1,\ldots,r_n\}$: $n>1$일 수 있지만 topological dimension은 0인 finite discrete record space.

세 번째 대상은 0차원이면서 여러 결과를 구별할 수 있다. 그러나 classical algebra는 $\mathbb C^n$이고 orthogonal quantum record register는 최소 $\mathbb C^n$이므로 첫째·둘째 대상과 같지 않다. “0차원”은 위상 차원을 말할 수도 있고 state cardinality를 말할 수도 있으며, 둘을 같은 말로 사용하면 channel 방향이 모호해진다.

## 2. informative instrument는 record 방향을 만든다

instrument $\{\mathcal I_r\}$의 effect를

$$
E_r=\mathcal I_r^*(I),
\qquad
\sum_rE_r=I
$$

로 두면 record channel은

$$
\mathcal C_{M\to R}(\rho)
=\sum_r\operatorname{tr}(E_r\rho)|r\rangle\langle r|.
$$

이 channel이 input을 실제로 측정하려면 어떤 $r,\rho,\sigma$에서

$$
\operatorname{tr}(E_r\rho)\ne\operatorname{tr}(E_r\sigma)
$$

여야 한다. 즉 completed record는 operational하게 $M\to R$ 정보 흐름을 가진다. 이는 energy 또는 dynamical feedback의 방향과는 구별되지만, strict no-signalling의 뜻에서는 분명한 channel이다.

## 3. no-signalling 정리

모든 input에서 record state가 같은 고정 $\omega_R$라고 하자. 그러면 outcome별로

$$
\operatorname{tr}(E_r\rho)=c_r
$$

가 모든 density operator $\rho$에서 성립한다. 따라서

$$
\operatorname{tr}[(E_r-c_rI)\rho]=0
\qquad\text{for all }\rho.
$$

$E_r-c_rI$는 Hermitian이고 모든 pure-state quadratic form이 0이므로 polarization identity에 의해

$$
\boxed{E_r=c_rI}
$$

다. 역도 즉시 성립한다. 따라서

$$
\mathcal C_{M\to R}\text{ constant for all inputs}
\quad\Longleftrightarrow\quad
E_r=c_rI\text{ for all }r.
$$

이 경우 $p_r=c_r$는 input과 독립이고 record는 input에 관해 uninformative하다. 그러므로

$$
\boxed{
R=Z_{\rm phys}
+\text{ informative measurement}
+\text{ strict no-}M\to Z_{\rm phys}
\quad\text{is inconsistent}
}
$$

이다. 이 결론은 strict conjunction에 대한 것이다. singleton을 다중 atom sector로 바꾸거나, no-signalling을 no-dynamical-feedback으로 약화하거나, 허용 input을 제한하면 전제가 바뀐다.

## 4. 직접 반례와 유한시간 구현

qubit projective measurement의 effects

$$
E_0=P_0=|0\rangle\langle0|,
\qquad
E_1=P_1=|1\rangle\langle1|
$$

는 $|0\rangle$에서 $(1,0)$, $|1\rangle$에서 $(0,1)$의 record를 만든다. 이 record는 두 inputs를 완전히 구분하므로 $M\to R$ signalling의 직접 witness다. 반대로 $E_0=0.3I$, $E_1=0.7I$는 no-signalling이지만 아무 input도 구분하지 못한다.

측정이 한순간일 필요도 없다. system과 apparatus qubit에 구간 $[t_0,t_0+\tau]$ 동안

$$
H_{SA}=\frac{\pi\hbar}{2\tau}P_1\otimes X
$$

를 적용하면

$$
U=P_0\otimes I-iP_1\otimes X
$$

이고 pointer readout의 Kraus operators는

$$
M_0=P_0,
\qquad
M_1=-iP_1.
$$

따라서 finite interval에서 정확한 projective instrument가 형성된다. duration은 $S+A$ interaction과 time-parameterized control schedule에 있다. 별도 physical clock을 반드시 실체화해야 한다는 결론은 아니지만, strict $\mathbb C$ sector만으로 distinguishable record dynamics를 만들 수는 없다.

## 5. 일방향 조건을 유지하는 최소 구조

권장 분리는

$$
Z_{\rm phys}\xrightarrow{\mathcal E_{Z\to M}}M,
\qquad
M\xrightarrow{\mathcal C}R,
\qquad
R\ne Z_{\rm phys}
$$

이다. $Z_{\rm phys}$는 strict one-way preparation boundary로 남고, $R$은 apparatus 또는 environment 안의 finite discrete outcome register다. 이때 둘 다 서로 다른 의미에서 0D일 수 있으나 동일한 physical object는 아니다.

사용자의 persistent fold-memory와 연결하려면

$$
R\xrightarrow{\mathcal R_{R\to F}}\mu_F\subset M
$$

라는 retention map을 별도로 둔다. 그러면 record는 $M$ 내부의 spatial-0D carrier로 남을 수 있고 external $Z_{\rm phys}$로 되돌아가지 않으므로 strict one-way 조건을 깨지 않는다. 하지만 $\mathcal R_{R\to F}$는 표준 quantum instrument에서 자동으로 나오지 않는 물리 사상이다.

## 6. 다른 선택지

$Z$를 $n$-atom topological-0D sector로 바꾸면 record를 저장할 수 있지만 strict singleton 정의를 포기한다. $M\to Z$ 정보를 허용하고 $Z\to M$ dynamical feedback만 막는다면 “일방향”을 no-feedback으로 재정의해야 한다. 허용 state subset에서만 record probability를 상수로 만들면 그 subset 안에서는 측정이 uninformative하고 superselection/control restriction이 필요하다. $Z_n\to M$ label-dependent preparation은 진짜 one-way지만 미지의 $M$ 상태를 읽는 measurement가 아니며, Born-dependent label distribution을 얻으려면 common cause, global constraint 또는 역방향 의존성을 추가해야 한다.

## 7. 물리적 지위

이번 정리는 external $Z_{\rm phys}$의 실재를 증명하지 않는다. 또한 $Z\to M$ open-channel dynamics, retention map, energy/stress action, dark matter clustering 또는 dark energy pressure를 공급하지 않는다. 가장 강한 결론은 type consistency다. measurement record를 topological-0D라고 부를 수 있지만, strict external physical singleton과 동일시해서는 informative measurement와 기존 no-$M\to Z$를 동시에 유지할 수 없다.

## 8. 재현성

```powershell
& '.codex\hooks\python.cmd' python '_workspace\ce\measurement-record-one-way-compatibility-20260825\artifacts\verify_one_way_record.py'
```

## 참고문헌

- E. B. Davies and J. T. Lewis, “An operational approach to quantum probability,” *Commun. Math. Phys.* 17 (1970), [doi:10.1007/BF01647093](https://doi.org/10.1007/BF01647093), accessed 2026-08-25.
- C. J. Fewster and R. Verch, “Quantum Fields and Local Measurements,” *Commun. Math. Phys.* 378 (2020), [doi:10.1007/s00220-020-03800-6](https://doi.org/10.1007/s00220-020-03800-6), accessed 2026-08-25.
