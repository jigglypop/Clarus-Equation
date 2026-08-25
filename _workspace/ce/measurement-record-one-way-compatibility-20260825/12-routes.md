# 일방향성과 0차원 기록의 대안 경로

Status: COMPLETE

| route | physical structure | one-way $Z_{\rm phys}\to M$ | informative record | verdict |
|---|---|---:|---:|---|
| R0 | strict singleton $Z_{\rm phys}$와 별도 finite discrete $R$ | 유지 | $M\to R$에서 가능 | 권장 최소 분리 |
| R1 | $R=Z_{\rm phys}$인 $n$-atom discrete sector | no-$M\to Z$ 포기 | 가능 | topological 0D이나 strict singleton 아님 |
| R2 | $M\to R=Z$ 정보는 허용하되 energy/dynamical feedback만 금지 | 약화된 의미에서 유지 | 가능 | 새 one-way 정의 필요 |
| R3 | 허용 state subset에서만 $p_r$ 상수 | subset에서만 no signal | subset 밖에서 가능 | superselection/restricted-control 조건 필요 |
| R4 | $Z_n\to M$ exogenous selector/preparation | 유지 | $M$을 읽지는 않음 | Born-dependent measurement가 아니라 준비 경로 |

## R0. strict physical 0D와 record 0D를 분리

$$
Z_{\rm phys}\xrightarrow{\mathcal E_{Z\to M}}M,
\qquad
M\xrightarrow{\mathcal C}R,
\qquad
R\ne Z_{\rm phys}.
$$

$Z_{\rm phys}=\{\star\}$는 fixed preparation boundary로 남고, $R=\mathcal O$는 apparatus/environment 안의 finite discrete record다. 둘 다 일상어로 “0D”라 부를 수 있지만 전자는 singleton physical input, 후자는 multi-atom topological-0D outcome space다. 이 경로는 이전 strict one-way 조건과 실제 informative measurement를 모두 보존한다.

persistent fold-memory에 연결하려면 retention map은 $R$에서 $M$ 내부의 carrier measure로 가야 한다:

$$
R\xrightarrow{\mathcal R_{R\to F}}\mu_F\subset M.
$$

이는 $M\to Z_{\rm phys}$를 만들지 않지만, standard quantum mechanics에서 자동으로 나오지 않는 별도 물리 사상이다.

## R1. multi-atom 0D sector로 재정의

$Z$를 singleton이 아니라 $Z_n=\{z_1,\ldots,z_n\}$로 바꾸면 topological dimension은 여전히 0이고 records를 저장할 수 있다. 그러나 physical record가 measurement input에 따라 바뀌면 $M\to Z_n$ 정보 channel이 생긴다. 따라서 strict no-$M\to Z$는 포기해야 한다. 이 경로는 “0차원”을 살리지만 이전 “strict singleton + 일방향” 정의를 바꾼다.

## R2. 정보 흐름과 dynamical feedback을 분리

record 정보 $M\to Z$는 허용하고, $Z$의 subsequent state가 $M$의 reduced dynamics나 stress에 되먹임하지 않는다고 one-way를 다시 정의할 수 있다. 그러면 readout은 가능하지만 정보 channel 자체는 양방향 구조의 일부다. no-signalling R4와 같은 명제가 아니므로 새로운 channel/network 공리가 필요하다.

## R3. 제한 상태 집합

허용 preparation 집합 $\mathcal S$에서만

$$
\operatorname{tr}(E_r\rho)=c_r
\qquad(\rho\in\mathcal S)
$$

를 요구하면 controllable signalling을 막을 수 있다. 하지만 $\mathcal S$ 밖의 상태를 준비할 수 없다는 superselection 또는 control restriction이 필요하며, $\mathcal S$ 안에서는 record가 그 상태들을 구분하지 못한다. 일반 measurement 해법은 아니다.

## R4. external selector

finite topological-0D alphabet $Z_n$이 label $z$에 따라 $\rho_z$를 $M$에 준비하는

$$
\mathcal E_{Z_n\to M}\left(\sum_zq_z|z\rangle\langle z|\right)
=\sum_zq_z\rho_z
$$

는 genuinely one-way다. 그러나 이는 $M$의 미지 상태를 읽는 measurement가 아니라 exogenous randomized preparation이다. $q_z$가 $M$의 상태에 맞춰 Born probability로 변하려면 common cause, global constraint 또는 $M\to Z$ dependence를 추가해야 한다.

## 선택

현재 정의를 가장 적게 바꾸는 경로는 R0이다. “0차원 측정”은 completed record의 topological dimension으로 남기고, 외부 strict physical 0D는 별도의 one-way preparation boundary로 둔다. **strict singleton, informative record, strict no-$M\to Z$를 동시에 요구하면서** 두 대상을 동일시하는 부모 주장만 제거한다. R1--R4처럼 전제를 바꾸는 경로까지 배제하지 않는다. 어떤 경로도 energy, stress 또는 dark-sector identity를 공급하지 않는다.
