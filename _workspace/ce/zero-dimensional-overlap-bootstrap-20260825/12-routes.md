# Alternative routes: one-way zero-dimensional boundary and quantum bootstrap

Status: COMPLETE  
Revision: 1 — 중심 화살표를 `Z -> M`으로 고정한다.

## 1. Route ZR-0: bare 0D state-preparing boundary

$Z=\{\star\}$를 동역학적 공간이 아니라 $M$의 초기/경계 상태를 준비하는
domain으로 둔다. 최소 입력 $\mathcal H_Z\cong\mathbb C$에서는

$$
\mathcal E(z)=z\rho_M
$$

이므로 fixed-state preparation이다. $\rho_M$에 복잡한 상관관계를 미리 넣을
수는 있지만, bare $Z$가 스스로 clock과 memory를 갱신하지는 않는다.

**Decision:** 가장 작은 일관된 0D 정의로 채택. 동역학과 history 생성은 다음
단계로 분리한다.

## 2. Route ZR-1: cascaded open quantum channel

상류 source operator $a$와 하류 target operator $b$에 대해

$$
\dot\rho=-i\left[H_A+H_B+\frac{b^\dagger a-a^\dagger b}{2i},\rho\right]
+\mathcal D[a+b]\rho
$$

를 쓴다. target을 trace하면 cross term이 사라져 source reduced dynamics가
target과 독립이고, source를 trace하면 target에는 상류 영향이 남는다.

**Decision:** exact feed-forward를 보이는 최소 수학적 구현으로 채택. 단,
travelling field/reservoir, Markov domain, noise와 energy current를 명시해야
한다. 이 route 자체가 우주론적 0D 원천을 증명하지는 않는다.

## 3. Route ZR-2: chiral/retarded spacetime channel

단방향 propagation을 chiral waveguide나 retarded local field로 구현한다.
근본적 우주론 모형이라면

$$
G_R(x,y)=0\qquad\text{for spacelike }x-y
$$

또는 대응하는 finite-speed network bound를 요구한다. delay를 버린 Markov
cascade는 선언된 coarse-grained 영역에서만 쓴다.

**Decision:** locality까지 요구할 때 선호되는 completion. strict 0D는 경계
label로 남고 실제 전달은 causal channel이 담당한다.

## 4. Route ZR-3: directed nonlinear edge reservoir

$M$ 안의 edge $j\to i$마다

$$
F_{ij}=\sigma_i^+n_j,
\qquad
\mathcal L_{\rm edge}\rho
=\sum_{j\to i}\kappa_{ij}\mathcal D[F_{ij}]\rho
$$

를 둔다. $\kappa_{ij}\ge0$이면 edge-basis Kossakowski matrix가 positive
semidefinite이고, diagonal/decohered sector에서 directed CTMC가 정확히
나온다.

**Decision:** “옆 양자가 다음 양자를 실행”한다는 operator를 직접 구현하는
조건부 route. microscopic nonlinear coupling, pump energy, reservoir spectrum과
edge channel count는 추가해야 한다.

## 5. Route ZR-4: causal measurement and feed-forward

$n_j$를 측정하고 causal record가 도착한 뒤에만 $i$를 구동한다. 이는 방향과
외부 work source를 명확히 노출한다. detector efficiency, latency, measurement
back-action, feedback noise와 record erasure가 추가된다.

**Decision:** 실험 가능한 operational analogue. 닫힌 우주 전체의 근본
bootstrap으로 바로 승격하지 않는다.

## 6. Route ZR-5: selected/nonselected co-output instrument

정규화 전 instrument outputs

$$
\widetilde\rho_a=\mathcal E_a(\rho_Z),
\qquad
\sum_a\mathcal E_a\ \text{CPTP}
$$

를 먼저 만든다. 비선택 record의 subprobability measure를 별도 사상으로

$$
\phi(x)=M_*\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\nu_{\rm ns}(d\gamma)
$$

에 보낸다.

**Decision:** 사용자의 암흑부문 핵심 가설을 가장 정확히 보존하는 route.
instrument까지는 표준 양자형식이고 residual-to-gravity 단계는 CE physical-map
axiom이다. local covariance와 no-double-counting은 아직 미완성이다.

## 7. Route ZR-6: covariant junction and cosmological readout

한 번의 initial preparation이면 이후 $M$에서 total stress conservation을
요구한다. 계속되는 주입이면

$$
\nabla_\mu T_M^{\mu\nu}=J_Z^\nu,
\qquad
\nabla_\mu T_{\rm total}^{\mu\nu}=0
$$

을 동시에 만족시키는 source/channel stress 또는 junction condition을
제시해야 한다. 그 뒤에야 scalar residual EFT의 DM-like oscillation과 DE-like
constant term을 Einstein--Boltzmann perturbations에 연결할 수 있다.

**Decision:** 우주론 검증을 위한 필수 completion. 현재는 미완성이다.

## 8. 비교용으로 내린 이전 common-bus route

reciprocal linear common mode를 제거해 얻는

$$
K=G\mathcal G G^\dagger,
\qquad\operatorname{rank}K\le r
$$

는 유효한 별도 정리다. 그러나 이 모형은 사용자가 정정한 `external 0D -> M`
one-way cascade가 아니다. single-mode rank-1, unwanted all-to-all tail,
instantaneous nonlocality 반례는 이 이전 후보에만 적용한다.

**Decision:** 중심 설명에서 제외하고 rejected comparison으로 보존.

## 9. 닫힌 부모 경로

다음은 완전한 반례 때문에 활성 주장으로 유지하지 않는다.

1. 점 $Z$ 내부에 공간적 한쪽 방향이 있다.
2. bare strict 0D가 외부 parameter 없이 time evolution과 memory update를 한다.
3. 단순 Hermitian exchange pair가 exact no-feedback cascade다.
4. 유한 DAG와 유한 seed가 외부 drive 없이 영원히 새 노드를 실행한다.
5. complete positivity 또는 neighbour occupation이 excitation energy를 공급한다.
6. $M\to Z$-only sink가 $M$ 안에 양의 dark stress를 되돌려 준다.
7. genealogy probability $q$ 또는 $1-q$가 곧바로
   $\Omega_{\rm DM}$ 또는 $\Omega_{\rm DE}$다.

## 10. 권장 폐쇄 순서

가장 작은 비모순 연구 순서는 다음이다.

1. $Z$를 static state-preparing boundary로 고정한다.
2. $Z\to M$을 cascaded/chiral open channel로 구현하고 no-feedback을 증명한다.
3. source, reservoir, noise, work와 heat current를 닫는다.
4. $M$ 안에서 directed neighbour jump의 microscopic origin을 유도한다.
5. 유한 graph와 무한 branching limit의 적용 영역을 분리한다.
6. subnormalized nonselected record를 residual field에 보내는 CE axiom을
   local-covariant하게 명시한다.
7. total stress junction과 no-double-counting을 닫는다.
8. normalization 및 perturbation equations를 제시한 뒤에만 CMB/BAO/lensing/
   structure likelihood를 계산한다.

이 경로는 사용자의 단방향 0D 아이디어와 이웃 bootstrap을 보존하면서,
수학적 존재 증명과 아직 필요한 새 물리를 분리한다.
