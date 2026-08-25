# 외부 0차원 경계에서 현재 시공간으로 향하는 단방향 양자 부트스트랩

Status: COMPLETE

## 초록

본 연구는 외부의 엄밀한 0차원 경계 $Z$가 현재 시공간 $M$으로만 작용한다는
가설을, 점 내부의 방향과 sector 사이의 채널 방향을 구별하여 다시 정의했다.
$\mathcal H_Z\cong\mathbb C$인 최소 모형은 고정 상태 준비 채널이며, 실제
단방향 동역학은 열린 cascaded GKSL channel로 조건부 구현할 수 있다. 현재
차원 안의 directed facilitated jump는 대각 부문에서 birth/death Markov
과정을 만들고, 별도의 무한 Poisson 계보 근사에서는
$q=0.048646719644028225$를 재현한다. 그러나 비선택 결과를 중력원으로 보내는
residual map은 새 물리 공리이고, 경계 에너지·응력 접합과 절대 abundance는
유도되지 않았다. 따라서 이 연구는 단방향 구조의 수학적 구성 가능성만
확립하며, 실제 암흑물질·암흑에너지의 동일성 증명으로 승격하지 않는다.

## 1. 문제와 핵심 결과

이 연구가 묻는 질문은 두 부분으로 나뉜다. 첫째, 시간과 공간 방향이 없는
0차원 경계를 어떻게 현재 시공간으로 향하는 일방향 원천으로 표현할 것인가.
둘째, 그 출력 뒤에 현재 차원 안의 양자들이 다음 양자의 전이를 여는
부트스트랩과 비선택 경로의 암흑부문 readout을 모순 없이 연결할 수 있는가.

핵심 결과는 조건부다. strict 0D 자체를 움직이는 물체로 만들지 않고 경계
자료로 제한하면, $Z\to M$이라는 방향은 채널의 domain/codomain 순서로
일관되게 정의된다. 양자광학에서 source의 출력이 downstream target의 입력이
되는 cascaded system은 이러한 no-feedback channel의 물리적 선례를 제공한다
[Gardiner 1993](https://doi.org/10.1103/PhysRevLett.70.2269),
[Carmichael 1993](https://doi.org/10.1103/PhysRevLett.70.2273). 다만 그 문헌은
CE의 우주론적 0D 경계나 암흑부문 해석을 뒷받침하지 않는다.

## 2. 정의와 표기

현재의 $3+1$차원 시공간을 $M$, 그 바깥의 strict zero-dimensional support를

$$
Z=\{\star\}
$$

로 쓴다. $Z$에는 내부 거리와 공간적 접선 방향이 없다. 물리적 화살표는

$$
Z\xrightarrow{\mathcal E_{Z\to M}}M,
\qquad M\not\to Z
$$

이며, 이는 경계 사상 또는 인과적 feed-forward의 순서다. spacetime dimension과
internal state dimension은 구별한다. 0D support에 내부 label을 둘 수는 있지만,
그 label을 시간에 따라 갱신하려면 $0+1$차원 dynamics나 외부 schedule을
추가해야 한다.

선택과 비선택 outcome은 CP trace-nonincreasing maps
$\mathcal E_{\rm sel}$과 $\mathcal E_{\rm ns}$로 쓴다. 합
$\mathcal E_{\rm sel}+\mathcal E_{\rm ns}$는 CPTP이며,

$$
\widetilde\rho_a=\mathcal E_a(\rho_Z),
\qquad p_a=\operatorname{Tr}\widetilde\rho_a
$$

는 정규화 전 상태와 그 확률이다. 이 표기는 outcome probability와 state
transformation을 함께 다루는 instrument 개념을 따른다
[Davies--Lewis 1970](https://doi.org/10.1007/BF01647093).

## 3. 채택한 공리와 외부 입력

**[공리: 모델 선택] A1.** $Z$는 동역학적 공간이 아니라 $M$의 경계 자료를
준비하는 외부 sector이고, 허용된 기본 방향은 $Z\to M$뿐이다.

**[공리: 물리 사상] A2.** 비선택 outcome의 subprobability measure
$\nu_{\rm ns}$는

$$
\phi(x)=M_*\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\nu_{\rm ns}(d\gamma)
$$

를 통해 $M$의 residual field로 읽힌다. 이 사상은 표준 조건화에서 유도한
결과가 아니다. kernel $\widehat K$, conversion scale $M_*$, matching과
no-double-counting은 모형의 추가 자료다.

**[공리: 유효 모형] A3.** residual field의 최소 readout은

$$
S_{\rm res}=\int d^4x\sqrt{-g}
\left[-\frac12(\nabla\phi)^2-\frac12m^2\phi^2-V_\Lambda\right]
$$

로 둔다. 이 작용은 residual map의 미시적 유도가 아니라, 그 사상을 채택한
뒤 시험할 수 있는 유효장이론이다.

**[공리: 외부 입력] A4.** 수치 예에서 쓰는 무차원 평균 자손수

$$
D=3.1777584234099736
$$

은 선행 CE 원장에서 상속한 등록값이다. 이 run은 $D$를 0D 원리에서 새로
유도하지 않는다.

## 4. 조건부 정리와 증명

### 4.1 최소 0D 입력의 상태 준비 정리

**[정리: 조건부]** $\mathcal H_Z\cong\mathbb C$이고
$\mathcal E:\mathbb C\to\mathcal B(\mathcal H_M)$가 CPTP이면 어떤 density
operator $\rho_M$에 대해

$$
\mathcal E(z)=z\rho_M
$$

이다.

증명. $\mathbb C$에서 나오는 모든 선형 map은
$\mathcal E(z)=z\sigma_M$ 꼴이다. complete positivity는
$\sigma_M\succeq0$을, trace preservation은
$\operatorname{Tr}\sigma_M=1$을 준다. 따라서 $\sigma_M=\rho_M$은 density
operator다. 이 채널은 하나의 고정 출력을 준비할 수 있지만 입력별 memory를
갱신할 자유도는 없다. □

고정 출력 자체가 $M$ 안의 time register나 path correlation을 포함할 수는
있다. 금지되는 것은 bare one-dimensional input이 서로 다른 입력 history를
판별하고 스스로 다음 출력을 선택한다는 해석이다.

### 4.2 열린 cascade의 no-feedback 정리

**[정리: 조건부 구성]** 상류 source $A$와 하류 target $B$의 rate-normalized
coupling operators를 $a,b$라 하고

$$
H_{\rm cas}=\frac{b^\dagger a-a^\dagger b}{2i}
$$

로 두자. 그러면

$$
\dot\rho=-i[H_A+H_B+H_{\rm cas},\rho]
+\mathcal D[a+b]\rho
$$

는 GKSL generator이고, source reduced dynamics는 target에 의존하지 않는다.

증명. $H_{\rm cas}$는 Hermitian이고 $\mathcal D[a+b]$는 Lindblad
dissipator이므로 전체 식은 GKSL 형태다. cross term을 전개하면

$$
\dot\rho=\mathcal L_A\rho+\mathcal L_B\rho
+[a\rho,b^\dagger]+[b,\rho a^\dagger]
$$

를 얻는다. $B$를 부분추적할 때 $B$ 연산자의 cyclicity로 마지막 두 항의 trace가
정확히 0이므로

$$
\dot\rho_A=\mathcal L_A\rho_A
$$

이다. 반대로 $A$를 부분추적한 cross term은 일반적으로 0이 아니므로 source는
target을 구동할 수 있다. □

이 구성은 chiral field, reservoir, collision ordering 또는 feed-forward 같은
열린 환경을 전제한다. modern input-output network 형식도 cascade와 feedback을
서로 다른 network operation으로 다룬다
[Combes--Kerckhoff--Sarovar 2017](https://arxiv.org/abs/1611.00375).
닫힌 단순 exchange Hamiltonian만으로 이 비대칭이 자동 발생한다는 주장은
허용하지 않는다.

### 4.3 directed neighbour jump의 대각 극한

**[정리: 조건부]** occupation basis에서 대각인 상태가 jump dynamics 아래
닫혀 있고

$$
L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\sigma_i^+n_j,
\qquad R_i=\sqrt{\gamma_i}\sigma_i^-
$$

이면 population은

$$
b_i(x)=(1-x_i)\sum_{j:j\to i}\kappa_{ij}x_j,
\qquad d_i(x)=\gamma_i x_i
$$

인 directed continuous-time Markov chain을 따른다.

증명. $n_j|x\rangle=x_j|x\rangle$이고
$\sigma_i^+|x\rangle$은 $x_i=0$일 때만 $i$를 켠다. 따라서 각 edge jump의
squared amplitude는 $(1-x_i)\kappa_{ij}x_j$이고, 같은 target으로 들어오는
독립 jump의 rate를 더하면 $b_i$를 얻는다. lowering jump도 같은 방식으로
$d_i=\gamma_ix_i$를 준다. □

이웃의 점유는 transition gate일 뿐 excitation energy의 공급원이 아니다.
열린 양자계의 에너지·일·열 장부에는 source와 reservoir가 포함되어야 한다
[Strasberg--Winter 2021](https://doi.org/10.1103/PRXQuantum.2.030202).

### 4.4 유한 DAG의 생존 한계

**[정리: 조건부 no-go]** 유한 DAG와 유한 seed만으로 무한한 새-node
activation을 얻을 수 없다.

증명. decay가 없고 exclusion $x_i\in\{0,1\}$이면 각 node는 최대 한 번
활성화되므로 birth 수는 유한하다. 모든 $\gamma_i>0$이고 지속 주입이 없으면
vacuum은 finite CTMC의 유일한 closed class이며 모든 상태에서 도달 가능하므로
거의 확실히 흡수된다. 영구 root나 반복 drive가 있으면 downstream activity가
지속될 수 있지만, 이는 명시된 sustaining source의 결과다. □

### 4.5 무한 Poisson 계보 정리

**[정리: 조건부]** 각 parent가 fresh target에 독립적으로 Poisson$(D)$개의
offspring을 만들고 collision과 target reuse를 무시하면 extinction probability
$q$는

$$
q=e^{-D(1-q)}
$$

의 최소 고정점이다.

증명. offspring generating function은 $G(s)=e^{D(s-1)}$이다. $q_n$을
$n$세대 안의 extinction probability로 두면 branching independence에서
$q_{n+1}=G(q_n)$이므로 그 극한은 $q=G(q)$의 최소해다. $D>1$일 때 작은 해는

$$
q=-\frac1D W_0(-De^{-D})
$$

이다. 이 재귀는 전방 genealogy이므로 reciprocity나 strongly connected
component를 요구하지 않는다. □

### 4.6 abundance 비식별성 정리

**[정리: no-go]** 위 instrument probability와 genealogy 자료만으로
$\Omega_{\rm DM}$, $\Omega_{\rm DE}$ 또는 그 분할을 식별할 수 없다.

증명. homogeneous scalar density는

$$
\rho_\phi=\frac12\dot\phi^2+\frac12m^2\phi^2+V_\Lambda
$$

이다. $(D,q)$를 고정한 채 $M_*$, kernel amplitude, $m$, field amplitude,
$V_\Lambda$ 또는 초기조건을 바꾸면 $\rho_\phi$가 연속적으로 변한다. 따라서
동일한 genealogy에 서로 다른 cosmic abundance가 대응한다. □

## 5. 산출과 조건부 우주론 readout

등록값 $D=3.1777584234099736$을 Poisson 고정점에 넣으면

$$
q=0.048646719644028225,
\qquad1-q=0.9513532803559718,
$$

$$
Dq=0.15458752312007412<1
$$

을 얻는다. deterministic checker에서 fixed-point residual은 $0$이었다. 이
값들은 각각 extinction과 survival probability다.

residual scalar EFT에서는 $H/m\ll1$ 및 $|\dot H|/m^2\ll1$일 때 빠른
quadratic oscillation이 평균적으로 $\rho\propto a^{-3}$, $w\simeq0$이 되어
DM-like하게 거동할 수 있다. 상수 $V_\Lambda$는
$T_{\mu\nu}=-V_\Lambda g_{\mu\nu}$와 $w=-1$을 주므로 DE-like하다. 이 결과는
선언한 action의 조건부 산출이지 실제 암흑성분과의 동일성이나 양의 예측이
아니다.

## 6. 관측 비교

이 0D/cascade 구조에는 아직 관측 가능한 abundance와 perturbation spectrum이
없으므로 직접 likelihood prediction을 만들 수 없다. 선행 관측 감사에서
Planck 2018 base flat $\Lambda$CDM은 $\Omega_m=0.3153\pm0.0073$을 보고했고
[Planck 2018](https://arxiv.org/abs/1807.06209), DESI DR2 2025의 DESI+CMB
결합은 $\Omega_m=0.3027\pm0.0036$을 보고했다
[DESI DR2 Results II](https://arxiv.org/abs/2503.14738). 이 수치와 가까운
계보 확률을 찾는 것은 residual map의 증거가 아니다.

선행 DSO run에서 역사적으로 고정된 density boundary, 외부 $H_0$와 $r_d$를
DESI DR2 13-vector에 넣은 결과는

$$
\chi^2=37.10026085715347,
\qquad\mathrm{dof}=13,
\qquad p=0.000399573259824
$$

로 그 고정 경계를 기각했다. 같은 자료에 global scale을 fit해 적합도를
개선한 결과는 사후 보정이며 abundance 예측이 아니다. 이 관측 실패는
$Z\to M$ channel의 존재를 직접 기각하거나 지지하지도 않는다. 아직 그
channel에서 해당 관측량으로 가는 닫힌 사상이 없기 때문이다.

## 7. 미완성 과제와 한계

가장 먼저 필요한 것은 $Z\to M$ channel의 물리적 실현이다. chiral propagation,
reservoir state, delay, Markov domain, noise와 energy current를 갖춘 microscopic
model이 필요하다. 열린 계의 reduced dynamics가 환경 결합과 근사에 의존한다는
점은 일반 open-system 이론에서도 분명하다
[de Vega--Alonso 2017](https://doi.org/10.1103/RevModPhys.89.015001).

경계가 초기자료만 준비하면 이후 $M$의 visible+residual+channel total stress가
보존되어야 한다. 계속 주입한다면

$$
\nabla_\mu T_M^{\mu\nu}=J_Z^\nu
$$

를 source/channel stress 또는 junction condition과 결합해 전체
$\nabla_\mu T_{\rm total}^{\mu\nu}=0$을 회복해야 한다. 현재는 $J_Z^\nu$,
matching surface와 noise kernel이 없다.

residual map도 미완성이다. $\widehat K$의 공변성과 국소성, 선택 outcome과의
no-double-counting, scalar action의 parameter matching, perturbation initial
conditions를 제시해야 한다. 그 뒤에야 CMB, BAO, lensing과 structure formation에
대한 사전 고정 예측을 만들 수 있다.

반대 화살표 $M\to Z$만 있는 sink는 $M$의 감쇠를 만들 수 있을 뿐, $M$ 안의
새 residual stress를 되돌려 주지 않는다. dark source를 얻으려면 사용자가
고정한 $Z\to M$ co-output 또는 별도 modified-gravity 항이 필요하다.

## 8. 재현성

연구 run은
`_workspace/ce/zero-dimensional-overlap-bootstrap-20260825`에 있다. 핵심
검산 명령은 다음과 같다.

```text
.codex\hooks\python.cmd python _workspace\ce\zero-dimensional-overlap-bootstrap-20260825\artifacts\verify_zero_dimensional_overlap.py
.codex\hooks\run.cmd check _workspace\ce\zero-dimensional-overlap-bootstrap-20260825 final
```

첫 명령은 one-dimensional-input Choi/instrument, cascade algebra, partial-trace
directionality, directed CTMC, branching fixed point, abundance counterexample와
dimensions를 검사한다. source Choi minimum eigenvalue와 upstream-feedback
residual은 모두 $0$, nonselected Choi minimum eigenvalue는
$0.08686291501015238$, summed-map trace-preservation residual은 $0$이었다.

## 9. 참고문헌

모든 링크의 최종 접근일은 2026-08-25이다.

1. C. W. Gardiner, “Driving a Quantum System with the Output Field From Another Driven Quantum System,” *Physical Review Letters* 70, 2269 (1993), [DOI](https://doi.org/10.1103/PhysRevLett.70.2269).
2. H. J. Carmichael, “Quantum trajectory theory for cascaded open systems,” *Physical Review Letters* 70, 2273 (1993), [DOI](https://doi.org/10.1103/PhysRevLett.70.2273).
3. J. Combes, J. Kerckhoff and M. Sarovar, “The SLH framework for modeling quantum input-output networks,” *Advances in Physics: X* 2, 784 (2017), [arXiv](https://arxiv.org/abs/1611.00375), [DOI](https://doi.org/10.1080/23746149.2017.1343097).
4. T. Ramos et al., “Quantum Optics of Chiral Spin Networks,” [arXiv:1411.2963](https://arxiv.org/abs/1411.2963).
5. E. B. Davies and J. T. Lewis, “An operational approach to quantum probability,” *Communications in Mathematical Physics* 17, 239 (1970), [DOI](https://doi.org/10.1007/BF01647093).
6. P. Strasberg and A. Winter, “First and Second Law of Quantum Thermodynamics: A Consistent Derivation Based on a Microscopic Definition of Entropy,” *PRX Quantum* 2, 030202 (2021), [DOI](https://doi.org/10.1103/PRXQuantum.2.030202).
7. I. de Vega and D. Alonso, “Dynamics of non-Markovian open quantum systems,” *Reviews of Modern Physics* 89, 015001 (2017), [DOI](https://doi.org/10.1103/RevModPhys.89.015001).
8. Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
9. DESI Collaboration, “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints,” [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
