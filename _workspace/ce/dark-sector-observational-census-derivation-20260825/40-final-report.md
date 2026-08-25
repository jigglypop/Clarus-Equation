# 비선택 양자경로 암흑부문 가설의 처음부터 유도와 관측 감사

Status: COMPLETE

## 초록

본 연구는 “선택되지 않은 양자경로가 암흑물질과 암흑에너지다”라는 CE의 중심
가설을 이웃 양자 부트스트랩에서 우주론적 관측량까지 단계별로 감사했다. 선언한
facilitated Lindblad jump는 대각 부문에서 정확한 directed Markov 과정을 만들고,
별도의 무한 Poisson 계보 근사에서는 Lambert-$W$ 소멸확률을 준다. 그러나 표준
양자 조건화는 비선택 outcome의 stress를 선택된 우주에 더하지 않으며, 계보
확률은 에너지 밀도분율을 결정하지 않는다. 비선택 history를 residual scalar로
보내는 새 공리를 채택하면 빠른 quadratic oscillation과 constant offset이 각각
DM-like와 DE-like하게 거동할 수 있지만, 그 양과 실제 동일성은 미완성이다. 기존
고정 density boundary는 DESI DR2 full-covariance 진단에서 기각됐으며, 구현은 이
실패를 숨기지 않도록 기본값·provenance·원자료 경로를 격리했다.

## 1. 질문과 판정 구조

연구 질문은 하나의 긴 등식이 아니라 네 개의 독립 다리다. 첫 번째 다리는
양자 $j$의 점유가 이웃 양자 $i$의 전이를 여는가를 묻는다. 두 번째는 그
directed 과정이 무한 계보 근사에서 어떤 소멸확률을 갖는가를 묻는다. 세 번째는
선택되지 않은 outcome이 현재 시공간의 stress tensor로 보존되는가를 묻는다.
마지막은 그 stress가 실제 암흑물질과 암흑에너지의 abundance와 perturbation을
예측하는가를 묻는다.

감사 결과 앞의 두 다리는 명시된 toy/open-system 조건에서 닫을 수 있었다.
세 번째 다리는 표준 양자역학의 결론이 아니라 CE가 추가해야 하는 physical-map
axiom이다. 네 번째 다리는 normalization과 미시 동역학이 없어 비식별적이다.
따라서 “암흑물질·암흑에너지가 비선택 경로다”라는 존재론적 문장은 아직 정리가
아니며, 검증 가능한 연구 가설로 남는다.

후속 0D 감사에서는 사용자의 정정에 따라 외부 경계를 $Z=\{\star\}$, 현재
시공간을 $M$으로 두고 방향을

$$
Z\to M,
\qquad M\not\to Z
$$

로 고정했다. 이는 reciprocal common bus가 아니라 open feed-forward channel이다.
양자광학의 cascaded system은 source output이 downstream target을 구동하는
조건부 구현을 제공한다
[Gardiner 1993](https://doi.org/10.1103/PhysRevLett.70.2269),
[Carmichael 1993](https://doi.org/10.1103/PhysRevLett.70.2273). 이 선례는
우주론적 0D 경계의 존재나 dark-sector identity를 입증하지 않는다.

## 2. 정의와 공리

각 두 준위 양자의 occupation operator를 $n_i$, raising과 lowering operator를
$\sigma_i^+$와 $\sigma_i^-$로 쓴다. directed edge $j\to i$는 이미 활성인
$j$가 $i$의 transition channel을 여는 인과 순서다.

**[공리: 모델 선택] B1.** 이웃 촉진 jump와 decay jump를

$$
L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\sigma_i^+n_j,
\qquad R_i=\sqrt{\gamma_i}\sigma_i^-
$$

로 선언한다. $\kappa_{ij}$와 $\gamma_i$는 $T^{-1}$ 차원의 rate다. 이웃은
전이의 gate이며 pump energy의 다른 이름이 아니다.

**[공리: branching limit] B2.** 무한 계보 계산에서만 fresh target, 독립
offspring, decorrelated bath record, 고정 parent window와 negligible collision을
가정한다. 이 근사는 유한 Lindblad graph와 동일하지 않다.

**[공리: 물리 사상] B3.** quantum instrument가 정의한 nonselected history의
subprobability measure $\nu_{\rm ns}$를

$$
\phi(x)=M_*\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\nu_{\rm ns}(d\gamma)
\tag{1}
$$

로 residual scalar에 보낸다. 식 (1)은 CE가 새로 채택하는 사상이다. 표준
conditioning에서 자동으로 나오지 않는다.

**[공리: 유효 모형] B4.** residual readout의 최소 action은

$$
S_{\rm res}=\int d^4x\sqrt{-g}
\left[-\frac12(\nabla\phi)^2-\frac12m^2\phi^2-V_\Lambda\right]
\tag{2}
$$

로 둔다. $m$, $M_*$, $V_\Lambda$와 초기조건은 아직 예측값이 아니다.

## 3. 이웃 양자 부트스트랩의 정확한 범위

### 3.1 대각 부문의 Markov 정리

**[정리: 조건부]** density matrix가 occupation basis에서 diagonal이고 그
부분공간을 깨는 coherent Hamiltonian이 없으면 B1의 Lindblad dynamics는

$$
b_i(x)=(1-x_i)\sum_{j:j\to i}\kappa_{ij}x_j,
\qquad d_i(x)=\gamma_i x_i
\tag{3}
$$

인 continuous-time Markov chain과 정확히 같다.

증명. $n_j|x\rangle=x_j|x\rangle$이고 $\sigma_i^+$는 $x_i=0$일 때만 node
$i$를 올린다. 따라서 edge $j\to i$의 squared jump amplitude는
$(1-x_i)\kappa_{ij}x_j$다. 같은 target으로 들어오는 독립 jump rates를 더하면
$b_i$가 된다. lowering jump는 $x_i=1$일 때만 작동해 $d_i=\gamma_ix_i$를
준다. □

coherent off-diagonal Hamiltonian이 있으면 population은 coherence와 결합해
일반적으로 닫히지 않는다. 따라서 식 (3)은 모든 양자계에 대한 보편식이
아니다.

### 3.2 유한계 생존 no-go와 에너지

**[정리: no-go]** 모든 $\gamma_i>0$인 유한 graph에 sustaining drive가 없으면
vacuum은 유일한 closed class이고 과정은 거의 확실히 흡수된다.

증명. vacuum에서 모든 facilitated upward jump는 parent occupation이 0이므로
사라지고, 모든 decay jump도 사라진다. 임의의 비진공 상태에서는 양의 decay
rates를 따라 유한한 순서로 vacuum에 도달할 확률이 양수다. 유한-state Markov
chain에서 다른 closed class가 없으므로 absorption probability는 1이다. □

strongly connected component는 mutual facilitation의 support만 말하며 생존을
보장하지 않는다. 또한 $L_{i\leftarrow j}$는 총 excitation number를 하나 늘릴
수 있으므로 필요한 에너지는 source field, pump 또는 bath에서 와야 한다. 열린
계의 energy/work/heat accounting은 환경과 control을 포함해야 한다
[Strasberg--Winter 2021](https://doi.org/10.1103/PRXQuantum.2.030202).

## 4. 무한 Poisson 계보와 고정점

B2에서 parent type $j$가 type $i$의 자손을 평균 $A_{ji}$개 만든다고 하자.
type별 extinction probability $q_j$는 generating function에서

$$
q_j=\exp\left[\sum_iA_{ji}(q_i-1)\right]
\tag{4}
$$

을 만족한다. irreducible branching class에서 nonzero survival에는
$\rho(A)>1$이 필요하고 seed가 그 class에 도달해야 한다.

uniform mean offspring $D$에서는

$$
q=e^{-D(1-q)}.
\tag{5}
$$

**[정리: 조건부]** $D>1$일 때 식 (5)의 extinction probability는 작은
고정점

$$
q_{\rm ext}=-\frac1D W_0(-De^{-D})
\tag{6}
$$

이다.

증명. offspring generating function $G(s)=e^{D(s-1)}$를 세대별 extinction
recursion에 적용하면 $q_{n+1}=G(q_n)$이다. $h(q)=\log q+D(1-q)$는 strict
concave이며 $D>1$일 때 $(0,1/D)$에 작은 root 하나와 $q=1$ root를 갖는다.
Lambert-$W$의 principal branch $W_0$가 작은 root를, $W_{-1}$가 $q=1$을
선택한다. □

CE가 상속한 $D=3.1777584234099736$을 넣으면

$$
q_{\rm ext}=0.048646719644028225,
\qquad s_{\rm branch}=1-q_{\rm ext}=0.9513532803559718,
$$

$$
Dq_{\rm ext}=0.15458752312007412.
\tag{7}
$$

식 (7)은 genealogy의 extinction와 survival probability다. mutual cycle이나
SCC가 필요하지 않다는 점에서 $Z\to M$의 전방 cascade와 구조적으로 맞지만,
그 사실이 $D$의 미시적 기원을 유도하지는 않는다. rate model에서
$A_{ji}=\kappa_{ij}\tau$처럼 무차원 window가 필요하다.

## 5. 표준 조건화가 residual gravity를 주지 않는 이유

outcome $r$을 갖는 quantum instrument를 $\{\mathcal I_r\}$라 하자. outcome
$0$이 기록되면 조건부 상태는

$$
\rho_0=\frac{\mathcal I_0(\rho)}
{\operatorname{Tr}\mathcal I_0(\rho)}.
\tag{8}
$$

**[정리: 반례]** 표준 조건화만으로 complementary output
$\mathcal I_1(\rho)$가 $\rho_0$의 local stress source에 더해지지 않는다.

증명. outcome $0$에서 local observable의 조건부 기대값은
$\operatorname{Tr}(O\rho_0)$다. 식 (8)에는 $\mathcal I_1(\rho)$가 없다.
따라서 “기록되지 않았다”는 이유만으로 complementary state를 더하는 규칙은
instrument formalism에서 나오지 않는다. 그러한 합을 원하면 별도의 map을
추가해야 한다. □

또한 같은 Lindblad master equation은 여러 unravelling을 가질 수 있으므로,
jump notation 자체가 우주의 실제 record ontology를 선택하지 않는다. 그러므로
식 (1)은 허용 가능한 새 모형 공리지만, 표준 양자역학에서 유도됐다고 부를 수
없다.

## 6. 확률에서 에너지로 가는 no-go

사건 $E$의 probability를 $q$라 하고 경로별 energy weight를 $W$라 하면 실제
energy fraction은

$$
\Omega_E=\frac{\mathbb E[W\mathbf 1_E]}{\mathbb E[W]}.
\tag{9}
$$

조건부 평균을 전개하면

$$
\Omega_E-q=
\frac{q(1-q)\left(\mathbb E[W\mid E]-
\mathbb E[W\mid E^c]\right)}{\mathbb E[W]}.
\tag{10}
$$

따라서 두 conditional mean energy가 같다는 추가 가정이 있을 때만
$\Omega_E=q$다. 예를 들어 $q=0.0486467$에서 두 outcome의 weights를 9와 1로
두면 energy fraction은 약 $0.3151661$로 바뀐다. probability normalization은
energy normalization이 아니다.

식 (2)의 homogeneous density도

$$
\rho_\phi=\frac12\dot\phi^2+\frac12m^2\phi^2+V_\Lambda
\tag{11}
$$

이므로 같은 $(D,q)$에서 $m$, amplitude, $V_\Lambda$를 바꾸면 연속적인 density
family를 얻는다. 감사 certificate는 같은 $q$와 $m=7$에서 $(A,V)=(1,0)$과
$(2,5)$를 택해 $24.5$와 $103.0$이라는 서로 다른 density를 재현했다.

**[정리: 비식별성 no-go]** $(D,q)$, graph와 instrument probability만으로
$\Omega_{\rm DM}$, $\Omega_{\rm DE}$ 또는 그 비를 결정할 수 없다. 따라서
$q\to\Omega_b$, $1-q\to$ dark abundance와
$\alpha_sD\to\Omega_{\rm DM}/\Omega_{\rm DE}$ 부모식은 활성 derivation에서
제거한다.

## 7. residual scalar의 조건부 DM-like와 DE-like 극한

식 (2)의 homogeneous field equation은

$$
\ddot\phi+3H\dot\phi+m^2\phi=0.
\tag{12}
$$

$\psi=a^{3/2}\phi$로 치환하면

$$
\ddot\psi+\left(m^2-\frac32\dot H-\frac94H^2\right)\psi=0.
\tag{13}
$$

**[정리: 조건부]** $H/m\ll1$ 및 $|\dot H|/m^2\ll1$인 WKB regime에서는

$$
\phi=a^{-3/2}\left[A\cos(mt+\delta)+O(H/m)\right],
$$

$$
\langle\rho_{\rm osc}\rangle\propto a^{-3},
\qquad\langle w_{\rm osc}\rangle\simeq0
\tag{14}
$$

이므로 quadratic component는 DM-like하다. 이는 모든 scale에서 exact CDM이라는
뜻이 아니다. perturbation에서는 scalar sound speed와 Jeans scale 때문에
$m$, fraction, transfer function, lensing과 structure growth가 독립 falsifier가
된다.

**[정리: 조건부]** constant $V_\Lambda$ component는

$$
T^{(\Lambda)}_{\mu\nu}=-V_\Lambda g_{\mu\nu},
\qquad p_\Lambda=-\rho_\Lambda,
\qquad w=-1
\tag{15}
$$

을 정확히 만족한다. 이 정리는 $V_\Lambda$의 관측 크기나 radiative stability를
설명하지 않는다.

## 8. 관측 census와 고정 경계의 실패

관측값은 residual map의 증거가 아니라 후속 모형이 통과해야 할 기준선이다.
서로 다른 likelihood와 데이터 결합을 한 행에 섞지 않았다.

| 자료 | 보고된 대표 결과 | 이 연구에서의 역할 |
|---|---|---|
| Planck 2018 base flat $\Lambda$CDM | $\omega_b=0.02237\pm0.00015$, $\omega_c=0.1200\pm0.0012$, $H_0=67.36\pm0.54$, $\Omega_m=0.3153\pm0.0073$, $\Omega_\Lambda=0.6847\pm0.0073$ | CMB 기준선; CE 증거 아님 |
| DESI DR2 2025, DESI+CMB | $\Omega_m=0.3027\pm0.0036$, $H_0=68.17\pm0.28$ | BAO+CMB 기준선 |
| DESI DR2 Ly$\alpha$ 2026 | $D_H/r_d=8.600\pm0.066$, $D_M/r_d=39.32\pm0.33$ at $z=2.33$ | 2025 분석과 독립된 업데이트 |
| DES Y6 2026 | Y6-only $S_8=0.789\pm0.012$; joint $w=-0.981^{+0.021}_{-0.022}$ | growth와 equation-of-state 기준선 |
| SH0ES 2024 | $H_0=73.17\pm0.86\;{\rm km\,s^{-1}\,Mpc^{-1}}$ | local distance-ladder 기준선 |

출처는 각각 [Planck 2018](https://arxiv.org/abs/1807.06209),
[DESI DR2 Results II](https://arxiv.org/abs/2503.14738),
[DESI DR2 Results IV](https://arxiv.org/abs/2607.27410),
[DES Y6](https://arxiv.org/abs/2601.14559),
[SH0ES](https://arxiv.org/abs/2404.08038)이다.

역사적 runtime tuple $(0.0487,0.2623,0.6891)$에서 late-time matter와
$\Lambda$만 flat-normalize하면

$$
\Omega_m=0.31096890310968905,
\qquad\Omega_\Lambda=0.6890310968903111
$$

이다. 이를 외부 $H_0=67.4$와 $r_d=147.09\,{\rm Mpc}$에 고정하고 DESI DR2
13-vector와 full covariance에 넣으면

$$
\chi^2=37.10026085715347,
\qquad\mathrm{dof}=13,
\qquad p=0.000399573259824.
\tag{16}
$$

따라서 이 **고정 경계 패키지**는 기각된다. 큰 covariance contributions는
$z=0.934$의 $D_M$, $z=0.706$의 $D_H$와 $D_M$, $z=0.510$의 $D_H$에서 나왔다.

한 global scale을 자료에 맞추면

$$
s=0.986476933470,
\qquad\chi^2=12.608346862,
\qquad\mathrm{dof}=12,
\qquad p=0.398138192515
\tag{17}
$$

로 개선된다. 이는 고정 $H_0$에서 $r_d=149.106375435\,{\rm Mpc}$ 또는 고정
$r_d$에서 $H_0=68.323949312$에 해당한다. 그러나 scale을 같은 자료에서 fit했기
때문에 식 (17)은 prediction이 아니다. 식 (16)과 (17)의 비교에서 기존 실패의
주된 위치가 redshift shape 전체보다 absolute BAO scale/calibration package에
있다는 진단만 얻는다.

## 9. 구현된 감사 경계

과거 runtime tuple은 이제 명시적 historical negative control로만 사용할 수
있다. ordinary forward model은 세 density boundary를 모두 받아야 하며, 그
provenance는 `adopted_or_external_boundary`다. `ce_prediction`이라는 density
role은 제거됐다.

DESI named dataset은 Cobaya `bao_data` v2.6의 raw mean과 covariance를 exact
bytes와 SHA-256으로 확인하고, vector order, covariance symmetry와 positive
definiteness를 검사한 뒤에만 읽는다. 고정된 값은

$$
\begin{aligned}
\text{mean: }&472\ \text{bytes},\quad
9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585,\\
\text{cov: }&2547\ \text{bytes},\quad
252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509.
\end{aligned}
$$

중복 embedded DESI constants와 public runtime-default aliases도 제거해 검증된
경로를 우회하지 못하게 했다. 이 구현은 provenance와 재현성을 보강한 것이며
물리 공리를 증명하는 software가 아니다.

## 10. 남은 미완성 다리

첫째, $Z\to M$ open channel의 microscopic source, reservoir, delay, noise와
energy current가 필요하다. strict 0D는 state-preparing boundary일 수 있지만
bare point 자체는 clock이나 pump가 아니다.

둘째, 식 (1)에 instrument/history algebra, local-covariant kernel, matching
hypersurface와 no-double-counting을 부여해야 한다. 연속 주입이면

$$
\nabla_\mu T_{\rm res}^{\mu\nu}=Q^\nu,
\qquad
\nabla_\mu(T_{\rm vis}^{\mu\nu}+T_{\rm channel}^{\mu\nu})=-Q^\nu
\tag{18}
$$

처럼 total stress를 닫아야 한다.

셋째, $m$, amplitude, $V_\Lambda$, $M_*$와 초기조건을 예측하는 normalization
law가 필요하다. 그 뒤에야

$$
(S_{\rm res},T_{\mu\nu},\text{initial data},\text{other species})
\to H(z),D(z),P(k,z)
\to\text{CMB/BAO/lensing/structure observables}
\tag{19}
$$

라는 Einstein--Boltzmann forward model을 닫을 수 있다.

현재 가장 강한 판정은 다음과 같다. **비선택 quantum history를 residual scalar로
보내는 CE physical-map axiom과 directed bootstrap motif는 함께 연구할 수 있고,
그 최소 scalar EFT에는 DM-like와 DE-like 극한이 존재한다. 그러나 실제
암흑물질·암흑에너지가 비선택 경로라는 동일성, 그 절대량, 영구 self-execution은
아직 확립되지 않았다.**

## 11. 재현성

연구 run은
`_workspace/ce/dark-sector-observational-census-derivation-20260825`에 있고,
검토용 구현 staging은 `.tmp/ce-cosmo-dso-20260825`에 있다. 감사된 16개 경로는
정본 `C:\dev\ce\ce-cosmo`의 base HEAD
`f78accbdd075454437e57ff39b6b6b0154088c10` 위에 미커밋 상태로 반영했다. 정본과
staging의 16개 해시는 모두 일치했다. 최종 집중 검증은 split dependency를 명시한
정책 허용 Python harness로 정본에서 다시 실행했다.

```powershell
$env:PYTHONPATH='C:\dev\ce\ce-cosmo\src;C:\dev\ce\ce-core\src'
& 'C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\.codex\hooks\python.cmd' pytest `
  tests\test_cosmology_registry.py `
  tests\test_cosmology_ratio_audit.py `
  tests\test_ce_residual_forward_model.py `
  tests\test_cosmology_closure_gate.py `
  tests\test_recombination_drag_adapter.py -q
```

결과는 `58 passed in 0.90s`였다. `git diff --check`와 독립 canonical
post-application audit도 남은 P0/P1 없이 통과했다. 연구 체인의 최종 구조 검사는
다음 명령으로 재현한다.

```text
.codex\hooks\run.cmd check _workspace\ce\dark-sector-observational-census-derivation-20260825 final
```

## 12. 참고문헌

모든 링크의 최종 접근일은 2026-08-25이다.

1. Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
2. DESI Collaboration, “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints,” [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
3. DESI Collaboration, “DESI DR2 Results IV: Lyman-alpha forest BAO,” [arXiv:2607.27410](https://arxiv.org/abs/2607.27410).
4. DES Collaboration, Year 6 cosmology analysis, [arXiv:2601.14559](https://arxiv.org/abs/2601.14559).
5. A. G. Riess et al., SH0ES distance-ladder analysis, [arXiv:2404.08038](https://arxiv.org/abs/2404.08038).
6. C. W. Gardiner, “Driving a Quantum System with the Output Field From Another Driven Quantum System,” *Physical Review Letters* 70, 2269 (1993), [DOI](https://doi.org/10.1103/PhysRevLett.70.2269).
7. H. J. Carmichael, “Quantum trajectory theory for cascaded open systems,” *Physical Review Letters* 70, 2273 (1993), [DOI](https://doi.org/10.1103/PhysRevLett.70.2273).
8. P. Strasberg and A. Winter, “First and Second Law of Quantum Thermodynamics: A Consistent Derivation Based on a Microscopic Definition of Entropy,” *PRX Quantum* 2, 030202 (2021), [DOI](https://doi.org/10.1103/PRXQuantum.2.030202).
