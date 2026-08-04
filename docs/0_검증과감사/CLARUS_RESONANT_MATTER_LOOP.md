# 클라루스 공명 물질생성 루프

작성일: 2026-08-04  
코드: `clarus_resonant_matter.py`, `casimir_carrier_target.py`,
`global_throat_exact_certificate.py`  
실행: `examples/physics/clarus_resonant_matter_gate.py`

## 1. 이번 루프의 질문

검사할 가설은 다음이다.

\[
\text{Clarus 장 중첩}
\rightarrow
\text{비선형 공명}
\rightarrow
\text{특정 모드의 물질 생성}
\rightarrow
\text{경계 응집상}
\rightarrow
\langle T_{\mu\nu}\rangle_{\rm ren}<0.
\]

이 사슬에서 앞 단계의 통과가 뒤 단계를 자동으로 증명하지 않도록 각 화살표를
독립 gate로 분리했다.

독립 toy branch의 최고 판정은

\[
\boxed{
\text{CONDITIONAL ASYMPTOTIC DAUGHTER EXCITATION}
\neq
\text{PHYSICAL CLARUS MATTER}
\neq
\text{NEGATIVE STRESS SOURCE}
}
\]

이다. 명시적으로 공급한 toy coupling 아래 smooth finite pulse가 bosonic daughter
mode를 여기할 수 있다는 수치 control은 닫혔다. CE-physical chain은 그보다 앞선
pole/vertex에서 막혀 있으며 현재 최고 지위는 target-scale calibration뿐이다.
경계물질과 재규격화 응력도 열려 있다.

## 2. ideal-planar 형식 scale 기준 통합

과거 코드에는 서로 다른 두 scale이 함께 있었다.

| 구분 | separation | wavelength | energy | 역할 |
|---|---:|---:|---:|---|
| 현재 full tensor, \(b'(r_0)=-1/3\) | \(4.0535640\times10^{-18}\) m | \(8.1071280\times10^{-18}\) m | \(152.9323309\) GeV | 현재 형식 scale |
| 과거 \(b'(r_0)=-1\) null control | \(3.6628086\times10^{-18}\) m | \(7.3256171\times10^{-18}\) m | \(169.2474456\) GeV | legacy control |

새 `exact_casimir_carrier_target()`은

\[
\frac{(\rho,p_r,p_t)}{C}
=\left(-\frac13,-1,+\frac13\right),
\qquad
C=\frac{c^4}{8\pi G r_0^2}
\]

인 현재 target만 반환한다. 169 GeV 값은
`legacy_bprime_minus_one_null_control()`로 분리했다. 물리 규모 모듈도 이 공통
target을 사용하도록 바꿔 수치 drift를 막았다.

여기서 Casimir density가 직접 정하는 것은 ideal parallel-plate separation
\(a\)다. \(\lambda_*=2a\)는 그 뒤에 붙인 planar cavity의 최저 normal-mode
선택이고,

\[
E_*=\frac{hc}{2a}
\]

는 그 선택에 따른 형식적 cavity energy scale이다. spherical throat의 실제
경계 고유모드에서 유도된 carrier가 아니며, Casimir stress를 한 photon 또는 한
파장이 만드는 것도 아니다. 전체 진공 spectrum의 재규격화 적분이 필요하다.

1 m target과 29.64757 MeV pole 후보의 에너지 비는

\[
\frac{152.9323309\ {\rm GeV}}{29.64757\ {\rm MeV}}
=5158.342857\ldots
\]

이다. 최근접 정수 5158을 곱해도 약 \(10.16\) MeV detuning이 남는다.

단, 이 비를 곧바로 “5158차 harmonic이 필수”라고 읽는 것도 틀리다.
29.64757 MeV가 실제 입자의 pole mass라면

\[
E_\Phi^2=p_\Phi^2+m_\Phi^2
\]

에 의해 고운동량 152.93 GeV Clarus mode가 원리상 가능하다. 현재 막힌 것은
주파수의 수학적 가능성이 아니라 pole, dispersion, 구동 vertex의 CE 유도다.
high-\(Q\)는 중심 주파수를 이동시키지 않으며 선폭과 저장시간만 바꾼다.

## 3. 세 에너지의 분리

다음 세 값을 동일시하지 않는다.

1. pump/mixing이 공급하는 총 에너지-운동량 \(Q^\mu\)
2. 생성되는 daughter의 질량과 운동량
3. \(\lambda=2a\) 선택에서 얻는 \(E_*=152.93\) GeV 형식적 cavity scale

공간적으로 균일한 pair control에서만

\[
Q^0\simeq2E_\chi,
\qquad
E_\chi=\sqrt{m_\chi^2+p_\chi^2}
\]

라고 쓸 수 있다. 152.93 GeV line으로 정지한 동일입자 쌍을 만든다면 각 daughter의
최대 질량은 76.465 GeV다. daughter 질량 자체를 152.93 GeV로 두면 최소
305.86 GeV의 pair line이 필요하다.

## 4. 위상까지 포함한 \(\Phi^2\) spectrum

1축 표기에서

\[
\Phi(x)=\sum_i A_i\cos(K_i\!\cdot x+\theta_i)
\]

이면 정확히

\[
\begin{aligned}
\Phi^2={}&\frac12\sum_iA_i^2
+\frac12\sum_iA_i^2\cos(2K_i\!\cdot x+2\theta_i)\\
&+\sum_{i<j}A_iA_j
\left[
\cos((K_i-K_j)\!\cdot x+\theta_i-\theta_j)
+\cos((K_i+K_j)\!\cdot x+\theta_i+\theta_j)
\right].
\end{aligned}
\]

따라서 가능한 quadratic line은

\[
2K_i,\qquad K_i+K_j,\qquad K_i-K_j,\qquad Q=0
\]

이다. 같은 \(Q\)로 모이는 항은 power를 더하지 않고 복소 phasor를 더해야 한다.
기본 구현은 정확히 같은 Fourier key만 병합한다. 유한 tolerance clustering을
명시적으로 켜면 finite-resolution 근사로 강등되며 downstream report는 이를
exact spectrum으로 받아들이지 않는다. 예를 들어 \(10^{11}\) eV와
\(10^{11}+0.05\) eV의 반대위상 mode는 장시간 beat를 가지므로 서로 지우지 않는다.
위상 cancellation tolerance도 기본값 0이다. 비영 tolerance로 작은 잔차를
버리면 exact flag가 내려간다. pump linewidth는 line-shape model 없이
Lorentzian처럼 단순 합산하지 않으며 quadratic-line linewidth는 미유도로 남긴다.

명시적 반례는

\[
\Phi=\cos t-\frac12\cos3t
\]

이다. self \(2t\) 성분과 difference \(3t-t\) 성분의 크기가 같고 위상이 반대라
\(\Phi^2\)의 \(2t\) line은 정확히 사라진다. 새 코드는 이 cancellation을 회귀
테스트로 고정했다.

## 5. 주파수 대신 collinear invariant pair gate

현재 코드는 횡운동량을 0으로 고정한 1+1차원 collinear control,
\(Q=(Q^0,Q_z)\)를 계산한다. 이 sector에서 동일질량 daughter 두 개의
kinematic threshold는

\[
\boxed{Q^0>0,\qquad Q^2\ge4m_\chi^2}
\]

이다. 이 식 자체는 exact지만 full 3+1차원 normalization과 횡운동량 spectrum은
아직 구현되지 않았다. collinear 질량중심계에서는

\[
E_{\chi,*}=\frac{\sqrt{Q^2}}2,
\qquad
|\mathbf p_{\chi,*}|=\frac12\sqrt{Q^2-4m_\chi^2}.
\]

strict PASS에는 tolerance를 더하지 않는다. 수치 tolerance 안에만 들어오는
channel은 별도 ambiguous flag일 뿐 threshold를 열지 못한다. 특히
\(Q^2=0,m_\chi=0\)인 null splitting은 kinematically open일 수 있어도 rest
frame이 없으므로 위 질량중심계 에너지·운동량을 반환하지 않는다.

이 gate는 다음 false positive를 제거한다.

- 같은 방향 massless pump 두 개는 총 에너지가 커도 \(Q^2=0\)이므로 massive pair를
  만들 수 없다.
- 동일질량 on-shell mode의 difference line은 spacelike 또는 null이라 massive pair에
  부적합하다.
- 단일 진행파의 self line \(Q=2K\)는 \(Q^2=4m_\Phi^2\)로 고정된다. 실험실
  에너지를 올리는 것만으로 무거운 pair threshold가 오르지 않는다.
- 반대방향 두 mode의 sum은 timelike이므로 heavy pair에 쓸 수 있다.

테스트 control \(K_1=(10,+10)\), \(K_2=(10,-10)\)에서는 self line 두 개가
null이라 실패하고 sum line \(Q=(20,0)\) 하나만 \(m_\chi=4\) pair gate를 통과한다.

## 6. 가장 건설적인 standing-wave 구성

\[
\Phi=A\cos(\omega t)\cos(kz)
\]

이면

\[
\Phi^2=\frac{A^2}{4}
\left[
1+\cos2\omega t+\cos2kz+\cos2\omega t\cos2kz
\right].
\]

이 한 구성에서

- \(2\omega\): timelike pair drive
- \(2k\): static effective-mass grating
- \((2\omega,2k)\): spacetime mixing line

이 동시에 생긴다.

pump wavelength를 현재 \(2a=8.1071280\times10^{-18}\) m로 잡으면 static
grating의 주기는 정확히 \(a=4.0535640\times10^{-18}\) m가 된다. 이때 pump
quantum은 약 152.93 GeV이고 timelike pair line은 약 305.86 GeV다. 29.64757
MeV pole mass를 가정하면 \(E_{\rm pump}=\sqrt{p^2+m_\Phi^2}\)라
\(E_{\rm pump}-E_*=2873.749\) eV,
\(2E_{\rm pump}-2E_*=5747.499\) eV가 남는다. 따라서 “정확한 factor 2”가
아니라 pair-line detuning과 실제 linewidth를 비교해야 한다. 현재 API의
linewidth는 사용자가 공급하는 비교 폭일 뿐이며 Lorentzian FWHM 또는 유한 지지
선모양이 유도된 값은 아니다.

## 7. smooth finite-pulse Bogoliubov control

선택적 bosonic EFT에서

\[
\mathcal L_{\rm int}=-\frac g2\Phi^2\chi^2
\]

를 받아들이면 실제 \(\Phi^2\)에는 DC mass shift와 oscillatory term이 함께
생긴다. 현재 solver는 그중 평균을 뺀 generic mass modulation control이다.
따라서 직접 \(\Phi^2\chi^2\) mapping에는 DC shift를 별도로 포함해야 한다.
prescribed pump 아래 단일 daughter mode는 무차원 시간 \(\tau\)에서

\[
u''+
\left[
\nu^2+q\,s(\tau)\cos(\tau+\theta)
\right]u=0
\]

을 따른다. envelope \(s(\tau)\)는 양 끝에서 0이고 sin-squared ramp를 사용한다.
초기 in-vacuum은

\[
u(0)=\frac1{\sqrt{2\nu}},
\qquad
u'(0)=-i\nu u(0)
\]

이며 out 영역에서

\[
n_k=
\frac{|u'|^2+\nu^2|u|^2}{2\nu}-\frac12
\]

를 계산한다. 정규화 control은

\[
i(u^*u'-uu'^*)=1
\]

이다.

현재 실행 control은

\[
\nu=0.5,
\qquad q=0.05,
\qquad 8\text{ cycles},
\qquad 2\text{ ramp cycles}
\]

이며 결과는 다음과 같다.

| 계산 | occupation |
|---|---:|
| N | 1.1814791657 |
| 2N | 1.1814792141 |
| 4N | 1.1814792170 |

\[
|n_{2N}-n_{4N}|=2.86\times10^{-9},
\qquad
\text{Wronskian residual}=3.29\times10^{-12}.
\]

no-drive control은 0이며 smooth switching, 보수적 tachyon-free 하한과
해상도 수렴을 동시에
통과한다. 따라서 지정 toy EFT에서 `CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION`
이다.

선택한 mode의 small-\(q\) leading-order first-band estimate는

\[
|4\omega_k^2-E_{\rm drive}^2|\le2|\delta m^2|
\]

이다. 이 부등식만으로는 통과시키지 않는다. full-amplitude periodic oscillator의
fundamental matrix를 한 주기 적분해

\[
M=Y(2\pi),\qquad
\det M\simeq1,\qquad
|\operatorname{Tr}M|>2
\]

가 N, 2N, 4N에서 수렴할 때만 Floquet instability를 인정한다.
\(q=0.1,\ 4\nu^2=1.2\) 반례는 leading estimate에는 걸리지만
\(\operatorname{Tr}M\simeq-1.99537\)로 안정이어서 실패한다.
\(m=1\) eV, \(p=0\), \(E_{\rm drive}=10\) eV 반례도 유한 pulse sideband
occupation이 있어도 off-resonant control로 남는다. 작은 occupation의
N, 2N, 4N 수렴은 상대오차와 별도 absolute floor를 함께 검사한다.

그러나 다음 네 항은 계산된 evidence object가 아니라 self-reported metadata라
모두 true로 넣어도 물리적 Clarus particle production으로 승격되지 않는다.

- `physical_clarus_pole_derived=False`
- `action_vertex_derived=False`
- `pump_backreaction_solved=False`
- `pump_work_energy_accounted=False`

또한 sudden switching과 \(g=0\)을 독립 반례로 검사한다. 현재
\(\omega_k^2-|\delta m^2|>0\)은 envelope와 phase에 무관한 보수적
tachyon-free 하한이다. 하한이 0 이하이면 실제 tachyon이라고 선언하지 않고
`TACHYON_STATUS_UNRESOLVED_BY_CONSERVATIVE_BOUND`로 잠근다.

## 8. 생성 입자가 바로 음의 source가 될 수 없는 이유

최소결합 canonical 자유 daughter의 고전 null projection은

\[
T_{\mu\nu}\ell^\mu\ell^\nu
=(\ell^\mu\partial_\mu\chi)^2\ge0.
\]

비음수 occupation을 가진 dephased particle distribution도

\[
\langle:T_{\ell\ell}:\rangle
=\int\frac{d^3p}{(2\pi)^3\omega_p}
(\ell\!\cdot p)^2n_p\ge0.
\]

따라서 보통 입자를 많이 만들수록 throat null source에는 불리하다. 같은 occupation
\(n_k\)라도 squeezed anomalous correlator \(\langle aa\rangle\)의 위상에 따라
국소 stress 부호가 달라질 수 있으므로

\[
n_k\not\Rightarrow
\operatorname{sign}\langle T_{\mu\nu}\rangle_{\rm ren}.
\]

현재 가장 일관된 경로는 생성 입자를 직접 음의 source로 쓰는 것이 아니라,
응집상·위상상·비물질 경계를 만들고 그 경계의 진공 subtraction을 계산하는 것이다.

## 9. 한 파장의 반사율로 Casimir를 통과시키지 않음

실제 경계 응력은 단일 \(R(E_*)\)가 아니라 전체 causal response를 쓴다.
아래 식은 zero-temperature, planar, specular, passive-equilibrium 경계의
scattering/Lifshitz 구조를 개략적으로 쓴 것이다.

\[
\frac{\mathcal E}{A}
=\frac{\hbar}{2\pi}
\int_0^\infty d\xi
\int\frac{d^2k_\parallel}{(2\pi)^2}
\sum_{p={\rm TE,TM}}
\ln\left[
1-r_{1p}(i\xi,k_\parallel)\,
r_{2p}(i\xi,k_\parallel)e^{-2\kappa a}
\right]
\]

여기서

\[
\kappa=\sqrt{k_\parallel^2+\xi^2/c^2}.
\]

이는 평면당 에너지이며 curved throat의 전체 \(T_{\mu\nu}\)가 아니다. 필요한
입력은

- imaginary-frequency continuum
- transverse momentum
- TE/TM polarization
- retarded susceptibility와 Kramers--Kronig
- material loss 또는 active gain/noise balance

다. `boundary_response_audit()`는 152.93 GeV 한 점에서 반사율이 1이어도 이 자료가
없으면 실패한다.

현재 API의 completeness 불리언은 metadata 진단만 만든다. 실제
\(r_{\rm TE/TM}(i\xi,k_\parallel)\) 배열과 적분 결과를 받지 않으므로 모든
불리언을 true로 넣어도 `physical_boundary_response_pass=False`로 잠근다.
마찬가지로 target tensor 숫자와 provenance 불리언만 복사해서 throat를
self-certify할 수 없다.

경계가 지속적으로 pump되는 active state라면 equilibrium Lifshitz 식도 금지한다.
그때는 gain, pump noise와 fluctuation을 포함한 non-equilibrium Keldysh stress가
필요하다.

마지막 stress에는

\[
T_{\mu\nu}^{\rm vacuum}
+T_{\mu\nu}^{\rm produced\ matter}
+T_{\mu\nu}^{\rm pump}
+T_{\mu\nu}^{\rm apparatus}
\]

를 모두 포함해야 한다. vacuum 음수항만 떼어내는 계산은 통과시키지 않는다.

## 10. 현재 stage ledger

| 단계 | 판정 | 이유 |
|---|---|---|
| ideal-planar scale calibration | `PASS` | \(a\)와 선택적 \(\lambda=2a\) scale을 169 GeV legacy와 분리 |
| physical Clarus pole | `OPEN` | 29.64757 MeV는 inverse-correlation bridge |
| optional portal local pair vertex | `EXACT CONDITIONAL` | 선택적 $Z_2$ action에서 $h\Phi^2,h^2\Phi^2,\chi^2\Phi^2$ bare derivative 재현; A1 유도 아님 |
| CE nonlinear production vertex | `OPEN` | A1 Hessian은 higher jet를 식별하지 못하고 직접 \(\Phi^2\chi^2\)는 toy EFT |
| phase-aware \(\Phi^2\) spectrum | `EXACT CONDITIONAL` | 정확히 같은 Fourier key를 쓰는 classical modes의 항등식 |
| collinear invariant pair threshold | `EXACT CONDITIONAL` | \(p_\perp=0\)인 공급 mode에 대해 exact |
| finite-pulse daughter excitation | `CONDITIONAL PASS` | same \(Q,m,p\) provenance, occupation·Wronskian·Floquet monodromy와 tachyon-free 하한 통과 |
| physical particle production | `OPEN` | pole, vertex, pump energy 장부 없음 |
| boundary matter/condensate | `NOT REACHED` | density, lifetime, phase transition 없음 |
| causal broadband response | `NOT REACHED` | \(r_{TE/TM}(i\xi,k_\parallel)\) 없음 |
| renormalized negative net stress | `NOT REACHED` | full source 합 미계산 |
| backreacted stable throat | `NOT REACHED` | EOM·QI·perturbation 미계산 |

독립 toy branch의 최대 단계는

```text
CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION
```

이다. wormhole realization은 `False`다.

CE-physical chain의 최대 단계는

```text
TARGET_SCALE_CALIBRATION_ONLY
```

이며 physical pole과 CE vertex는 `OPEN`이다.

### 10.2 Q0.4–Q0.5 후속 결과

선택적 portal control에서는

\[
G_F(p)=\frac{i}{p^2-m_0^2-\lambda_{HP}v^2+i0},
\qquad \operatorname{Res}(G_F/i)=+1
\]

과 local pair vertex를 닫았다. 그러나 \(m_0^2\ge0\)이면

\[
m_{\rm pole}\ge v\sqrt{\lambda_{HP}}=43.7677\,\mathrm{GeV}
\]

여서 29.64757 MeV same-field pole은 불가능하다. light pole을 강제로 만들면
\(m_0^2=-1915.6085\,\mathrm{GeV}^2\)이고 portal shift 중
\(4.5885\times10^{-7}\)만 남는 상쇄가 필요하다. 이 light benchmark의
공급된 비가시폭 gate도 \(\mathrm{BR}_{inv}=0.8253>0.11\)로 실패한다.

따라서 현재 살아있는 생산 구조는 조건부
\(\Phi\Phi\to h^*\to\mathrm{SM}\) 후보이며, physical external \(\Phi\) pole,
renormalized 1PI form factor, pump 분포와 phase space가 없어 rate는 아직
`OPEN`이다. 상세 감사는 `CE_TWO_POINT_AND_VERTEX_LOOP.md`다.

### 10.1 경전 비유에서 얻은 별도 가설: private dressing과 public scaffold

종교 서사는 물리 증거로 사용하지 않는다. 다만 “그에게는 물이 땅이고 옆
사람에게는 물”이라는 구조는 새 물질 생성보다 probe-selective interaction으로
번역할 수 있다. 측정 가능한 controller state \(z_A\)를 가진 probe \(A\)의
propagator를

\[
G_A^{-1}(\omega,k;z_A)
=G_{A,0}^{-1}(\omega,k)-\Sigma_A[\Phi,z_A]
\]

로 쓰면 같은 환경에서도 \(A\)와 결합하지 않는 probe \(B\)가 서로 다른
dispersion·impedance를 볼 수 있다. 이는 의식이 현실을 바꾼다는 주장이 아니라
state-dependent coupling을 갖는 평범한 open/control-system 가설이다.

위상잠금 order parameter는 예를 들어

\[
R=\left|\frac1T\int_0^T e^{i\Delta\phi(t)}dt\right|
\]

로 둘 수 있다. noise로 \(R\)이 임계값 아래로 내려갈 때 transport가 사라지는지
검사하면 “결의가 흔들리면 다시 가라앉는다”는 서사를 물리적 feedback
stability 반례로 바꿀 수 있다.

두 branch는 다음처럼 분리한다.

| branch | 필수 판정 | 승격 금지 |
|---|---|---|
| private dressed-state / 전변 | \(A\)의 transport 변화, phase-lock·energy ledger | pump-off에서 사라지고 \(B\)가 못 보면 matter/boundary 아님 |
| public scaffold / 화작 | controller 제거 뒤 환경 response 지속, 독립 \(B\)도 재현 | 현재는 held-out·별도 chain 선언뿐이며 제3자 전달·공변 stress가 없으면 throat source 아님 |

따라서 다음 pilot gate는 `probe selectivity → phase/noise sweep → pump-off
persistence → held-out separate-chain response → energy balance`다. 실제
third-party transfer는 `OPEN`이다. Casimir/GR 경로에는
observer-relative effective metric만으로 부족하며, 마지막에는 모든 probe가
공유하는 causal response와 보존된 \(T_{\mu\nu}\)가 필요하다.

### 10.2 다음 pilot의 구현 결과

위 구분은
[`PROBE_SELECTIVE_DRESSING_AND_PUBLIC_SCAFFOLD_LOOP.md`](PROBE_SELECTIVE_DRESSING_AND_PUBLIC_SCAFFOLD_LOOP.md)
와 `probe_scaffold_pilot.py`에 구현했다. 단순 A/B on/off가 아니라 pump×controller
2×2 factorial contrast를 쓰고, phase/noise level 하나를 held-out으로 지정한다.
이 지정은 아직 외부 manifest로 검증된 사전등록이 아니다. 또한 held-out에서
측정한 \(R_{\rm bc}\)에 조건부인 response를 예측할 뿐 noise에서 \(R(D)\)를
예측하지 않는다. public branch는 서로 다른 calibration \(c_p\)를 가진 세 probe에서

\[
d_p=c_pK_{\rm post}+\epsilon_p
\]

의 rank-one scalar kernel을 두 probe로 맞춘 뒤 세 번째 probe를 예측한다. private
branch와 public branch는 서로의 선행조건이 아니다.

현재 합성 control은 private 쪽
`CONDITIONAL_PHASE_LOCKED_PRIVATE_DRESSING`, public 쪽
`CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE`까지 도달한다. 그러나 실제 raw
measurement, physical pump-off, apparatus-memory 제거와 blinded sample swap이
없으므로 `conditional_public_scaffold_candidate=False`, new matter와 physical
boundary도 `False`다.

### 10.3 (g,R,S,\tau) 결합 시공간 응답 마스크

위 네 축을 각각 통과시키는 것으로는 joint response를 증명할 수 없다. 후속 루프는
[`CAUSAL_MASK_AND_SPACELIKE_MARGINAL_LOOP.md`](CAUSAL_MASK_AND_SPACELIKE_MARGINAL_LOOP.md)
에서 paired raw tensor와 사전 고정한 전체 설계 tensor

\[
D_{npfxta}=Y^{\rm matched}_{npfxta}-Y^{\rm sham}_{npfxta},
\qquad
M_{pfxta}=g_pR_fS_{xa}T_t(\tau_{xa})
\]

를 직접 비교한다. training cell에서는 manifest design으로만 정해진 fixed-weight
amplitude 하나를 맞추고, pre-arrival·off-support·target을 포함한 disjoint heldout
cell을 예측한다.
unique paired block ID, 전처리·calibration hash, 최소 64 block과 Student-t Bonferroni
simultaneous bound를 요구하며 validator가 raw rows에서 인증서 전체를 재계산한다.
별도 spacelike marginal gate는 selector 전후의 전체 finite-bin 결과분포에
simultaneous TV bound를 적용한다.

합성 control의 최대 단계는
`CONDITIONAL_DECLARED_BLOCK_SPATIOTEMPORAL_RESPONSE_MASK`다. 이는 block 독립성을
외부에서 검증했다는 뜻이 아니다. caller-supplied pre-arrival mask는
causal proof가 아니므로 relativistic causality도 hard-false다. factor normalization은
scale-gauge 때문에 비식별이고, 실제 tensor 자료가 없으므로 public scaffold, CE
coupling, new matter, causal boundary와 재규격화 stress는 계속 `False`다.

## 11. 전역 throat source 재감사

기존 explicit target을

\[
t=e^{1-x},\qquad
\frac{b}{r_0}=\frac{2+t}{3},\qquad
\Phi=\frac t2
\]

로 쓰면 shape gap, finite lapse, asymptotic flatness와 각 end의
\(M_{\rm ADM}/r_0=1/3\)은 cutoff와 무관하게 exact다. 역정의한 Einstein
source의 Bianchi identity도 exact다. 그러나 이는 독립 matter action의 EOM
증명이 아니다.

더 중요한 결함은

\[
\rho+p_r<0\quad(x\ge1),\qquad
x^3p_r\to-\frac23,\qquad x^3p_t\to+\frac13
\]

이라 radial tension이 전구간 \(r^{-3}\) tail을 가진다는 점이다. Killing
energy를 1로 고정한 양쪽 affine ANEC는

\[
\mathcal A_{\rm null}=-2.4975554173
\]

로 유한·음수지만 coordinate/proper volume NEC는
\(-\frac23\log X\)로 발산한다. 따라서 finite ADM은 localized exotic source를
뜻하지 않는다.

redshift만

\[
\Phi_{\rm match}
=\frac12\ln\left(1-\frac{2}{3x}\right)+\frac32e^{1-x}
\]

로 바꾼 후보는 같은 throat tensor, horizon-free lower bound, 각 end의
\(M_{\rm ADM}/r_0=1/3\)을 유지하면서 stress tail을 지수감쇠시킨다. 이 후보의
양쪽 affine ANEC는 \(-2.2927281338\), 한 end의 coordinate/proper volume NEC는
각각 \(-4.2189353455\), \(-6.0917872476\)로 유한하다.

그러나 비최소 scalar reconstruction은 throat에서 \(K/F=7/12>0\)인 반면
\(x=37/32\)에서 \(K/F<-1.8\), 수치 최소는 약 \(-1.83055\)다. 따라서
localized target geometry는 개선됐지만 healthy global scalar, potential과
perturbative stability는 여전히 실패한다.

## 12. 다음 루프 우선순위

1. inverse-correlation scale과 선택적 portal field를 같은 field로 볼지 분기 확정
2. 같은 field라면 full CE action·counterterm·matching scale에서 renormalized
   \(\Gamma_{CE,R}^{(2)}\), pole, residue, cut과 LSZ를 실제 유도
3. 같은 action에서 \(h\Phi^2\) 1PI form factor와
   \(\Phi\Phi\to h^*\to\mathrm{SM}\) rate를 계산
4. 1축 collinear 단일 mode에서 3차원 momentum spectrum으로 확장하고
   \(n_\chi=\int k^2n_kdk/(2\pi^2)\) 수렴
5. prescribed pump를 버리고 pump depletion 및 에너지 보존을 동시 적분
6. 생성물의 응집상, lifetime, coherence length, layer thickness 유도
7. causal broadband response와 passive/active 분기 계산
8. full renormalized stress, conservation, ANEC/QI, backreaction과 perturbation 검사
9. drive-frequency·pulse-cycle·ramp scan으로 switching excitation과 Floquet
   growth를 분리

## 13. 재현

```powershell
uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_casimir_carrier_target.py `
  tests/test_clarus_resonant_matter.py `
  tests/test_global_throat_exact_certificate.py -q

uv --cache-dir .uv-cache run python examples/physics/clarus_resonant_matter_gate.py
uv --cache-dir .uv-cache run python examples/physics/ce_two_point_vertex_gate.py
```

위 3-file primary suite는 현재 69 passed다. 기존 carrier·multimode·stress
연결 회귀 3개를 더한 6-file focused suite는 91 passed다.

추가된 반례에는 위상 cancellation, co-propagating null line, zero coupling,
sudden switching, tachyon/resonance 혼동, 단일 reflectivity false positive, 정확한 throat
tensor 숫자만 복사한 provenance failure, NaN/Inf/bool/fractional-step 입력이 포함된다.

## 참고 문헌

- Kofman, Linde, Starobinsky, [Reheating after Inflation](https://arxiv.org/abs/hep-th/9405187)
- Greene, Kofman, Linde, Starobinsky, [Structure of Resonance in Preheating after Inflation](https://arxiv.org/abs/hep-ph/9705347)
- Wilson et al., [Observation of the Dynamical Casimir Effect in a Superconducting Circuit](https://arxiv.org/abs/1105.4714)
- Rahi et al., [Scattering Theory Approach to Electrodynamic Casimir Forces](https://arxiv.org/abs/0908.2649)
- Fewster and Eveson, [Bounds on Negative Energy Densities in Flat Spacetime](https://arxiv.org/abs/gr-qc/9805024)
