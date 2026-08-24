# 비선택 양자 역사에서 공통 암흑 부문으로

Status: COMPLETE

## 초록

본 연구는 “암흑물질과 암흑에너지는 선택되지 않은 양자경로의 두 표현”이라는
CE의 중심 제안을 공변 유효장이론으로 좁혀 보강했다. 표준 양자 조건부화는
비선택 outcome을 선택된 branch의 추가 중력원으로 만들지 않으므로, 그 강한
부모 주장은 완전 반례에 따라 제거했다. 대신 비선택 history data를 하나의
residual scalar로 보내는 국소-공변 물리 사상을 독립 공리로 두었다. 이 공리와
최소 scalar action 아래에서 빠른 이차 진동은 평균적으로
$w\simeq0$, $\rho\propto a^{-3}$인 암흑물질형 성분을, 상수 진공항은
$w=-1$인 암흑에너지형 성분을 정확히 준다. 다만 미시적 map, 절대 abundance,
두 성분의 분할과 관측 forward model은 아직 미완성이므로 본 결과는 공통
기원의 조건부 EFT이며 완성된 암흑 우주 예측은 아니다.

## 1. 문제와 살아남은 핵심 주장

CE의 물리 서사는 끼임, 접힘, 암흑 표현의 세 단계로 읽는다. 환경과 검출기가
가시적 outcome 하나를 기록하게 하는 과정이 끼임이다. 기록되지 않은 history
data를 새 물리 자유도로 보내는 과정이 접힘이다. 그 자유도의 서로 다른
우주론적 한계를 암흑물질과 암흑에너지로 읽는 것이 암흑 표현이다.

이번 보강에서 가장 중요한 구분은 “비선택 outcome이 존재한다”와 “그
outcome이 우리 branch의 stress tensor에 더해진다”가 같은 문장이 아니라는
점이다. quantum instrument $\{\mathcal I_a\}$에서 outcome $0$을 기록했다면
조건부 상태는

$$
\rho_0=
\frac{\mathcal I_0(\rho)}
{\operatorname{tr}\mathcal I_0(\rho)}.
\tag{1}
$$

선택 branch의 국소 관측량은 $\rho_0$로 계산한다. 보완 operation
$\mathcal I_1(\rho)$를 별도 중력원으로 더하는 규칙은 표준 instrument
조건부화에서 나오지 않는다. 따라서 “표준 양자역학에서 선택되지 않은
경로가 자동으로 우리 branch에서 중력한다”는 부모 주장은 제거한다.

그 반례 뒤에도 사용자가 고정한 중심 생각은 더 정확한 형태로 남는다.

**[공리: 물리 사상]** CE는 비선택 history data를 새 residual sector로
보내는 물리 map을 채택한다. 이 sector는 표준 조건부 상태에 숨어 있던
에너지가 아니라, 별도의 source rule과 보존법칙을 요구하는 새 자유도다.

## 2. 비선택 history에서 residual field로

비선택 history 공간을 $\Gamma_{\rm ns}$라 하고, 그 위의 차원 없는
subprobability를 $\nu_{{\rm ns},\beta}$라 하자. 시공간 점 $x$와 history
$\gamma$를 연결하는 차원 없는 kernel을 $\widehat K(x,\gamma)$, 질량 차원
1의 변환 척도를 $M_*$라 둔다. CE가 채택한 최소 map은

$$
\phi(x)=M_*
\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)
\nu_{{\rm ns},\beta}(d\gamma)
\tag{2}
$$

다. $\nu$와 $\widehat K$가 무차원이므로 $[\phi]=M$이다. 식 (2)가
국소-공변 map이 되려면 history measure가 공변적으로 정의되고,
$\widehat K$가 $x$에서 scalar로 변환하며, 선언한 국소 자료만으로 그 값이
정해져야 한다. 임의의 전역 history functional도 적분 자체는 정의할 수
있으므로 적분가능성만으로 locality가 증명되지는 않는다.

$M_*$는 probability를 에너지로 자동 환산하는 수가 아니다. instrument,
history space, kernel, transition hypersurface와 source rule을 미시
이론에서 얻기 전까지 식 (2)는 `[공리: 물리 사상]`이다. 가시 sector와
환경의 에너지를 이미 센 뒤 같은 에너지를 residual sector에서 다시 세지
않도록 no-double-counting 조건도 필요하다.

## 3. 최소 공변 residual action

자연단위 $c=\hbar=1$과 metric signature $(-,+,+,+)$에서 다음 최소
작용을 채택한다.

$$
S_{\rm res}=\int d^4x\sqrt{-g}
\left[
-\frac12g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi
-\frac12m^2\phi^2-V_\Lambda
\right],
\qquad V_\Lambda\geq0.
\tag{3}
$$

$[\phi]=[m]=M$, $[V_\Lambda]=M^4$이고 $d^4x$의 차원은 $M^{-4}$이므로
작용은 무차원이다. 표시된 kinetic 부호는 ghost-free canonical branch다.

**[정리]** 식 (3)의 metric variation은

$$
T^{\rm res}_{\mu\nu}
=\nabla_\mu\phi\nabla_\nu\phi
-g_{\mu\nu}
\left[
\frac12\nabla_\alpha\phi\nabla^\alpha\phi
+\frac12m^2\phi^2+V_\Lambda
\right]
\tag{4}
$$

를 준다. 장방정식은 $\Box\phi-m^2\phi=0$이며,

$$
\nabla_\mu T^{{\rm res}\,\mu}{}_{\nu}
=(\Box\phi-m^2\phi)\nabla_\nu\phi=0
\tag{5}
$$

가 on-shell에서 성립한다. 이는 지정한 작용 내부의 정확한 보존 정리다.

visible field $\psi$와의 interaction $-U_{\rm int}(\phi,\psi)$를 되살리면
residual stress는 일반적으로 따로 보존되지 않는다. 이때
$\nabla_\mu T_\phi^{\mu}{}_{\nu}=Q_\nu$라면 다른 sector가 $-Q_\nu$를
가져야 총 stress가 보존된다. 접힘 transition에서 residual energy가
생긴다고 주장하려면 같은 matching current 또는 hypersurface stress를
명시해야 한다.

## 4. 하나의 residual sector가 만드는 두 암흑 한계

평탄 FLRW metric에서 균일한 $\phi$는

$$
\ddot\phi+3H\dot\phi+m^2\phi=0,
\tag{6}
$$

$$
\rho_\phi
=\frac12\dot\phi^2+rac12m^2\phi^2+V_\Lambda,
\qquad
p_\phi
=\frac12\dot\phi^2-rac12m^2\phi^2-V_\Lambda
\tag{7}
$$

를 만족한다. 이 식이 공통 residual 기원 안에서 두 암흑 표현을 분리한다.

### 4.1 암흑물질형 진동 성분

$\psi=a^{3/2}\phi$로 두면 운동방정식은 정확히

$$
\ddot\psi+
\left[m^2-\frac32\dot H-\frac94H^2\right]\psi=0
\tag{8}
$$

가 된다. $H/m\ll1$, $|\dot H|/m^2\ll1$이고 배경이 한 진동주기 동안
천천히 변하면 WKB 해는

$$
\phi=a^{-3/2}
\left[A\cos(mt+\delta)+O(H/m)\right]
\tag{9}
$$

다. 한 주기 평균에서 kinetic과 quadratic potential이 같아지므로

$$
\frac{\langle p_{\rm osc}\rangle}
{\langle\rho_{\rm osc}\rangle}
=O\!\left(\frac{H^2}{m^2},\frac{\dot H}{m^2}\right),
\qquad
\langle\rho_{\rm osc}\rangle
\propto a^{-3}
\left[1+O\!\left(\frac{H^2}{m^2},\frac{\dot H}{m^2}\right)\right].
\tag{10}
$$

따라서 빠른 이차 진동은 배경에서 암흑물질형이다. 이 결론은
$m\lesssim H$, 비단열 생성, 붕괴·에너지 교환 또는 비이차 potential에서
바뀐다.

배경의 $w\simeq0$만으로 cold dark matter가 완성되지는 않는다.
비상대론적 scalar mode의 선행 sound speed는

$$
c_s^2\simeq\frac{k^2}{4m^2a^2}
\tag{11}
$$

이며, CDM형 성장을 주장하는 mode는 $k/a\ll m$이고 scalar Jeans scale
$k_J/a\sim(mH)^{1/2}$보다 충분히 길어야 한다. 작은 규모 power, halo,
lensing과 CMB는 이 조건의 직접 falsifier다.

### 4.2 암흑에너지형 상수 성분

**[정리]** 식 (3)의 상수항은

$$
T^{(\Lambda)}_{\mu\nu}=-V_\Lambda g_{\mu\nu},
\qquad
\rho_\Lambda=V_\Lambda,
\qquad
p_\Lambda=-V_\Lambda,
\qquad
w_\Lambda=-1
\tag{12}
$$

을 준다. $V_\Lambda$가 상수이면 이 부분은 별도로 보존된다. 따라서 식
(2)의 공통 residual 기원을 채택한 뒤에는 같은 sector의 진동 부분이
암흑물질형, 상수 offset이 암흑에너지형이라는 결론이 식 (3)에서
조건부로 따라온다.

식 (12)는 관측된 vacuum scale의 값이나 radiative stability를 설명하지
않는다. $V_\Lambda$의 절대값은 독립 입력으로 남는다.

## 5. Poisson 고정점이 정하는 것과 정하지 않는 것

CE의 기존 Poisson branch에서 평균 offspring 수 $D>1$의 최소
소멸확률은

$$
q_{\rm ext}=e^{-D(1-q_{\rm ext})}
=-\frac1D W_0(-De^{-D})
\tag{13}
$$

이다. $D_{\rm eff}=3.1777584234$에서는
$q_{\rm ext}=0.0486467196$이고
$s_{\rm branch}=1-q_{\rm ext}=0.9513532804$다. 이 결과는 선언한
branching model의 조건부 정리다.

그러나 $95.14\%$는 residual energy fraction이 아니다. 식 (13)은
$M_*$, $m$, 초기 진폭, $V_\Lambda$와 $M_{\rm Pl}$을 포함하지 않는다.
고정된 $m$에서도 초기 진폭을 바꾸면 oscillatory density가 연속적으로
변하고, $V_\Lambda$는 별도의 연속 입력이다. 따라서 Poisson root는
$\Omega_{\rm DM}$, $\Omega_\Lambda$ 또는 그 비를 식별하지 않는다.

기존 고정점 수치는 composition/readout diagnostic으로 보존한다. 이를
absolute abundance 예측으로 되돌리려면 식 (2)의 normalization, 초기조건,
transition과 관측 forward model을 독립적으로 닫아야 한다.

## 6. 관측 비교와 반증 조건

oscillatory scalar의 물질형 배경은 wave pressure 때문에 scale-dependent
growth를 보일 수 있다. Hui, Ostriker, Tremaine과 Witten의 ultralight-scalar
분석은 작은 halo와 power suppression이 질량·fraction을 강하게 시험함을
보였다. 따라서 residual scalar는 CMB, Lyman-$\alpha$, dwarf/halo
abundance, lensing과 galaxy clustering을 함께 통과해야 한다.

DESI DR1과 full-shape 분석은 암흑에너지와 성장 제약이 BAO, CMB와
supernova 조합 및 선택한 모형에 의존함을 보여 준다. 일부 동적
dark-energy parameterization의 fit 개선이나
$\Omega_m=0.3056\pm0.0049$ 같은 flat-$\Lambda$CDM posterior는
residual-history 해석의 증거가 아니다. 이 값들은 frozen likelihood의
비교 대상으로만 사용한다.

다음 네 경로를 순서대로 닫아야 관측 주장을 시작할 수 있다.

1. 미시 action에서 instrument와 conditional-gravity/source rule을 유도한다.
2. 식 (2)의 locality, covariance, $M_*$와 transition matching을 정한다.
3. scalar perturbation을 포함한 Einstein--Boltzmann forward model을
   사전에 고정한다.
4. $m$, 초기 amplitude와 $V_\Lambda$를 관측 density를 넣지 않고 정하거나
   독립 자료로 제한한다.

## 7. 형식 판정과 한계

이번 보강에서 닫힌 부분은 다음과 같다. 식 (2)의 차원 회계, 식 (3)의
metric stress와 on-shell 보존, adiabatic quadratic scalar의 dust-like
평균, 상수 offset의 $w=-1$은 각각 명시한 전제 아래의 정리 또는 산출이다.
기존 전자약 혼합, Hodge $d=3$, Poisson/Lambert $W$ 수학도 보존했다.

아직 닫히지 않은 부분은 더 중요하다. 비선택 history가 왜 식 (2)의 물리
field가 되는지, transition에서 에너지가 어디에서 어디로 이동하는지,
visible/environment와 어떻게 이중계산을 피하는지, 그리고 오늘의 absolute
dark abundance가 왜 그 값을 갖는지는 `[미완성]`이다. 공통 암흑 기원은
명시적 `[공리: 물리 사상]`이며 표준 decoherence의 정리가 아니다.

따라서 현재의 가장 강한 정직한 결론은 다음 문장이다.

> CE는 선택되지 않은 양자 history data를 새 residual sector로 보내는
> 물리 사상을 채택한다. 그 sector를 최소 공변 scalar EFT로 구현하면,
> 이차 진동과 상수 진공항은 각각 암흑물질형과 암흑에너지형 거동을 한다.

## 8. 재현성

독립 residual-EFT 계산은
`artifacts/verify_residual_eft.py`에 있다. 실행 명령은 다음과 같다.

`.codex/hooks/python.cmd python _workspace/ce/nonselected-quantum-dark-sector-20260825/artifacts/verify_residual_eft.py`

등록된 finite check는 `oscillator_w=1.249e-17`, `rho=220.5`,
`vacuum_w=-1`을 얻었다. 집중 회귀 명령은 다음과 같다.

`.codex/hooks/python.cmd pytest tests/test_dimensionless.py tests/test_cosmology_closure_gate.py tests/test_density_bridge_variational_audit.py -q`

결과는 `41 passed in 0.87s`였다. 이 검사는 수치·차원·기존 no-go의
회귀를 확인하며 물리 사상 자체를 증명하지 않는다.

## 참고문헌

- E. B. Davies and J. T. Lewis, “An operational approach to quantum
  probability,” *Communications in Mathematical Physics* 17 (1970). 접근일
  2026-08-25.
- M. Ozawa, “Quantum measuring processes of continuous observables,” *Journal
  of Mathematical Physics* 25 (1984). 접근일 2026-08-25.
- L. Hui, J. P. Ostriker, S. Tremaine and E. Witten, “Ultralight scalars as
  cosmological dark matter,” *Physical Review D* 95, 043541 (2017),
  https://doi.org/10.1103/PhysRevD.95.043541. 접근일 2026-08-25.
- DESI Collaboration, “DESI 2024: Constraints on Physics-Focused Aspects of
  Dark Energy using DESI DR1 BAO Data,” arXiv:2405.13588. 접근일 2026-08-25.
- DESI Collaboration, full-shape cosmology analysis, arXiv:2411.12022. 접근일
  2026-08-25.
