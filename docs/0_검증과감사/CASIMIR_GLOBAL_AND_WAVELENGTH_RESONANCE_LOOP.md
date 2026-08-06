# Casimir 전역 연장과 특정 파장 공명 루프

## 1. 전역 연장 질문

국소 throat series는 닫혔지만, 이상적 Casimir equation of state

\[
p_r=3\rho,
\qquad p_t=-\rho
\]

를 유지한 채 source 진폭만 공명 envelope로 줄여 유한 에너지의
asymptotically flat 해를 만들 수 있는지 검사했다.

## 2. 고정 EoS의 asymptotic no-go

에너지 밀도 꼬리를

\[
|\rho(r)|\sim r^{-n}
\]

이라 하자. anisotropic conservation은

\[
\varphi(r)\sim\left(\frac{3n}{4}-2\right)\ln r
\]

를 준다.

- 유한 redshift는 $n=8/3$을 요구한다.
- $b(r)/r\to0$은 $n>2$이면 가능하다.
- 유한 총 source energy와 표준 finite ADM mass는 $n>3$을 요구한다.

따라서 유한 redshift의 $n=8/3$과 유한 질량의 $n>3$은 양립하지 않는다.
고정 Casimir 압력비에서 진폭 profile만 바꾸는 resonance envelope는 모든
전역 gate를 동시에 통과할 수 없다.

throat의 요구 slope와 유한 redshift를 동시에 맞추는 예

\[
f(x)=x^{-8/3}
\exp\left[\frac23\left(1-\frac1x\right)\right]
\]

도 구성했다. $f'(1)/f(1)=-2$이고 $b/r\to0$이지만 총 음의 에너지와
ADM mass가 유한하지 않다. 이는 국소해를 전역 장치로 자동 승격할 수
없다는 양성 반례다.

## 3. 특정 파장 공명 역산: legacy control

과거 \(b'(r_0)=-1\) 1 m null-scale control의 이상적 Casimir 요구량을
맞추는 separation은

\[
a_0=3.6628\times10^{-18}\,\mathrm m
\]

였다. 평행판 fundamental standing wavelength를 $\lambda=2a_0$로 두면

\[
\lambda_{\rm req}=7.3256\times10^{-18}\,\mathrm m,
\]

\[
f_{\rm req}=\frac c\lambda,
\qquad
hf_{\rm req}=1.6925\times10^{11}\,\mathrm{eV}
\simeq169\,\mathrm{GeV}.
\]

과거 snapshot의 CE light-pole ansatz \(29.65\,\mathrm{MeV}\)와 비교하면

\[
\frac{169\,\mathrm{GeV}}{29.65\,\mathrm{MeV}}
\simeq5708.
\]

이는 두 에너지의 수치비일 뿐이다. 물리적 pole, 분산관계와 해당 vertex가
없으므로 5708번째 harmonic의 존재를 뜻하지 않는다.

### 3.1 최신 full-tensor target 교정

위의 \(a_0=3.6628\times10^{-18}\,\mathrm m\), \(169\,\mathrm{GeV}\)
값은 \(b'(r_0)=-1\)인
과거 negative-null scale control이다. 현재 정확한 (b'(r_0)=-1/3) full
Casimir tensor에는

\[
a_*=4.0535640\times10^{-18}\ {\rm m},\qquad
\lambda_*=2a_*=8.1071280\times10^{-18}\ {\rm m},
\]

\[
E_*=152.9323309\ {\rm GeV}
\]

를 사용한다. canonical \(29.6991596174\,\mathrm{MeV}\)
inverse-correlation 후보와의 비는 \(5149.382436\)이다. 다만 이 scale은
physical pole로 확인되지 않았고, pole mass라 해도 propagating mode의 에너지
상한이 아니므로 이 비 자체가 5149차 harmonic을
강제하지 않는다. 가능한 경로는 실제 pole·dispersion을 갖는 고운동량 mode 또는
명시적 비선형 mixing vertex이며, 둘 다 현재 CE에서는 `OPEN`이다.

또한 \(E_*\)는 \(\lambda=2a\)를 추가로 택한 ideal-planar 최저 mode의 형식
scale이지 spherical throat에서 유도된 boundary eigenmode나 생성 daughter의
질량이 아니다. 현재 1+1D collinear pair control에는
\(Q^0>0,\ Q^2\ge4m_\chi^2\)가 필요하다.
세부 gate는 `CLARUS_RESONANT_MATTER_LOOP.md`에 둔다.

## 4. $Q$와 harmonic의 구분

quality factor $Q$는 resonance linewidth, 저장시간과 정상상태 amplitude를
바꾸지만 선형계의 중심 frequency를 \(5149.382436\)배 올리지 않는다. 그러므로

```text
29.6991596 MeV inverse-correlation 후보 + high Q
```

만으로 현재 152.93 GeV 형식 mode scale을 만들 수 없다. 가능한 경로에 필요한 것은

```text
물리적 pole·residue·dispersion
+ 고운동량 mode 또는 유도된 비선형 mixing vertex
+ 전체 causal boundary response
+ pump·matter·apparatus를 포함한 renormalized net T_mn
```

이다. 구동 공명은 보통 양의 실재 입자를 축적하므로, 높은 amplitude가
음의 Casimir vacuum stress를 증폭한다는 결론도 별도로 유도해야 한다.

## 5. 공명으로 해결 가능한 범위

| 문제 | 특정 파장 공명의 역할 |
|---|---|
| 필요한 microscopic length scale | ideal-planar \(\lambda=2a\) 선택에서 \(152.93\) GeV 형식 scale |
| CE \(29.6991596\,\mathrm{MeV}\) inverse-correlation scale과의 동일성 | 동일성 미유도; 등록 scale, 물리 질량과 mode 에너지를 구분해야 함 |
| high-$Q$만으로 주파수 상승 | 불가능 |
| 음의 stress 부호 | 미유도 |
| Casimir EoS의 전역 finite-mass no-go | amplitude-only 공명으로 해결 불가 |
| 압력비를 공간적으로 전이시키는 다중 mode | 열린 후보 |

따라서 특정 파장 공명은 **부분 후보**다. 단일 mode의 amplitude가 아니라
radial 위치에 따라 $(p_r/\rho,p_t/\rho)$를 바꾸는 multi-mode stress
engineering이 필요하다. throat 근처에서는 Casimir 비율을 유지하고,
외부에서는 finite-energy vacuum으로 전이해야 한다.

## 6. 판정

| 명제 | 판정 |
|---|---|
| 고정 Casimir EoS + amplitude envelope의 finite-redshift·finite-mass 전역 해 | `REFUTED` |
| ideal-planar \(\lambda=2a\) 형식 scale | \(8.1071\times10^{-18}\)m / \(152.932\)GeV |
| 과거 \(b'=-1\) null control | \(7.3256\times10^{-18}\)m / \(169.247\)GeV, legacy |
| CE light pole의 high-$Q$ 공명만으로 요구 mode 생성 | `REFUTED` |
| 고운동량 Clarus mode 또는 비선형 mixing vertex | `OPEN` |
| smooth-pulse toy daughter excitation | `CONDITIONAL PASS` |
| causal boundary와 renormalized negative net stress | `OPEN` |
| multi-mode anisotropic stress transition | `NEW FRONTIER` |

별도 전역 재감사에서는

\[
\Phi_{\rm match}
=\frac12\ln\left(1-\frac{2}{3x}\right)+\frac32e^{1-x}
\]

가 같은 throat tensor와 각 end \(M_{\rm ADM}/r_0=1/3\)을 유지하면서 기존
\(r^{-3}\) stress tail과 volume-NEC 로그 발산을 지수감쇠 유한 적분으로
바꾼다는 점을 확인했다. 다만 비최소 scalar \(K/F\)가 \(x=37/32\)에서
음수이므로 이는 개선된 target geometry이지 physical CE matter completion은
아니다.

후속 공명 루프는 위상 합산, invariant pair threshold와 smooth finite-pulse
occupation까지 구현했다. 다음 병목은 입자수를 직접 음의 source로 승격하는
것이 아니라, 생성 상태의 causal response를 구해 renormalized net stress와
전역 backreaction을 동시에 검사하는 것이다.

## 7. 실행

```powershell
uv run --extra dev python -m pytest tests/test_casimir_global_resonance.py -q
uv run python examples/physics/casimir_global_resonance_gate.py
```
