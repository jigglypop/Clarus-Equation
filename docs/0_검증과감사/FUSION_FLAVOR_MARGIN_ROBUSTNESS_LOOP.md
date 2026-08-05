# 핵융합 flavor-aligned 후보의 여유 강건성 루프

코드: `reality_stone/python/reality_stone/clarus/fusion_flavor_margin_robustness_loop.py`  
실행: `examples/physics/fusion_flavor_margin_robustness_gate.py`  
테스트: `tests/test_fusion_flavor_margin_robustness_loop.py`

## 결론

29.64757 MeV flavor-aligned scalar는 여전히 가장 가까운 조건부 후보지만, 유한 핵 크기와
제약의 이론 오차를 반복해도 물리 gate를 열 수 없다.

- D/T Gaussian one-body 접힘을 1% thermal/WKB 목표까지 다시 풀면 필요한 charge
  product는 최대 2.10%, nucleon coupling은 1.05% 감소한다.
- Gaussian·3차원 exponential·균일구 밀도와 두 반지름 세트를 약한 퍼텐셜의 선형응답으로
  비교하면 product 감소 범위는 1.65--2.41%, coupling 감소 범위는 0.83--1.21%다.
- neutron--Pb의 각분포 재투영은 결합 상한을 완화하지만, scalar bound에 더 가까운
  저에너지 총단면적 재투영은 상한을 강화한다. 서로 반대인 결과를 평균내지 않는다.
- rare-kaon 중앙 곡선이 견디는 하향 NLO tightening은 점핵에서 6.637배, 가장 유리한
  one-body proxy에서도 6.719배뿐이다. 최신 NA62의 BR 개선 범위는 coupling bound를
  추가로 `1/sqrt(1)--1/sqrt(3)`배 줄인다. Figure 2 벡터 곡선을 CE 질량에서 보간하면
  중앙 개선 proxy는 1.331배이고 중앙 NLO 임계는 5.752--5.823배로 내려간다.
  두 BR 판독값에 독립 5% box를 전파해도 점핵 임계는 5.471--6.047배다. 인정된
  최대 10배 NLO 범위에서는 닫힌다.

따라서 `margin_robustness_gate_pass=False`와
`physical_ce_fusion_branch_accepted=False`를 유지한다. 임의의 오차폭이나 단순 proxy
통과는 실험 likelihood 통과로 세지 않는다.

## 1. D/T one-body 접힘

각 핵의 정규화 scalar density를 rms 반지름 `R_D`, `R_T`인 Gaussian으로 두면 상대
Gaussian 폭은

\[
a^2=\frac{R_D^2+R_T^2}{6}
\]

이고 Yukawa Green function의 정확한 접힘은

\[
Y_a(r)=\frac{e^{a^2\mu^2}}{2r}
\left[
e^{-\mu r}\operatorname{erfc}\left(a\mu-\frac{r}{2a}\right)
-e^{\mu r}\operatorname{erfc}\left(a\mu+\frac{r}{2a}\right)
\right],
\qquad \mu=\frac{m_\phi}{\hbar c}.
\]

이를 기존 10 keV thermal/WKB 루프에 넣고 charge product를 1% gain까지 이분법으로
다시 풀었다.

| D rms / T rms (fm) | 필요한 product / 점핵 | 필요한 coupling / 점핵 |
|---:|---:|---:|
| point | 1.000000 | 1.000000 |
| 1.50 / 1.30 | 0.98579 | 0.99287 |
| 1.975 / 1.59 | 0.98032 | 0.99011 |
| 2.20 / 1.90 | 0.97902 | 0.98946 |
| 2.40 / 2.10 | 0.97975 | 0.98982 |

반지름만 넓히는 것으로 density-shape 의존성을 가릴 수 없으므로, 구조 반지름
`(1.97507, 1.5978) fm`과 보수적 반지름 외피 `(2.12799, 1.7591) fm`에서 세 밀도를
별도로 접었다. 1% 지점 부근의 선형응답 product 비는 다음과 같다.

| 반지름 세트 | Gaussian | exponential | 균일구 |
|---|---:|---:|---:|
| 구조 반지름 | 0.98028 | 0.98345 | 0.97772 |
| 보수적 외피 | 0.97923 | 0.98314 | 0.97586 |

이 값은 작은 probe coupling에서 얻은 선형응답이다. 별도 고해상도 nonlinear 재계산과의
차이는 coupling 기준 약 `1e-5`였지만, hard gate에는 사용하지 않는다. 특히 charge
radius가 scalar-density radius라는 보장은 없고 chiral-EFT two-body scalar current,
D 사중극, T neutron profile, overlap deformation과 density covariance가 빠져 있다.

## 2. neutron--Pb 유한 전파자와 form factor

25 keV neutron의 Pb 중심질량계 운동량을 사용하면 30--150도에서

\[
q=3.531\text{--}13.178\ \mathrm{MeV},\qquad
\frac{m_\phi^2}{m_\phi^2+q^2}=0.835\text{--}0.986.
\]

Pb rms 반지름 5.40--5.75 fm와 Gaussian, exponential, 균일구 form factor를 사용했다.
contact-template의 국소 기울기 응답

\[
K_{\rm loc}(q)=\frac{m_\phi^2}{m_\phi^2+q^2}
\left[1+\frac{m_\phi^2(1-F(q))}{q^2}\right]
\]

은 0.925--1.109로, 점핵 D/T의 통과 임계 `K <= 1.00633`을 양쪽에서 가로지른다.
가장 유리한 D/T morphology proxy를 쓰면 임계는 `1.03122`다.

그러나 국소값의 평균은 원 실험 fit이 아니다. 자유 intercept를 둔 25 keV 각분포의
선형 기울기로 재투영하면

\[
K_{\rm angular}=0.910\text{--}0.923,
\]

이어서 결합 상한을 완화한다. 반대로 scalar 제약에 사용된 저에너지 총단면적
`sigma(k)=sigma_0+sigma_2 k^2`를 10 eV--10 keV에서 재투영하면

\[
K_{\sigma_2}=1.050\text{--}1.080,
\]

이고 `k -> 0`에서는

\[
K_{\sigma_2}(0)=1+\frac{m_\phi^2R_{\rm Pb}^2}{6(\hbar c)^2}
=1.110\text{--}1.124,
\]

이어서 상한을 강화한다. 두 recast가 통과 경계의 반대편에 놓인다. 원 분석이 이미 유한
nuclear density와 distorted wave를 포함했는지 불명확하여 재적용하면 double counting일
수도 있다. 각도·에너지 covariance, acceptance, strong-amplitude phase와 원 fit의
finite-density provenance 없이는 어느 proxy도 hard bound가 아니다. 특히 후보 질량은
간단한 원문 recast의 명시 범위보다 높다.

finite-window 적분은 181×241 격자와 1001×1001 격자를 함께 계산한다. 정제 후 범위는
`1.049888--1.079729`, 최대 상대 이동은 `3.82e-5`로 선언한 `1e-4` 수치 허용치 안이다.
gate에는 안정한 여섯 자리까지만 출력한다.

이 구분은 scalar differential bound의 strong p-wave cancellation과 저에너지 총단면적
대안을 논의한 [Barger et al.](https://arxiv.org/abs/1011.3519)을 따른다. 반지름 proxy의
규모는 [PREX-II](https://doi.org/10.1103/PhysRevLett.126.172502), D/T 구조 반지름은
[deuteron structure radius](https://doi.org/10.1103/PhysRevA.83.042505)와
[triton point-charge extraction](https://arxiv.org/abs/1512.03805)을 참고했다.

## 3. rare-kaon NLO 축과 결합 조건

D/T 접힘 때문에 필요한 결합이 `sqrt(p)`만큼 변한다고 두면 `p=P/P0`에 대한 두 proxy
통과 조건은

\[
K_{\rm Pb}\le
\left(\frac{g_{\rm bound}}
{g_{\rm Pb,0}\sqrt p}\right)^2,
\qquad
K_{\rm NLO}\le
\frac{182\,d}{27.4217\sqrt p}.
\]

여기서 `d`는 digitized kaon 곡선 multiplier다. 중앙 `d=1`에서 오른쪽 임계값은 점핵
6.637, 가장 유리한 one-body proxy 6.719다. 하단 `d=0.88`과 점핵을 동시에 요구하면
5.841이다. 따라서 인정된 `K_NLO=10`에서는 Pb proxy를 어떻게 고르더라도 kaon 조건이
닫힌다.

[NA62 2016--2022 hidden-sector search](https://arxiv.org/abs/2507.17286)는 0--110 MeV와
150--260 MeV에서 `K+ -> pi+ X` invisible peak를 스캔한다. 2021--2022 자료 추가로
single-event sensitivity는 약 2배 좋아졌지만, SM 배경 때문에 observed BR 상한의 개선은
질량 가설에 따라 1--3배다. decay rate가 aligned coupling의 제곱에 비례한다고 두면

\[
B_{\rm coupling,new}=\frac{B_{\rm coupling,old}}{\sqrt{I_{\rm BR}}},
\qquad I_{\rm BR}=1\text{--}3.
\]

따라서 기존 digitized coordinate 182는 범위만 전달하면 105.08--182가 되고, 점핵의
허용 NLO tightening은 3.832--6.637, 가장 유리한 D/T proxy에서는 3.879--6.719다.
최신 자료만으로는 tree-level 후보가 이 범위 전체에서 중앙 상한 아래지만, 이것도 exclusion
또는 pass 판정이 아니다.

범위만 쓰는 데서 한 단계 더 나아가 JHEP/arXiv v2 PDF의 Figure 2-a 벡터 좌표를 직접
기록했다. 축과 후보 질량을 선형 변환하고 후보를 감싸는 두 선분을 보간하면

\[
\mathcal B_{2016\text{--}22}^{90\%}(29.64757\,\mathrm{MeV})
\simeq2.4763\times10^{-11},\qquad
\mathcal B_{2016\text{--}18}^{90\%}
\simeq3.2968\times10^{-11}.
\]

따라서 그림 readout의 **중앙값** 개선은 `I_BR=1.33137`, coupling multiplier는
`0.866665`, 갱신된 uds 중앙 coordinate는 `157.73`이다. 중앙 점핵 NLO 임계는
`5.75212`, 가장 유리한 D/T proxy에서는 `5.82283`이다. 코드에는 PDF page index,
축 좌표와 두 벡터 선분 끝점을 그대로 넣어 이 보간을 재현한다.

그림 판독 자체의 5% 상대 오차는 old/new BR에 독립 box로 전파했다. 그 결과

\[
I_{\rm BR}=1.20457\text{--}1.47151,
\quad B_{\rm coupling,new}/B_{\rm coupling,old}=0.82436\text{--}0.91114,
\]

점핵 NLO 임계는 `5.47136--6.04729`, 가장 유리한 D/T proxy에서는
`5.53862--6.12163`다. 이 전 구간도 10보다 작다.

그러나 논문은 1.4 MeV 간격의 mass hypothesis를 사용하고 이 값은 공개 그림의 선분
보간이다. 29.64757 MeV의 tabulated CLs 값·acceptance table과 flavor-aligned uds
amplitude 재매칭은 아직 없다. 따라서 `figure2_candidate_mass_curve_interpolation_entered=True`
이지만 `exact_candidate_mass_observed_limit_entered=False`이고, 그림 보간을 exact
mass-bin likelihood로 간주하지 않는다.

결합식 양쪽의 셀을 평가하는 API도 제공하지만, 셀이 대수적으로 열려도
`experimental_likelihoods_supplied=False`와 `physical_gate_pass=False`다.

## 4. gate를 열기 위해 필요한 입력

1. ab-initio D/T one- 및 two-body scalar current와 밀도 covariance,
2. 29.65 MeV 유한 전파자를 직접 넣은 Pb differential/total likelihood, 각도·에너지
   covariance, strong phase nuisance와 원 분석의 finite-density 처리 내역,
3. 완전한 `O(p^4)` weak-ChPT 진폭, 저에너지상수 covariance, NA62/E949 mass-bin
   likelihood.

## 5. 실행

```bash
uv run python examples/physics/fusion_flavor_margin_robustness_gate.py
uv run --extra dev python -m pytest tests/test_fusion_flavor_margin_robustness_loop.py -q
```
