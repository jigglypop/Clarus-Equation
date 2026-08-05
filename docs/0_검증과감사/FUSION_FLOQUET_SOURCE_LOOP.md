# 핵융합 Floquet/source 식-수정 루프

코드: `reality_stone/python/reality_stone/clarus/fusion_floquet_source_loop.py`  
실행: `examples/physics/fusion_floquet_source_gate.py`  
테스트: `tests/test_fusion_floquet_source_loop.py`

## 1. 이번 반복의 판정

이번 단계에서는 시간의존 분기를 실제 단면적까지 닫았다. 결과는 두 문장으로
분리해야 한다.

1. 공개된 표준 QED Floquet--Volkov 식을 10 keV까지 외삽하면 D--T
   Maxwellian 반응률을 1% 높이는 수치해가 나온다. 다만 논문의 thermal
   benchmark는 1 keV이므로 10 keV 점은 공개 검증 통과점이 아니다.
2. 이 전자기 해는 29.64757 MeV CE scalar의 해가 아니다.

대표 외삽점과 CE 판정은 다음과 같다.

| 항목 | 값 | 판정 |
|---|---:|---|
| $kT$ | 10 keV | 고정 |
| $\hbar\omega$ | 0.3 keV | 공개 FV--CN 대조 하한 |
| 1% 임계 전기장 | $4.861597077\times10^{15}$ V/m | 10 keV FV 식 외삽 |
| $10^{16}$ V/m 반응률 증가 | 4.223237599% | 수치 회귀 `PASS` |
| 임계장 에너지밀도 | $1.046349192\times10^{20}$ J/m³ | pump 장부 입력 |
| 10 fs, 반지름 10 nm 입사 에너지 | $9.85479\times10^{-2}$ J | 명시적 장부 |
| 추가 fusion / 입사 pulse | $7.65470\times10^{-9}$ | net-energy `FAIL` |
| CE $Z_2$ beat 필요 에너지밀도 | $5.65818\times10^{40}$ J/m³ | scalar source `FAIL` |

따라서 “시간의존 장으로 1%가 수학적으로 가능한가”에는 `YES`이지만,
“현재 CE scalar가 그 효과를 내는가”와 “reactor 순에너지가 좋아지는가”에는
여전히 `NO`이다.

## 2. QED 상대좌표 작용

D와 T의 상대좌표에서 최소결합 Hamiltonian은

\[
H(t)=\frac{[\mathbf p+q_{\rm eff}\mathbf A(t)]^2}{2\mu}+V_C(r),
\qquad
\frac{q_{\rm eff}}e=\frac{m_T-m_D}{m_T+m_D}=0.1992318073
\]

이다. 이는 임의의 저에너지 연산자를 붙인 것이 아니라 gauge-invariant QED의
두 입자 상대좌표 축약이다.

\[
U_p=\frac{(q_{\rm eff}E_0)^2}{4\mu\omega^2},
\quad
u=\frac{q_{\rm eff}E_0\sqrt{2\mu E}\cos\theta}
{\mu\hbar\omega^2},
\quad
v=\frac{U_p}{2\hbar\omega}.
\]

Volkov 위상을 generalized-Bessel sideband로 전개하면

\[
P_n=|J_n(u,v)|^2,
\qquad
E_n=E+U_p+n\hbar\omega,
\]

\[
\sigma_{\rm FV}(E,\theta)
=\sum_nP_n(u,v)\,\sigma_{\rm BH}(E_n)
\]

을 얻는다. 구현은 음의 Volkov 위상을 `ifft`로 전개한다. 확률합

\[
\left|\sum_nP_n-1\right|<4.5\times10^{-16}
\]

을 매번 검사해 sideband truncation을 잠갔다. 부호를 뒤집고 같은 FFT 방향을
쓰면 (U_p) 때문에 다른 답이 나오므로 이 회귀는 물리 gate의 일부다.

근거는 [Lindsey et al., Phys. Rev. C 109, 044605](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.109.044605)와
[accepted manuscript](https://link.aps.org/accepted/10.1103/PhysRevC.109.044605)이다.
그 논문은 0.1--10 keV 충돌에너지에서 Floquet--Volkov와
Crank--Nicolson을 대조하고, thermal 결과는 1 keV D--T plasma를
benchmark로 사용했다. 그 benchmark의 0.3 keV photon에서 1% 임계장은
(8.680352\times10^{14}) V/m이며 공개-support gate를 통과한다. 반면 이 문서의
10 keV 목표는 Gamow saddle이 30.92 keV이므로 그 CN 대조범위를 벗어난다.
0.3 keV 아래 photon도 공개 검증범위 밖이므로 더 낮은 임계장이 나오더라도
채택하지 않는다.

## 3. Maxwellian 평균

각도분포는 3차원 등방성으로, 에너지는 Maxwell--Boltzmann으로 적분했다.

\[
\langle\sigma v\rangle_{\rm FV}
=\sqrt{\frac{8}{\pi\mu}}(kT)^{-3/2}
\int dE\,E e^{-E/kT}\,
\frac12\int_{-1}^{1}dc\,\sigma_{\rm FV}(E,c).
\]

field-free 단면적은 [Bosch--Hale](https://www.osti.gov/etdeweb/biblio/5161054)의
D--T (S(E)) fit을 사용했다. 무구동 (E_0=0)에서 비율 1을 (2\times10^{-14})
이내로 복원한다. 1% 임계장에서 계산한 주요 값은

```text
required field                 4.861597077e15 V/m
ponderomotive energy           9.022551070e1 eV
Gamow saddle energy            3.0917656e1 keV
Keldysh-Gamow parameter        1.3089518e1
coarse/default/fine gain       0.01 / 0.01 / 0.01
maximum grid spread            4.44e-16
weighted out-of-fit mass       5.6e-13 이하
```

이다. (gamma_G>1)이므로 이 점은 quasi-static WKB가 아니라 multiphoton/FV
영역이다. 에너지 121/181/361, 각도 12/16/24, 위상 128/256/512 격자에서
임계점 gain이 수렴했다.

10 keV 증가분을 에너지별로 분해하면 10 keV 이하가 4.17%에 불과하고,
20 keV 이하 39.0%, 30 keV 이하 82.5%다. 즉 전체 증가의 대부분이 공개된
0.1--10 keV CN/FV 대조구간 밖에서 생긴다. 따라서 수치 수렴과 공개 검증을
서로 다른 gate로 기록한다.

## 4. source와 pump 장부

임계장의 평면파 하한은

\[
u_{\rm EM}=\frac12\epsilon_0E_0^2
=1.046349192\times10^{20}\ {\rm J/m^3},
\]

\[
I=cu_{\rm EM}=3.136875961\times10^{28}\ {\rm W/m^2}
\]

이다. geometry를 숨기지 않기 위해 10 fs, 반지름 10 nm의 선언된 pulse를
사용했다. 이는 약 725 optical cycles, (0.09855) J의 입사 에너지다.

동수 D/T 총 ion density (10^{31}\,\mathrm{m^{-3}})의 선언된 microvolume에서
Bosch--Hale 반응률을 그대로 적용하면, 1% 증강으로 얻는 추가 fusion 에너지는
입사 pulse의 (7.65\times10^{-9})뿐이다. 흡수, plasma propagation, ramp,
pump recovery는 아직 풀지 않았다. 따라서 이 장부는 source 비용을 감춘 것이
아니지만 reactor upgrade도 아니다.

## 5. CE scalar와 전자기 장은 같지 않다

등록 scalar의 on-shell quantum은

\[
\hbar\Omega_\Phi=29.64757\ {\rm MeV}=29647.57\ {\rm keV},
\]

즉 QED 외삽점의 0.3 keV보다 98,825배 높다. 0.3 keV로 구동한 같은 massive
field는 macroscopic propagating mode가 아니라 reduced Compton length
6.65576 fm에서 감쇠하는 off-shell near field다.

더구나 질량비례 scalar gradient의 상대좌표 결합은

\[
g_{\rm rel}=\mu\left(\frac{g_D}{m_D}-\frac{g_T}{m_T}\right)=0
\]

이다. 균일한 scalar mass shift도 상대운동에는 공통 위상만 준다. 그러므로
QED의 V/m를 scalar의 V/m로 이름만 바꾸는 mapping은 금지했다.

## 6. exact-\(Z_2\) 두-mode beat 우회로

두 on-shell mode를

\[
\Phi=a\cos(m_\Phi t)+a\cos[(m_\Phi+\Delta)t-\mathbf k\cdot\mathbf x]
\]

로 두면 (Phi^2)에는 실제로 저주파 (Delta) beat가 생긴다. 0.3 keV 차주파수의
두 번째 mode는

\[
k=0.133374\ {\rm MeV},\qquad \bar\lambda_{\rm beat}=1479.50\ {\rm fm}
\]

이고 10 keV Gamow saddle turning radius 46.57 fm보다 길다. 따라서 장벽
구간에서 국소 균일하다는 kinematic gate는 통과한다.

그러나 이것은 전기력이 아니라 reduced-mass modulation이다. asymptotic toy

\[
H_s(t)\simeq[1-\epsilon\cos(\Delta t)]\frac{p^2}{2\mu}+V_C
\]

를 같은 Bosch--Hale Maxwell 평균에 sideband로 연결하면 1%에

\[
\epsilon=0.302439606
\]

이 필요하다. 이미 선형화 범위 밖이다. invisible-width가 허용하는 portal
(lambda=0.005110743)의

\[
C_N^{\rm pair}=\frac{2\lambda f_Nm_N}{m_h^2}
=1.83844\times10^{-10}\ {\rm MeV^{-1}}
\]

는 산란 Feynman vertex 계수다. 고전 배경 질량항에는 동일입자 미분에서
생기는 2를 쓰지 않으므로

\[
C_N^{\rm background}=\frac{C_N^{\rm pair}}2
=9.1922\times10^{-11}\ {\rm MeV^{-1}}
\]

를 사용한다. 정정된 background 계수에서는

\[
a=1.7570\times10^6\ {\rm MeV},\qquad
u_\Phi=5.65818\times10^{40}\ {\rm J/m^3}
\]

가 필요하다. 같은 크기의 30.2% DC mass shift와 약 59.3 MeV sum-frequency
성분도 피할 수 없다. 이 toy는 scalar 전용 finite-pulse Crank--Nicolson
계산도 아니므로 물리적 CE upgrade로 승격하지 않는다.

## 7. 남은 gate

현재 판정은 다음과 같다.

| gate | 판정 |
|---|---|
| gauge-invariant QED action | `PASS` |
| published FV sideband formula | `PASS` |
| 1 keV published thermal benchmark support | `PASS` |
| modified Bosch--Hale Maxwell average | `PASS` |
| prescribed x-ray field에서 10 keV 1% | `FORMULA EXTRAPOLATION ONLY` |
| 입사 source/pump 수치 장부 | `PASS` |
| 선언 microvolume net energy | `FAIL` |
| QED 결과의 CE scalar 귀속 | `REJECT` |
| exact-(Z_2) beat kinematics | `CONDITIONAL PASS` |
| exact-(Z_2) beat source/energy/CN | `FAIL` |
| 물리적 CE scalar 1% gain | `NOT REACHED` |

다음 CE gate는 scalar source의 유한 pulse와 공간 profile, DC 및 sum-frequency
항을 모두 유지한 scalar-specific time-dependent D--T 계산이다. 현재 허용 portal
계수에서는 에너지밀도만으로 이미 강한 negative control이므로, 임의의 (Q)
배율로 이 단계를 건너뛰지 않는다.

## 8. 실행

```bash
uv run python examples/physics/fusion_floquet_source_gate.py
uv run --extra dev python -m pytest tests/test_fusion_floquet_source_loop.py -q
```
