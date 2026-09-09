# 24. 접힌 먼지와 상수 진공의 FLRW 전파

CE의 물리 서사는 **끼임 → 접힘 → 암흑 표현**이다. 환경이 여러 가능성 중 하나의 기록을 강제하는 것이 끼임이고, 선택되지 않은 성분을 장부에서 보존하는 방식이 접힘이며, 보존된 에너지를 우주론 변수로 읽는 후보가 암흑 표현이다. 이 장은 이 세 단계 전체를 유도하지 않는다. 이미 공급한 comoving initial data를 homogeneous FLRW에 넣었을 때의 시간 전개와, 상수 진공을 전역 action으로 별도 채택했을 때의 전개를 분리해 적는다.

## 24.1 접힌 기록이 먼지처럼 전파하려면

기존 causal-record 구성은 supplied mode-to-phase-space map 아래에서 한 시각의 dust 초기자료를 주는 존재 witness다. 그것만으로 다음 시각의 stress tensor, 선택법칙, 또는 우주 팽창이 나오지는 않는다. `QD-M5-K`는 여기에 homogeneous, source-free, $p=0$, dust-only, flat expanding FLRW branch를 **추가로** 둔다.

scale factor를 $a(t)$, Hubble rate를 $H=\dot a/a$, 물리 부피를 $V=a^3V_c$라 하자. 고정 comoving 부피 $V_c$에서 continuity equation은

$$
\dot\rho+3H(\rho+p)=0
$$

이다. $p=0$이면

$$
n(a)=n_*a^{-3},\qquad \rho(a)=\rho_*a^{-3},\qquad
H(a)=H_*a^{-3/2},
$$

이고 flat dust Friedmann branch는

$$
a(t)=\left[1+\frac32H_*(t-t_*)\right]^{2/3}
$$

로 닫힌다. 즉 $N=na^3V_c$와 $\rho a^3V_c$가 보존된다. 이것이 ‘접힌 에너지’가 자동으로 먼지가 된다는 뜻은 아니다. 먼지로 읽는 mass-shell/profile과 위의 FLRW 가정이 모두 입력이다. 빠르게 진동하는 quadratic scalar가 평균적으로 dust가 되는 별도 표준 경로는 [Turner (1983)](https://doi.org/10.1103/PhysRevD.28.1243)에 있지만, 그 경로 역시 CE 기록 자체의 유도가 아니다.

## 24.2 한 장면의 진공과 계속되는 진공은 다르다

한 slice에서 에너지를 맞춘 것은 그 slice의 장부다. 이후 모든 시각에 같은 vacuum energy가 존재한다는 전역 명제는 거기서 따라오지 않는다. `QD-M5-L`은 global constant covariant vacuum action을 별도로 채택하고, flat expanding vacuum-only branch를 다룬다. 이때

$$
p=-\rho,\qquad \rho=\rho_*,\qquad
H^2=\frac{8\pi G\rho_*}{3},\qquad
a(t)=e^{H(t-t_*)}.
$$

그 차이는 열역학 형태에서도 읽힌다. $E=\rho V$에 대해

$$
dE+p\,dV=Vd\rho+(\rho+p)dV=0.
$$

먼지에서는 부피가 커질수록 고정 총에너지가 희석된다. 상수 진공에서는 밀도는 그대로지만 $p=-\rho$가 증가한 부피의 에너지와 균형을 이룬다. 이 설명은 global action을 채택한 뒤의 결과이며, one-slice record가 그 action을 산출했다는 주장은 아니다.

## 24.3 암흑 표현이 되려면 남은 장부가 있다

두 branch는 background 방정식만 닫는다. renormalized total stress tensor와 그 보존, 어느 record가 선택되는지, 절대 density와 abundance, 비균질 perturbation·structure formation, 그리고 CE 고유의 독립 관측 예측은 아직 없다. 반고전 중력에서 쓸 stress tensor 자체가 state·renormalization·counterterm을 포함해 보존되도록 정의되어야 한다는 점은 [semiclassical stress 검토 (2020)](https://arxiv.org/abs/2011.05947)를 따른다.

상수 진공 branch는 비교를 위한 조건부 기준선이다. 최신 DESI DR2 분석은 결합 자료에서 $w_0>-1$, $w_a<0$인 진화하는 암흑에너지 해가 더 잘 맞는다는 보고를 포함한다([DR2 논문 목록](https://data.desi.lbl.gov/doc/papers/dr2/), [DESI 2025](https://arxiv.org/abs/2503.14738)). 이것은 CE의 증거가 아니다. CE가 자료와 비교하려면 $w(a)$, perturbation과 sound speed/anisotropic stress, 독립 파라미터 관계, 그리고 사전 고정한 forward likelihood를 별도로 만들어야 한다.

## 24.4 재현 범위

먼지 branch의 continuity/Friedmann/Raychaudhuri 전파는 [homogeneous_dust_flrw_propagation.py](../../examples/physics/homogeneous_dust_flrw_propagation.py), 상수 진공 branch는 [constant_vacuum_flrw_propagation.py](../../examples/physics/constant_vacuum_flrw_propagation.py)에 있다. 원장의 focused 결과는 각각 `16 passed`, `14 passed`다. 이 회귀는 조건부 homogeneous background만 재현하며 density, selection, renormalization, abundance 또는 CE-specific prediction을 증명하지 않는다.

다음 [25장](25_공유_영수증_먼지_상수진공_FLRW.md)은 두 branch를 한 source receipt 안에서 함께 쓸 때 필요한 분할 규칙과 혼합 해를 다룬다.
