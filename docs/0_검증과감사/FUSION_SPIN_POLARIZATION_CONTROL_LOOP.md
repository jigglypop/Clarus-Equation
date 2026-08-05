# D–T 스핀 편극 표준모형 대조 루프

작성일: 2026-08-05  
코드: reality_stone/python/reality_stone/clarus/fusion_spin_polarization_control_loop.py  
실행: examples/physics/fusion_spin_polarization_control_gate.py  
테스트: tests/test_fusion_spin_polarization_control_loop.py

## 1. 판정

저에너지 D–T에서 이상적인 $J^\pi=3/2^+$, $s$-wave quartet만 반응한다고 두면

\[
A_J=\frac{\sigma_{\rm pol}}{\sigma_0}
=1+\frac{P_DP_T}{2}
\]

이고 1%에는 $P_DP_T=0.02$가 필요하다. 이 상수 배율은 Bosch–Hale Maxwellian
평균과 교환되므로 이상적 projector 대조군의 반응률 비는 1.01이다. 이때
Bosch–Hale D–T fit은 공개 적용 범위인 $0.2\le T\le100$ keV 안에서만 평가하며,
범위 밖 입력은 외삽하지 않고 거부한다. Quartet
지배와 최대 약 50% 증가는 chiral-EFT 힘을 쓴 NCSMC 계산이 지지한다
([Hupin–Quaglioni–Navrátil](https://www.nature.com/articles/s41467-018-08052-6)).

그러나 실제 편극 단면적은 spin correlation $C_{zz}(E)$와 고차 partial wave를
포함한다. $C_{zz}$는 직접 측정되지 않았고, 이 구현도 energy-dependent 편극
단면적을 새로 Maxwellian 적분한 것이 아니라 **이상적 상수 projector를 표준
반응률에 곱한 control**이다. 이 projector를 실제 energy-dependent 반응 연산자로
유도한 provenance도 아직 없다. 목표점이 정확히 1.000%라 이론 불확실성 여유도
없다.

또한 이 효과는 CE가 아니라 표준 핵스핀 선택규칙이다. 2026년 현재 reactor-rate
편극 tritium source, burn-weighted D–T 편극 보존 실측, pair당 wall-plug 장부가
없다. 따라서 최대 단계는

    STANDARD_MODEL_IDEAL_QUARTET_ONE_PERCENT_CONTROL_
    SOURCE_THROUGHPUT_AND_RETENTION_FAIL_CLOSED

이며 물리적 spin-fusion branch와 CE branch는 모두 False다.

## 2. 밀도행렬, projector, Maxwellian 경계

Deuteron $S_D=1$, triton $S_T=1/2$에 대해

\[
\mathcal O=\mathbf S_D\cdot(2\mathbf S_T),\qquad
\Pi_{3/2}=\frac{\mathcal O+2}{3}.
\]

서로 상관 없는 축방향 vector 편극이면

\[
\langle\Pi_{3/2}\rangle=\frac{2+P_DP_T}{3},\qquad
\langle\Pi_{3/2}\rangle_0=\frac23,
\]

이므로 위의 $A_J$가 나온다. 더 일반적인 편극 단면적은

\[
\frac{d\sigma_{\rm pol}}{d\Omega}
=\frac{d\sigma_0}{d\Omega}
\left[1+\frac12p_{zz}A_{zz}
+\frac32P_DP_TC_{zz}\right].
\]

따라서 실제 10 keV 판정에는 $A_{zz}(E),C_{zz}(E)$를 넣은 수정 단면적의
Maxwellian 적분이 필요하다. 코드는 이를
energy_dependent_polarized_cross_section_integrated=False와
spin_correlation_czz_directly_measured=False로 잠근다.

설계 여유 대조점은 $P_D=0.50,\ P_T=0.10$이다. 이상적 주입 배율은 1.025이고,
1%를 남기려면 편극 곱의 40%, 두 종의 보존율이 같다면 종별 63.25% 이상이
burn volume에서 유지되어야 한다. 이상적 상수-quartet **retention 하위 gate**는
단순 수명이 아니라

\[
\overline{P_DP_T}_{\rm burn}
=\frac{\int dVdt\,n_Dn_T\langle\sigma v\rangle_0P_D(V,t)P_T(V,t)}
{\int dVdt\,n_Dn_T\langle\sigma v\rangle_0}
\ge 0.02
\]

이다. 실제 물리 gate는 이 조건만으로 끝나지 않으며, $A_{zz}(E),C_{zz}(E)$ 또는
편극 단면적 자체를 에너지·공간·시간에 대해 burn-weight해 검증해야 한다.

## 3. 10 keV 열평형 자화 경로의 no-go

이상적 비상호작용 Zeeman 분포에서

\[
P_D=\frac{2\sinh x_D}{1+2\cosh x_D},\quad
P_T=\tanh x_T,\quad
x_i=\frac{\mu_iB}{kT}.
\]

$T=10$ keV와 $P_DP_T=0.02$를 동시에 만족시키면

    B_required          3.5013566e10 T
    P_D                 0.0630012
    P_T                 0.3174543
    B^2/(2 mu0)         4.8778993e26 J/m^3

이다. 총 D–T 이온밀도 $10^{20}\,\mathrm{m^{-3}}$의 동온 이온+전자
열에너지밀도보다 약 $1.0\times10^{21}$배 크고, 전자 QED 임계장
$4.414\times10^9$ T의 약 7.93배다. 즉 고전적 plasma 모델의 적용범위가 먼저
끝나며 결론은 더 강한 no-go다.

수치 solver는 자기장 bracket을 적응적으로 늘리되 $10^{14}$ T fail-safe cap을
두고 최종 $P_DP_T$ residual을 검사한다. D–T fusion-fit 범위인 0.2--100 keV 밖
온도는 거부한다. 따라서 49% 같은 높은 공개 입력도 cap에 걸린 가짜 수렴으로
기록하지 않는다.

## 4. 준비와 보존의 2026 상태

LiD DNP는 1 K, 5 T에서 수 시간 뒤 deuteron 약 50%, 0.2–0.3 K와 6–8 T에서
최대 약 70%를 달성한 기술 근거가 있다. 다만 “수 시간”을 코드의 정확한 3시간
측정값으로 해석하지 않는다
([Heidbrink et al.](https://doi.org/10.3389/fphy.2024.1355212)).

2026년 고주파 Alfvén-wave 계산은 특정 파동 조건에서 depolarization이 작음을
보였지만 ionization, pellet ablation, edge transport, metal-wall recycling 전체를
닫지 않는다
([Cook et al.](https://doi.org/10.1088/1741-4326/ae5ab8)).
Laser-driven He-3 plasma에서 일부 편극 보존의 실험 증거도 생겼지만 최종 절대
편극을 측정하지 못했고 D–T tokamak retention 증명은 아니다
([Zheng et al.](https://www.cambridge.org/core/journals/high-power-laser-science-and-engineering/article/preservation-of-3he-ion-polarization-after-laserdriven-acceleration-in-plasma/E6FC7622A7130299AF43998AC5B48A31)).
최신 JLab 프로그램은 D/He-3 DIII-D 최종 측정을 2027년 단계로 둔다
([Jefferson Lab, 2026](https://www.jlab.org/news/releases/fusion-energy-project-gets-boost)).

따라서 시뮬레이션·대리종 증거와 실제 burn-weighted D–T 실측은 별도 Boolean이다.

## 5. source·pump 에너지 장부

임의 처리량 대신 기준 fusion power 500 MW, single-pass burn fraction 1%를
선언한다. $Q_{DT}=17.6$ MeV에서

\[
\dot N_{\rm fus}=1.7731\times10^{20}\ {\rm s^{-1}},\qquad
\dot N_{\rm pair}=1.7731\times10^{22}\ {\rm s^{-1}},
\]

이며 D+T 질량 처리량은 약 $1.48\times10^{-4}$ kg/s다. 공개된 이상적 편극
deuterium neutral-beam **개발 목표** $10^{18}$ D atoms/s와 per-species deuterium
처리율을 비교하면 약 $1.77\times10^4$배다. 이 수치는 달성된 source 성능도,
D–T pair source rate도 아니다.

Pair당 기준 기대 융합에너지는 176 keV다. 선형화하면 1% 반응률 향상의 원시
증분은 1.76 keV/pair, 즉 5 MW다. 하지만 고정 노출에서 기준 hazard/exposure
$\tau=-\ln(1-f)$를 1.01배 하여 $f'=1-(1-f)^{1.01}$로 바꾸는 보수적 장부는
$\Delta f=9.9493325\times10^{-5}$, 1.7510825 keV/pair, 4.9746663 MW다.
이를 wall-plug 예산으로 곧바로 쓰면 안 된다. 전기변환효율 40%를 선언하면

    linearized break-even      704 eV/pair = 2.0 MW
    fixed-exposure break-even  700.433 eV/pair = 1.98987 MW
    10x engineering margin     70.0433 eV/pair = 0.198987 MW

이다. 극저온 COP, DNP microwave 효율, tritium handling, ionization, exhaust와
repolarization을 모두 포함한 측정값이 이 상한 아래인지 아직 모른다. 따라서
net_incremental_energy_positive_demonstrated와 source_and_pump_gate_pass는
모두 False다.

최종 one-percent gate는 reaction, source, retention과 pump가 같은 목표를 쓰고
그 목표가 1% 이상인지 다시 확인한다. burn-weighted 편극 곱은 유한한 [0,1]
값이면서 source 편극 곱 이하, pump 비용은 유한·비음수이며 700.433 eV/pair보다
엄격히 작아야 한다. 경계와 정확히 같으면 순이득이 0이므로 통과하지 않는다.
직렬화된 audit의 저장된 배율이나 break-even을 그대로 신뢰하지 않고 온도, 편극,
fusion power, burn fraction, 효율에서 Bosch–Hale 반응률·pair 처리량·고정노출
증분·wall-plug ceiling을 다시 계산한다.

## 6. 남은 필수 입력

1. $C_{zz}(E)$ 또는 동등한 편극 D–T 관측량과 covariance,
2. energy-dependent 편극 단면적의 10 keV Maxwellian 적분,
3. 편극 tritium을 포함한 연속 source의 처리량과 wall-plug 계측,
4. 주입·이온화·burn·배기·재순환을 포함한
   $\overline{P_DP_T}_{\rm burn}$ 실측,
5. 선언한 효율과 burn fraction에서 전체 source·pump 장부.

이 입력 전에는 표준모형 편극 공학 branch도 물리적 성공으로 승격하지 않으며,
CE branch는 계속 False다.
