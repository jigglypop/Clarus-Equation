# CE 미시적 상태준비의 동역학: 공통 환경, 보존된 교환량, 대칭평균의 유도

CE-CS1 · 2026-09-19 · 새로운 관측 피팅 없음.

## 0. 결론과 범위

기존의 “네 상보적 기저에서 24개 상태쌍을 동등하게 평균한다”는 준비를, **명시적인 공통 환경 충돌의 시간진화가 접근하는 유일한 조건부 정상상태**로 구현했다. 같은 과정에서 방향당 가지분산 delta/8, 교환사건확률 delta, 가법적 에너지분율 delta가 함께 회수된다. 수렴률과 미시 오차에서 거시 표본오차로 가는 상계를 붙였다.

그러나 공통 환경은 초기 교환량 delta를 보존한다. 따라서 delta의 수치 자체, p=4 alpha_s^(4/3), 실제 PMNS/Yukawa, 생성된 최종 바리온량 또는 진공 절대값을 선택한 것은 아니다. 또한 이 환경의 실제 사건 기록은 기존 Poisson(3+delta) 분기와 다르다. 둘을 같은 사건이라고 부르는 강한 동일시는 이 구성에서 성립하지 않는다.

이는 표준 집단 소산·Schur–Weyl 구조를 CE의 기존 준비에 적용한 조건부 구성이다. 표준 수학의 최초 발견을 주장하지 않는다. 이전의 관측 근접값과 이번 구현 정리를 독립 관측 성공처럼 합산하지 않는다.

## 1. 보존한 입력과 추가 공리

기존 입력은 alpha_s=0.11789, p=4 alpha_s^(4/3), delta=p(1-p)다.

- p=0.23122206826075514
- delta=0.17775842340997383
- delta/8=0.02221980292624673
- D_legacy=3.1777584234099736
- q_legacy=0.04864671964402821

기존 h의 진단 고유값은 sqrt(0.5901923788646684), sqrt(1.1098076211353316), sqrt(1.3)이다. 실제 렙톤 질량으로 동정하지 않는다.

추가 공리 A1: 두 구별되는 qutrit 준비에 같은 환경이 작용한다. 초기 R(0)=rho_p tensor rho_p, rho_p=diag(p,1-p,0)이며 두 준비는 조건부 독립이다. 초기 rank-two 기록을 만드는 과정은 상속한 준비조건이다.

추가 공리 A2: 공통 환경은 8개의 traceless Hermitian 방향을 동등하게 결합하고, 신선한 환경 앙실라와 짧은 충돌을 반복한다. 방향별 강도를 별도로 고르지 않는다. gamma>0는 전체 시간척도이며 그 자연값은 미정이다. 초기 p나 최종 delta에 맞춰 gamma를 선택하지 않는다.

추가 공리 A3: hbar=1에서 H0=h tensor I+I tensor h를 유지한다. 이 미시계는 열린계다. 환경의 준비·교체 및 스위칭 비용을 포함한 자율적 우주 Hamiltonian은 아직 주어지지 않았다. 감소한 계 에너지를 소멸했다고 처리하지 않으며 총 에너지 장부의 환경/일 부분은 별도다.

## 2. 유한 충돌에서 연속 생성자로

Tr(Ta Tb)=delta_ab/2로 정규화한 SU(3) 생성자로

    La=Ta tensor I+I tensor Ta

를 정의한다. 앙실라 qubit를 |0>로 준비해

    Ua(dt)=exp[-i sqrt(gamma dt) La tensor sigma_x]

로 충돌시킨다. 환경을 부분추적하면 정확히

    Phi_a(R)=Ca R Ca+Sa R Sa,
    Ca=cos(sqrt(gamma dt) La), Sa=sin(sqrt(gamma dt) La).

Ca^2+Sa^2=I이므로 각 유한 충돌은 완전양성·trace-preserving이다. 전체 계+환경의 충돌은 유니터리다. 환경을 무조건 없앤 비유니터리 기본법칙을 가정한 것이 아니다.

Taylor 전개는

    Phi_a(R)=R-(gamma dt/2)[La,[La,R]]+O(dt^2)

를 준다. 여덟 충돌과 자유진화를 합한 뒤 유한 시간에서 dt->0으로 보내면

    dR/dt = -i[H0,R] -(gamma/2) sum_a [La,[La,R]] = L(R).      (1)

를 얻는다. 반복 상호작용에서 Lindblad 극한을 얻는 일반적인 도구는 [R1]이다. 이 특정 결합 및 환경 통계는 본 구성의 공리다.

## 3. 보존량과 조건부 정상상태의 유일성

두 qutrit 교환을 S, Pi±=(I±S)/2라 하자. [S,La]=[S,H0]=0이므로 모든 유한 충돌과 연속 진화에서

    d Tr(Pi- R)/dt=0.

독립 초기 두 준비에서는 Tr(S rho_p tensor rho_p)=Tr(rho_p^2), 따라서

    Tr(Pi- R(0))=(1-Tr rho_p^2)/2=p(1-p)=delta.                (2)

주어진 delta에서 후보 정상상태는

    Sigma_delta=(1-delta) Pi+/6 + delta Pi-/3.                 (3)

실제로 [La,Sigma_delta]=[H0,Sigma_delta]=0이다.

유일성을 보이자. Hilbert–Schmidt 내적에서 Hermitian La에 대해

    Re <X,L(X)> = -(gamma/2) sum_a ||[La,X]||_HS^2.

정상 연산자이면 모든 [La,X]=0이어야 한다. 3 tensor 3=6 plus 3bar의 두 불가약 표현이 서로 동등하지 않으므로 Schur 정리에 따라 가환대수는 span(Pi+,Pi-)다. trace=1과 식 (2)가 두 계수를 고정한다. 따라서 같은 보존 교환량 안의 정상상태는 (3) 하나다. 원래의 24개 MUB 상태쌍 평균도 (3)과 정확히 같다 [R2].

일반 초기 R에 대한 극한은

    T(R)=Tr(Pi+ R)Pi+/6+Tr(Pi- R)Pi-/3.

이다. 특정 초기 방향을 관측값으로 골라야만 생기는 극한이 아니다.

## 4. 방향당 1/8도 같은 동역학에서 유도

A=P1-P2, Tr A=0, Tr A^2=2라 하고, 같은 공통 회전 과정이 A tensor A를 진화시킨 연산자를 B(t)라 하자. 이는 밀도행렬이 아니라 방향 공분산을 계산하는 보조 연산자다.

    B(infinity)=S/4-I/12.

이는 Tr(A tensor A)=0와 Tr[S(A tensor A)]=Tr A^2=2 및 위 두 불가약 부분공간만으로 정해진다.

조건부 가지차이 d_a=Tr[(U A Udagger)Ta]의 앙상블 공분산은

    C_ab(t)=delta Tr[B(t)(Ta tensor Tb)].

교환 trace identity Tr[S(A tensor B)]=Tr(AB)와 traceless성을 사용하면

    C_ab(infinity)=delta delta_ab/8.                         (4)

따라서 1/8을 수치로 공급하지 않았다. 같은 집단 환경이 기존의 균등 방향평균을 실제로 생성한다. 단, C는 조건부 가지 사이의 분산이며 물리적 PMNS 행렬원소가 아니다.

중요한 구분: 최종 한-사본 주변상태는 I/3, 순도는 1/3이다. 초기 한-사본 순도 0.6444831531800523가 그대로 남는 것이 아니다. 보존된 것은 두-사본 교환상관 Tr(SR)=1-2delta다. 초기 순도와 최종 한-사본 순도를 같은 대상으로 바꾸면 기존 식을 잘못 사용한다.

## 5. 수렴률의 증명과 정확도

SU(3) 표현 (a,b)의 Casimir는 [a^2+b^2+ab+3a+3b]/3이다. 연산자 공간은

    End(6 plus 3bar)=2*1 plus 4*8 plus 10 plus 10bar plus 27.

이에 따라 소산 생성자의 고유값은

    0 (2개), -3gamma/2 (32개), -3gamma (20개), -4gamma (27개).

Hamiltonian commutator는 Hilbert–Schmidt 노름에 기여하지 않고 보존 부문을 유지한다. X=R-Sigma_delta에서

    ||X(t)||_HS <= exp(-3gamma t/2)||X(0)||_HS.               (5)

이는 특정 적분기 결과가 아니라 스펙트럼/에너지 추정으로 얻는 상계다. 방향 공분산에는

    |C_ab(t)-delta delta_ab/8|
      <= (delta/2) sqrt(7/2) exp(-3gamma t/2)

가 따른다. 유한 충돌수의 근사와 연속식 자체의 진화시간을 구분한다.

고정 진단 결과:

|gamma t|상태 HS 오차|식 (5)의 상계|방향 공분산 최대 오차|
|---:|---:|---:|---:|
|1|0.1063646231|0.1206029656|0.01013029048|
|4|0.001180458862|0.001339777928|0.00008813731854|
|8|0.000002926064974|0.000003320977454|0.0000002184371016|
|12|7.252989892e-9|8.231880093e-9|5.414514635e-10|

상태 오차 1e-8 이하를 보장하는 충분조건은 이 초기상태에서 gamma t>=11.8702862263이다. 실제 우주시간이나 자연의 gamma를 산출한 값이 아니다.

연속식 대신 원래 유한 충돌의 곱으로 gamma t=1을 계산하면, 단계수16/32/64/128/256에서 HS 차이는 .00232898144/.00116313035/.00058128408/.00029057906/.00014527469다. dt->0의 1차 수렴이 회수된다. 각 유한 충돌에서도 확률과 교환량은 정확히 보존된다.

## 6. 미시 준비에서 거시 에너지 통계까지

H0=h tensor I+I tensor h, h>0라 하자. 두 정상 부문의 한-사본 주변상태가 모두 I/3이므로

    E+=E-=E0=2 Tr(h)/3,
    Tr(Sigma_delta H0 Pi-)/Tr(Sigma_delta H0)=delta.            (6)

이는 서로 다른 h의 고유값을 같게 설정하지 않고 얻는다. 유도된 정상상태가 직전 에너지–사건 공분산0 조건을 구현한다.

더 나아가 집단 생성자 작용에서 H0-E0 I는 adjoint Casimir 3을 가지므로,

    E(t)=E0+[E(0)-E0] exp(-3gamma t/2),
    A(t)=delta E0+[A(0)-delta E0] exp(-3gamma t/2),
    A(t)=Tr(R(t) H0 Pi-), f_E(t)=A(t)/E(t).                   (7)

가 정확히 성립한다. 분모와 분자가 같은 미시 이완률을 갖는다.

|gamma t|사건확률|반대칭 에너지분율|
|---:|---:|---:|
|0|0.177758423410|0.163958429120|
|1|0.177758423410|0.174678682905|
|4|0.177758423410|0.177724208862|
|8|0.177758423410|0.177758338601|
|12|0.177758423410|0.177758423200|
|극한|0.177758423410|0.177758423410|

기존 비축퇴 h에서 E(0)=1.9750432244895724, E0=1.974592860009483이며 계 에너지 변화는 -0.0004503644800891715다. 환경/구동의 에너지 장부가 필요하다. 열린계의 준비 비용이0이거나 닫힌 우주의 총에너지까지 증명됐다고 하지 않는다.

같은 정상상태에서 v_h=Tr(h^2)/3-(Tr h/3)^2이면 Var_-(H0)=v_h, Var_+(H0)=5v_h/2도 회수한다.

독립적으로 준비된 M개 셀에서 에너지 E_i와 교환기록 C_i를 함께 측정하자. [H0,Pi-]=0이므로 이 공동 측정은 정의된다. 정상상태에서는

    fhat_E=sum_i E_i C_i / sum_i E_i -> delta.

양의 에너지 범위 E_min<=E_i<=E_max에 대해 Hoeffding의 표준 bounded-variable 부등식을 적용하면

    Pr(|fhat_E-delta|>=epsilon)
      <= 2 exp[-2 M epsilon^2 E_min^2/E_max^2].               (8)

유한 준비시간이면 (7)의 편향을 먼저 더한다. 이는 독립 셀 표본의 거시적 평균 정확도이지 우주 밀도요동의 공간 power spectrum이나 As 예측이 아니다.

## 7. 같은 미시법칙의 사건수까지 계산하면 기존 Poisson 분기는 자동으로 나오지 않는다

원래 앙실라의 |1> 기록을 각 방향의 사건으로 센다는 측정 규약을 고정한다. 연속극한의 jump는 sqrt(gamma)La다. 정확한 생성자 완비성은

    sum_a La^2=(10/3)Pi+ +(4/3)Pi-.

따라서 한 부문 안의 총 사건율은 항상 10gamma/3 또는 4gamma/3이고, 부문을 바꾸는 jump는 없다. 사건수 N_t의 생성함수는

    F_count(z,t)=(1-delta)exp[(10gamma t/3)(z-1)]
                   +delta exp[(4gamma t/3)(z-1)].             (9)

같은 Lindblad 식의 tilted generator에서 독립 계산해 확인했다.

    E N_t=gamma t(10/3-2delta),
    Var N_t=E N_t+4(gamma t)^2 delta(1-delta).

이는 보존된 내부 부문에 조건화한 두 Poisson 분포의 혼합이다. gamma t=1이라는 진단에서는 평균2.977816486513386, 분산3.562457951780483, Fano1.1963322682626532다. 기존 평균3+delta=3.1777584234099736 및 단일 Poisson과 다르다. gamma t를 조절해 평균만 맞춰도 추가 분산이 남는다.

따라서 이 환경의 jump를 곧바로 바리온 분기과정의 offspring라고 선언할 수 없다. 기존 q를 틀렸다고 보편적으로 결론내리는 것이 아니라, **이번 microscopic event와 기존 branching event의 직접 동일시가 성립하지 않는다**는 범위의 검사다. 다른 quantum unraveling을 다른 물리 계수 없이 같은 관측기록으로 바꿔 부르지 않는다.

## 8. 미시적 기원에 대한 정확한 판정

이번에 대체한 가정은 “균등한 24상태 평균을 직접 공급한다”는 준비다. 공통·등방·Markov 충돌이라는 명시적 microscopic protocol이 그 평균과 에너지 통계를 안정적으로 생성함을 증명했다.

그러나 다음은 남는다.

1. delta 값 자체: 식 (2)는 보존식이다. 임의의 다른 초기 delta도 해당 정상상태를 만든다. 초기 p와 alpha_s matching은 아직 선택되지 않았다.
2. PMNS: 방향 공분산이 실제 질량행렬의 unitary mixing 원소가 되는 연산자는 없다. delta/8을 물리적 mixing으로 새로 증명하지 않았다.
3. 생성량: 식 (9)는 원래 Poisson(D)와 다르다. 사건수·최종 물질에너지·우주 임계밀도의 대응은 따로 증명해야 한다.
4. 진공: H0에 상수를 더해도 commutator와 상태진화는 같지만 에너지는 이동한다. 이 동역학만으로 절대 진공밀도는 정해지지 않는다.
5. 에너지와 압력: 정상상태의 모든 additive 한-사본 관측량은 두 부문에서 같은 평균이다. 따라서 이 같은 준비만으로 한 부문 w=0, 다른 부문 w=-1을 지정할 수 없다.

상태가 entropy를 증가시키며 정상화하는 것은 주어진 교환량 제약 아래의 최대엔트로피 선택이다. log(Sigma_delta)=aI+bS이므로 같은 delta를 가진 R에서 D(R||Sigma_delta)=S(Sigma_delta)-S(R)>=0이다. 이는 유일한 constrained maximum이다. 에너지 제약을 가진 유한온도 Gibbs 평형과는 다르다. 특히 비축퇴 h에서는 Sigma_delta가 일반적인 Gibbs(H0+J S)가 아니다.

기존 main의 88–89장 Higgs–Clarus Gaussian portal은 별도 연구 가지다. 이 충돌 생성자가 그 장론의 미시적 유도라고 주장하지 않으며 기존 원고를 수정하지 않는다.

## 9. 실제 검산과 게시

state_selection.py는 원래 Gell-Mann 생성자와 12개 MUB 사영에서부터 9x9 상태·81x81 생성자를 구성한다. 20개 unittest를 실행했다. Casimir 스펙트럼, 유일 정상공간, 모든 충돌의 교환보존, 노름상계, 유한 충돌수 수렴, 별도 원래 MUB 평균, 에너지의 닫힌 시간해, 조건부 에너지분산, 실제 기록 PGF의 tilted-generator 대조, 거시 비율의 분산을 검사했다.

초기상태 다섯 무작위 대조에는 고정 seed19092026을 사용했다. 관측으로 seed나 초기방향을 골랐다는 뜻이 아니다. 이번에 과거 BAO/CMB/재결합/인플레이션 또는 뮤온 계산은 재실행하지 않았다.

실행: `OPENBLAS_NUM_THREADS=1 python state_selection.py --output results.json`.

Git의 앞선 인계 커밋 a7e9476은 요약·핵심 수치·원본 ZIP 식별자 게시다. 대용량 ZIP 전체 업로드를 의미하지 않는다. 이번 신규 코드·원고·결과는 별도로 main에 게시한다.

## 출처

첨부 CE_symmetry_energy_closure_2026-09-18 및 CE_micro_macro_bridge_2026-09-18의 상태·평균·에너지 범위를 상속했다. CE_gravity_radiation_closure는 이전 전달 계산의 인계 출처이며 이번 상태준비 계산의 물리 증거로 쓰지 않았다.

[R1] S. Attal, Y. Pautrat, From repeated to continuous quantum interactions, math-ph/0311002. https://arxiv.org/abs/math-ph/0311002

[R2] A. Klappenecker, M. Roetteler, Mutually Unbiased Bases are Complex Projective 2-Designs, quant-ph/0502031. https://arxiv.org/abs/quant-ph/0502031

[R3] P. Zanardi, M. Rasetti, Noiseless Quantum Codes, quant-ph/9705044, Phys. Rev. Lett. 79,3306. 집단 환경과 보존 대수의 기존 연구. https://arxiv.org/abs/quant-ph/9705044

이 문서의 구체적인 두-qutrit 스펙트럼·시간해·사건수는 본문에서 직접 유도하고 코드에서 대조했다. 외부 논문이 CE 전체나 새 관측 성공을 증명한다는 인용은 하지 않는다.
