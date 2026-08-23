# CE 우주론 전면 닫힘과 정본 통합 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/cosmology-theory-repository-audit-20260815  
PREDECESSOR: _workspace/ce/cosmology-density-bridge-derivation-20260815

## 1. 최종 목표

이 run의 목표는 CE 우주론의 바리온·암흑물질·암흑에너지·배경팽창·성장률·
$H_0$ readout·원시 스펙트럼·우주상수 절대척도를 하나의 일관된 정본과
계산 경로로 닫는 것이다. 기존
경로에 완전 반례가 있으면 목표량 자체를 낮추지 않고, 반례가 죽이는
구현만 분리한 뒤 구조적으로 다른 작용·current·통계·관측 경로를 끝까지
탐색한다.

정본과 과거판본은 다음 원칙으로 통합한다.

1. 목표 주장과 그것을 구현하던 특정 경로를 별도 Claim ID로 관리한다.
2. 과거판본은 즉시 삭제하거나 이동하지 않고, 정본 주장과 수치·기호·
   전제의 대응표를 먼저 만든다.
3. 하나의 활성 정의·상수 원장·배경 방정식·검증 진입점을 정한 뒤 과거
   문서는 그 정본을 참조하도록 제자리 정리한다.
4. 완전 반례가 죽이는 것은 해당 route claim뿐이다. 원래 물리 목표는
   대체 경로의 `[미완성]` 연구 명제로 보존한다.
5. 역사적 계산과 실패 증인은 git과 run artifacts에 보존한다. 사용자
   승인 없이 광범위 삭제·이동을 하지 않는다.

## 2. 닫을 주장군

### U1 — 정본과 수치 원장

- 기호 $q_{\rm ext}$, $s_{\rm branch}$, $D$, $\Omega_b$, $\Omega_c$,
  $\Omega_m$, $\Omega_\Lambda$, $H_0$의 뜻을 전역에서 하나로 고정한다.
- exact chain, rounded constants와 관측 snapshot을 서로 다른 원장으로
  분리하고 각 수치의 생성 경로와 허용오차를 둔다.
- LO/three-layer/default와 과거 문서의 중복 배경을 하나의 canonical
  configuration 및 명시적인 named alternatives로 통합한다.

### U2 — 확률에서 바리온 abundance로

- 선행 scalar 존재구성을 재사용하되 scalar 값의 직접 density 동일시는
  사용하지 않는다.
- local species label, conserved 또는 명시적으로 reacting current,
  equal-energy 조건, freeze-out hypersurface, total yield와 entropy
  normalization을 갖는 공변 미시 후보를 유도한다.
- composition과 critical-density fraction을 구분하고
  $\Omega_b=q_{\rm ext}\Omega_m$ 이후의 $\Omega_m$ 및 absolute abundance를
  독립 계산한다.

### U3 — 암흑물질·암흑에너지 분할

- baryon, cold dark matter, radiation와 dark energy를 각각 stress tensor와
  continuity equation으로 정의한다.
- 정적 비율 readout이 아니라 하나의 공변 작용 또는 명시된 effective
  fluid system에서 배경과 섭동을 함께 산출한다.
- 기존 CE 비율이 composition, 초기조건, attractor 또는 관측 fit 중 어느
  층에 속하는지 판정하고, 다른 epoch의 교차 산출을 요구한다.

### U4 — 배경팽창과 성장

- radiation을 포함한 평탄 FLRW 식, Ricci scalar와 거리 적분을 정확히
  구현한다.
- 비균일 grid 적분 오류를 제거하고 성장 ODE 또는 검증된 quadrature를
  사용한다.
- $a\to0$, radiation-, matter-, dark-energy-dominated 극한과 독립 수치
  해를 교차 검사한다.

### U5 — $H_0$ readout

- $\omega_b=\Omega_bh^2$, $\omega_c=\Omega_ch^2$, radiation/neutrino,
  recombination과 sound horizon이 실제로 사용되는 forward model을 둔다.
- 사용하지 않는 입력이나 toy curvature 대입으로 $H_0$를 읽지 않는다.
- compressed approximation을 쓰면 full Boltzmann/likelihood 대비 오차와
  유효범위를 고정하고, 독립 관측량을 함께 산출한다.

### U6 — 원시 스펙트럼과 우주상수 절대척도

- 원시 스펙트럼은 projector 숫자 선택이 아니라 perturbation action,
  Mukhanov--Sasaki evolution, vacuum, horizon crossing, reheating과 amplitude
  normalization에서 $A_s,n_s,r$을 함께 산출한다.
- holographic/de Sitter entropy 관계는 동일 $H$ scale의 항등식과 독립
  절대척도 예측을 분리하고, 후자를 주장하면 scale을 고르는 미시 원리와
  radiative-stability 조건을 제시한다.
- 이미 본 관측량에 가장 가까운 후보를 고르는 방식 대신 부수 스펙트럼,
  running, tensor 또는 epoch dependence 중 하나를 사전 교차량으로 둔다.

### U7 — 관측 provenance와 blind 판정

- 관측값은 공식 release, likelihood 또는 공개 chain의 단일 조합으로
  고정하고 hybrid posterior를 만들지 않는다.
- covariance와 상류 입력 오차를 포함하며 snapshot 날짜를 기록한다.
- 이미 본 자료와 같은 likelihood는 holdout으로 세지 않고, 독립 holdout
  또는 아직 보지 않은 교차량을 사전 고정한다.

### U8 — 구현과 정본 통합

- gate 승인 뒤 canonical theory document, constants/config, forward model,
  tests와 CLI를 하나의 진입점으로 통합한다.
- 기존 이름을 유지해야 하면 compatibility alias와 deprecation note를
  사용하고 값이 다른 두 기본값을 조용히 병존시키지 않는다.
- 실패를 exit 0의 성공 문자열로 숨기지 않는 fail-closed validation을
  제공한다.

## 3. 성공 기준

“전부 성공”은 기존 숫자를 같은 숫자로 재출력하는 것이 아니라 다음을
모두 만족하는 상태로 정의한다.

1. U1--U8 각각에 하나 이상의 활성 경로가 있고 정의·공리·정리·산출·
   경험식·미완성·예측 지위가 고정된다.
2. 기존 반례를 우회하는 새 경로는 같은 숨은 자유도를 이름만 바꾸어
   재도입하지 않으며, action/EOM/current/stress/초기·경계조건이 닫힌다.
3. 모든 exp, log, fixed-point와 fraction core가 무차원이고 action density
   차원이 맞는다.
4. analytic limit, 독립 solver와 관련 회귀가 일치한다.
5. 관측 비교에는 provenance·오차·covariance가 있고 target-aware 선택과
   blind test를 분리한다.
6. 정본에는 활성 정의와 살아 있는 경로만 남고, 과거판본의 유효 결과는
   출처를 잃지 않은 채 정본으로 흡수된다.

한 항목이 아직 닫히지 않으면 목표를 삭제하지 않고 정확한 missing lemma,
필요한 데이터 또는 다음 action을 `[미완성]`으로 남긴다. 단, 닫히지 않은
항목을 관측 `[예측]`이나 제1원리 산출로 부르지는 않는다.

## 4. 금지 사항

1. $x:=\Omega_b$ 같은 정의만으로 bridge를 증명했다고 쓰기.
2. 관측 $\Omega_b$, $H_0$ 또는 현재 cosmic time을 coupling·경계조건에
   숨겨 넣기.
3. radiation, baryon loading, neutrino 또는 covariance를 생략한 값을 full
   cosmology 결과로 부르기.
4. 여러 posterior의 성분을 섞어 하나의 공개 chain처럼 표기하기.
5. 과거 문서를 먼저 삭제한 뒤 정본을 추정하거나, 사용자 변경을 덮어쓰기.
6. 수치 테스트 통과를 이론적 닫힘으로 해석하기.

## 5. 허용오차와 검증

- 닫힌 대수·고정점·원장 일치는 절대 또는 상대오차 $10^{-12}$ 이하를
  기본으로 한다.
- quadrature와 ODE는 grid doubling 또는 독립 방법 비교를 기록한다.
- likelihood 비교는 dataset covariance와 upstream uncertainty를 포함한다.
- radiation/matter/de Sitter 극한은 별도 단위시험으로 강제한다.
- canonical 통합 전후 모든 변경 파일과 기존 dirty worktree를 분리해
  기록한다.

## 6. 산출물

- `10-sources.md`: 공식 우주론 자료·Boltzmann/유체/current 1차 출처
- `11-math.md`: U1--U7 유도, 반례 경계와 대체 경로 검산
- `12-routes.md`: 구조적으로 다른 action/readout/likelihood 경로 비교
- `20-audit.md`: 목표 claim과 route claim을 분리한 형식 지위 게이트
- `30-implementation.md`: 승인된 정본·코드 통합
- `31-validation.md`: 수치·회귀·provenance 검증
- `40-final-report.md`: 자기완결 결론과 남은 재개 조건
- `artifacts/`: 판본 대응표, 수치 스크립트, 후보 작용과 재현 로그
