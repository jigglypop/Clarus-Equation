# CE 우주론 이론·저장소 구현 전수 감사 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/_archive/repository-code-analysis-20260815

## 1. 질문

현재 dirty checkout에서 CE의 우주론 주장을 생성·전달·검사하는 문서, Python
구현, 관측 fixture, 사전등록과 테스트의 의존 경로 전체를 읽었을 때 다음을
판정한다.

1. 고정점, 성분비, FLRW 배경, 거리, 성장, 원시 스펙트럼, 우주상수와
   $H_0$ readout에 관한 식이 명시된 전제에서 수학적으로 성립하는가.
2. 각 식의 가지·정의역·경계·근사 오차와 exp/log/확률/고정점 인자의
   무차원성이 닫혀 있는가.
3. 실행 코드와 테스트가 문서의 살아 있는 좁은 주장을 실제로 재현하며,
   실패·fixture 결손·미실행 경로를 녹색으로 오인하지 않는가.
4. 2026-08-15에 접근 가능한 1차 관측 자료와 같은 likelihood·공분산 조건에서
   CE의 수치 관계가 무엇을 지지하거나 배제하는가.
5. 각 결론의 지위를 정의·정리·공리·산출·경험식·미완성·예측 중 하나로
   분류할 수 있는가.

## 2. 감사 범위

포함 범위는 저장소 전체 중 우주론 주장에 도달하는 모든 살아 있는 경로다.

- 루트 README와 우주론·상수·고정점·작용·형식 원장 문서
- `examples/physics`의 cosmology, residual forward model, $H_0$,
  primordial spectrum, cosmological constant 및 discrimination gate
- `reality_stone/python/reality_stone/clarus`의 bootstrap, dimensionless,
  core model-selection 및 위 스크립트가 실제 import하는 공용 코어
- 대응 테스트, scorecard, validation runner, 관측 registry/fixture와
  `experiments/preregistration/cosmology_*`
- 문서가 참조하지만 현재 checkout에 없거나 이름이 어긋난 파일도 provenance와
  재현성 결함으로 포함

우주론 경로와 연결되지 않는 AGI, brain, fusion, Guard 제품 구현과 삭제 상태인
`RBE/`는 수학 감사 대상에서 제외한다. 선행 run의 저장소 전체 inventory와
일반 release 판정은 재사용하되, 우주론 코드·문서·테스트는 현재 snapshot에서
새로 추적한다.

## 3. 고정할 명제

- **C1 (고정점):** 무차원 $D>1$에서
  $q=\exp[-D(1-q)]$는 $q=1$ 외에 $(0,1/D)$의 유일한 최소 해를 가지며,
  선택한 $D_{\rm eff}$의 수치해는 코드와 일치한다.
- **C2 (밀도 사상):** $q_{\rm ext}\mapsto\Omega_b$와 추가 성분 분할은
  고정점 정리의 결론인지, 독립 물리 사상·경험식인지 구분한다.
- **C3 (조건부 우주론):** 명시된 평탄 FLRW/CPL/GR 및 외부
  $(H_0,r_d,\sigma_{8,0})$ 아래 배경·거리·성장 계산은 내부적으로
  일관되고 단위가 맞는다.
- **C4 (초기우주·절대척도):** 원시 스펙트럼, $H_0$ readout과 우주상수
  절대척도에 관한 코드가 독립 예측인지, 입력 재표현·사후 보정·미완성
  bridge인지 판정한다.
- **C5 (관측 판별):** 동결된 자료·공분산·nuisance·자유도·선택 규칙이
  공급된 항목만 관측 예측으로 평가한다. 중심값 근접과 같은 자료에서의
  재보정은 예측 증거로 세지 않는다.
- **C6 (저장소 보장):** 테스트 통과는 등록된 계산 계약의 구현 무결성만
  보장하며 C2/C4의 물리 사상이나 자연의 참을 증명하지 않는다.

## 4. 정의역·기호·독립 전제

- $q\in(0,1]$, $D\geq0$; 최소 비자명 해를 말할 때는 $D>1$.
- $a>0$, $z=a^{-1}-1$, $E(a)=H(a)/H_0$; 곡률과 방정식상태는 각 branch가
  명시한 범위만 허용한다.
- $\Omega_i=\rho_i(t_0)/\rho_{\rm crit}(t_0)$는 무차원이며, 평탄 closure,
  GR, CPL, scalar potential, 가지 선택과 물리 동일시는 독립 공리다.
- 관측값, 외부 scale, 재결합·sound horizon, nuisance, calibration과
  covariance는 외부 입력이다.
- 물리 단위는 명시적으로 환산하며 $c=\hbar=1$ 사용 여부와 reduced/non-reduced
  Planck mass를 혼용하지 않는다.

## 5. 판정 기준과 허용 오차

- 대수 항등식·고정점 잔차는 달리 명시되지 않으면 절대오차 $10^{-12}$ 이하,
  수치 적분은 구현 테스트에 사전 고정된 tolerance와 수렴 검사를 적용한다.
- 관측 잔차는 한 데이터 release의 평균·오차·공분산만 결합한다. 공분산이
  없으면 독립 sigma 주장을 만들지 않는다.
- branch, 단위, 오차항 또는 정의역이 빠져 결론을 바꿀 수 있으면 P0,
  결론은 보존되나 숨은 공리·provenance·검증 공백이면 P1, 문서 정리 문제는 P2다.
- 허용 정의역 안의 완전한 반례가 있는 부모 주장은 폐기하고, 정확히 좁힌
  후손만 별도 정리로 보존한다.
- 현재 소스 수정은 요청 범위 밖이다. 감사 산출물과 재현 로그만 추가하고
  기존 사용자 변경을 보존한다.

## 6. 산출물

- `10-sources.md`: 1차 이론·관측 자료와 로컬 fixture provenance
- `11-math.md`: 명제별 유도, 반례, 무차원 감사
- `12-routes.md`: P0/미완성 bridge의 대안 경로와 salvage 가능 범위
- `20-audit.md`: 형식 지위와 심각도 gate
- `30-implementation.md`: 소스 변경 여부와 범위
- `31-validation.md`: 명령, exit code, 잔차, 표본·제외·fixture 기록
- `40-final-report.md`: 자기완결적인 최종 판정과 재현 절차
