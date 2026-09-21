# CE-EI1: 보존한 경험식의 역산과 교차예측

2026-09-22. 암흑에너지·암흑물질·뮤온·허블에서 과거 가까운 수치를 냈던
원식을 Git 이력에서 별도로 보존하고, 같은 입력으로 전방 재현·역산·교차예측한다.
원고: [후속 38장](../../paper/후속연구_기록과_상태선택/38_고정밀_경험식의_복원과_교차_역예측.md).

## 실행

저장소 루트의 PowerShell에서:

```powershell
.venv\Scripts\python.exe experiments/ce_empirical_inverse_20260922/recover_sources.py
.venv\Scripts\python.exe experiments/ce_empirical_inverse_20260922/inverse_predictions.py
.venv\Scripts\python.exe experiments/ce_empirical_inverse_20260922/hubble_flow_inverse.py
.venv\Scripts\python.exe experiments/ce_empirical_inverse_20260922/dark_energy_reconstruction.py
.venv\Scripts\python.exe experiments/ce_empirical_inverse_20260922/common_background.py
```

Python, numpy, scipy, mpmath를 사용한다. 외부 서비스나 유료 계산은 필요 없다.
`recover_sources.py`에만 기존 Git 객체가 필요하며, 보존 후 나머지 계산은
`sources/`의 해시가 고정된 사본으로 재현한다.

## 산출물

- `sources.json`, `sources/*.txt`: 원본 commit·blob·SHA256·행 범위와 9개 역사 자료.
  옛 문서의 과장·오류도 원문 그대로 남아 있으므로 현재 주장으로 인용하지 않는다.
  사용자가 삭제한 옛 runtime 디렉터리를 복원하지 않는다.
- `results.json`: 정확한 3계층 밀도분할, 홀로그래피 척도, 뮤온 접촉·유한핵,
  대안 스칼라 핵, 역산별 교차예측, 공통 입력 공분산, 필요한 보정의 역산.
- `hubble_flow_results.json`: 옛 H0 결과, 같은 toy 방정식의 수치 수렴,
  CMB→국소 허블 역예측. 지평선 척도 분기와 별도다.
- `dark_energy_results.json`: 보존한 w0·wa 식에서 정준 스칼라 퍼텐셜을
  매개형식으로 구성하고 Klein–Gordon 방정식을 검산한다.
- `common_background_results.json`: DE·DM 배경에 곡률 기반 ε flow를 함께
  구현하는 양의 운동계수와 퍼텐셜을 구성한다. 옛 허블 읽기와의 불일치,
  같은 배경에서의 acoustic 역조건, 바리온 밀도–H 범위도 산출한다.

## 판정 범위

숫자 보존, 조건부 역함수 정리, 관측 간 일치조건, 이를 재현하는 저에너지
구성의 명시가 목적이다. 한 원리에서 계수·상태·여러 장이 필연적으로 발생한다는
증명, 실제 전체 likelihood, 새 독립 관측 검증은 아직 없다.

옛 접촉값 248.996×10^-11과 유한핵 134.854×10^-11을 보존한다.
WP25+2025 최종 실험 기준의 잔차 38.5±63.7×10^-11도 별도로 비교한다.
2026-04-28 BMW/DMZ v2 기준의 잔차 19.5±38.8×10^-11도 별도 비교한다.
서로 공통 자료를 쓰는 SM 기준들이므로 두 점수를 독립 자료처럼 합산하지 않는다.
새 기준에 맞춰 옛 계수나 입자 질량을 덮어쓰지 않는다.

DESI 압축 비교는 명시한 평탄 LCDM posterior의 2차원 Gaussian 진단이다.
여기서 DM은 비바리온 비상대론적 물질 전체로 읽는다. CDM만의 역산, 진화하는
암흑에너지 배경, 후기 flow는 다른 식별/분기이므로 이 점수를 공유하지 않는다.
