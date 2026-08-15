Status: COMPLETE

# 31 — 우주론 감사 검증 기록

Snapshot은 `5414336ae2ff20197efe3bf8a92ec5183ad079aa`와 현재 dirty
worktree다. 실행 환경은 Windows/PowerShell, Python 3.11.9, uv 0.11.9이며
기준일은 2026-08-15 (Asia/Seoul)이다. 제품 소스는 수정하지 않았다.

이 파일의 PASS는 등록된 산술·schema·회귀 계약의 재현만 뜻한다. 물리
사상, 독립 예측 또는 자연에 대한 참을 승격하지 않는다. 원문에 가까운
전체 명령 원장은 `artifacts/validation-command-ledger.md`, 독립 반례
수치는 `artifacts/verify_cosmology_math.py`에 있다.

## 1. 실행 결과

| 검사 | exit | 결과 | 허용 해석 |
|---|---:|---|---|
| 독립 수학 검산 artifact | 0 | 고정점·FLRW·Hubble·비균일 적분·covariance·entropy 반례 재현 | 수학 감사 증인 |
| 확장 우주론 pytest 묶음 | 0 | 91 passed | 등록 구현 회귀 통과 |
| CE core 검증 묶음 | 0 | 58 passed, warnings 2 | core 구현 회귀 통과 |
| canonical document policy | 0 | 5 passed | 문서 정책 회귀 통과 |
| manuscript validator | 0 | 47/47; active `[예측]` 0; CE-specific physical closure 0 | 문서·산술·라우팅 검사만 통과 |
| future holdout validator | 0 | manifest VALID; holdout unassigned; evaluation NOT_READY | 관측 검증 미실행 |
| 전체 저장소 pytest | 1 | 49 failed, 2500 passed, 14 skipped, 41 errors, warnings 2 | clean-repository/release 주장 불가 |
| 우주론 경로 Ruff | 1 | 9 findings; 8 mechanical, 1 material unused `h2` | 정적 부채 존재 |

전체 회귀의 주된 실패는 범위 밖 ScienceDB fusion payload와 AGI/Q0
manifest·sealed artifact 결손이다. focused cosmology 묶음에서는 실패가
없었지만, 이는 아래의 완전 반례와 양립한다.

## 2. 결정적 독립 반례

- Hubble 경로에서 선언한 `omega_b h2`를 0.001과 0.1로 바꿔도 계산된
  acoustic angle 차이는 0이었다. 복사 지배기 `a=10^-6`에서 Ricci readout은
  코드 11.9664, 정확식 0.0111918로 재현됐다. 현재 H0 closure 수치는
  검증 결과에서 제외한다.
- log-grid에 균일-grid Simpson 공식을 적용한 cumulative branch는
  `a` 약 0.1에서 기준 적분보다 약 +33.9%였다. 이 branch와 의존 growth를
  제외한다.
- covariance parser는 양의 대각을 가진 비-SPD 행렬 `[[1,2],[2,1]]`을
  받아들였고, 잔차 `(1,-1)`의 이차형식은 -2가 됐다. 내장 DESI 행렬은
  별도 검산에서 SPD였으므로 그 특정 fixture의 계산만 보존한다.
- phase-area 경로의 두 절대척도는 같은 Friedmann--entropy scale을
  재표현하며 여러 외부 선택에 의존한다. 항등식만 보존한다.

## 3. 관측 진단

동결된 DESI DR2 13점과 내장 full covariance에서 외부
`rd=147.09 Mpc` 고정 모형은 `chi2=37.100260857`, 자유도 13,
`p=0.0003995733`이었다. 같은 자료에서 scale 하나를 fit하면
`chi2=12.608346862`, 자유도 12, `p=0.3981382`이나 이는 사후 진단이지
CE 예측이 아니다. 해당 자료는 preregistration 전에 이미 열람됐으므로
두 결과 모두 exploratory다.

`scorecard.py`, `run_validation.py`, residual CLI는 각각 출력에
`CAUTION` 또는 `REJECT`가 있어도 process exit 0일 수 있다. 따라서 이
감사에서는 exit 0을 과학적 PASS로 해석하지 않았다.

## 4. 핵심 재현 명령

```powershell
python _workspace\ce\cosmology-theory-repository-audit-20260815\artifacts\verify_cosmology_math.py

python -m pytest tests\test_bootstrap_solver.py tests\test_core_model_selection.py tests\test_cosmology_ratio_audit.py tests\test_ce_residual_forward_model.py tests\test_recombination_drag_adapter.py tests\test_primordial_spectrum_readout_gate.py tests\test_dimensionless.py tests\test_holdout_preregistration.py -q -p no:cacheprovider

python docs\2_경로적분과_응용\validate_manuscript.py

python experiments\preregistration\validate_holdout_manifest.py experiments\preregistration\cosmology_future_holdout_v2.json
```
