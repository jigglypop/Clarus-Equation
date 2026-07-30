# 0. 검증과 감사

이 폴더는 CE 문서군의 주장 강도, 미해결 항목, 검산 상태를 관리하는 정본 카테고리다. 여기 있는 문서는 새 물리식을 추가하기보다, 기존 문서가 무엇을 `Exact`, `Selection`, `Bridge`, `Phenomenology`, `Open`, `Open test`로 주장할 수 있는지 정리한다.

## 살린 문서

| 문서 | 역할 | 판정 |
|---|---|---|
| [PROOF_STATUS_MATRIX.md](PROOF_STATUS_MATRIX.md) | 전체 증명 등급표 | 정본 |
| [PROOF_VALIDATION_LEDGER.md](PROOF_VALIDATION_LEDGER.md) | 실제 검산 이력과 판정 변경 기록 | 정본 |
| [VALIDATION_FRAMEWORK.md](VALIDATION_FRAMEWORK.md) | 남은 증명 요구사항과 tier별 blocker | 참고 정본 |
| [MATHEMATICAL_PHYSICS_ISSUES.md](MATHEMATICAL_PHYSICS_ISSUES.md) | 수학/물리 문제점 감사 목록 | 참고 정본 |
| [BRIDGE_B2_DERIVATION.md](BRIDGE_B2_DERIVATION.md) | \(P_{\text{survive}}\leftrightarrow\Omega_b\) bridge 보강 기록 | bridge 보조 |
| [CORE_STRENGTHENING_LOOP.md](CORE_STRENGTHENING_LOOP.md) | 코어 반례군을 최소 원리·정리·코드 gate로 줄이는 반복 보강 정본 | working canonical |
| [A1_Q0_COVARIANT_ACTION_LOOP.md](A1_Q0_COVARIANT_ACTION_LOOP.md) | 보통 Hessian 반례, 공변 장공간 Hessian, metric variation과 Q0 통과 조건 | working canonical |
| [Q0_0_Q0_3_MINIMAL_MANIFEST.md](Q0_0_Q0_3_MINIMAL_MANIFEST.md) | 깨진 \(U(1)\)+\(Z_2\) 싱글릿 통제 절단에서 범위·장공간·배경·게이지/ghost를 한 convention으로 고정 | control manifest |
| [문서_전체_완성도_감사.md](문서_전체_완성도_감사.md) | docs 전체 완성도와 보강 순서 | 운영 감사 |
| [미해결_난제_목록.md](미해결_난제_목록.md) | 인류/CE 미해결 난제 카탈로그 | open-problem index |

## 죽인 문서

다음 문서는 정식 카테고리 문서가 아니라 임시 산출물, 생성 덤프, stale roadmap이므로 제거한다.

| 문서/폴더 | 이유 |
|---|---|
| `TEMP.md` | 비공식 대화/임시 사고 로그 |
| `llm_extract/` | working tree에서 생성한 LLM 컨텍스트 덤프 |
| `ANALYSIS_SUMMARY.txt` | stale validation 요약, 정본 문서로 흡수됨 |
| `IMPLEMENTATION_ROADMAP.md` | 오래된 구현 로드맵, 현재 코드/테스트 상태와 불일치 |
| `MASTER_ACTION_PLAN.md` | 루트 임시 계획 문서, 카테고리 문서로 대체 |
| `PROJECT_AUDIT_FINAL_REPORT.md` | 과거 단발 감사 산출물, 현재 ledger/status matrix와 중복 |
| `ALTERNATIVE_PROOFS_AND_DATA_UPDATE.md` | 과거 보강 초안, 현재 proof ledger와 audit 문서로 흡수 |
| `README_VALIDATION.md` | 삭제된 요약/로드맵 파일을 가리키던 stale index |

## 읽는 순서

1. [코어 독자 가이드](../코어_독자_가이드.md): 수식 전에 환기구 비유,
   다공간 재귀, 증명과 bridge의 경계를 잡는다.
2. [A1/Q0 공변 작용 루프](A1_Q0_COVARIANT_ACTION_LOOP.md): Hessian,
   stress tensor와 보존법칙 사이의 열린 조건을 먼저 본다.
3. [Q0.0–Q0.3 최소 manifest](Q0_0_Q0_3_MINIMAL_MANIFEST.md):
   전체 CE+SM과 통제 절단의 통과 표시를 왜 분리하는지 본다.
4. [CORE_STRENGTHENING_LOOP.md](CORE_STRENGTHENING_LOOP.md): 코어를 어떤 반례와 gate로 강화하는지 본다.
5. [PROOF_STATUS_MATRIX.md](PROOF_STATUS_MATRIX.md): 지금 무엇이 닫혔고 무엇이 열려 있는지 본다.
6. [PROOF_VALIDATION_LEDGER.md](PROOF_VALIDATION_LEDGER.md): 왜 그 판정이 되었는지 검산 이력을 본다.
7. [미해결_난제_목록.md](미해결_난제_목록.md): CE가 아직 못 닫은 외부/내부 난제를 확인한다.
8. [VALIDATION_FRAMEWORK.md](VALIDATION_FRAMEWORK.md): 남은 blocker와 필요한 증명/실험을 본다.
9. [문서_전체_완성도_감사.md](문서_전체_완성도_감사.md): 문서군 정리 우선순위를 본다.

## 운영 규칙

- 새 주장이 나오면 먼저 `PROOF_VALIDATION_LEDGER.md`에 검산 근거를 남긴 뒤 status matrix를 올린다.
- 수학적으로 닫힌 항목과 관측 readout bridge는 반드시 분리한다.
- 임시 분석, 긴 대화 덤프, 생성된 LLM context는 루트나 `docs/`에 남기지 않는다.
- 검산 코드 없이 "해결"로 부르는 문서는 이 폴더의 audit 대상이다.
