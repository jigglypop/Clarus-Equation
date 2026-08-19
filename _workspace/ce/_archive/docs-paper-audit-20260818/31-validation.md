# Validation

Status: COMPLETE

## 실행 결과

새 스킬 두 개에 공식 `quick_validate.py`를 실행했고 두 스킬 모두 유효했다.

문서 정책의 집중 회귀 검사는 다음 명령과 동등한 격리 실행으로 수행했다.

```text
python -B -m pytest tests/test_canonical_document_policy.py -q -p no:cacheprovider --basetemp <unique-temporary-directory>
```

후속 정리까지 포함한 최종 결과는 `11 passed in 0.50s`였다. 실행별 임시 디렉터리를 저장소 밖에 두었고 pytest cache를 만들지 않았다.

수식 정규화 도구의 dry-run 결과는 `files=0 block-delimiters=0 inline-delimiters=0`이었다. 문서 전체에서 로컬 상대 링크 479개를 확인했으며 누락 대상은 없었다.

원고 검증기는 `RESULT: PASS (47/47)`을 냈다. 구분자 정규화에 따라 원고 원문 digest와 정확 문자열 검사를 갱신했으며, 의미 manifest digest는 바뀌지 않았다. 인접 원고 루프 테스트 결과는 `5 passed in 14.44s`였다.

두 새 역할 TOML은 Python 표준 `tomllib`로 읽어 이름을 확인했으며 결과는 `TOML_OK=2`였다.

수학 레인의 독립 계산은 $s_W^2$, $\delta$, $D$, $q_{\rm ext}$, $Dq_{\rm ext}$와 밀도 closure를 정본 수치와 같은 정밀도에서 재현했다. 이 수치 일치는 구현 무결성 확인이며 미완성 물리 다리를 증명하지 않는다.

표준 라이브러리만 사용하는 bootstrap 수학 회귀는 `7 passed in 0.10s`였다. 이 검사는 낮은 소멸근, 기계 정밀도 잔차, 축소 조건, 중앙차분 도함수와 잘못된 유효 깊이 거부를 확인한다. ML package facade를 통과하는 `tests/test_bootstrap_solver.py`는 현재 환경에 `torch`가 없어 `1 skipped in 0.04s`로 명시적으로 제외됐다. 계산 실패와 선택 의존성 부재가 이제 구분된다.

전체 pytest, 전체 benchmark와 release 검증은 사용자가 요청하지 않았으므로 실행하지 않았다.

## 후속 논문 초안 검증

`60-paper-draft.md`에는 전용 검사기 `validate_paper_draft.py`를 실행했다.
절 10개의 순서와 산문 시작, 식 번호 1–21, 정리 증명 7쌍, 산출 유도 1개,
상대 링크, 수식 구분자, 형식 경계와 수치 사슬이 모두 통과했다. 독립
재계산의 고정점 잔차는 binary64에서 `0.0`, 축소인자는
`Dq=0.1545875231200741`이었다.

무차원 회귀는 `7 passed, 8 skipped in 0.07s`였다. 순수 차원 대수 7개는
통과했고, 식 registry 8개는 현재 환경에 선택 의존성 `sympy`가 없어
명시적으로 제외됐다. 제외는 차원 불일치 판정이 아니다.
