# 검증 기록: 등호·무차원 결함·대안 차원

Status: COMPLETE

## 집중 기계 검증

| 검사 | 결과 | 범위 |
|---|---|---|
| `.codex/hooks/python.cmd doctor` | PASS | system Python 3.11.9, bytecode disabled, repository `PYTHONPATH` |
| `artifacts/verify_equality_dimension_math.py` | `OK equality/dimension certificate` | typed zero, SPD 반례, Pi domain, E3 퇴화/비선형 반례, Hodge/binomial/power counting |
| `.codex/hooks/python.cmd pytest tests\\test_dimensionless.py -q` | **23 passed** | 기존 무차원 API와 새 E1--E3 production guards |
| focused canonical-document policy | **2 passed** | relative links 및 renderable math delimiters |
| 두 서사 문서 UTF-8 strict decode | PASS | 독자 문서 인코딩 |
| `git diff --check` | PASS | whitespace error 없음; 기존 line-ending warning만 존재 |
| research contract/lanes/gate | PASS | 독립 source/math/routes lane 및 최종 형식 gate |

최종 numeric spot-check는 다음을 확인했다.

- `log_equality_defect(1e308, nextafter(1e308,+inf))`
  = `1.9958403095347196e-16 > 0`이며 역순도 같다.
- 최소 subnormal과 다음 subnormal의 결함은 `log(2)`다.
- `1e308 / 5e-324`의 결함은 유한한 `1453.6362805635472`다.
- 정확히 같은 값에서만 log 결함이 정확히 `0.0`이다.
- linear/Mahalanobis/beta 보상의 representational overflow는 `ValueError`로
  거부된다.

## 수학적 판정과 기계 판정의 분리

테스트와 certificate는 선언한 차원 character, 결함, 정규화 및 유한정밀도
구현의 일관성을 보인다. 이것은 다음을 증명하지 않는다.

- 같은 dimension character를 가진 서로 다른 physical bundle/kind의 동일성
- 무한 상태공간 Gibbs 분포의 정규화 가능성이나 beta-positive 농축
- Hodge/cross-product/critical-dimension 특수성이 실제 우주의 차원을 선택함
- compact/warped/large extra dimension의 물리적 존재

전체 pytest는 실행하지 않았다. 변경 범위의 focused test와 문서 policy node가
green이었고, 저장소 규약은 명시적 full/release 요청 없이는 전체 회귀를 금지한다.
hook가 소유하는 고유 basetemp를 사용했으며 repository cache/bytecode residue를
만들지 않았다.

## 관측 자료의 ceiling

PDG 2025와 CMS SUS-23-016의 수치는 특정 모형과 채널의 제약이다. CMS 결과는
13 TeV, 총 137 fb^-1, monophoton+missing-transverse-momentum 검색에서 유의한
편차가 없었고, ADD `n=3--6` benchmark의 `M_D`를 최대 3.2 TeV까지 95% CL로
제외한다. 이는 추가차원의 검출도, 모든 compactification의 배제도 아니다.

