# 구현 기록: 등호·무차원 결함·대안 차원

Status: COMPLETE

## 정본 변경

원장 동결 뒤 독자 문서와 production API를 갱신했다.

1. `docs/검증_원장/참조_무차원_감사_수학.md`
   - E1의 dimension character, typed zero, 단위 공변 등호를 등록했다.
   - E2의 linear/log/SPD Mahalanobis 결함과 Buckingham--Pi의
     positive/nonzero·constant-rank 국소 범위를 고정했다.
   - E3의 affine 결함과 beta 보상, 기준척도 보상, beta-zero 및
     constant/one-level·two-level 예외, three-level 비선형 반례를 고정했다.
   - `typed_equal`은 dimension-character gate이며 physical bundle/kind까지
     판정하지 않는다는 구현 ceiling을 남겼다.
2. `docs/검증_원장/참조_차원_분류_원장.md`
   - 공간 `d`, 시공간 `D`, 형식 차수 `p`, internal/configuration/path,
     compact, spectral/effective dimension을 별도 타입으로 분리했다.
   - Hodge·binomial·cross-product 분류와 일반 `D` power counting을
     정리/조건부 정리로 기록했다.
   - KK·ADD·RS·10D/11D/26D를 모형/일관성 경로로, 추가차원의 물리적
     존재를 미완성으로 기록했다.
   - PDG 2025와 CMS SUS-23-016을 model/channel/luminosity/95% CL 의존
     제약으로 동결했다.
3. `docs/9_등호이전/04c_PreEq_보편스킴.md`
   - metric defect보다 먼저 typed equality와 단위 재척도화 증명을 배치했다.
   - 결함의 정의역, SPD 반례, Pi의 국소 한계 및 finite-beta 정규화
     의존성을 독자가 재현할 수 있게 연결했다.
   - E1--E3의 정의·정리·공리·미완성 지위를 기존 형식표에 추가했다.
4. `docs/2_경로적분과_응용/01_차원의_유일성.md`
   - `Lambda^1`--`Lambda^2` 동수 문제의 유효 domain에서는 `d=3`만
     남기고, 정의역 밖 `d=0` 연장값을 활성 해에서 제거했다.
   - 일반 Hodge `d=2p`, 인접차수 `d=2p+1`, signature/reality,
     `d=2,3,4,5`, cross product와 division algebra의 차이를 보강했다.
   - 중심력, CKM, CE toy bridge의 전제·유도·미완성 조건을 복원했다.
   - 일반 `D` 장론과 5D/10D/11D/26D 경로를 관측 사실과 분리했다.
5. `reality_stone/python/reality_stone/clarus/dimensionless.py`
   - `same_dimensions`, `require_same_dimensions`, `typed_zero`,
     `typed_equal`을 추가했다.
   - linear, positive-domain log, normalized-SPD Mahalanobis 결함을 추가했다.
   - affine 결함 및 기준척도 변화의 beta 보상 함수를 추가했다.
   - NaN/Inf, 산술 overflow, 비SPD covariance를 거부하고, log defect는
     `log1p` 상대차와 extreme-ratio fallback으로 정확한 zero set을 지킨다.
6. `tests/test_dimensionless.py`
   - 단위 불일치, typed zero, 재척도화, log 정의역과 인접-float zero set,
     SPD/indefinite/singular/asymmetric covariance, overflow 및 beta 보상을
     focused regression으로 고정했다.

## 수정 라운드

첫 감사의 P0는 `dim Lambda^1 = dim Lambda^2` 계산에서 형식 차수 정의역 밖의
`d=0`을 해처럼 남긴 오류였다. 유효 domain `d>=2`에서 `d=3`만 남기고
certificate에 domain guard를 추가했다.

두 번째 라운드는 E3의 빠진 퇴화 전제와 수치 구현 경계를 닫았다. 고정 beta의
유일성은 `beta != 0`이고 서로 다른 결함 수준이 있을 때만 성립한다. 이후 finite
입력이 중간 계산에서 Inf가 되는 경우와 큰 인접 float의 log 차가 0으로 반올림되는
경우까지 찾아 production guard 및 회귀 테스트로 잠갔다.

어떤 production 코드도 추가차원의 물리적 존재를 참으로 인코딩하지 않는다.

