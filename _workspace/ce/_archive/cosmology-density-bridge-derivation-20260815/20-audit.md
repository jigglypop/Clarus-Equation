# CE 우주 밀도 사상 1단계 형식 지위 감사

Status: COMPLETE

Gate: PASS

감사 기준일: 2026-08-15  
감사 범위: `00-contract.md`, `10-sources.md`, `11-math.md`,
`12-routes.md`와 지정된 보조 근거 3개

## 0. 게이트 판정의 정확한 뜻

이 PASS는 **자연에서 \(q_*=\Omega_b\)가 유도되었다는 승인**이 아니다.
계약이 허용한 두 종료 방식 가운데, B1--B3의 좁은 변분·국소 안정성
결과는 닫혔고 B4--B5의 긍정 부모 명제는 완전 반례로 제외되었으며 그
자리에 정확한 no-go와 조건부 존재구성이 남았다는 뜻이다. 따라서 승인
범위는 다음뿐이다.

1. 고정점 식을 정지조건으로 갖는 canonical scalar 작용의 **변분적
   존재구성**.
2. 선언된 \(D>1\), \(0<x\leq1\), \(F^2>0\)에서 두 정지 가지와 작은
   가지의 **국소** 고전 안정성.
3. 확률을 energy-weighted fraction으로 바꿀 때의 covariance 항등식과
   equal-conditional-mean **필요충분 정리**.
4. A1--A7을 독립 공리로 넣은 two-dust 모형에서
   \(f_b^{(m)}=q_*\)가 되는 **조건부 조성 존재구성**.
5. scalar 값만으로 density fraction을 정할 수 없고, 혼합 우주에서
   conserved dust의 \(\Omega_b\)를 상수 \(q_*\)에 계속 묶을 수 없다는
   **no-go 정리**.

활성 관측 예측은 0개다. 현재 \(\Omega_b\)의 자연 유도는 `[미완성]`으로
남으며 이 게이트의 승인 범위에 포함되지 않는다.

입력 상태는 `00-contract.md`, `11-math.md`, `12-routes.md`와 두 보조 연구
artifact가 COMPLETE이고 `10-sources.md`만 관측 인용 없음으로 SKIPPED다.
관측 결론이 없는 이번 수학 게이트에는 비차단이지만, 실제 current나
예측을 승격할 때는 정식 source lane을 다시 열어야 한다.

## 1. B1--B6 최소 주장 원장

아래 24개를 이번 감사의 최소 비자명 단위로 고정했다. “삭제”는 완전
반례가 있는 부모 해석을 활성 결론에서 제외한다는 뜻이다.

| Claim ID | 계약상/현재 주장 | 실제 지위 | 근거와 범위 | 판정 |
|---|---|---|---|---|
| B1-a | \(D>1\), \(0<x\le1\), 차원 원장 | `[정의]` | `00-contract.md:50-57`, `11-math.md:38-67` | 유지 |
| B1-b | 특정 \(S_x,v_D,F,M,C\) 선택 | `[공리: 모델 선택]` | `11-math.md:38-56`; \(C\)도 중력 stress에 영향 | 유지하되 산출로 세지 않음 |
| B1-c | 균일 E-L 정지조건이 고정점 식과 동치 | `[정리: 존재구성]` | `11-math.md:69-102`, `action-route-study.md:127-157` | 승인 |
| B1-d | 고정점 식이 이 potential/action을 유일하게 고름 | 삭제 | 양의 임의 함수 \(w(x)\)가 같은 stationary set을 보존, `11-math.md:91-102` | 제외 |
| B2-a | 작은 근은 선언 영역의 유일한 전역 최소 | `[정리: 선언 영역]` | Hessian·부호·\(x\to0^+\) 경계, `11-math.md:104-137` | 승인 |
| B2-b | \(x=1\)은 한쪽 최대이자 tachyonic 가지 | `[정리]` | `11-math.md:104-137`, `:180-199` | 승인 |
| B2-c | 모든 초기값에서 작은 근이 전역 attractor | `[미완성]` | field space 불변성·벽·UV completion 부재, `11-math.md:138-143` | 활성 주장 아님 |
| B3-a | dimension-four action의 EOM, stress, FLRW 식 | `[정리]/[산출]` | `11-math.md:145-178` | 승인 |
| B3-b | 작은 근의 no-ghost/no-gradient/no-tachyon | `[정리: 국소 고전]` | \(F^2>0\), 고정점 주위 이차작용, `11-math.md:180-204` | 승인 |
| B3-c | constant-\(H>0\) 선형 감쇠 시간척도 | `[산출: 조건부]` | 자유로운 \(m_*/H\), `11-math.md:205-236` | 승인; 보편 attractor로 확대 금지 |
| B3-d | 전역 안정·양자/EFT·radiative 안정성 | `[미완성]` | cutoff, loop, domain completion 부재, `11-math.md:201-204`, `:460-465` | 미승인 |
| B4-a | scalar의 \(x=q_*\)만으로 \(\Omega_b\)가 결정됨 | 삭제 | 같은 \(q_*\)에서 \(C\) 이동으로 \(\Omega_x=0,0.2\), `11-math.md:240-269` | P0 해소: 부모 제외 |
| B4-b | 상수 scalar stress 자체가 baryon dust | 삭제 | 비영 정지 stress는 \(w_x=-1\), dust는 \(w_b=0\), `11-math.md:264-269` | P0 해소: 부모 제외 |
| B4-c | weighted fraction covariance 항등식 | `[정리]` | `symmetry-bridge-attempt.md:38-78` | 승인 |
| B4-d | \(\Omega_E^{(W)}=q\) iff 두 conditional mean energy가 같음 | `[정리: 필요충분]` | `symmetry-bridge-attempt.md:79-113` | 승인 |
| B4-e | product state+label-blind energy이면 equality | `[정리: 충분조건]` | 측도·경계상태까지 factorize, `symmetry-bridge-attempt.md:136-224` | 승인; 동적 중력계에 자동 확대 금지 |
| B4-f | constrained equal-energy two-dust에서 \(f_b^{(m)}=q\) | `[산출: 조건부 존재구성]` | A1--A7 및 boundary constraint, `symmetry-bridge-attempt.md:303-389` | 승인; 자연 유도 아님 |
| B4-g | CE core가 state map, product preparation, current와 freeze-out을 자동 생성 | `[미완성]` | A1--A7 미유도, `symmetry-bridge-attempt.md:532-574` | 미승인 |
| B5-a | conserved fixed-mass baryon dust는 \(\rho_b\propto a^{-3}\) | `[정리]` | `11-math.md:335-343` | 승인 |
| B5-b | flat GR에서 \(d\ln\Omega_b/d\ln a=3w_{\rm tot}\) | `[정리]` | `11-math.md:344-361`, `symmetry-bridge-attempt.md:454-496` | 승인 |
| B5-c | 혼합 우주에서 상수 \(q_*\)가 \(\Omega_b(t)\)를 계속 고정 | 삭제 | radiation/matter/DE 반례와 유일한 source law, `11-math.md:362-402` | P0 해소: 부모 제외 |
| B5-d | \(f_b^{(m)}=q\)이면 \(\Omega_b=q\Omega_m\) | `[정리: 항등식]` | `symmetry-bridge-attempt.md:414-452` | 승인 |
| B5-e | \(f_b^{(m)}=q\)이면 추가 조건 없이 \(\Omega_b=q\) | 삭제 | 정확히 \(\Omega_m=1\) 또는 A8이 더 필요, 같은 근거 | P0 해소: 부모 제외 |
| B6-a | 현재 \(\Omega_b\)의 자연 유도와 독립 예측 | `[미완성]`; `[예측]` 0개 | normalization, \(\Sigma_*\), \(\Omega_m\), blind protocol 부재, `11-math.md:404-423`, `symmetry-bridge-attempt.md:575-612` | 미승인 |

## 2. 완전 반례와 삭제 범위

| ID | 완전 반례 | 삭제하는 부모 범위 | 보존하는 좁은 결과 |
|---|---|---|---|
| P0-R1 | 같은 \(q_*\)에서 \(C\) 이동만으로 \(\Omega_x=0,0.2\), `11-math.md:240-263` | scalar 값만으로 density fraction 결정 | E-L 정지식과 Hessian |
| P0-R2 | 상수 scalar는 \(w=-1\), baryon dust는 \(w=0\), `11-math.md:264-269` | scalar stress 자체가 baryon stress | 별도 current를 둔 조건부 구성 |
| P0-R3 | \(d\ln\Omega_b/d\ln a=3w_{\rm tot}\), `11-math.md:335-402` | 혼합 우주의 지속적 \(q_*=\Omega_b(t)\) | 단일 hypersurface 경계 공리와 no-go |
| P0-R4 | 같은 \(q\)에서 weight 비로 energy fraction이 0--1 변화, `symmetry-bridge-attempt.md:114-134` | 확률의 자동 energy-fraction 동일시 | covariance 및 equal-mean iff 정리 |
| P0-R5 | 정확한 관계가 \(\Omega_b=q\Omega_m\), `symmetry-bridge-attempt.md:414-452` | \(f_b^{(m)}=q\Rightarrow\Omega_b=q\) | 조건부 matter-composition 구성 |

다섯 부모를 활성 범위에서 제거했으므로 열린 P0는 없다.

## 3. 공리·자유도 회계

고정점 core 밖의 scalar-side 독립 자료는 적어도 7개 논리 범주다:
\(D\)의 외부 입력, potential 함수형, \(F\), \(M\), \(C\), 초기자료/basin,
field-space/UV completion. 이 가운데 \(F,M,C\)는 stationary location을
바꾸지 않으면서 질량·relaxation·stress를 바꾼다 (`11-math.md:404-423`).

matter-composition 존재구성에는 별도로 다음 A1--A7이 들어간다.

1. extinction tail event를 국소 baryon species label로 바꾸는 state map.
2. label-blind energy Hamiltonian.
3. product boundary state 또는 정확한 zero covariance.
4. homogeneous stationary-ergodic many-cell preparation.
5. generation clock과 독립적으로 고른 freeze-out \(\Sigma_*\).
6. 두 conserved dust sector와 equal rest/conditional energy.
7. complement가 모든 matter라는 species 해석.

\(\Omega_b=q\)까지 직접 가려면 A8—\(\Omega_m=1\), tuned interaction 또는
critical-density boundary closure—이 더 필요하다
(`symmetry-bridge-attempt.md:532-574`). 따라서 검사된 공리 의존 범주는
scalar 쪽 7개와 bridge 쪽 A1--A8의 8개, 합계 15개다. 이들은 현재 CE
core의 정리에서 나온 산출이 아니다.

## 4. 남은 P1/P2와 허용 문구

### P1 — 승인 범위를 넘으면 다시 차단하는 항목

1. \((0,1]\)이 Lorentzian 시간발전에 대해 불변이라는 전역 증명과
   bounded-below UV/EFT completion이 없다.
2. exact product factorization은 공통 동적 metric과 Hamiltonian constraint가
   상관을 만들 수 있어 우주론에 자동 적용되지 않는다
   (`symmetry-bridge-attempt.md:211-223`).
3. A1--A8, reaction/freeze-out, total charge와 reheating normalization이
   core에서 유도되지 않았다.
4. 후보 다섯 개를 이미 본 target-aware 탐색이므로 현재 equality를
   새로운 blind 예측이라고 부를 수 없다.

이 P1들은 좁은 수학 정리의 참값을 바꾸지 않으므로 PASS를 막지 않지만,
\(\Omega_b\) 자연 유도나 관측 예측을 활성화하는 순간 열린 P0로 승격한다.

### P2 — 문서·검증 라우팅

1. 이론 출처가 `10-sources.md`가 아니라 보조 artifact의 참고문헌에 있다.
   다음 물리 레인에서는 정식 source lane으로 옮겨야 한다.
2. dimensionless 회귀 15개는 공통 checker가 통과했다는 뜻일 뿐 신규
   reacting/projector 식이 registry에 등록됐다는 뜻은 아니다
   (`action-route-study.md:43-58`). 구현 시 신규 식을 별도 등록해야 한다.

허용 문구는 “구성할 수 있다”, “명시한 공리 아래 따른다”, “국소적으로
안정하다”, “필요충분조건은 ...이다”까지다. “자연히 유도된다”,
“바리온 밀도를 예측한다”, “무입력 우주론 산출이다”는 허용하지 않는다.

## 5. 수량 요약

이번 감사에서 고정한 최소 명제는 24개다.

| 분류 | 수 |
|---|---:|
| `[정의]` | 1 |
| `[공리: 모델 선택]`인 주장 단위 | 1 |
| 승인된 `[정리]`/조건부 `[산출]` | 13 |
| `[미완성]` | 4 |
| 완전 반례로 활성 범위에서 삭제 | 5 |
| 활성 `[예측]` | 0 |

별도 의존성 회계에서 드러난 공리 범주는 15개다. 이는 위 주장 단위
분류의 “B1-b 한 행”을 scalar/bridge 세부 자료로 다시 펼쳐 센 값이므로
두 표의 숫자를 더하지 않는다.

## 6. \(\Omega_b\) 자연 유도 재개 조건

다음 run의 첫 목표는 \(\Omega_b=q\)가 아니라

\[
\text{local branching label current}
\Longrightarrow
\nabla_\mu J_b^\mu=0,
\quad
\mathbb E[W\mid E]=\mathbb E[W\mid E^c],
\quad
f_b^{(m)}=q
\]

이어야 한다. 자연 유도 감사를 다시 열려면 최소한 다음이 모두 필요하다.

1. 관측 \(\Omega_b,H_0\)를 사용하지 않고 \(D\), local species label과
   reaction law를 산출하는 Lorentz-covariant microscopic action.
2. conserved baryon current, partner current, EOS, entropy current와
   positive entropy production의 완결된 변분/비평형 구조.
3. label-blind energy와 zero covariance를 정의가 아니라 대칭과 준비
   동역학에서 산출하고, gravity·Standard Model interaction·RG correction
   아래 오차경계를 제시하는 증명.
4. 현재 시각을 보고 고르지 않은 covariant freeze-out hypersurface와
   total charge/reheating entropy normalization.
5. dark-sector 동역학에서 \(\Omega_m\)을 독립 계산한 뒤
   \(\Omega_b=q\Omega_m\)을 읽는 Einstein--Boltzmann 경로.
6. 후보군, priors, 비교량과 kill threshold를 새 독립 자료 공개 전에
   동결한 blind 검증.

어느 단계에서든 \(x:=\Omega_b\), boundary constraint
\(\rho_b=q\rho_{\rm crit}\), tuned \(C,M/F,\kappa,\Sigma_*\)를 다시 넣으면
자연 유도는 재개되지 않고 공리화된 존재구성으로 남는다.

## 7. 최종 승인 범위와 종료 체크

최종적으로 B1--B3은 “선택한 scalar model의 존재 및 국소 안정성”으로만
승인한다. B4--B5는 긍정 density 부모 주장이 아니라 exact no-go,
weighted-event iff 정리와 A1--A7 아래 matter-composition 구성으로만
승인한다. B6의 관측 예측 승격은 승인하지 않는다.

- [x] 모든 비자명 주장을 24개 Claim ID로 분해하고 실제 지위를 판정했다.
- [x] 존재구성, 자연 유도, 물리 사상과 관측 예측을 분리했다.
- [x] 완전 반례 다섯 개의 부모 삭제 범위와 보존 가능한 좁은 정리를 적었다.
- [x] 열린 P0가 없고 PASS 범위가 좁은 수학 결과로 제한됐다.
- [x] 현재 \(\Omega_b\) 자연 유도에 구체적인 재개 조건을 남겼다.
- [x] Status와 Gate 기계 문자열을 기록했다.
