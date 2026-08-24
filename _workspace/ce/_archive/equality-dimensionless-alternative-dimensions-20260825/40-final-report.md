# 최종 보고: 등호의 무차원 수학과 가능한 차원들의 위치

Status: COMPLETE

## 결론

등호의 수학은 먼저 “두 수치가 같은가”가 아니라 “두 항이 같은 변환 타입에
있는가”를 묻는다. 기본 단위 재척도화 군의 character가 다르면 등호는 단위 선택에
따라 참거짓이 바뀐다. 영도 맨 숫자 0이 아니라 같은 표적의 typed zero여야 한다.
같은 character는 dimension gate를 닫지만 energy와 torque 같은 semantic kind까지
동일하게 만들지는 않는다.

이 gate 뒤에서만 다음 무차원 결함들이 정당화된다.

$$
\delta_{\rm lin}=\frac{|F-G|}{S},
\qquad
\delta_{\log}=\left|\log\frac FG\right|,
\qquad
\delta_\Sigma=r^{\mathsf T}\Sigma^{-1}r.
$$

linear 결함에는 같은 차원의 양의 유한 기준척도, log 결함에는 양의 유한 입력,
Mahalanobis 결함에는 SPD covariance가 필요하다. Buckingham--Pi의 매끄러운 불변량
분류도 positive/nonzero constant-rank 국소 층에서만 닫힌다.

finite-beta PreEq는 영점집합만의 이론이 아니다. `delta'=a+c delta`, `c>0`이면
`beta'=beta/c`가 정규화 분포를 보존하고, 기준척도 `S'=kS`에는 `beta'=k beta`가
필요하다. 그러나 beta zero나 one-level support는 퇴화 예외이고, two-level support는
우연한 보상이 가능하다. 세 수준 `(0,1/2,2)`에 제곱 변환을 적용하면 한 beta로 모든
확률비를 맞출 수 없다. 따라서 defect, scale, beta, base measure는 추가 구조다.

## 다른 차원이 가능한 곳

| 위치 | 특수 차원 | 정확한 뜻 |
|---|---|---|
| Hodge 같은 차수 | `d=2p` | `Lambda^p`가 같은 차수로 돌아오는 조건; signature가 real self-duality를 추가 제한 |
| Hodge 인접 차수 | `d=2p+1` | `Lambda^p`와 `Lambda^(p+1)`의 대응; `p=1`이면 `d=3` |
| binary normed cross product | `0,1,3,7` | 대수적 존재 분류, 우주론적 차원 선택 아님 |
| scalar/Yang--Mills power counting | `D=4`, `D=6` 등 | marginality/engineering dimension 분류 |
| 중력 | `D=2,3,4`의 서로 다른 경계 | EH 위상성, 국소 편광 0/2 등의 조건; UV 완성 주장 아님 |
| compact/warped 모형 | 5D, `4+n`D | KK·ADD·RS의 조건부 모형 경로 |
| string/M-theory | 10D, 11D, 26D | 지정한 이론의 일관성 또는 저에너지 limit |

`dim Lambda^1 = dim Lambda^2`는 비자명한 형식 차수 domain `d>=2`에서 `d=3`만
허용한다. 정의역 밖 다항식 연장의 `d=0`은 형식 해나 물리 상태가 아니다. 반대로
Hodge, cross product, power counting, string critical dimension 중 어느 것도
관측된 시공간 차원을 단독으로 선택하지 않는다.

CE의 활성 branch는 계속 `d=3,D=4`다. `D_eff`는 무차원 control readout이며
spectral/Hausdorff/compact/spacetime dimension이 아니다. internal fiber,
configuration/path-space, compact, spectral/effective dimension도 각각 별도 타입이다.

## 관측 상태와 남은 다리

PDG 2025와 CMS 최종 결과에서 검토한 추가차원 검색은 제약만 보고하며 확인 신호를
보고하지 않는다. 이 문장의 ceiling은 “검토한 모형·채널에서 유의한 편차가 없다”다.
모든 추가차원의 부재라는 수학적 no-go가 아니다.

물리적 존재 주장으로 넘어가려면 geometry/topology, compactification 또는 warp
parameter, field localization, observable, likelihood/data set, confidence level,
그리고 4D matched control을 고정해야 한다. 이 operational bridge는 현재
`[미완성]`이다.

최종 형식 판정은 두 revision 뒤 `Gate: PASS`다. 이 판정은 좁혀진 수학·구현·출처
범위의 완결성이지, 추가차원이 발견됐다는 판정이 아니다.

