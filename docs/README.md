# Clarus Equation 문서 안내

이 문서는 Clarus Equation의 통합 논문, 유도·강의, 검증 원장, 구현 문서를 처음 독자가 목적에 맞게 찾도록 안내한다. 독자는 문서의 수식·관측 비교·코드 검증이 서로 다른 지위를 가진다는 점에서 출발하며, 한 경로의 통과를 다른 경로의 증명으로 읽지 않는다.

이론의 서사를 알고 싶으면 통합 논문과 강의를, 수학 전제를 재현하려면 유도와 형식수학을, 주장·상수·판본을 감사하려면 검증 원장을, 실행 contract를 확인하려면 구현/검증 문서를 읽는다. 아래 지도는 각 진입점과 의존성을 설명하며, 원장은 읽기 전용 사실·지위 입력으로 사용한다.

CE 문서는 하나의 수치식이 아니라 서로 다른 이론 층의 모음이다. 먼저
정의·정리·공리·산출을 읽고, 그 뒤 경험식과 관측 비교를 읽는다.

## 1. 형식 출처

각 문장은 정의·정리·공리·산출·경험식·미완성·예측 중 어디에서 오는지 구분해야 한다. 형식 출처 표지는 외부 데이터나 코드 실행을 수학적 증명으로 바꾸지 않는다.

| 표지 | 의미 |
|---|---|
| **[정의]** | 기호·대상·정의역 |
| **[정리]** | 적힌 전제에서 증명된 명제 |
| **[공리]** | 모형·가지·경계조건·물리 사상의 선택 |
| **[산출]** | 정리와 공리를 대입한 직접 결과 |
| **[경험식]** | 자료·보정·유효계수를 사용하는 관계 |
| **[미완성]** | 작용·사상·증명 또는 자료가 더 필요한 항목 |
| **[예측]** | 입력과 판정 기준을 미리 고정한 독립 관측량 |

관측 적합성은 수학적 정리의 진위를 결정하지 않는다. 완전한 반례가 있는
부모 명제는 활성 문서에 보존하지 않지만, 전제를 좁혀 참이 되는 정리,
조건부 toy/EFT와 정확한 no-go는 보존한다.

우주론 수치는 먼저 [우주론 판본·주장 원장](검증_원장/상수_우주론_원장.md)에서
정밀 코어, 과거 재현, 런타임 호환과 관측 외부 입력의 역할을 확인한다.

## 2. 한눈에 보는 이론 층

이론 층은 동기 서사, 조건부 수학, 채택 공리, 관측·구현 입력을 분리해 읽기 위한 구조다. 한 층의 결과가 다른 층의 미완성 bridge를 자동으로 닫지 않는다.

| 층 | 대상 | 현재 닫힌 부분 |
|---|---|---|
| 순수 수학 | 함수방정식·외대수·고정점 | 자족 증명 |
| 공변 장론 | $Z_2$ singlet-portal EFT | 대칭·안정성·운동방정식·stress 보존 |
| Euclidean 적분 | scalar--Higgs bosonic 유한 격자 cutoff | partition function·모멘트의 존재 |
| 양자-고전 bridge | reduced dynamics | 조건 목록만 있으며 미시 유도는 열림 |
| 확률 코어 | 다형 Poisson offspring | 최소 소멸 고정점·균일 축약 |
| 우주론 | GR+canonical scalar·dust+$\Lambda$·inflation branch | 배경식·no-go·조건부 해 |
| flavor | Yukawa·Majorana·Koide toy potential | 존재구성·기하 동치 |
| 관측 | CKM·PMNS·$A_s$·$H_0$ readout | 경험식 또는 재현 자료 대기 |

## 3. 핵심 수학 정리

핵심 정리는 명시한 정의역·정규성·경계조건 아래에서만 성립한다. 통합 논문은 결과의 역할을 설명하고, 유도·증명 문서는 재현 가능한 전제를 제공한다.

[핵심 정리 증명 원장](검증_원장/참조_핵심_정리_증명.md)은 다음을 한곳에 모은다.

1. 연속 곱함수의 지수형
2. 외대수 성분 수와 $d=0,3$
3. $s(1-s)$ 범위
4. Poisson 최소 소멸 고정점과 Lambert $W$
5. 연속 곱적 readout $I(x)=x^c$
6. Koide 조건과 민주축 각도
7. Gleason 정리의 정확한 가정
8. Euler 항등식
9. Hodge $2$-form/$1$-form 폐쇄
10. $Z_2$ portal quartic 안정성
11. 유한 격자 Euclidean 측도
12. diffeomorphism Noether stress 보존
13. Euclidean 수축 semigroup
14. 다형 Poisson 공통 행합 축약
15. canonical scalar의 $w\geq-1$
16. CKM·PMNS 질량행렬 존재구성
17. 상수 진공항의 $w=-1$
18. logistic 흐름의 전역 안정성
19. Starobinsky형 slow-roll 근사
20. mixture-affine 이분할 complement kernel
21. irreducible 다형 Poisson의 Perron 임계값
22. 유한차원 Euclidean Laplace 근사
23. 고전 secretary 문제의 $1/e$ 극한
24. 평탄 dust+$\Lambda$ 우주의 나이
25. 빠른 quadratic scalar의 dust 극한

## 4. 공변 장론 branch

공변 장론 branch는 action·대칭·field 정의를 다루는 조건부 수학과 모델 선택을 포함한다. 물리적 실재나 관측 적합은 외부 입력·비교 절을 함께 읽어야 한다.

[공리계](axium.md)는 다음 작용을 하나의 일관된 EFT로 채택한다.

$$
S_{\rm EFT}=\int d^4x\sqrt{-g}\left[
\frac12(M_{\rm Pl}^2-\xi\phi^2)R-\Lambda_0
-\frac12(\nabla\phi)^2+\mathcal L_{\rm SM}^{\rm kin+gauge+Yuk}
-V(H,\phi)\right],
$$

$$
V(H,\phi)=V_H(H)+\frac12m_\phi^2\phi^2
+\frac{\lambda_\phi}{4}\phi^4
+\frac{\lambda_{H\phi}}2\phi^2H^\dagger H.
$$

$Z_2:\phi\mapsto-\phi$와

$$
M_{\rm Pl}^2-\xi\phi^2>0,\qquad
\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}
$$

를 적용 정의역으로 둔다.

이 모형은 특정 $m_\phi$나 portal coupling을 예측하지 않지만 다음은
정확히 계산한다.

- $\phi$ 운동방정식과 tree-level 곡률질량
- $Z_2$-보존 진공의 무혼합과 안정한 odd 입자
- on-shell 총 stress 보존
- $\xi=0$, 고정 배경 scalar--Higgs bosonic 유한 Euclidean
  cutoff의 측도 존재
- $\xi=0$ 최소 결합 canonical scalar FLRW의 방정식상태 경계

## 5. 확률 코어

확률 코어는 후보분포·정규화·농축을 수학적으로 정의한다. 이를 물리적 확률·측정·agent belief로 읽는 해석에는 별도의 bridge와 반례 조건이 필요하다.

### 5.1 다형 모형

다형 모형은 일반 구조와 가정의 범위를 드러내는 형식 도구다. 각 branch의 parameter·prior·normalization은 문서에 명시된 입력을 따른다.

type $i$ 개체가 type $j$ 자손을 독립
$\operatorname{Poisson}(A_{ij})$로 만든다고 두면 최소 소멸확률은

$$
q_i=\exp\!\left[-\sum_jA_{ij}(1-q_j)\right]
$$

의 최소 고정점이다.

### 5.2 균일 CE toy sector

균일 toy sector는 계산 가능한 축약 사례이며 전체 CE 물리 모형의 대체물이 아니다. toy 결과를 관측 예측으로 승격하려면 추가 closure gate가 필요하다.

$$
A=dI+\delta B,\qquad
B\geq0,\quad B\boldsymbol1=\boldsymbol1
$$

를 택하면

$$
D_{\rm eff}=d+\delta,\qquad
\boldsymbol q=q_{\rm ext}\boldsymbol1
$$

로 정확히 닫힌다. 이는 내부적으로 완결된 stochastic toy model이다.

양자 진폭에서 $A\geq0$를 유도하려면 완전양성 reduced dynamics,
Markov 근사와 genealogy가 필요하다. 이 bridge는 열려 있다.

## 6. 물리 사상

물리 사상은 수학 객체를 현상 언어로 연결하는 채택 공리 또는 미완성 bridge다. 이 절의 연결은 정의·정리와 구별해 관련 검증 원장과 외부 비교를 함께 확인한다.

다음 화살표는 서로 다른 지위다.

$$
\text{SM neutral mass matrix}
\longrightarrow \delta
$$

는 지정 기저의 **[산출]**이다.

$$
\delta\longrightarrow A=dI+\delta B
$$

는 stochastic toy family의 **[공리]**다.

$$
q_{\rm ext}\longrightarrow\Omega_b
$$

는 `C-B-LEGACY-01`의 과거 확률–에너지 **[공리]**다. conditioned
$Dq$ 조성과 전이 면 $1/D$를 결합하는 새 목표는
[우주론 판본·주장 원장](검증_원장/상수_우주론_원장.md)에 별도로 둔다.

$$
\text{quantum amplitude}\longrightarrow A_{ij}
$$

는 **[미완성]** bridge다.

이 네 단계를 하나의 “유도”라고 부르지 않는다.

## 7. 우주론

우주론 문서는 background cosmology·데이터·nuisance·covariance라는 외부 입력을 사용한다. 수치 근접은 조건부 forward comparison이며 이론의 형식 증명이 아니다.

[우주론 모형군](3_상수/7_우주론.md)은 다음 branch를 구분한다.

| branch | 보존되는 결과 |
|---|---|
| $Z_2$ stable scalar | 안정한 암흑물질 후보 |
| $m\gg H$ 최소 결합 quadratic coherent scalar | adiabatic leading order에서 평균 $w=0$, $\rho\propto a^{-3}$ |
| 최소 결합 canonical scalar | $w\geq-1$ no-go |
| 상수 potential | $w=-1$ |
| flat dust+$\Lambda$ | 정확한 $H_0t_0$ 적분 |
| Starobinsky형 potential | $V_0>0$, 정준 단일장 지배의 leading slow-roll $n_s,r,\alpha_s^{\rm(run)}$ |
| flat GR bounce | NEC 위반 필요 |

원시 진폭 projected readout은 **[경험식]**이다. 반면
Mukhanov--Sasaki 방정식은 지정한 단일장 perturbation theory의 정확한
선형식이다.

## 8. flavor와 질량

flavor와 질량 branch는 대칭·parameterization·측정 비교의 의존성을 가진다. 관측값과의 일치 여부는 source role과 fit 절차를 보존한 채 읽어야 한다.

[입자물리](3_상수/4_입자물리.md)는 세 CKM 경험값을 정확히 unitary인
표준 각 매개화로 완성하고 Yukawa 행렬에 embedding한다.

[PMNS](3_상수/5_PMNS.md)는 경험각을 unitary PMNS 행렬과 대칭 Majorana
질량행렬로 완성하고 차원 5 Weinberg operator에 embedding한다.

[질량](3_상수/6_질량.md)은 두 toy EFT를 보존한다.

- flavor spurion으로 정수 지수 계층 구현
- nonnegative flavon potential로 Koide cone 구현

이 구성들은 이론적으로 일관되지만 charge, 계수와 진공 방향을 입력하므로
아직 매개변수 예측은 아니다.

## 9. 문서 지도

문서 지도는 목적별 진입점을 연결한다. 통합 논문은 서사와 기여를, 강의·유도는 정의와 증명을, 검증 원장은 상태·provenance를, 구현 문서는 실행 범위를 담당한다.

최신 0차원 측정·접힘·암흑부문 연구는 큰 단일 보고서 대신 [최신 연구 조립 목차](6_최신_연구/00_읽기_지도.md)에서 장별로 읽는다. 작업공간 보고서는 과거 근거이고 이 목차와 연결된 `docs/` 문서가 독자용 최신본이다.

### 9.1 먼저 읽기

처음 읽기는 전체 서사와 형식 지위 규약을 잡는 경로다. 세부 수식은 뒤 문서로 미루되, 공리와 미완성 다리를 건너뛰지 않는다.

1. [최신 연구 조립 목차: 측정, 접힘과 암흑 표현](6_최신_연구/00_읽기_지도.md)
2. [CE 통합 논문: 선택, 접힘, readout과 조건부 응용의 현재 지형](CE_통합_논문.md)
3. [선택과 접힘: 핵심 유도 사슬](5_유도/00_선택과_접힘.md)
4. [극장 개장과 서로 다른 좌석 무게의 에너지 장부](5_유도/08_극장과_좌석_에너지장부.md)
5. [양자 극장 개장: 다종 모드와 매끈한 질량 변화](5_유도/09_양자_극장_개장/00_읽기_순서.md)
6. [코어 독자 가이드](코어_독자_가이드.md)
7. [공리계와 기호](axium.md)
8. [경로적분과 조건부 장론](경로적분.md)
9. [상수·매개변수 원장](상수.md)
10. [핵심 정리 증명](검증_원장/참조_핵심_정리_증명.md)

### 9.2 강의

강의는 처음 독자가 기호·정의·유도를 따라가도록 쓴 논문형 문서다. 원장 표의 판정값은 강의의 설명을 대체하지 않는다.

- [연역 구조](1_강의/A_연역적_유도.md)
- [귀납 구조](1_강의/B_귀납적_유도.md)
- [다섯 상수의 문법](1_강의/C_다섯_상수.md)

### 9.3 상수와 물리 branch

상수와 물리 branch는 외부 입력·경험식·산출을 분리해 읽는 경로다. source role과 비교 절차를 확인하지 않은 수치 인용은 피한다.

- [격자 기본량](3_상수/1_격자기본량.md)
- [혼합 매개변수](3_상수/2_혼합매개변수.md)
- [부트스트랩](3_상수/3_부트스트랩.md)
- [입자물리](3_상수/4_입자물리.md)
- [PMNS](3_상수/5_PMNS.md)
- [질량](3_상수/6_질량.md)
- [우주론](3_상수/7_우주론.md)
- [차원 스케일링](3_상수/8_차원스케일링.md)
- [우주론 수식의 의미](3_상수/9_우주론_수식_의미와_후보.md)

### 9.4 논문과 형식수학

통합 논문은 핵심 기여와 의존관계를, 형식수학은 정리의 가정·증명을 제공한다. 구현 검증과 과학적 지위 판정은 각각 별도 문서에서 확인한다.

- [뇌 검증기준](검증_원장/뇌_검증기준.md)
- [검증 규약](검증_원장/경로적분_검증_규약.md)
- [형식 구조 원장](검증_원장/경로적분_전체_진리값_감사.md)
- [등호 이전](9_등호이전/README.md)
- [형식적 수학 모델](참조/형식적_수학_모델과_증명.md)
- [이론물리 보존 원장](검증_원장/참조_이론물리_보존_원장.md)

## 10. 재현

재현 경로는 코드·fixture·입력 snapshot·검증 명령의 contract를 따른다. 기계적 성공은 문서 링크와 수식 무결성을 확인하지만 물리적 참을 판정하지 않는다.

핵심 수치와 정책 검사는 다음 순서로 실행한다.

    .codex\hooks\python.cmd doctor
    .codex\hooks\python.cmd source
    .codex\hooks\python.cmd pytest tests\test_ckm_vcb_nlo_gate.py -q

코드가 방정식을 높은 정밀도로 푸는 것과 그 변수를 자연의 물리량으로
식별하는 것은 다른 검증이다.

## 11. 현재 결론

현재 결론은 닫힌 수학, 채택 공리, 외부 비교, 미완성 bridge의 교집합이 아니라 분리된 상태다. 독자는 검증 원장의 최신 지위와 관련 반례 조건을 함께 확인해야 한다.

CE에는 순수 수학, 공변 EFT, scalar--Higgs bosonic 유한 cutoff 측도,
다형 분지과정, 조건부 GR·flavor 모형으로 살릴 수 있는 구조가 있다.
가장 큰 열린 문제는

$$
\text{quantum field theory}
\longrightarrow
\text{positive stochastic process}
\longrightarrow
\text{cosmological readout}
$$

의 두 bridge다. 현재 정본에 사전 고정된 독립 관측 예측은 없다.
