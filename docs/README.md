# Clarus Equation 문서 안내

CE 문서는 하나의 수치식이 아니라 서로 다른 이론 층의 모음이다. 먼저
정의·정리·공리·산출을 읽고, 그 뒤 경험식과 관측 비교를 읽는다.

## 1. 형식 출처

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

## 2. 한눈에 보는 이론 층

| 층 | 대상 | 현재 닫힌 부분 |
|---|---|---|
| 순수 수학 | 함수방정식·외대수·고정점 | 자족 증명 |
| 공변 장론 | \(Z_2\) singlet-portal EFT | 대칭·안정성·운동방정식·stress 보존 |
| Euclidean 적분 | scalar--Higgs bosonic 유한 격자 cutoff | partition function·모멘트의 존재 |
| 양자-고전 bridge | reduced dynamics | 조건 목록만 있으며 미시 유도는 열림 |
| 확률 코어 | 다형 Poisson offspring | 최소 소멸 고정점·균일 축약 |
| 우주론 | GR+canonical scalar·dust+\(\Lambda\)·inflation branch | 배경식·no-go·조건부 해 |
| flavor | Yukawa·Majorana·Koide toy potential | 존재구성·기하 동치 |
| 관측 | CKM·PMNS·\(A_s\)·\(H_0\) readout | 경험식 또는 재현 자료 대기 |

## 3. 핵심 수학 정리

[핵심 정리 증명 원장](참조/핵심_정리_증명.md)은 다음을 한곳에 모은다.

1. 연속 곱함수의 지수형
2. 외대수 성분 수와 \(d=0,3\)
3. \(s(1-s)\) 범위
4. Poisson 최소 소멸 고정점과 Lambert \(W\)
5. 연속 곱적 readout \(I(x)=x^c\)
6. Koide 조건과 민주축 각도
7. Gleason 정리의 정확한 가정
8. Euler 항등식
9. Hodge \(2\)-form/\(1\)-form 폐쇄
10. \(Z_2\) portal quartic 안정성
11. 유한 격자 Euclidean 측도
12. diffeomorphism Noether stress 보존
13. Euclidean 수축 semigroup
14. 다형 Poisson 공통 행합 축약
15. canonical scalar의 \(w\geq-1\)
16. CKM·PMNS 질량행렬 존재구성
17. 상수 진공항의 \(w=-1\)
18. logistic 흐름의 전역 안정성
19. Starobinsky형 slow-roll 근사
20. mixture-affine 이분할 complement kernel
21. irreducible 다형 Poisson의 Perron 임계값
22. 유한차원 Euclidean Laplace 근사
23. 고전 secretary 문제의 \(1/e\) 극한
24. 평탄 dust+\(\Lambda\) 우주의 나이
25. 빠른 quadratic scalar의 dust 극한

## 4. 공변 장론 branch

[공리계](axium.md)는 다음 작용을 하나의 일관된 EFT로 채택한다.

\[
S_{\rm EFT}=\int d^4x\sqrt{-g}\left[
\frac12(M_{\rm Pl}^2-\xi\phi^2)R-\Lambda_0
-\frac12(\nabla\phi)^2+\mathcal L_{\rm SM}^{\rm kin+gauge+Yuk}
-V(H,\phi)\right],
\]

\[
V(H,\phi)=V_H(H)+\frac12m_\phi^2\phi^2
+\frac{\lambda_\phi}{4}\phi^4
+\frac{\lambda_{H\phi}}2\phi^2H^\dagger H.
\]

\(Z_2:\phi\mapsto-\phi\)와

\[
M_{\rm Pl}^2-\xi\phi^2>0,\qquad
\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}
\]

를 적용 정의역으로 둔다.

이 모형은 특정 \(m_\phi\)나 portal coupling을 예측하지 않지만 다음은
정확히 계산한다.

- \(\phi\) 운동방정식과 tree-level 곡률질량
- \(Z_2\)-보존 진공의 무혼합과 안정한 odd 입자
- on-shell 총 stress 보존
- \(\xi=0\), 고정 배경 scalar--Higgs bosonic 유한 Euclidean
  cutoff의 측도 존재
- \(\xi=0\) 최소 결합 canonical scalar FLRW의 방정식상태 경계

## 5. 확률 코어

### 5.1 다형 모형

type \(i\) 개체가 type \(j\) 자손을 독립
\(\operatorname{Poisson}(A_{ij})\)로 만든다고 두면 최소 소멸확률은

\[
q_i=\exp\!\left[-\sum_jA_{ij}(1-q_j)\right]
\]

의 최소 고정점이다.

### 5.2 균일 CE toy sector

\[
A=dI+\delta B,\qquad
B\geq0,\quad B\boldsymbol1=\boldsymbol1
\]

를 택하면

\[
D_{\rm eff}=d+\delta,\qquad
\boldsymbol q=q_{\rm ext}\boldsymbol1
\]

로 정확히 닫힌다. 이는 내부적으로 완결된 stochastic toy model이다.

양자 진폭에서 \(A\geq0\)를 유도하려면 완전양성 reduced dynamics,
Markov 근사와 genealogy가 필요하다. 이 bridge는 열려 있다.

## 6. 물리 사상

다음 화살표는 서로 다른 지위다.

\[
\text{SM neutral mass matrix}
\longrightarrow \delta
\]

는 지정 기저의 **[산출]**이다.

\[
\delta\longrightarrow A=dI+\delta B
\]

는 stochastic toy family의 **[공리]**다.

\[
q_{\rm ext}\longrightarrow\Omega_b
\]

는 확률–에너지 **[공리]**다.

\[
\text{quantum amplitude}\longrightarrow A_{ij}
\]

는 **[미완성]** bridge다.

이 네 단계를 하나의 “유도”라고 부르지 않는다.

## 7. 우주론

[우주론 모형군](3_상수/7_우주론.md)은 다음 branch를 구분한다.

| branch | 보존되는 결과 |
|---|---|
| \(Z_2\) stable scalar | 안정한 암흑물질 후보 |
| \(m\gg H\) 최소 결합 quadratic coherent scalar | adiabatic leading order에서 평균 \(w=0\), \(\rho\propto a^{-3}\) |
| 최소 결합 canonical scalar | \(w\geq-1\) no-go |
| 상수 potential | \(w=-1\) |
| flat dust+\(\Lambda\) | 정확한 \(H_0t_0\) 적분 |
| Starobinsky형 potential | \(V_0>0\), 정준 단일장 지배의 leading slow-roll \(n_s,r,\alpha_s^{\rm(run)}\) |
| flat GR bounce | NEC 위반 필요 |

원시 진폭 projected readout은 **[경험식]**이다. 반면
Mukhanov--Sasaki 방정식은 지정한 단일장 perturbation theory의 정확한
선형식이다.

## 8. flavor와 질량

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

### 9.1 먼저 읽기

1. [코어 독자 가이드](코어_독자_가이드.md)
2. [공리계와 기호](axium.md)
3. [경로적분과 조건부 장론](경로적분.md)
4. [상수·매개변수 원장](상수.md)
5. [핵심 정리 증명](참조/핵심_정리_증명.md)

### 9.2 강의

- [연역 구조](1_강의/A_연역적_유도.md)
- [귀납 구조](1_강의/B_귀납적_유도.md)
- [다섯 상수의 문법](1_강의/C_다섯_상수.md)

### 9.3 상수와 물리 branch

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

- [검증 규약](2_경로적분과_응용/00_검증_규약.md)
- [형식 구조 원장](2_경로적분과_응용/전체_진리값_감사.md)
- [등호 이전](9_등호이전/README.md)
- [형식적 수학 모델](참조/형식적_수학_모델과_증명.md)
- [이론물리 보존 원장](참조/이론물리_보존_원장.md)

## 10. 재현

핵심 수치와 정책 검사는 다음 순서로 실행한다.

    .\.venv\Scripts\python.exe -m pytest tests\test_bootstrap_solver.py -q
    .\.venv\Scripts\python.exe -m pytest tests\test_dimensionless.py -q
    .\.venv\Scripts\python.exe tests\run_validation.py
    .\.venv\Scripts\python.exe -m pytest tests\test_canonical_document_policy.py -q

코드가 방정식을 높은 정밀도로 푸는 것과 그 변수를 자연의 물리량으로
식별하는 것은 다른 검증이다.

## 11. 현재 결론

CE에는 순수 수학, 공변 EFT, scalar--Higgs bosonic 유한 cutoff 측도,
다형 분지과정, 조건부 GR·flavor 모형으로 살릴 수 있는 구조가 있다.
가장 큰 열린 문제는

\[
\text{quantum field theory}
\longrightarrow
\text{positive stochastic process}
\longrightarrow
\text{cosmological readout}
\]

의 두 bridge다. 현재 정본에 사전 고정된 독립 관측 예측은 없다.
