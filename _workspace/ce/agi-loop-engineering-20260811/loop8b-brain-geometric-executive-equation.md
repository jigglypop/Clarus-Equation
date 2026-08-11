# Loop 8B — Brain-derived geometric executive dynamics

## 0. 판정과 범위

**[미완성: 구현 전 후보]** 이 문서는 Loop 8의 일반 POMDP식을 대체하는
전전두엽(PFC)–내측배측 시상(MD)–기저핵/시상하핵(BG/STN)–해마 회로의
연속 동역학 가설이다. 뇌와 우주가 같은 물리계라고 주장하지 않는다. CE
우주론에서 가져오는 것은 특정 성분비가 아니라 **기하, 잔차 회계, 결합
안정성**이라는 수학적 조직 원리뿐이다.

목표는 분기문을 늘리는 것이 아니라 네 기능을 한 동역학에서 분리해 측정하는
것이다: PFC 재귀회로는 기억을 유지하고, MD는 PFC 유효결합을 바꾸며,
BG/STN은 행동 확정 경계를 조절하고, 해마–PFC replay는 아직 흡수되지 않은
예측잔차를 다시 주입한다.

## 1. 생물학적 제약

- **[경험식]** PFC의 문맥 선택과 증거 적분은 재귀적 집단동역학 안에서 함께
  나타난다. Mante et al. (2013), <https://www.nature.com/articles/nature12742>
- **[경험식]** PFC의 비선형 mixed selectivity가 복잡한 과제의 고차원 표현을
  지지한다. Rigotti et al. (2013), <https://www.nature.com/articles/nature12160>
- **[경험식]** MD 시상은 규칙을 단순 전달하기보다 국소 PFC 연결을 증폭하고
  관련 표현을 유지한다. Schmitt et al. (2017),
  <https://www.nature.com/articles/nature22073>; Rikhye et al. (2018),
  <https://www.nature.com/articles/s41593-018-0269-z>
- **[경험식]** 전두피질의 지속활동은 끌개 끝점으로 수렴하며, 교란이 다른
  끌개로 넘기면 오류가 뒤따른다. Inagaki et al. (2019),
  <https://www.nature.com/articles/s41586-019-0919-7>
- **[경험식]** PFC 작업기억 오차는 확산하는 bump attractor의 예측과 맞는다.
  Wimmer et al. (2014), <https://www.nature.com/articles/nn.3645>
- **[경험식]** 인간 피질 활동은 피질 형상의 Laplace–Beltrami 고유모드로 잘
  분해된다. 이는 기하 기반 표현기저를 지지하지만 열확산 자체를 입증하지는
  않는다. Pang et al. (2023),
  <https://www.nature.com/articles/s41586-023-06098-1>
- **[경험식]** 갈등 상황의 mPFC theta–STN 상호작용은 결정 임계값 상승과
  연관된다. Cavanagh et al. (2011),
  <https://pubmed.ncbi.nlm.nih.gov/21946325/>
- **[경험식]** 규칙 전환 중 mPFC replay는 수행과 상관한다. Kaefer et al.
  (2020), <https://pubmed.ncbi.nlm.nih.gov/32032512/>

**[미완성]** MD 신호를 metric 자체로 동일시하는 것, replay를 아래 CE
잔차장과 동일시하는 것, 하나의 에너지가 모든 PFC 과제를 설명한다는 것은
아직 모델 가설이다.

**[삭제]** CE 우주론 고정점의 4.87/26.2/68.9 비율을 뉴런 활성·구조·배경
비율로 옮기지 않는다. 신경생물학적 유도가 없다.

## 2. 무차원 상태

기준 시간 \(\tau_0\), 기준 발화율 \(r_0\), 기준 입력 \(I_0\)를 잡고

\[
\hat t=t/\tau_0,\qquad z=r/r_0,\qquad \hat o=o/I_0
\]

로 정규화한다. \(z\in\mathcal M\)은 신경집단 상태, \(s\)는 느린 기억흔적,
\(\theta\in\Delta^{K-1}\)는 MD 조절좌표, \(\varphi\)는 미해결 예측잔차,
\(y\)는 행동 축적기다. 모두 무차원이다. \(g_\theta\)는 \(\mathcal M\) 위
metric이며
\(L_{g_\theta}:=-\operatorname{div}_{g_\theta}\nabla_{g_\theta}\succeq0\)로
부호를 고정한다.

## 3. 결합식

### 3.1 PFC: 끌개 drift + 기하 확산

**[공리: 모델 선택]**

\[
\boxed{
d z=-M_z\nabla_{g_\theta}\mathcal E\,d\hat t
+\sqrt{2D_z(\theta)}\,dW_{g_\theta}}
\tag{B1}
\]

\[
\boxed{
\mathcal E
=V_{\rm att}(z;s,g)
+\frac{\alpha}{2}\langle z,L_{g_\theta}z\rangle
-\langle B\Phi(\hat o,g),z\rangle
-\langle R\varphi,z\rangle}
\tag{B2}
\]

\(M_z\succ0\), \(D_z\succeq0\)다. \(\Phi\)는 감각·목표·문맥의 비선형
mixed-selective 특징, \(V_{\rm att}\)는 기억 우물이다. LBO는 거친 공간모드를
누르는 정규화이고, 확산텐서는 방향별 노이즈와 탐색이다. 둘 다 기억 저장소가
아니다.

### 3.2 MD 시상: 연결기하의 연속 변조

**[공리: 모델 선택]**

\[
g_\theta^{-1}=g_0^{-1}+\sum_{k=1}^{K}\theta_kG_k,
\qquad
\tau_\theta\dot\theta
=\Pi_{T_\theta\Delta}
\left[-\theta+\operatorname{softmax}(K_z z+K_g g+K_e\varepsilon)\right].
\tag{B3}
\]

\(G_k\succeq0\)이고 \(\Pi_{T_\theta\Delta}\)는 simplex 접공간 투영이다.
규칙별 `if` 대신 \(\theta\)가 PFC 모드 사이 거리와 유효결합을 계속 바꾼다.

### 3.3 느린 기억과 CE 잔차

**[공리: 모델 선택]**

\[
\tau_s\dot s=-s+H_s(z),
\qquad
\tau_\varphi\dot\varphi=-\varphi+C_\varepsilon\varepsilon-C_z\dot z.
\tag{B4}
\]

첫 식은 발화가 약해져도 남는 느린 흔적 후보이다. 두 번째 식은 현재 PFC
갱신으로 설명되지 않은 오차만 남기며, \(-C_z\dot z\)는 흡수된 변화를
이중계상하지 않게 한다. replay는 별도 분기가 아니라 (B2)의
\(-\langle R\varphi,z\rangle\)를 통해 연속적으로 되먹임된다.

### 3.4 BG/STN: 행동 확정 경계

**[공리: 모델 선택]**

\[
d y_a=\mu_a(z,s,g)\,d\hat t+\sqrt{2D_a}\,dB_a,
\qquad
\tau_b\dot b=b_0+k_c\mathcal C(y)-b.
\tag{B5}
\]

\(\mathcal C(y)\)는 행동갈등, \(b>0\)는 흡수경계다. 첫 \(y_a=b\) 도달에서
행동이 확정된다. 갈등이 크면 STN 대응항 \(b\)가 올라가 더 많은 증거를
요구한다.

## 4. 디퓨전 무망 정리

**[정리]** \(\mathcal M\)이 compact, connected이고 경계가 없으며
\(D>0\)라 하자. 끌개 drift와 입력이 없는 순수 열확산

\[
\partial_{\hat t}p=-D L_g p
\tag{B6}
\]

에서 초기 기억대비 \(p_0-\bar p\)는 사라진다. \(L_g\psi_k=\lambda_k\psi_k\),
\(0=\lambda_0<\lambda_1\le\cdots\)이면

\[
p(\hat t)=\bar p+\sum_{k\ge1}c_k e^{-D\lambda_k\hat t}\psi_k,
\qquad
\|p(\hat t)-\bar p\|_2
\le e^{-D\lambda_1\hat t}\|p_0-\bar p\|_2.
\tag{B7}
\]

**증명.** LBO 직교 고유기저에 전개하면 각 비상수 계수가
\(\dot c_k=-D\lambda_kc_k\)를 만족한다. connected 조건에서
\(\lambda_1>0\)이므로 결론이 따른다. \(\square\)

**[산출]** CE의 LBO smoothing만으로는 작업기억도 executive state도 구현할
수 없다. \(V_{\rm att}\), 구조화된 drift 또는 시간의존 metric이 필요하다.

## 5. drift가 있을 때

등방확산 \(D_z=DI\), 가역 gradient flow 아래 Fokker–Planck 식은

\[
\partial_{\hat t}p
=\operatorname{div}_{g_\theta}
\left(pM_z\nabla_{g_\theta}\mathcal E+D\nabla_{g_\theta}p\right).
\tag{B8}
\]

**[정리: 조건부]** \(M_z=I\), \(Z<\infty\), zero-flux 경계조건이면

\[
p_\infty(z)=Z^{-1}\exp[-\mathcal E(z)/D]
\tag{B9}
\]

은 정상해다. 지수의 \(\mathcal E/D\)는 무차원이다.

**[예측: 근사]** Kramers 영역에서 기억 우물 탈출시간은 대략
\(T_{\rm escape}/\tau_0\propto\exp(\Delta\mathcal E/D)\)다. 따라서 장벽을
동시에 높이지 않고 확산만 강화하는 기존 CE 제어는 기억수명을 줄여야 한다.

## 6. CE 채택/폐기 경계

| CE 구조 | 뇌 모델에서의 사용 | 형식 지위 |
|---|---|---|
| metric/LBO 고유모드 | 피질 상태 공간기저와 국소 정규화 | 모델 선택; 기하모드는 경험 지지 |
| residual field | 흡수되지 않은 예측오차 회계 | 미완성 가설 |
| small-gain/스펙트럼 | PFC–MD–replay 피드백 폭주 방지 | 조건부 정리 |
| 우주론 고정점 성분비 | 사용하지 않음 | 신경 사상 삭제 |
| 곡률 임계 뒤 상수배 | 사용하지 않음 | 분기 제어 삭제 |

고정점 근방 연속 Jacobian을 \(J_*\)라 하면

\[
\boxed{\max_i\Re\lambda_i(J_*)<0}
\tag{B10}
\]

가 국소 지수안정 조건이다. step \(h\)의 이산 구현에서는

\[
\boxed{\rho(I+hJ_*)<1}
\tag{B11}
\]

을 검사한다. 기존 CE의 고정 수치 \(0.155\)는 가져오지 않는다.

## 7. 무차원 감사

| 코어 인자 | 판정 | 정규화 |
|---|---|---|
| \(\hat t=t/\tau_0\) | 통과 | 기준 시간 \(\tau_0\) |
| \(z=r/r_0\) | 통과 | 기준 발화율 \(r_0\) |
| \(\hat o=o/I_0\) | 통과 | 기준 입력 \(I_0\) |
| softmax 인자 | 조건부 통과 | 각 \(K\)가 역단위를 포함 |
| \(\exp[-\mathcal E/D]\) | 통과 | 동일 무차원 에너지 단위 |
| \(\exp(\Delta\mathcal E/D)\) | 통과 | 에너지비 |
| \(\rho(I+hJ_*)\) | 통과 | \(hJ_*\) 무차원 |

차원 통과는 생물학적 타당성을 증명하지 않는다.

## 8. 최소 구현과 반증

첫 구현은 2차원 PFC 상태, 두 행동, 두 MD 모드로 제한한다.

\[
V_{\rm att}(z;g)=\frac{a}{4}(z_1^2-1)^2
+\frac{a}{4}(z_2^2-1)^2+c_gz_1z_2,
\tag{B12}
\]

\(g_\theta^{-1}=g_0^{-1}+\theta G_1+(1-\theta)G_2\),
\(\theta\in[0,1]\)로 둔다. 비교군은 순수 diffusion, 고정 attractor,
MD 조절 attractor, MD+residual, 전체식의 다섯 개다.

반증 과제는 지연시간 증가, 방해자극, 잠복 규칙전환, 단서 고갈, 갈등 조작이다.

- pure diffusion이 긴 지연에서 attractor와 동등하면 구현/과제가 잘못됐다.
- \(\theta\) 제거가 문맥전환에 영향이 없으면 MD 항은 무효다.
- residual 제거가 stationary와 switch 과제에 똑같이 작용하면 replay 해석은
  기각한다.
- \(\max\Re\lambda(J_*)\ge0\) 또는 \(\rho(I+hJ_*)\ge1\)이면 성능과 무관하게
  탈락시킨다.
- 전체식이 고정 attractor 기준선보다 seed 신뢰구간에서 우월하지 않으면
  구조 추가는 실패다.

## 9. 구현 전 유망성

**[예측]** `78/100`.

- 생물학적 구조 적합성: 22/25
- 수학적 폐쇄성과 무차원성: 20/25
- Loop 6–7 실패 설명력: 18/20
- 작고 반증 가능한 첫 실험: 12/15
- 과매핑 위험: 6/15

감점 이유는 MD=metric, replay=residual 대응이 아직 직접 관측된 동일성이
아니기 때문이다. 다음 단계는 큰 모델이 아니라 (B12)의 작은 SDE에서
diffusion no-go와 MD ablation부터 재현하는 것이다.

## 10. Loop 8B/8C 수치 상태

- **[산출]** 고정된 합성 벤치에서 MD 조절 끌개는 고정 끌개를 ID/OOD 모두
  통과 기준 이상으로 이겼다 (`100/100 GO`).
- **[산출]** 행동 후 문맥 예측오차를 저장한 좁은 residual은 cue-depleted
  전환에서 checkpoint를 이기고 stationary 중립성도 통과했다
  (`100/100 GO`).
- **[미완성]** 두 산출은 각각 MD=metric, residual=해마 replay라는 생물학적
  동일성을 증명하지 않는다. 특히 Loop 8C residual은 spontaneous replay가
  아니라 action-feedback error recurrence다.
- **[산출·실패]** Loop 8D의 선형 경계식
  `b=b0+k_c C`는 낮은 고정경계보다 고갈등 정확도를 높였지만, 동일 평균경계,
  conflict-shuffle, OOD 저갈등 반응시간과 utility 게이트를 함께 통과하지
  못했다 (`0/100 STOP`). (B5)의 생물학적 해석과 구현 승격은 보류한다.
