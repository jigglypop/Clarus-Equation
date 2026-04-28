# CE 게임 밸런스: 마스터 공식의 응용 투영

> 관련: `agi.md`(AGI 작용 범함수), `homeomorphism.md`(구조 유비), `axium.md`(공리/기호 규약)
>
> 비개발자 가이드: `guide.md`

이 문서는 CE 마스터 공식을 게임 설계 공간에 **응용 투영**해 보는 가설적 문서다. 벨만 방정식/마르코프 체인을 직접 대체한다고 주장하기보다, 밸런스 지형을 읽는 하나의 구조적 좌표계로 사용하는 것이 안전하다.

---

## 1. 설계 공간 작용 범함수

설계 변수 $x \in \mathcal{X}$ (맵, 유닛 스펙, 팩션 구성)가 승률 지형 $P(x)$를 정의한다. CE 마스터 공식을 설계 공간 $(\mathcal{X}, g)$에 적용:

$$S_{\text{balance}} = \int_{\mathcal{X}} d^n x \sqrt{|g|} \left[ \underbrace{L_{\text{win}}(P)}_{\text{밸런스 비용}} + \underbrace{\alpha|\nabla P|^2}_{\text{기울기 안정화}} + \underbrace{\lambda|\Delta_g P|^2}_{\text{곡률 평탄화}} + \underbrace{\gamma S_{\text{meta}}}_{\text{메타 다양성}} \right]$$

| 항 | CE 대응 | 게임 의미 |
|---|---|---|
| $L_{\text{win}}$ | $\mathcal{L}_{\text{Physical}}$ | 승률 편차 $\sum_f(W_f - 1/F)^2$ |
| $\alpha\|\nabla P\|^2$ | 1차 클라루스 | 스탯 미세 변경에 대한 승률 민감도 억제 |
| $\lambda\|\Delta_g P\|^2$ | 2차 클라루스 (LBO) | "살얼음판 밸런스" 방지 -- 곡률이 큰 설계 배제 |
| $\gamma S_{\text{meta}}$ | 정보 엔트로피 | 메타 고착 방지 -- 전략 다양성 보존 |

$\delta S = 0$의 해가 "CE 최적 설계"이다.

---

## 2. LBO 안정성: CE 2차 항에서 직접

CE 마스터 공식의 2차 항 $\lambda|\Delta_g\Phi|^2$를 승률 지형에 적용하면, $\Delta_g P$가 크면 패널티를 받는다.

$$\Delta_g P = \frac{1}{\sqrt{|g|}} \partial_i\!\left(\sqrt{|g|}\, g^{ij} \partial_j P\right)$$

이산 설계 공간에서는 그래프 라플라시안 $L = D - W$로 근사:

$$\Delta_g P \approx L P, \qquad |\Delta_g P|^2 \approx P^\top L^2 P$$

**안정성 판정**: 설계 $x$에서 승률 $P$의 곡률을 2차 차분으로 측정:

$$\kappa(x, e) = \frac{P(x + \sigma e) + P(x - \sigma e) - 2P(x)}{\sigma^2}$$

$|\kappa|$가 크면 "그 방향으로 설계가 예민하다" = 살얼음판.

---

## 3. 3x3+1 격자의 게임 해석

`agi.md` 2절의 3x3+1 격자를 게임 설계에 대입:

| 층 | 게이지 | 비율 | 게임 의미 |
|---|---|---|---|
| 3 | SU(3) | $74.1\%$ | **유닛 합성(binding)**: 유닛 조합/시너지 |
| 2 | SU(2) | $21.1\%$ | **전략 분기(decision)**: 전투/우회/방어 선택 |
| 1 | U(1) | $4.9\%$ | **자원 배분(attention)**: 집중/분산 |
| $\Phi$ | 중력 | 전역 | **LBO 안정화**: 밸런스 곡률 평탄화 |

설계 파라미터 공간도 이 비율을 따라야 한다.

$$\dim(\text{유닛 스펙}) : \dim(\text{전략 옵션}) : \dim(\text{자원 변수}) \approx 74 : 21 : 5$$

유닛 조합의 자유도가 전략 분기보다 3.5배 커야 하고, 자원 배분은 전체의 5% 미만이어야 한다.

---

## 4. 유니타리 조건과 지배 전략 억제

$$|\det \mathbf{T}|^2 \leq 1$$

전이 행렬 $\mathbf{T}$의 유니타리 조건 = "정보가 증폭되지 않는다."

게임 번역: **어떤 전략도 정보(승률)를 무한히 증폭할 수 없다.**

$$\sigma_1(M_{\text{strategy}}) \leq 1$$

여기서 $M_{\text{strategy}}$는 전략 $\to$ 승률 변환 행렬. 최대 특이값이 1 이하이면:
- 카운터가 존재하지 않는 지배 전략 불가
- 메타가 순환(가위바위보)하거나 수렴(50:50)

**판정**: 모든 팩션 쌍 $(f,g)$에 대해 승률행렬 $W_{f,g}$의 최대 특이값 확인.

---

## 5. 부트스트랩 분배와 게임 경제

CE 부트스트랩 고정점(`homeomorphism.md` 보조정리 3.2):

$$\varepsilon^2 = \exp\!\big(-(1-\varepsilon^2) D_{\text{eff}}\big) \implies 4.87\% / 26.2\% / 68.9\%$$

게임 경제에 적용:

| CE 성분 | 비율 | 게임 해석 |
|---|---|---|
| 활성 ($\varepsilon^2$) | $4.87\%$ | 결정적 교전 (킬/데스) |
| 구조 | $26.2\%$ | 포지셔닝/자원 관리 |
| 배경 | $68.9\%$ | 이동/대기/탐색 |

**설계 제약**: 게임 플레이 시간의 ~5%만 결정적 교전이어야 한다. 교전 비율이 이보다 높으면 혼란, 낮으면 지루함.

---

## 6. 최적화: CE 경로적분에서 ES로

CE 경로적분(`경로적분.md`):

$$Z = \int \mathcal{D}\Phi\, e^{-S[\Phi]}$$

설계 공간의 경로적분:

$$Z_{\text{design}} = \int \mathcal{D}x\, e^{-S_{\text{balance}}[x]}$$

이 적분을 몬테카를로로 근사하면 자연스럽게 ES(Evolution Strategy)가 된다.

$$\nabla_x S_\sigma = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\!\left[S_{\text{balance}}(x + \sigma\epsilon)\, \epsilon\right]$$

가우시안 평활화 $S_\sigma(x) = \mathbb{E}[S(x + \sigma\epsilon)]$는 CE의 확산 방정식 해:

$$\partial_t S_\sigma = \Delta S_\sigma, \qquad t = \sigma^2/2$$

**벨만 방정식이 불필요한 이유**: CE 경로적분 자체가 "모든 설계 경로의 가중 합"이다. 재귀적 분해(DP)가 아니라 전역 적분으로 최적을 찾는다.

분산 감소:

$$\hat{\nabla} = \frac{1}{K\sigma} \sum_{k=1}^{K/2} \left(S(x + \sigma\epsilon_k) - S(x - \sigma\epsilon_k)\right) \epsilon_k$$

---

## 7. 평가 프로토콜

설계 후보 $x$의 CE 밸런스 점수:

$$\hat{S}(x) = \frac{1}{N} \sum_{s=1}^{N} \left[ L_{\text{win}}(x; \omega_s) + \lambda\,\kappa^2(x; \omega_s) + \gamma\,H_{\text{meta}}(x; \omega_s) \right]$$

| 지표 | 정의 | 기준 |
|---|---|---|
| $L_{\text{win}}$ | $\max_{f \neq g} \|W_{f,g} - 0.5\|$ | $\leq 0.05$ |
| $\kappa^2$ | $P^\top L^2 P$ (곡률 에너지) | 단조 감소 |
| $H_{\text{meta}}$ | $-\sum_i p_i \ln p_i$ (전략 엔트로피) | $\geq \ln F$ (균등 이상) |

성공 판정: 3개 동시 만족 + 시드 분산 $\text{Var}[W] < \epsilon$.

---

## 8. 퇴화 해 방지

CE 정보 엔트로피 항 $\gamma S_{\text{Info}}$가 퇴화를 구조적으로 방지한다.

- **교전 없는 50:50**: $S_{\text{meta}} \to 0$ (정보 소멸) → 엔트로피 항이 패널티
- **극단적 게임 길이**: $|\nabla P|^2 \to \infty$ → 1차 항이 패널티
- **메타 고착**: $H_{\text{meta}} \to 0$ → 엔트로피 항이 패널티

제약은 별도 추가가 아니라 **CE 작용 범함수에 이미 내장**되어 있다.
