# Layer A--E 방정식 정본

> 위치: `12_Equation.md`의 canonical runtime 5계층에 대한 **수식 전용** 참조 문서.
> 의존: `14_BrainRuntimeSpec.md`(설계 사양), `6_뇌/evidence.md`(근거 판정), `6_뇌/proof.md`(검증 매트릭스)
>
> 이 문서에는 서사 설명을 넣지 않는다. 식, 정의, 증명/검증 기준만 둔다.

---

## A. 순수 셀 동역학 (kernel dynamics)

### A.1 상태 정의

$$s_i^t = (a_i^t,\;r_i^t,\;m_i^t,\;b_i^t) \in \mathbb{R}^3 \times \{0,1\}$$

| 변수 | 의미 | 범위 |
|---|---|---|
| $a_i$ | activation | $(-1, 1)$ (tanh 이미지) |
| $r_i$ | refractory / inhibition | $\ge 0$ |
| $m_i$ | local memory trace | $\mathbb{R}$ |
| $b_i$ | hysteretic bit | $\{0,1\}$ |

### A.2 입력 합산

$$I_i^t = u_i^t + \sum_j W_{ij}(g)\,a_j^t - \lambda_r(M_t)\,r_i^t + \lambda_m(M_t)\,m_i^t + \eta_i^t$$

- $u_i^t$: 외부 입력
- $W_{ij}(g)$: 결합 가중치 (Layer B 참조)
- $\lambda_r(M_t)$: 모드 의존 억제 계수
- $\lambda_m(M_t)$: 모드 의존 기억 주입 계수
- $\eta_i^t \sim \mathcal{N}(0, \sigma_\eta^2)$: 확률적 잡음 (선택)

### A.3 활성 갱신

$$a_i^{t+1} = (1-\gamma_a(M_t))\,a_i^t + \kappa_a(M_t)\,\tanh(I_i^t)$$

| 파라미터 | 의미 | 안정 조건 |
|---|---|---|
| $\gamma_a \in (0,1]$ | decay rate | $\gamma_a > 0$ 필수 |
| $\kappa_a > 0$ | gain | $\kappa_a < \gamma_a$ 이면 영점 수렴 보장 |

### A.4 억제 갱신

$$r_i^{t+1} = (1-\gamma_r(M_t))\,r_i^t + \kappa_r(M_t)\,(a_i^t)^2$$

$r_i$는 활성의 제곱에 비례하여 축적되며, 높은 활성이 곧 강한 억제를 만든다.

### A.5 기억 흔적 갱신

$$m_i^{t+1} = (1-\gamma_m(M_t))\,m_i^t + \kappa_m(M_t)\,a_i^t$$

기존 `memory EMA`를 재정의. 해마 구현 전 국소 trace cache로 사용.

### A.6 히스테리시스 비트 갱신

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \tau_i^- \le a_i^{t+1} \le \tau_i^+ \end{cases}$$

$\tau_i^+ > \tau_i^-$: 양방향 임계가 달라야 히스테리시스가 성립. 같으면 단순 threshold로 퇴화.

### A.7 형식 검증: 유계성

**정리 (A-bound).** $\gamma_a \in (0,1]$이고 $\kappa_a(1+\gamma_a^{-1}\kappa_a\|W\|_\infty) \le C < \infty$이면 모든 $t$에 대해 $|a_i^t| < 1$.

*증명 스케치.* $a_i^{t+1} = (1-\gamma_a)a_i^t + \kappa_a\tanh(I_i^t)$. $|\tanh(x)| < 1$이므로 $|a_i^{t+1}| \le (1-\gamma_a)|a_i^t| + \kappa_a$. 부등식을 반복하면 $\limsup_{t\to\infty}|a_i^t| \le \kappa_a/\gamma_a$. $\kappa_a < \gamma_a$이면 $|a_i^t| < 1$ 보장. $\square$

### A.8 형식 검증: 억제 유계

**정리 (R-bound).** $\gamma_r \in (0,1]$이고 $|a_i^t| < 1$이면 $\limsup r_i^t \le \kappa_r/\gamma_r$.

*증명.* $r_i^{t+1} \le (1-\gamma_r)r_i^t + \kappa_r$. 선형 재귀 상계. $\square$

### A.9 형식 검증: 영점 안정성

**정리 (Zero-attract).** $\kappa_a < \gamma_a$이고 $u_i^t = 0,\;\eta_i^t = 0,\;W = 0$이면 $(a_i^t, r_i^t) \to (0, 0)$.

*증명.* 결합과 입력이 없으면 $|a_i^{t+1}| \le (1-\gamma_a)|a_i^t| + \kappa_a|\tanh(0)| = (1-\gamma_a)|a_i^t|$. 기하급수 감소. $r_i$는 $a_i^2 \to 0$이므로 후속 감소. $\square$

---

## B. 필드 결합 (coupling / geometry)

### B.1 결합 행렬

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right) \cdot \chi_{ij}$$

| 기호 | 의미 |
|---|---|
| $d_g(i,j)$ | 리만 다양체 $(M, g)$ 위에서 $i$와 $j$ 사이의 측지선 거리 |
| $\sigma$ | 결합 반경 (kernel width) |
| $\chi_{ij} \in \{0,1\}$ | sparse mask |

### B.2 대안: k-nearest neighbor

$$\chi_{ij} = \mathbf{1}[j \in \text{knn}(i, k)]$$

### B.3 에너지 함수 (선택적)

$$E(\{a_i\}) = -\frac{1}{2}\sum_{i,j} W_{ij}\,a_i\,a_j - \sum_i u_i\,a_i + \sum_i V(a_i)$$

$V(a)$: 국소 potential (self-coupling term).

### B.4 형식 검증: 에너지 감소

**정리 (E-decrease).** 동기적 업데이트에서 $a_i^{t+1}$이 $E$의 좌표별 최소화이면 $E(\{a_i^{t+1}\}) \le E(\{a_i^t\})$.

### B.5 spectral bound

$$\|W\|_2 \le \lambda_{\max}$$

$\lambda_{\max}$를 clamp하여 증폭을 제한한다. 이것은 `12_Equation.md`의 spectral constraint와 동일.

---

## C. 전역 모드 (mode update)

### C.1 모드 집합

$$M_t \in \{\mathrm{WAKE},\;\mathrm{NREM},\;\mathrm{REM}\}$$

### C.2 모드 전환

$$M_{t+1} = \Pi(M_t,\;Q_t,\;\Psi_{\text{global}}(t),\;U_t)$$

- $Q_t = (p_{\text{sleep}},\;\text{arousal},\;\text{autonomic},\;\text{endocrine},\;\text{immune},\;\text{metabolic})$: body-loop 제어벡터
- $\Psi_{\text{global}}(t) = \sum_i \omega_i\,a_i^t$: 전역 활성 관측량
- $U_t$: 외부 입력 하중

### C.3 모드별 파라미터 해석

| 파라미터 | WAKE | NREM | REM |
|---|---|---|---|
| $\gamma_a$ | 중간 (0.1--0.3) | 큼 (0.5+) | 작음 (0.05--0.1) |
| $\kappa_a$ | 큼 | 작음 | 중간 |
| $\lambda_r$ | 약함 | 강함 | 약함 |
| $\lambda_H$ (replay) | 약함 | 중간 | 강함 |
| $\sigma_\eta$ (noise) | 작음 | 거의 0 | 중간 |
| $B_t$ (energy budget) | 큼 | 작음 | 중간 |

### C.4 초기 규칙 기반 전환

```
if Q_t.sleep_pressure > theta_sleep and norm(U_t) < theta_input:
    M_{t+1} = NREM
elif M_t == NREM and elapsed > T_nrem:
    M_{t+1} = REM
elif M_t == REM and elapsed > T_rem:
    if Q_t.sleep_pressure > theta_cont:
        M_{t+1} = NREM
    else:
        M_{t+1} = WAKE
else:
    M_{t+1} = M_t
```

---

## D. 해마/기억 (hippocampus / replay)

### D.1 해마 상태

$$H_t = (K_t,\;V_t,\;P_t)$$

### D.2 인코딩

$$K_{t+1} = K_t \cup \{k_{\text{new}}\}, \quad V_{t+1} = V_t \cup \{v_{\text{new}}\}, \quad P_{t+1} = \text{update}(P_t)$$

$k_{\text{new}} = h(A_t)$: 활성 패턴의 해시/임베딩
$v_{\text{new}} = A_t$: 활성 스냅샷

### D.3 회상

$$R_t = \mathcal{R}(H_t, c_t) = V[\arg\max_j \text{sim}(c_t, K_j)]$$

### D.4 재주입

$$I_i^t \leftarrow I_i^t + \lambda_H(M_t)\,R_{i,t}$$

### D.5 Priority replay

$$P_j \leftarrow P_j \cdot \rho + (1-\rho)\,\text{surprise}_j$$

replay 확률은 $P_j / \sum P$에 비례.

---

## E. 자아/전역 상태 (global runtime summary)

### E.1 전역 관측 상태

$$G_t = (M_t,\;A_t^{\text{summary}},\;H_t,\;Q_t,\;\mu_t)$$

- $A_t^{\text{summary}} = \frac{1}{|A_t|}\sum_{i \in A_t} s_i^t$: 활성 모듈 평균 상태
- $\mu_t$: self-bias / identity trace (장기 누적)

### E.2 자아 함수

$$\text{Self}_t = \mathcal{S}(G_t)$$

초기에는 $\text{Self}_t = G_t$ (identity). 후기에 higher-order summary.

---

## F. 자기참조 재귀 (agent loop)

### F.1 최소 재귀

$$S_{t+1} = U(S_t,\;a_t,\;o_t,\;c_{t+1})$$

$$z_t = R(S_t) \quad\text{(relax/converge)}$$

$$a_t = \pi(z_t) \quad\text{(action)}$$

$$o_t = E(a_t) \quad\text{(observe)}$$

$$c_{t+1} = C(S_t, a_t, o_t) \quad\text{(critique)}$$

### F.2 에너지 기반 자기참조

$$E_t(z) = E_{\text{task}}(z;g) + \lambda_m E_{\text{mem}}(z;m_t) + \lambda_c E_{\text{crit}}(z;c_t) + \lambda_h E_{\text{hist}}(z;h_t)$$

$$z_t^* = \arg\min_z E_t(z)$$

### F.3 Clarus 통합 재귀

$$X_{t+1} = B\big[X_t + \lambda_1 R_{\text{self}}(X_t) + \lambda_2 R_{\text{obs}}(X_t) + \lambda_3 C(X_t) - \lambda_4 S(X_t)\big]$$

---

## G. 형식 증명 요약

| 정리 | 주장 | 조건 | 상태 |
|---|---|---|---|
| A-bound | $\|a_i^t\| < 1$ 유계 | $\kappa_a < \gamma_a$ | **closed** |
| R-bound | $r_i^t$ 유계 | $\gamma_r > 0$, A-bound | **closed** |
| Zero-attract | 입력 없으면 영점 수렴 | $W=0$, $u=0$, $\kappa_a < \gamma_a$ | **closed** |
| E-decrease | 에너지 비증가 | 좌표별 최소화 업데이트 | **closed** |
| Sleep-stabilize | sleep 후 에너지/잡음 감소 | NREM: $\gamma_a$ 증가, $\kappa_a$ 감소 | **closed** |
| Local-contract | 국소 Lipschitz 수축 | $\max((1-\gamma_a) + \kappa_a\|W_i\|_1) < 1$ | **closed** |
| Sparse-energy | 활성 수 제한 | $B_t$ 유한, threshold $\theta_i > 0$ | **closed** |
| Hysteresis-invariant | $b_i$ 전환 조건 비대칭 | $\tau^+ > \tau^-$ | **closed** |

---

## H. 검증 게이트 대응

`proof.md`의 4중 게이트를 이 문서의 식에 적용.

| 게이트 | 적용 대상 | 상태 |
|---|---|---|
| $G_{\text{formal}}$ | A-bound, R-bound, E-decrease, Zero-attract, Local-contract | pass |
| $G_{\text{obs}}$ | EEG/fMRI에서 $a_i^t$ proxy 추출 | partial |
| $G_{\text{causal}}$ | 약물/수면박탈/자극 실험에서 모드 전환 방향 일치 | partial |
| $G_{\text{pred}}$ | 모델 시뮬레이션 vs 실제 뇌파 스펙트럼 비교 | pending |

---

## I. 관측 가능량 매핑

| formal 변수 | 뇌 관측량 | 데이터 소스 |
|---|---|---|
| $a_i^t$ | local field potential, spiking rate | intracranial EEG, multi-electrode array |
| $\Psi_{\text{global}}(t)$ | scalp EEG power | EEG datasets (PhysioNet, LEMON) |
| $r_i^t$ | refractory period, inhibitory postsynaptic potential | in vitro recordings |
| $b_i^t$ | UP/DOWN state | cortical slice recordings |
| $M_t$ | wake/NREM/REM scoring | polysomnography |
| $Q_t$ | autonomic, hormonal | heart rate variability, cortisol |
| $W_{ij}$ | structural/functional connectivity | DTI, resting-state fMRI |
