# 11-math — agi-clarus-field-20260812

Status: COMPLETE

## 대상·정의역·전제

- 계약: `00-contract.md` (Status: COMPLETE 확인). PREDECESSOR: `_workspace/ce/agi-v14-binding-design-20260812` — C1(무손실 슬롯+bilinear 충분성)·C2(가법/선형 판독 불가능성)·A-SAL(짝함수 게이트 공리)은 선행 run CONFIRMED 결론으로 인용하고 재유도하지 않는다 (선행 11-math §C1·C2, 12-routes §3).
- 기질: 유한 연결 그래프 $G=(V,E)$, $|V|=N$, 정규화 라플라시안 $\Delta_G=D^{-1/2}(D-A)D^{-1/2}\succeq 0$ (`docs/7_AGI/20_DiffusionOrchestration.md`의 Exact 부분만 상속: $\lambda>0$이면 $\Delta_G+\lambda I\succ 0$, 정상해 유일).
- 노드 상태 $s_i\in\mathbb R^w$, 보조장 $\phi_i\ge 0$. tick $t\in\mathbb N$마다 게이트 갱신, tick 사이 구간에서 $\phi$는 선형 디퓨전.
- 상 분류(계약 정의 그대로): 활성 $\iff g_i(t)>\theta_g$; 구조 $\iff$ 비활성이고 binding 참여 지표 $>\theta_s$; 동결 $\iff$ 그 외. 점유율 $\pi(t)$는 시간·노드 평균.
- $p^*$ 정본: `reality_stone/python/reality_stone/clarus/constants.py` (0.0487, 0.2623, 0.6891); 유도 후보는 `docs/7_AGI/12_Equation.md` 8.1·A.2.
- 수치 검산은 전부 독립 경로(정의에서 재구현)로 수행, 스크립트·로그는 `artifacts/`.

### 결합계의 정식화에 필요한 공리 (계약 식의 두 결손을 먼저 닫음)

계약의 장 방정식 $\partial_\tau\phi=-\Delta_G\phi+s-\lambda\phi$는 그대로는 정의되지 않는다: $s_i\in\mathbb R^w$는 벡터이고 $\phi_i$는 스칼라(차원 불일치), 그리고 $\phi_i\ge 0$ 제약은 임의 소스에서 보존되지 않는다. 다음을 공리로 선언하고 그 위에서 증명한다.

- **공리 D1 (스칼라 readout 소스).** 디퓨전 소스는 부호불변·유계·Lipschitz readout $r:\mathbb R^w\to[0,R]$ (예: $r(s)=\min(\lVert s\rVert,R)$)를 통해서만 진입한다: $\partial_\tau\phi=-(\Delta_G+\lambda I)\phi+r(s(t))$, $\lambda>0$.
- **공리 S1 (분리).** 디퓨전 연산자는 $\phi$에만 작용한다. $s$의 갱신식에 $\Delta_G$ 항이 없다.
- **공리 S2 (게이트 경유 결합).** $\phi$는 $s$ 동역학에 오직 $g_i$와 쓰기값 $\tilde s_i$의 인수로만 진입한다: $s_i^+=(1-\hat g_i)s_i+\hat g_i\tilde s_i$, $\lVert\tilde s_i\rVert\le 1$.
- **공리 S3 (경성 게이트).** 유효 게이트는 $\hat g_i=g_i\,\mathbf 1[g_i>\theta_g]$. 로지스틱 $\sigma$는 $g>0$을 항상 주므로, S3 없이는 "동결 갱신 = 정확히 항등"이 성립하지 않는다(아래 CF-2(iv)가 정량화). 상 분류 임계 $\theta_g$와 동역학 임계를 일치시키는 이 공리는 계약 정의(활성 $\iff g>\theta_g$)와 정합.

## CF-1 — 결합계 well-posedness (지위: **조건부 정리** — 공리 D1·S1–S3 하 증명 완료)

**정리 CF-1.** 공리 D1·S1–S3 하에서, 임의의 유계 입력열과 임의의 초기조건 $(s(0),\phi(0))$, $\phi(0)\ge 0$에 대해:

(a) *존재·유일성*: tick 사이 $\phi$-흐름은 상수 강제항의 선형 ODE이므로 해가 정확히
$$\phi(t{+}h)=e^{-(\Delta_G+\lambda I)h}\phi(t)+(\Delta_G+\lambda I)^{-1}\big(I-e^{-(\Delta_G+\lambda I)h}\big)\,r(s(t))$$
로 닫힌다. 전역 해 존재·유일은 자명.

(b) *$s$ 전역 유계*: $\lVert s_i(t)\rVert\le\max(\lVert s_i(0)\rVert,1)$ 모든 $t,i$. (증명은 CF-2(i).)

(c) *$\phi$ 전역 유계·양성*: $\Delta_G\succeq 0$이므로 $\lVert e^{-(\Delta_G+\lambda I)h}\rVert_2\le e^{-\lambda h}$, 따라서 $\lVert\phi(\tau)\rVert_2\le\max\big(\lVert\phi(0)\rVert_2,\ \sqrt N\,R/\lambda\big)$. 또한 $-(\Delta_G+\lambda I)$는 비대각 성분이 $D^{-1/2}AD^{-1/2}\ge 0$인 Metzler 행렬이므로 $e^{-(\Delta_G+\lambda I)h}\ge 0$ (성분별), $r\ge 0$과 함께 $\phi(0)\ge 0\Rightarrow\phi(\tau)\ge 0$ 보존.

(d) *동결 항등*: 동결상 노드는 $g_i\le\theta_g\Rightarrow\hat g_i=0\Rightarrow s_i^+=(1-0)s_i+0=s_i$ — IEEE 부동소수에서도 비트 단위 항등(스칼라 1.0 곱과 0 덧셈은 정확). 디퓨전이 latch의 고유값-1 성질을 파괴하지 않을 충분조건이 정확히 S1+S2+S3이다: S1 위반(예: $s^+=(1-\hat g)s+\hat g\tilde s-\epsilon\Delta_G s$)은 닫힌 게이트의 Jacobian을 $I-\epsilon\Delta_G$로 바꿔 고유값 1을 즉시 깨고, S3 위반은 tick당 $O(\theta_g)$ 누설을 준다(CF-2(iv)). $\square$

*주의(경계)*: (c)의 $\ell_\infty$ 버전은 일반 그래프에서 실패한다 — 성형(star) 그래프 중심에서 $D^{-1/2}AD^{-1/2}$의 행합이 $\sqrt d>1$이므로 행합 논증이 안 통함. 유계성 인증서는 2-노름으로 제시한다.

**V14 route L과의 경계.** 선행 toy의 실제 route L 갱신은 $h^+=(1-g)h+g\{h\circledast(\delta+\tanh(Vx))+\tanh(Ux)\}$이고 게이트는 경성 임계를 통과하지 않은 sigmoid다. 따라서 쓰기 후보가 상태 $h$에 의존하고 노름 상계도 없어서 S2·S3을 만족하지 않는다. 1차원에서 $\tanh(Vx)=v>0$, $\tanh(Ux)=0$, $g>0$이면 $h^+=(1+gv)h$이므로 반복 시 지수 발산한다. 따라서 CF-1~3은 route L 원형에 상속되지 않는다. 구현 승인은 경성 게이트와 유계 투영을 둔 별도 변형, 또는 무손실 슬롯에 저장한 뒤 HRR을 readout으로만 계산하는 변형으로 제한한다.

수치: `artifacts/verify_cf1_cf2.log` — $N=16$, $T=4000$, 정확 행렬지수 경로. 동결 tick 비트 단위 항등 PASS, $\sup_t\lVert s\rVert=1.3550\ldots=\max(\lVert s(0)\rVert,1)$ 경계 정확 포화, $\sup\lVert\phi\rVert_2=11.303\le 11.429=\sqrt N R/\lambda$, $\phi\ge 0$ 유지.

## CF-2 — 게이트-스케줄 안정성 정리 (지위: **정리** — 증명 완료)

**설정.** 노드별 이벤트 시퀀스: 열림 tick 집합 $\mathcal O_i\subset\mathbb N$. 닫힘 tick에서 $s_i^+=U_ts_i$ ($U_t$ 선형 등거리; 본 장에서는 $U_t=I$), 열림 tick에서 $s_i^+=(1-g)s_i+g\,\tilde s$, $g\in(\theta_g,1]$, $\lVert\tilde s\rVert\le 1$.

**정리 CF-2.** 이벤트 시퀀스에 대한 귀납으로:

**(i) 전역 유계 (T-균일, 스케줄 자유).** $\lVert s(t)\rVert\le\max(\lVert s(0)\rVert,1)$. *증명.* 닫힘: 등거리로 노름 보존. 열림: $\lVert(1-g)s+g\tilde s\rVert\le(1-g)\lVert s\rVert+g\le\max(\lVert s\rVert,1)$. 귀납 완료. 특히 단위구는 불변집합. $\square$

**(ii) 쓰기 섭동의 비확대.** 같은 스케줄·같은 $g$, 쓰기값만 $\varepsilon$ 이하로 다른 두 궤적의 차 $e$: 닫힘에서 $\lVert e\rVert$ 보존(등거리), 열림에서 $\lVert e^+\rVert\le(1-g)\lVert e\rVert+g\varepsilon\le\max(\lVert e\rVert,\varepsilon)$. 따라서 $\sup_t\lVert e(t)\rVert\le\max(\lVert e(0)\rVert,\varepsilon)$ — $T$·$\rho$ 무관.

**(iii) 이벤트당 망각과 $\rho$-비례 결함 전파.** 열림 이벤트에서 $g\ge g_{\min}>0$이고 $k$번째 이벤트에 추가 결함 $\lVert\eta_k\rVert\le\bar\eta$가 주입되면(닫힘 tick 결함 0 — CF-1(d)의 정확 항등), 이벤트 수 $N(t)$에 대해
$$\lVert e(t)\rVert\le(1-g_{\min})^{N(t)}\lVert e(0)\rVert+\varepsilon+\frac{\bar\eta}{g_{\min}},$$
그리고 $g_{\min}$ 하한이 없으면 최악 누적은 $\lVert e(t)\rVert\le\lVert e(0)\rVert+N(t)\bar\eta\le\lVert e(0)\rVert+\rho\,t\,\bar\eta$ — 열림 빈도 $\rho=N(t)/t$에 비례. *증명.* 이벤트 재귀 $e_{k+1}\le(1-g_{\min})e_k+g_{\min}\varepsilon+\bar\eta$의 고정점 상계와, 상계 없는 경우의 단순 합산. $\square$

**(iv) 정확 항등의 필요성 (조건부 정리; S3의 부하).** 다음 스칼라 갱신 모형을 추가로 가정한다. 각 tick은 현재 오차와 독립인 Bernoulli$(\rho)$ 열림 사건이고, 열림 tick에는
$$e_{t+1}=(1-g)e_t+g\varepsilon+\eta_o,$$
닫힘 tick에는 $e_{t+1}=e_t+\eta_c$가 정확히 성립하며 $g\in(0,1]$, $\eta_o,\eta_c\ge0$는 상수다. 그러면 평균 재귀는
$$\mathbb E[e_{t+1}]=(1-\rho g)\mathbb E[e_t]+\rho g\varepsilon+\rho\eta_o+(1-\rho)\eta_c$$
이고 유일한 정상 평균은
$$\mathbb E[e_\infty]=\varepsilon+\frac{\eta_o}{g}+\frac{(1-\rho)\eta_c}{\rho g}.$$
따라서 $\eta_c>0$이면 정상 평균은 $\rho\downarrow0$에서 $1/\rho$ 규모로 발산한다. 이 결론은 위의 갱신 순서·독립성·정확한 비음수 결함을 둔 모형에만 대한 정리다. 임의 스케줄에서는 평균 빈도 $\rho$만으로 최대 닫힘 간격을 제어할 수 없고, 결함의 상계만으로 실제 발산을 추론할 수도 없다. 그러므로 일반 장에 대한 $1/\rho$ 법칙은 **경험식/검증 예측**이며, (i)–(iii)의 스케줄-자유 정리와 구분한다. $\square$

**NISCC-6A와의 관계 — 대체이며 특수화가 아님.** NISCC-6A(`docs/7_AGI/28_Nested_Infinite_SCC_V9.md`)는 level-독립 균일 $q<1$을 가정하고 유일 고정점 + 지수 수렴을 결론한다. 본 계는 닫힘 구간에서 $q=1$(등거리)이므로 6A의 가설 밖이고, 결론도 다르다: CF-2는 유일 고정점을 주지 **않는다**(게이트가 영원히 닫히면 모든 상태가 고정 궤적 — 고정점 연속체가 존재하며 이는 결함이 아니라 latch의 정의적 성질). CF-2가 주는 것은 (i) T-균일 유계 + (ii)(iii) 이벤트-단위 망각·$\rho$-비례 결함 제어이고, 망각률은 벽시계 시간이 아니라 이벤트 수 $N(t)\approx\rho t$로 측정된다. 따라서 6A의 결론 집합과 CF-2의 결론 집합은 서로 포함하지 않는다 — 인증서 **교체**.

**NISCC-7C와의 정합.** 7C의 가중 후진 시프트 $B_n$은 닫힘 구간 사상이 항등이 아니므로($q_n=(n-1)/n$) CF-2의 가설 밖 — 충돌 없음. 반대로 CF-2의 인증 상수($\max(\lVert s_0\rVert,1)$, $\varepsilon$, $g_{\min}$, $\rho$)는 스펙트럼량을 전혀 쓰지 않고 level-독립 가설(등거리·$\lVert\tilde s\rVert\le 1$ 균일)에서 나오므로, $q_n\to 1$로 균일 수축 인증서가 죽는 direct-limit 상황에서도 그대로 생존한다. V9의 옛 인증서 $q=0.54$는 default Jacobi fixture의 global_coordinate_sup metric 전용 unit certificate(28_NISCC §V9 구현 절)로, latch 동역학($q=1$)에는 적용 자체가 불가 — CF-2가 이를 **대체**한다(포함 아님).

수치: `artifacts/verify_cf1_cf2.log` — (ii) $\varepsilon=10^{-3}$에서 $\sup\lVert e\rVert=9.849\times10^{-4}\le\varepsilon$ PASS; (iii) $\bar\eta=10^{-4}$, $g_{\min}=0.998$에서 $\sup\lVert e\rVert=1.0005\times10^{-4}\le\bar\eta/g_{\min}=1.0010\times10^{-4}$ PASS (정상 상계 포화); (iv) 연성 게이트의 닫힘 tick 누설 실측 $2.08\times10^{-1}$ vs 경성 게이트 정확히 0.

## CF-3 — 점유율 평형 존재 (지위: **조건부 정리** — 공리 A-E1·A-E2R·A-E3 하 증명 완료; 내생 게이트는 미완성으로 분리)

**공리 (명시 선언).**
- **A-E1 (입력 에르고딕성).** 입력열 $(x_t)$는 i.i.d. (정상 에르고딕이면 별도 skew-product 조건이 필요하며, 여기서는 주장하지 않는다).
- **A-E2R (외생 게이트 + 재생 쓰기).** $g_i(t)$와 쓰기값 $\tilde s_i(t)=\psi_i(x_t)$는 현재 상태 $(s_t,\phi_t)$와 무관한 같은 입력 $x_t$의 함수다. 노드별 열림 확률은 $\rho_i>0$이고 열림 시 $\hat g_i\ge g_{\min}>0$다. 특히 두 초기조건을 같은 입력으로 결합하면 열림 tick의 쓰기값이 동일하다.
- **A-E3 (임계 비원자).** 유일 정상 법칙 하에서 분류 임계 집합 $\{g=\theta_g\}$, $\{\text{binding 지표}=\theta_s\}$의 측도는 0이고 binding 지표는 상태·입력의 연속 함수다. 이는 상태 결합의 수렴을 불연속 상 지표의 시간평균 수렴으로 옮기는 데 필요하다.

**정리 CF-3.** A-E1·A-E2R·A-E3 하에서 시간 평균 점유율 $\pi(t)$는 $t\to\infty$에서 초기조건에 무관한 극한 $\bar\pi$로 a.s. 수렴한다.

*증명.* (1) *경로별 수축 결합*: 같은 입력열을 받는 두 초기조건은 A-E2R 때문에 열림 tick에서 동일한 $\psi_i(x_t)$를 쓴다. 따라서 닫힘 tick에서는 차이가 보존되고 열림 tick에서는 정확히 $s_i^+-s_i^{\dagger+}=(1-\hat g_i)(s_i-s_i^\dagger)$다. 이에 따라 $\lVert s_i(t)-s_i^\dagger(t)\rVert\le(1-g_{\min})^{N_i(t)}\lVert s_i(0)-s_i^\dagger(0)\rVert$. LLN으로 $N_i(t)/t\to\rho_i>0$ a.s.이고, 유한 노드 모두에서 차이는 지수적으로 0으로 간다. $\phi$ 성분의 차는 감쇠율 $\lambda$의 지수안정 선형 필터가 Lipschitz 입력 $r(s)-r(s^\dagger)$로 구동되므로 역시 0으로 간다. (2) *정상 법칙의 존재·유일*: CF-1의 유계 불변집합에서 같은 입력을 쓰는 후진 반복의 상 지름은 모든 노드의 재생 사건 수가 발산하므로 0으로 간다. 그 극한은 초기점에 무관한 i.i.d. 입력열의 가측 함수이고, 그 법칙 $\mu$는 유일 정상 법칙이다. (3) *시간평균*: 정상 과정은 i.i.d. shift의 가측 factor이므로 에르고딕이고 Birkhoff 정리가 유계 상 지표의 시간평균 수렴을 준다. A-E3에 의해 수렴하는 두 상태가 임계 경계의 서로 다른 쪽에 머무는 시간밀도는 0이므로 임의 초기조건도 같은 $\bar\pi$를 갖는다. $\square$

**필요성 정리 (재생 쓰기 없는 A-E2의 한계).** 외생 게이트만으로는 결론이 나오지 않는다. 허용된 쓰기값을 $\tilde s_i=s_i$로 두면 열림 여부와 무관하게 $s_i^+=s_i$다. 두 초기조건 $s_i(0)=0$과 $s_i(0)=1$은 영원히 분리되고, $r(s)=s$이면 $\phi$도 각각 $0$과 $1/\lambda$로 수렴한다. $0<\theta_s<1/\lambda$에서 두 궤적의 구조·동결 점유율은 서로 다르다. 따라서 A-E2R의 공통 재생 쓰기 또는 이를 대신하는 전체 열린 사상의 명시적 joint contraction/small-gain 조건은 필수다. $\square$

**경계 (정직 선언).** 게이트 또는 쓰기값이 $\phi$·$s$에 의존하면 A-E2R이 깨진다. 콤팩트 불변집합과 Feller 연속성까지 추가하면 Krylov–Bogolyubov로 정상 법칙의 **존재**를 보일 수 있지만, CF-1의 유계성만으로 콤팩트성·Feller 성질이 자동으로 나오지는 않는다. 완전 결합계의 존재·유일성·초기조건 무관성은 **미완성**이며, 구현 baseline은 A-E2R을 그대로 상속해야 한다.

수치: `artifacts/verify_cf3.log` — 원거리 두 초기조건이 동일 입력에서 유한 시간 내 정확 합류(차 $0.0$), $\pi(t)$ 연속 차분 $1.02\times10^{-3}\to 1.88\times10^{-4}$ 단조 감소, 초기조건 간 $|\bar\pi_A-\bar\pi_B|_\infty=2.03\times10^{-5}$.

## CF-4 — $p^*$ 자기수렴의 지위 판정 (지위: **자유 예측** — 유도 경로 부재, killing test 명세; 유도 후보의 정본 결함 P1 보고)

**탐색한 유도 경로와 판정 (`artifacts/verify_cf4.log`).**

1. *부트스트랩 스칼라 자체는 닫힌다*: $A_d=4/(e\pi)^{4/3}=0.2291575578$, $D_{\text{eff}}=3+A_d(1-A_d)=3.1766443715$, $a^\star=\exp(-(1-a^\star)D_{\text{eff}})=0.0487077473$ (잔차 $<10^{-12}$; 12_Equation 8.1의 0.04865와는 $5.8\times10^{-5}$ 차 — 반올림 관행, P2). 그러나 이 유도의 입력은 공간 차원 $d=3$과 결합상수 $A_d$이며, 클라루스장의 구조 상수($\theta_g$, 신호 희소성, 그래프 차수)와의 강제된 동일시는 존재하지 않는다. 그래프 유효 차원을 3으로 놓는 것은 선택이지 유도가 아니다.
2. *정본의 3-simplex 사상은 $p^*$를 고정하지 않는다*: 12_Equation A.2.1의 $B(p)_a=\exp(-(1-p_a)D_{\text{eff}})$, $B(p)_b=\alpha_sD_{\text{eff}}$, $B(p)_s=1-B_a-B_b$의 실제 고정점은 $(a,s,b)=(0.048708,\ 0.796565,\ 0.154727)$로, $p^*=(0.0487,0.2623,0.6891)$과 $\lVert\cdot\rVert_\infty$ 거리 **0.5344** — A.2.2의 "$p^*$는 $B$의 유일 내부 고정점" 진술은 쓰인 그대로는 수치적으로 성립하지 않는다 (**P1**, 아래). 라벨 교환으로도 구제 불가. 부수적으로 A.2.2의 Jacobian 식 $D_{\text{eff}}p_a(1-p_a)=0.1472$는 실제 도함수 $D_{\text{eff}}p_a=0.1547$과 다르다(둘 다 $<1$이라 국소 수축 결론은 유지, P2).
3. *CF-1/2/3 장 동역학 자체에는 자기수렴 메커니즘이 없다*: toy에서 $\pi_A$는 외생 신호율을 그대로 추적한다 — 신호율 0.049/0.120/0.300에서 $\pi_A=0.0488/0.1207/0.2994$. 즉 $\pi_A$는 입력 통계량이고, $\pi_S/\pi_F$ 분할은 자유 임계 $\theta_s$ 대 $\phi$ 스케일($\approx\bar r/\lambda$)의 함수다. 자기수렴이 성립하려면 임계의 내생 적응(12_Equation A.2.3 조건 2의 자기측정 루프) 같은 **추가** 메커니즘이 필요하며 이는 계약의 장 정의에 없다. 이 판정은 canon의 F1 게이트(메커니즘 결손)·5_Sparsity 8.5(transformer 기질 falsified)와 정합.

**판정: 자기수렴 주장은 정리가 아니라 자유 예측(Hypothesis)이다.** 유도가 안 되는 것을 되는 것처럼 쓰지 않는다.

**Killing test 프로토콜 (구현 단계 사전등록 대상).**
- 측정량: 사전등록된 $\theta_g,\theta_s,\lambda$, 입력 분포, seed 16개에서 burn-in 후 창 $[T/2,T]$의 $\bar\pi$ 성분별 값과 seed 부트스트랩 CI.
- 통과 기준: $\lVert\bar\pi-p^*\rVert_\infty\le 0.02$ (절대) — 사전등록 후 변경 금지.
- 대조군과 사망 조건: (i) **무작위 게이트** — $\hat g$를 열림률 매칭된 Bernoulli($\hat\rho$)로 교체; 대조군도 허용 오차 내로 $p^*$에 도달하면 예측은 비특이적으로 사망. (ii) **셔플 입력** — 시간 구조 파괴; 동일 판정. (iii) **신호율 섭동** — 입력 신호율을 2배/절반으로 변경; $\bar\pi_A$가 신호율을 추적하면(오차 대역을 넘는 이동) 자기수렴 사망 — 본 run의 toy는 이 test에서 **이미 죽는 형태**임을 기록한다. (iv) 처리군이 허용 오차 밖이면 즉시 사망.

## CF-5 — 선형 게이트 불가능성의 장 일반화 (지위: 정적 임계 **정리**, 적응 임계 **조건부 정리** — 증명 완료)

**설정.** 신호 tick 입력 $x=z\,\xi+\eta$ ($\xi\ne 0$ 고정 패턴, $z=\pm1$ 등확률), 노이즈 tick 입력 $x=\eta$다. 현재 잡음 $\eta$는 신호/노이즈 계급 및 $z$와 독립이고 두 계급에서 같은 임의 분포를 갖는다(대칭성은 불요). 선형 게이트의 열림 사건은 $\sigma(u^\top x+\beta)>\theta_g\iff u^\top x>c$. $m=u^\top\xi$, $S(v)=\Pr(u^\top\eta>v)$ (비증가).

**정리 CF-5 (interlacing).** $p_\pm=\Pr(\text{열림}\mid\text{신호},z=\pm1)=S(c\mp m)$, $q=\Pr(\text{열림}\mid\text{노이즈})=S(c)$에 대해
$$\min(p_+,p_-)=S(c+|m|)\ \le\ q\ \le\ S(c-|m|)=\max(p_+,p_-).$$
*증명.* $c$는 $c-|m|$과 $c+|m|$ 사이에 있고 $S$는 비증가. $\square$

**따름정리 1 (정렬 불가능, 정적 임계).** $\delta<\tfrac12$에 대해 "$p_+\ge 1-\delta$이고 $p_-\ge 1-\delta$이고 $q\le\delta$"는 불가능: $1-\delta\le\min(p_+,p_-)\le q\le\delta$는 $\delta\ge\tfrac12$를 강제. 즉 활성상은 신호 tick 집합을 노이즈 tick 집합과 분리하지 못한다 — 한쪽 부호의 신호를 노이즈 오탐률 이상으로 놓치거나, 노이즈를 약한 부호의 검출률 이상으로 오탐해 latch를 덮어쓴다(V14 C1 불완전 게이트 분석의 오염 경로). $\pi_A$ 언어로: 불리한 부호 계급에 제한한 활성 점유율은 노이즈 tick 활성 점유율을 넘지 못한다.

**따름정리 2 (장 일반화, 적응 임계).** 게이트 편향 $c_t=c_0+\kappa\phi_i(t)$는 판정 전에 정해지는 $\mathcal F_{t-1}$-가측량이라고 가정한다. 현재 계급과 $z_t$는 $\mathcal F_{t-1}$와 독립이고, $\eta_t$의 과거 조건부 법칙은 계급과 $z_t$에 무관하게 동일하다고 가정한다. 그러면 공통 조건부 생존함수 $S_t$에 대해 interlacing이 tick별로 성립한다. $\min(a,b)\ge a+b-1$과 기대값을 취하면 $\bar q\ge\bar p_++\bar p_--1$이고, 따라서 $\bar p_\pm\ge 1-\delta$, $\bar q\le\delta$는 $\delta\ge\tfrac13$을 강제한다. 이 예측가능성과 조건부 동일분포가 없으면 결론은 성립하지 않는다. 예를 들어 $u=\xi=1,c=0$에서 신호 잡음을 $z=+1$일 때 $-0.5$, $z=-1$일 때 $2$로 두고 노이즈 잡음은 그 혼합분포로 두면 $p_+=p_-=1$이지만 $q=1/2$라 interlacing이 깨진다. $\square$

**A-SAL과의 관계.** 짝함수(에너지) 게이트 $g=\sigma(a\lVert m\odot x\rVert^2+b)$는 $z$-불변이므로 $p_+=p_-$이고 interlacing 양끝이 붕괴하지 않는다 — 진폭 분리가 있으면 $p_\pm\to1$, $q\to0$ 동시 달성(존재 증명은 수치 I4). V14 A-SAL 비형식 논증("$u^\top x$ 부호 반전, 상시 열림 = 노이즈 덮어쓰기")이 이로써 분포-자유 정리로 승격된다.

수치: `artifacts/verify_cf5.log` — I1 가우시안 정확 CDF 격자($61^2$)에서 interlacing 위반 0; I2 Laplace/uniform/student-t(3), 무작위 $u,\xi$, 20 trial × $4\times10^5$ 표본에서 위반 0/20; I3 적응 임계의 $\bar q\ge\bar p_++\bar p_--1$ 위반 0/20; I4 에너지 게이트 $p_\pm=1.000000$, $q=0.0$ (진폭/표준편차 = 8, V14 스케일).

## 숨은 공리·자유도

1. **D1 (스칼라 readout)** — 계약 장 방정식의 차원 불일치를 닫는 공리. CF-3 결합에는 Lipschitz 성질까지 필요하다. $r$의 선택(노름, 성분, 임계)은 자유도이며 $\phi$ 스케일 $\bar r/\lambda$를 통해 $\theta_s$ 분류에 직접 전파된다 — CF-4 killing test (iii)의 근거.
2. **S3 (경성 게이트)** — 계약 식의 로지스틱 $\sigma$와 "동결 = 정확 항등" 요구는 그대로는 모순; S3가 유일하게 확인된 해소책이고 CF-2(iv)가 그 비용($\eta_c/(\rho g_{\min})$ 발산)을 정량화한다.
3. **A-E2R (외생 게이트·공통 재생 쓰기)** — CF-3 유일성의 부하 공리. $\tilde s=s$ 반례가 외생 게이트만으로 부족함을 증명한다. 내생 결합에서는 존재·유일성이 미완성이다.
4. **CF-5의 공통 잡음 법칙·예측가능성** — 현재 잡음은 계급과 $z$에 조건부 동일분포이고 적응 임계는 현재 표본을 보기 전에 정해져야 한다. 이를 버리면 위의 명시적 반례가 정리를 깨뜨린다.
5. $\theta_g,\theta_s$는 사전등록 자유 모수 — 어떤 정리도 이 값들을 유도하지 않는다.

## 경계·반례·교차 예측

- **반례(의도된 것)**: 게이트가 영원히 닫힌 궤적은 전부 고정 궤적 — CF-2가 유일 고정점을 주장하지 않는 이유이며 NISCC-6A 교체의 실체.
- **브리지 반례**: V14 route L 원형은 1차원에서 $h^+=(1+gv)h$가 되어 유계성에 반례를 갖는다. "route L에 CF-1~3이 자동 상속된다"는 부모 주장은 삭제하고 경성 게이트+투영 변형 또는 슬롯-readout 변형만 보존한다.
- **경계**: $\phi$ 유계는 2-노름(성형 그래프에서 $\ell_\infty$ 논증 실패); CF-3 유일성은 외생 게이트와 공통 재생 쓰기까지; CF-2(iv)의 $1/\rho$ 법칙은 명시한 Bernoulli 스칼라 모형에만 정리; CF-5 따름정리 2의 상수는 $\tfrac13$(개선 여지, 결론 불변).
- **교차 예측 1**: 구현체에서 닫힘 구간 길이를 늘려도($\rho\downarrow$) 상태 오차가 늘지 않아야 한다(경성 게이트, CF-2(iii)); 연성 게이트로 바꾸면 오차가 $1/\rho$에 비례해 증가해야 한다(CF-2(iv)) — 구현 단계 판별 실험.
- **교차 예측 2**: 열림률 매칭 무작위 게이트 대조군은 CF-3의 $\bar\pi$ 수렴은 재현하되(공리 동일) CF-4 통과는 재현하지 말아야 예측이 특이적.
- **교차 예측 3**: 선형 게이트 구현체는 부호 균형 신호에서 임계를 어떻게 조정해도 (미탐 + 오탐) 합을 $\tfrac12$(정적)/$\tfrac13$(적응) 아래로 못 내린다 — 음성 대조.

## P0 / P1 / P2

- **P0: 없음.** CF-3의 과대 전제는 $\tilde s=s$ 완전 반례로 제거하고 A-E2R·A-E3 하의 좁은 정리로 교체했다. CF-4는 계약 스스로 예측으로 정식화할 것을 요구했고 그대로 판정했다.
- **P1-1 (정본 12_Equation.md A.2.2, CF-4 유도 경로 후보의 붕괴).** A.2.1의 사상 $B$의 실제 고정점 $(0.048708,0.796565,0.154727)\ne p^*$, $\lVert\cdot\rVert_\infty$ 오차 0.5344. $B_b=\alpha_sD_{\text{eff}}=0.1547$은 $p^*_b=0.6891$이 아니라 수축상수 $\rho=0.155$와 일치 — 전사 오류 추정. 닫는 데 필요한 최소 보조정리: $p^*$ 전체(특히 $0.2623, 0.6891$)를 고정점으로 갖는 사상 $B$의 올바른 정의와 그 유도, 또는 A.2.2를 $a$-성분 한정으로 축소 정정. 이 P1은 본 run 결론(자유 예측 판정)을 바꾸지 않고 오히려 강화한다.
- **P1-2 (계약 장 방정식의 차원 결손).** $\phi$(스칼라) 방정식의 소스로 $s\in\mathbb R^w$(벡터)가 직접 들어감 — 공리 D1로 해소하고 명시. 구현 계약서에 D1의 $r$ 선택을 사전등록할 것.
- **P1-4 (V14 route L 직접 상속 결손).** route L 원형은 S2·S3 밖이고 1차원 발산 반례가 있다. 직접 상속 주장은 제거했다. 구현은 경성 게이트+유계 투영 또는 외생 슬롯+HRR readout 변형만 허용한다.
- **P2**: (i) 12_Equation 8.1의 $a^\star=0.04865$ vs 재계산 $0.0487077$ ($5.8\times10^{-5}$, 반올림 관행 — constants.py의 0.0487은 정합); (ii) A.2.2 Jacobian 식의 잉여 $(1-a)$ 인자 (결론 불변); (iii) $\theta_s$-상 분류가 $\phi$ 스케일 $\bar r/\lambda$에 종속 — toy에서 $\pi_F=0$이 그 증상, 사전등록 시 $\theta_s$를 $\bar r/\lambda$ 단위로 무차원화할 것; (iv) CF-2(iv)의 일반 $1/\rho$ 문구를 조건부 정리와 검증 예측으로 분리함.

## 재현

```
cd C:/Users/dongh/OneDrive/Desktop/Clarus-Equation
./.venv/Scripts/python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf1_cf2.py
./.venv/Scripts/python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf3.py
./.venv/Scripts/python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf4.py
./.venv/Scripts/python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf5.py
```

로그: `artifacts/verify_cf1_cf2.log`, `artifacts/verify_cf3.log`, `artifacts/verify_cf4.log`, `artifacts/verify_cf5.log` (numpy 2.3.5, 단일 스레드, 결정적 시드).

Status: COMPLETE
