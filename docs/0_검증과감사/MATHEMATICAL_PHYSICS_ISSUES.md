# CE 수학·물리 정합성 문제와 실제 교체 결과

## 1. 감사 범위

2026-08-06 최신본은 과거 오류를 상태표지만 바꾸어 보존하지 않고, 성립하는
정리·작용·측도·수치 사슬로 교체했다. 기계 검증은
[`run_full_consistency_gate.ps1`](run_full_consistency_gate.ps1), 단일 수치
원천은
[`CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`](CANONICAL_NUMERIC_MANIFEST_2026-08-06.json)이다.

## 2. 수학 교체 장부

| ID | 과거 불일치 | 최신 교체 | 검증 |
|---|---|---|---|
| M1 | $S(D)$ 지수형에 정칙성 가정 누락 | 연속/측정가능 양의 character 정리와 optical-depth 단위 분리 | Cauchy functional proof |
| M2 | $d(d-1)/2=d$에서 $d=0$ 삭제 | $d\ge1$ nonzero natural-isomorphism model class를 명시 | 해 전수검사 |
| M3 | $P$와 energy-weighted Gamma CDF 동일시 | $P_a-Q_a=t^ae^{-t}/\Gamma(a+1)$와 에너지 편향 측도 | 수치 적분 회귀 |
| M4 | $D(x)=-\ln(1-x)/x$ 변수 혼용 | $D(x)=-\ln x/(1-x)$; complement 변수 별도 표기 | 역함수 residual |
| M5 | bootstrap “유일해” | $D\le1$ 한 해, $D>1$ 두 해; $I_D$의 저분율 해만 수축 유일 | Lambert-$W$/bisection |
| M6 | 반복식을 물리 시간으로 해석 | $V_D=x^2/2-e^{-D(1-x)}/D$와 logit gradient flow | Lyapunov 감소 |
| M7 | Lorentzian에 elliptic 최대원리 사용 | Lorentzian Cauchy energy와 Euclidean elliptic/heat/Poisson 분리 | 부호·경계 gate |
| M8 | 감쇠계의 에너지 보존 주장 | 폐쇄계 보존, bath 포함 총보존, Euclidean gradient flow 분리 | stress 장부 |
| M9 | 곱적 사상에서 $I(P)=P$ 자동 | 일반해 $P^c$, $c=1$에는 접선 정규화 추가 | 함수방정식 검산 |
| M10 | $\epsilon\leftrightarrow1-\epsilon$ 대칭 부호 오류 | $g(1-\epsilon)=-g(\epsilon)$와 대칭/방향전이 branch 분리 | 대칭 대입 |

## 3. 공변 물리 교체 장부

### 3.1 장·곡률·readout

과거에는 독립 scalar, Hessian readout과 Ricci scalar가 같은 기호로
오갔다. 최신 타입은

$$
\phi(x)\ne\Phi_H[\gamma,\eta]\ne R\ne q
$$

로 고정한다. Lorentzian EFT는

$$
\mathcal L_\phi=-\frac{Z_\phi}{2}(\nabla\phi)^2-V(\phi)
-\frac\xi2R\phi^2-\frac{\lambda_{H\phi}}2\phi^2H^\dagger H+\cdots
$$

이고 stress tensor는 metric variation으로만 정의한다.

### 3.2 결합상수 순환

세 관계를 “3식 3미지수”라고 했지만 $s_W^2$까지 세면 네 변수다. 최신본은
두 트랙으로 닫는다.

- Track A: $\alpha_s^{\overline{\rm MS}}(M_Z)=0.1180$ 입력,
  등록 출력 $s_A^2:=4\alpha_s^{4/3}=0.2315097758$
- Track B: $\alpha_{em}^{\overline{\rm MS}}(M_Z)=1/127.95$ 입력,
  두 양의 근과 hierarchy branch 출력

입력으로 쓴 관측량을 같은 트랙의 성공으로 재집계하지 않는다.

### 3.3 기저 의존 혼합량

질량행렬 한 원소 대신

$$
\delta_N=\operatorname{Tr}(P_WP_Z)
\operatorname{Tr}(P_BP_Z)=s_A^2(1-s_A^2)
$$

를 사용한다. charged projector는 neutral 부분공간과 직교하므로 이
functional에 정확히 0을 주며 전체 $W^\pm$ spectrum에서는 보존된다.
여기서 \(s_A^2\)를 물리적 \(s_W^2\)의 어느 scheme과 동일시할지는
RG·threshold·scheme matching을 요구하는 별도 bridge다.

### 3.4 B2 바리온 연결

경로 개수와 바리온 에너지를 동일시하던 문장을 폐기했다. 관측면에서

$$
x=\frac{\langle E_b\rangle}{\langle E_{\rm tot}\rangle}
$$

로 energy-weighted sample space를 정의하고, 평탄 FLRW에서
$\Omega_b=x$를 증명한다. unitary two-sector 실현과 다른 epoch 전달식도
[`BRIDGE_B2_DERIVATION.md`](BRIDGE_B2_DERIVATION.md)에 포함했다.

### 3.5 strong CP

instanton action을

$$
S_{\rm inst}=8\pi^2/g_s^2=2\pi/\alpha_s
$$

로 교정했다. CP-even singlet은 spectator이고, 해결 branch는
shift-symmetric pseudoscalar의

$$
(\bar\theta+a/f_a)G\widetilde G
$$

와 QCD potential을 사용한다. explicit breaking 품질조건과 nEDM gate를
같이 둔다.

### 3.6 flavour·중성미자

CKM/PMNS 원소별 거듭제곱식을 한 unitary matrix construction으로
교체했다. 중성미자는 gauge-invariant matrix Weinberg operator와
scotogenic UV benchmark를 사용한다. 질량차를 넣어 얻은
$\sum m_\nu$는 변환값으로 장부화한다.

### 3.7 인플레이션

large-$\xi$ 공식을 $\xi\simeq0.49$에 그대로 쓰던 계산을 finite-$\xi$
Einstein-frame 식으로 교체했다. 최신 exact benchmark는

$$
\xi=0.4904868132,
\quad n_s=0.96617114,
\quad r=0.00434561,
\quad\lambda_4=1.3434991\times10^{-10}.
$$

$A_s$는 $\lambda_4$ 정규화 입력이다. exact $Z_2$ 재가열은 single-particle
decay가 아니라 annihilation/preheating 방정식을 사용한다.

### 3.8 바리오제네시스

$d=3$이나 CP-even wall이 CP source를 자동 생성한다는 주장을 제거했다.
완성 EFT는 $(H^\dagger H)W\widetilde W/\Lambda_{\rm CP}^2$, thermal
bounce, diffusion과 sphaleron washout을 한 수송계로 연결한다.

## 4. 최신 수치 사슬

Track A에서

$$
\begin{aligned}
s_A^2&=0.2315097758079336,\\
\delta_N&=0.1779129995132939,\\
D_N&=3.177912999513294,\\
x&=0.04863825851598632,\\
R&=0.3782386966438831.
\end{aligned}
$$

따라서

$$
\boxed{(\Omega_b,\Omega_{DM},\Omega_{DE})
=(0.0486382585,0.2610881744,0.6902735671)}
$$

이고 합은 1이다. 과거 $R=0.38063$은 표시된 식에서 나오지 않으므로
current vector가 아니다.

## 5. 통계·관측 정합성

개별 marginal 중앙값 근접을 full model 적합으로 부르던 관행을 교체했다.
최신 density vector의 explicit-override 결과는

| 설정 | $\chi^2$ | dof | $p$ | 판정 |
|---|---:|---:|---:|---|
| external $H_0,r_d$ | 40.20145 | 13 | $1.2828\times10^{-4}$ | reject |
| EH-hybrid $r_d$ | 41.19455 | 13 | $8.8602\times10^{-5}$ | reject |

따라서 고정 background benchmark는 내부적으로 계산 가능하지만 현 공동
자료 gate를 통과하지 않는다. 이 실패는 문서에서 숨기지 않으며, 다음
모형은 공변 dark stress의 background와 perturbation을 함께 풀어야 한다.

## 6. 공학·계산 교체

| 분야 | 과거 오류 | 최신 교체 |
|---|---|---|
| Casimir | $E/A$를 두 판의 총에너지로 사용 | massive-scalar $E/A$와 $E=A(E/A)$ 분리 |
| 효율 | 출력과 일부 입력의 서로 다른 단위 | 한 cycle의 drive·reset·loss를 joule로 합산 |
| 초전도 | 45 meV를 45 neV로 변환 | band cutoff·Wilson coefficient·Eliashberg/Keldysh 연결 |
| AGI | simplex를 보존하지 않는 affine map | 양의 normalized self-map과 실제 Jacobian bound |
| Riemann | phase convention 혼용 | canonical phase와 real/conjugate spectral weight |
| 블랙홀 | 측정 $G_N$과 bare coupling 혼용 | scalar-tensor BVP와 measured $G_N$ matching |

## 7. 자동 gate

```powershell
powershell -ExecutionPolicy Bypass -File docs/0_검증과감사/run_full_consistency_gate.ps1
```

통과조건은 다음과 같다.

- 212개 Markdown의 H1, 링크, math/code fence
- 알려진 strong-CP·단위·기호 회귀 0
- Track A/B 근과 bootstrap branch 재계산
- Gamma count/energy 측도 차이
- 밀도 합과 $\eta_b$ 변환
- finite-$\xi$ 인플레이션 재계산

## 8. 완성의 정확한 경계

문서의 수학적·작용 수준 정합성은 명시된 모형 안에서 닫혔다. 자연에 대한
최종 판정은 별도다. 검증을 통과하지 않은 fixed cosmology를 통과했다고
쓰지 않고, 아직 측정하지 않은 EDM·GW·collider 결과도 성공으로 세지 않는다.
이 구분은 주장을 약하게 만드는 표지 작업이 아니라, 거짓 등식 없이 실제
계산과 반증이 가능한 이론 문서가 되기 위한 완성 조건이다.
