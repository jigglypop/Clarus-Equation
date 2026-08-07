# 1장 정합성 원장: 입력·유도·상태·검증

## 0. 목적과 범위

이 문서는 1장 A·B·C에서 반복되는 수치와 주장 지위를 한곳에 고정한다.
삭제된 전역 JSON manifest나 과거의 212문서 gate 없이도 현재 checkout의
1장을 독립 재계산하는 것이 목적이다. 이 원장은 다음을 구분한다.

1. 외부 관측 또는 benchmark 입력
2. 정의와 대수로 재계산되는 수치
3. 모형 선택
4. 물리량과의 bridge
5. 관측 gate의 통과·미완성·기각 상태

이 문서의 통과는 전체 CE 문서군이나 자연에 대한 검증을 뜻하지 않는다.

## 1. convention과 입력 장부

| 항목 | 이 장의 값 | 역할 | 출처·주의 |
|---|---:|---|---|
| metric signature | $(-,+,+,+)$ | `Convention` | A·C 공통 |
| reduced Planck mass | $M_P=(8\pi G)^{-1/2}$ | `Convention` | unreduced mass와 혼용 금지 |
| 단위계 | EFT·inflation에서 $c=\hbar=1$ | `Convention` | 경로적분 지수에는 $\hbar$ 표시, 수밀도 변환에서는 SI 복원 |
| $C_{\rm CE}:=\alpha_s+\alpha_w+\alpha_{em}$ | $1/(2\pi)$ | `Selection` | 같은 scale·scheme에서만 쓰는 경계 ansatz |
| $\alpha_s^{\overline{\rm MS}}(M_Z)$ | $0.1180$ | Track-A calibration input | [PDG 2025 QCD review](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-qcd.pdf)의 $0.1180\pm0.0009$ |
| $\alpha_{em}^{\overline{\rm MS}}(M_Z)$ | $1/127.95$ | Track-B benchmark input | 동일 scale·scheme에서만 결합상수 합 규칙 시험 |
| $H_0$ | $67.4\ {\rm km\,s^{-1}Mpc^{-1}}$ | 밀도 변환 input | [Planck 2018 cosmological parameters](https://doi.org/10.1051/0004-6361/201833910)의 base-$\Lambda$CDM 값; 모형 의존 |
| $T_{\rm CMB}$ | $2.7255\ {\rm K}$ | 광자수밀도 input | [NASA 자료가 인용하는 FIRAS 온도](https://ntrs.nasa.gov/api/citations/20140011029/downloads/20140011029.pdf?attachment=true) |
| $N_{\rm eff}$ | $3.044$ | EH radiation `Selection` | precision recombination 결과가 아닌 표준 방사 성분 가정 |
| $A_s$ | $2.10\times10^{-9}$ | $\lambda_4$ calibration | inflation amplitude를 다시 예측으로 세지 않음 |
| $N_*$ | $57.1999$ | inflation benchmark input | reheating completion 전에는 독립 유도값 아님 |
| $(s_{12},s_{23},s_{13}),\delta_q$ | $(0.22724210,0.04168209,0.00372494),1.2\,{\rm rad}$ | flavour `Calibration input` | B의 CKM benchmark; 독립 예측 아님 |

$\alpha_s$의 불확도를 전파하지 않고 열 자리 숫자를 표시한 값은 **중앙값
재계산 계약**이다. 열 자리 물리 예측 정밀도를 뜻하지 않는다.

## 2. Track-A 등록 사슬

입력 $\alpha_s=0.1180$에서

$$
s_A^2:=4\alpha_s^{4/3}
=0.2315097758079336,
$$

$$
\delta_A:=s_A^2(1-s_A^2)
=0.17791299951329392,
$$

$$
D_A:=3+\delta_A
=3.177912999513294
$$

를 얻는다. $s_A^2$와 $\delta_A$는 Track-A 내부 등록량이다. 물리적
projector의

$$
\delta_{\rm proj}=s_W^2(1-s_W^2)
$$

와 정의상 동일시하지 않는다. $\delta_{\rm fold}:=\delta_A$는 모형 `Selection`이고,
$\delta_A\leftrightarrow\delta_{\rm proj}$는 RG·threshold·scheme
`Open Bridge`다.

$D_A$를 next-generation 행렬 $\mathsf K$의 공통 행합으로 쓰는 단계도
CE+SM 작용에서 아직 유도되지 않은 별도 `Open Bridge`다. 다음 절은 이
공통 행합을 조건으로 넣은 수학적 결과를 기록한다.

## 3. vector bootstrap과 저분율 근

유한 비음수 next-generation 행렬 $\mathsf K$와 type별 독립 Poisson
offspring를 조건으로 최소 소멸확률 벡터는

$$
x_i=\exp\!\left[-\sum_j\mathsf K_{ij}(1-x_j)\right]
$$

을 만족한다. 한 type 또는
$\mathsf K\boldsymbol1=D_A\boldsymbol1$인 공통 행합 균일 sector에서만

$$
x=e^{-D_A(1-x)}
$$

로 줄어든다. 저분율 근은

$$
x_\star=-\frac{W_0(-D_Ae^{-D_A})}{D_A}
=0.04863825851598631.
$$

독립 검산량은

$$
e^{-D_A(1-x_\star)}-x_\star=0
$$

이고 표시 정밀도 밖의 수치 잔차만 허용한다. 안정성 multiplier는

$$
D_Ax_\star=0.1545681540116411<1
$$

이다. 이는 고정점 반복의 안정성이다. 물리 시간진화의 안정성이 아니다.

## 4. 에너지 readout의 조건부 동일시

수학적 근 $x_\star$와 에너지분율 $x_E$는 먼저 분리한다. 양의 정규화
측도 $\mu_D$, 생존 사건 $\mathcal A_D$, 양의 총에너지 readout $H_D$에
대해

$$
x_E(D):=
\frac{\int_{\mathcal A_D}H_Dd\mu_D}{\int H_Dd\mu_D}
$$

로 정의한다. 다음 조건이 모두 있어야 한다.

생존 character의 namespace는
S1(정규화), S2(연결 곱성), S3(정칙성), S4(비자명성)다. 이 네 조건에서
$S(D)=e^{-\kappa_{\rm surv}D}$가 따르며, 아래 에너지 조건 E1--E4와
혼용하지 않는다.

- E1: $H_{b,D}=H_D\mathbf1_{\mathcal A_D}$ 또는 동등한 baryon sector
  projector가 있다.
- E2: 분자와 분모가 같은 초곡면·같은 comoving 영역을 쓴다.
- E3: $x_E:[0,\infty)\to(0,1]$, $x_E(0)=1$이고 곱적·연속이다.
  비자명성은 어떤 $D>0$에서 $x_E(D)<1$이라는 뜻이다.
- E4: $c_E>0$를 fold-to-energy depth 변환으로 두고
  $D_{\rm act}=c_E(1-x_E)D_A$다.
- E3에 같은 함수형 정리를 독립 적용하면
  $x_E(D)=e^{-\kappa_E D}$인 고유 rate $\kappa_E>0$를 얻는다.
  $\kappa_E$와 $\kappa_{\rm surv}$는 자동으로 같은 상수가 아니다.
- invariant rate: $\beta_E:=\kappa_E c_E$이고
  $x_E=e^{-\beta_E D_A(1-x_E)}$다.
- 단위율 `Convention`: energy-depth 좌표에서 $\kappa_E=1$로 둘 수 있지만
  이것만으로 $c_E$ 또는 $\beta_E$가 1이 되지는 않는다.
- matching `Selection`: $\beta_E=1$을 채택한다. 이 값의 동역학적 검증은
  `Open Bridge`다.
- branch `Selection`: 공통 행합 sector의 저분율 근을 고른다.

이 조건과 $\beta_E=1$ 아래에서만 같은 수축구간의 유일성으로

$$
x_E=x_\star
$$

가 따른다. 이 E1--E4와 CE+SM 동역학의 연결은 현재 `Open Bridge`다. 또한

$$
\frac{\Omega_b}{\Omega_{\rm phys}}=x_E
$$

이며 $\Omega_{\rm phys}:=\rho_{\rm tot}/\rho_c$는 곡률을 제외한다.
flat-slice $\Omega_k=0$, 따라서 $\Omega_{\rm phys}=1$을 추가할 때에만
$\Omega_b=x_E=x_\star$다.

## 5. 완전한 밀도 장부와 3-sector 절단

neutrino를 다른 항과 중복 계상하지 않는 convention에서

$$
\Omega_{\rm phys}:=\Omega_b+\Omega_{\rm cdm}+\Omega_{\rm DE}
+\Omega_r+\Omega_\nu,
\qquad
1=\Omega_{\rm phys}+\Omega_k.
$$

Track-A 분할 ansatz는

$$
R_{\rm dark}:=\alpha_sD_A(1+x_\star\delta_A)
=0.3782386966438831
$$

이다. 이는 `Phenomenology/Bridge`다. 일반식은

$$
\Omega_{\rm rem}:=\Omega_{\rm phys}-\Omega_b-\Omega_r-\Omega_\nu
=1-\Omega_k-\Omega_b-\Omega_r-\Omega_\nu,
$$

$$
\Omega_{\rm cdm}=\Omega_{\rm rem}
\frac{R_{\rm dark}}{1+R_{\rm dark}},
\qquad
\Omega_{\rm DE}=\Omega_{\rm rem}
\frac1{1+R_{\rm dark}}.
$$

$\Omega_r=\Omega_\nu=\Omega_k=0$인 late-time truncated 3-sector
benchmark에서는

$$
\Omega_b=0.04863825851598631,
$$

$$
\Omega_{\rm cdm}=0.26108817435761356,
\qquad
\Omega_{\rm DE}=0.6902735671264001.
$$

세 수의 합이 1인 것은 절단 정의의 대수 검산이다. 현재 실행된 cosmology
검증은 CMB·SN·growth 공동 likelihood가 아니라 로컬 `desi-dr2-all`의
13-component compressed DESI DR2 BAO mean/covariance만 쓴 partial gate다.
데이터 벡터·공분산과 forward 식은
[로컬 구현](../../examples/physics/ce_residual_forward_model.py)에 고정돼 있다.
이 결과는 현 문서 수정에 사용한 validation/diagnostic이며 새 untouched
holdout으로 다시 세지 않는다.
external branch는 위 late-time 3-sector 밀도를 쓴다. EH branch는 방사 성분을
그 합 1 위에 중복 가산하지 않고

$$
\Omega_{{\rm rad},0}^{({\rm EH})}
:=\Omega_{\gamma,0}+\Omega_{\nu,{\rm rel},0}
=9.192332265998932\times10^{-5},
$$

$$
\Omega_{\rm rem}^{({\rm EH})}:=1-\Omega_b-\Omega_{{\rm rad},0}^{({\rm EH})},
$$

$$
\Omega_{{\rm cdm},0}^{({\rm EH})}
:=\Omega_{\rm rem}^{({\rm EH})}\frac{R_{\rm dark}}{1+R_{\rm dark}}
=0.26106294726317864,
$$

$$
\Omega_{{\rm DE},0}^{({\rm EH})}
:=\frac{\Omega_{\rm rem}^{({\rm EH})}}{1+R_{\rm dark}}
=0.6902068708981751
$$

로 $\Omega_b+\Omega_{{\rm cdm},0}^{({\rm EH})}
+\Omega_{{\rm DE},0}^{({\rm EH})}
+\Omega_{{\rm rad},0}^{({\rm EH})}=1$을 맞춘다. 이 방사량은 photon과
relativistic neutrino의 합이다. sound horizon과 BAO distance 계산은 모두
이 동일한 4-sector Friedmann 배경

$$
E_{\rm EH}^2(z)
=\Omega_{{\rm rad},0}^{({\rm EH})}(1+z)^4
+\bigl(\Omega_b+\Omega_{{\rm cdm},0}^{({\rm EH})}\bigr)(1+z)^3
+\Omega_{{\rm DE},0}^{({\rm EH})}
$$

을 쓴다. 따라서 $E_{\rm EH}(0)=1$이다. 적합 파라미터는 0개라 dof는
13이고, 판정 경계는 $p\geq0.05$ 통과 구간,
$0.0027\leq p<0.05$ 긴장 구간, $p<0.0027$ 기각 구간이며,
마지막 구간을 validation 상태 `Rejected`로 기록한다.
$w_0=-1,w_a=0,H_0=67.4$에서 결과는

| background·$r_d$ branch | $r_d$ [Mpc] | $\chi^2$ | dof | $p$ | 상태 |
|---|---:|---:|---:|---:|---|
| external input | 147.09 | 40.20145086 | 13 | $1.28283168\times10^{-4}$ | `Rejected` |
| 4-sector EH `Selection` | 151.50842877 | 41.90607733 | 13 | $6.78476334\times10^{-5}$ | `Rejected` |

다. 4-sector EH branch는 $T_{\rm CMB}=2.7255$ K,
$N_{\rm eff}=3.044$와
Eisenstein--Hu drag fit에 의존하며 precision recombination 계산이 아니다.
따라서 두 고정 background는 **지정 BAO-only partial gate에서** `Rejected`다.
CMB·SN·growth 공동 gate는 실행되지 않아 `Open`이다.

## 6. Track-B 조건부 교차근

$\alpha_{em}=1/127.95$를 입력하고 같은 scale·scheme에서

$$
s_{W,B}^2:=4(\alpha_s^{(B)})^{4/3},
\qquad
\alpha_{w,B}:=\frac{\alpha_{em}}{s_{W,B}^2}
$$

라는 matching을 조건부 채택한다. 이 matching은 validation 상태가
`Open`인 `Bridge`이므로 아래 근도 조건부 benchmark다. 그때

$$
f_B(\alpha_s^{(B)})=\alpha_s^{(B)}
+\frac{\alpha_{em}}{4(\alpha_s^{(B)})^{4/3}}
+\alpha_{em}-\frac1{2\pi}=0
$$

을 풀면 양의 근은

$$
\alpha_{s,{\rm low}}^{(B)}=0.0528678687103,
\qquad
\alpha_{s,{\rm SM}}^{(B)}=0.1173186646973
$$

이다. SM-like hierarchy를 사전 선택한 가지에서

$$
s_{W,B}^2=0.22972916798,
\qquad
\alpha_{w,B}=0.03402072544
$$

를 얻는다. Track A 입력과 Track B 출력을 동시에 독립 성공으로 세지 않는다.

## 7. inflation branch 계약

core와 inflation은 scalar field와 비최소결합 기호를 공유하지 않는다.

$$
F_{\rm core}(\phi)=M_P^2-\xi_{\rm core}\phi^2,
$$

$$
F_{\rm inf}(\varphi)=M_P^2+\xi_{\rm inf}\varphi^2,
\qquad
\xi_{\rm inf}:=\alpha_s^{1/3}=0.4904868132.
$$

두 번째 식은 별도 `Selection`이다. plus-sign·reduced-$M_P$·quartic
potential·$N_*=57.1999$·$A_s=2.10\times10^{-9}$ 아래의 finite-$\xi$
배경 적분과 leading-order slow-roll 관측식은

$$
n_s=0.96617114,
\qquad
r=0.00434561,
\qquad
\lambda_4=1.3434991\times10^{-10}
$$

을 준다. $A_s$는 $\lambda_4$ calibration이며 reheating·RG·고차 보정은
`Open`이다.

## 8. 바리온-광자 비 변환

E1--E4 뒤의 $\Omega_b$를 사용하고

$$
\eta_b=
\frac{\Omega_b[3H_0^2/(8\pi G)]}
{m_p[2\zeta(3)/\pi^2][k_BT_{\rm CMB}/(\hbar c)]^3}
$$

를 계산하면

$$
\eta_b^{\rm density}=6.041176330\times10^{-10}
$$

이다. 이는 $H_0,T_{\rm CMB},m_p$를 사용한 density conversion이지
바리오제네시스의 독립 생성 예측이 아니다.
여기서 $3H_0^2/(8\pi G)$는 SI mass-equivalent density다. 에너지밀도
convention에서는 분자와 한 바리온 에너지에 같은 $c^2$가 붙어 상쇄된다.

## 9. 1장 수용 gate

1장은 다음 조건을 모두 만족해야 한다.

1. A·B·C·D의 상대 링크가 모두 존재한다.
2. $R_g$와 $R_{\rm dark}$, $q_E$와 $\varphi$, $\phi$와 $\varphi$,
   $x_\star$와 $x_E$가 분리된다.
3. Hodge 절에 metric·orientation·회전 등변성이 모두 있다.
4. S1--S4와 E1--E4가 서로 다른 namespace로 정의된다.
5. vector-to-scalar 축약에는 한 type 또는 공통 행합 조건이 있다.
6. $\delta_A$, $\delta_{\rm proj}$, $\delta_{\rm fold}$의 지위가 분리된다.
7. core minus-sign와 inflation plus-sign branch가 별도 field/coupling을 쓴다.
8. portal은 no-$1/2$ 규약을 쓰거나 factor-2 변환을 표시한다.
9. 양의 Euclidean 측도를 쓸 때 정칙화·positivity·유한 정규화를 표시한다.
10. $\kappa_E$, $c_E$, $\beta_E=\kappa_E c_E$를 분리하고
    $\beta_E=1$을 `Selection`으로, 그 동역학적 validation을
    `Open Bridge`로 표시한다.
11. 완전한 밀도 장부는 곡률 제외 $\Omega_{\rm phys}$와 $\Omega_k$를 분리한다.
12. 표시 작용에 없는 재가열 채널은 추가 portal을 조건으로만 말한다.
13. DESI DR2 BAO-only partial gate의 수치를 재현하고 이를 full joint로
    과대 해석하지 않는다.
14. 현행 수치 독립 재계산과 기존 구조 회귀 테스트가 모두 통과한다.

집중 검증 명령은 다음과 같다.

두 등록 우주론 branch 자체를 재현할 때는 반올림된 legacy 기본 밀도 대신
반드시 `chapter1` density preset을 쓴다.

```powershell
.\.venv\Scripts\python.exe examples\physics\ce_residual_forward_model.py `
  --density-preset chapter1 --rd-mode external `
  --bao-dataset desi-dr2-all

.\.venv\Scripts\python.exe examples\physics\ce_residual_forward_model.py `
  --density-preset chapter1 --rd-mode early-universe `
  --bao-dataset desi-dr2-all
```

이 두 CLI 경로와 독립 수치 oracle은 공용
`chapter1_canonical_params` factory를 호출한다. 일반 legacy EH 입력의 총밀도가
1에서 벗어나면 forward model은 남는 항을
$\Omega_k=1-\Omega_{\rm rad}-\Omega_m-\Omega_{\rm DE}$로 보존하여 조기·후기
배경을 같은 Friedmann 식으로 닫는다. 등록 Chapter-1 EH preset에서는
$\Omega_k=0$이다.

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_chapter1_document_contract.py `
  tests\test_chapter1_numeric_contract.py `
  -q -p no:cacheprovider
```

기존 구조 회귀까지 합친 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_chapter1_document_contract.py `
  tests\test_chapter1_numeric_contract.py `
  tests\test_core_axioms.py `
  tests\test_bootstrap_solver.py `
  tests\test_dimensionless.py `
  tests\test_core_model_selection.py `
  tests\test_cosmology_ratio_audit.py `
  tests\test_a1_q0_action_bridge.py `
  tests\test_ce_residual_forward_model.py `
  tests\test_clarus_negative_source_search.py `
  -q -p no:cacheprovider
```

통과 개수는 테스트 추가 때마다 바뀌므로 원장에 고정하지 않는다. 두 명령이
현재 checkout에서 exit code 0이어야 한다. 기존 회귀 파일에 남은 과거
benchmark 숫자는 구조 회귀용이며, 현행 1장 수치의 authority는
`test_chapter1_numeric_contract.py`다.
