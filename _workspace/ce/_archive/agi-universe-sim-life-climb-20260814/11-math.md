# 11-math — universe-simulator life-climb

Status: COMPLETE

대상: 계약 `_workspace/ce/agi-universe-sim-life-climb-20260814/00-contract.md`의 등록 주장.
정의역: $Z=(m,b,q)\in[0,1]^3$, $\kappa\in[0,1]$, 명목 유리 모수
$r=9/2$, $\lambda=5/2$, $\rho=1/5$, $\delta=1/10$, $s=1/2$, $\mu=3/32$, $\eta=1$, $\theta_D=3/4$, $K=1$.
전제: 12개 exact obligation은 재증명하지 않는다. L4--L8은 판정 대상이 아니다.

## 판정표

| ID | 지위 | P | 판정 |
|---|---|---|---|
| P-D0 | 정의 | — | 유한 커널 $U$는 이산 한 스텝. agent API·보상 항 없음. |
| P-D1 | 정의 | — | 사다리 L0--L8은 계획. 이 run 범위는 L0--L3 구성. L4--L8 승격 없음. |
| P-D2 | 정의 | — | $q_{\mathrm{label}}$만 사용. $q_{\mathrm{homeo}}$, $q_{\mathrm{ext}}$ 부재. $q$는 유전자가 아님 (총정리 해석 반증을 재인용). |
| P-D3 | 정의 | — | $F_0$와 명목 모수는 출처 식과 일치. |
| P-D4 | 정의 | — | $U_t=(t,E_t,\mathcal Z_t,\phi_t)$. $\phi_t=0$은 이 run에서 허용. |
| P-D5 | 정의 | — | $\nu=0$, $E_\star=1$이면 $E_t\equiv E_0$. $E_0=1$이 출처 chemostat. |
| P-D6 | 정의 | — | `step(E)` 인터페이스. $E=1$에서 $Z\mapsto F_0(Z)$로 두는지가 호스트 항등식. |
| P-D7 | 정의 | — | $F_0=F_{\kappa=0}$는 점별 정규화. $\kappa>0$ 법칙은 가설. |
| P-A0 | 공리 | — | 설계 공리. 정리·우주론 도출이 아님. |
| P-A1 | 공리 | — | 설계 공리. 연산자 추가 기준. 정리로 승격하지 않음. |
| P-A2 | 공리 | — | occupancy $(0.0487,0.262,0.689)$와 $q_{\mathrm{ext}}\mapsto\Omega_b$는 이 run 밖. |
| P-C1 | 정리 (인용) | — | 출처 분류 인용은 충실. $q$는 (P.1)--(P.3)에 없음. 12 obligation 재증명 없음. |
| P-C2 | 정리 (인용) | — | $F_0$에 $q\to(m,b)$ 채널 없음. 진화 정리로 올리지 않음. |
| P-C3 | 산출 | P1 | 호스트 항등식은 well-posed (같은 맵, $E\equiv 1$). 구현하지 않음. |
| P-E1 | 미완성 | P1 | 존재 주장. 후보 대수는 아래. 닫지 않음. |
| P-E2 | 미완성 | — | 결합은 자율 $A$를 함의하지 않음. 접지 금지. |
| P-H1 | 미완성 | P0 해소(전칭 삭제) | 부모 $\forall\kappa\in(0,1]$ 충분구성은 삭제. 남은 가설은 $I_r=(0,86/315)$ 상자. 정리 승격 없음. |
| P-H2 | 미완성 | P1 | $\kappa\in\{1/4,1/2\}$에서 양성 분열 근 존재. $\kappa=1$ 끝점만 분열 게이트 실패. |

예측·경험식 승격 없음. AGI·우주론·역사적 최초 생명·세 항의 보편 필요성 없음.

## P-C1 인용 충실성

출처 한 줄 (`08_원시생명_존재정리.md`):

> 주어진 deterministic, chemostatted, selected-daughter hybrid map에는 매 step 분열하는 양의 고정상태가 정확히 3개 있고, 그중 transmitted-state $q=1/4,3/4$인 두 점이 국소 점근 안정하다. 두 점으로 수렴하는 양의 부피 basin과, 반대로 즉시 소멸하는 양의 부피 basin이 모두 존재한다.

정리 2: cube 고정점은 정확히 6개,
$(m,b)\in\{(0,0),(1/2,1/2)\}$, $q\in\{1/4,1/2,3/4\}$.
총정리 표: (2) 6개, (3) 양의 매-step 분열 3개, (4) LAS는 $q=1/4,3/4$ 두 점, (8) 재귀 basin 부피 하한 $2/99$, (9) 소멸 wedge 면적 $1/10$.

계약 문장(여섯 고정점 / 양성 분열 3개 / LAS는 $q\in\{1/4,3/4\}$ / 양부피 재귀 basin / 양면적 소멸 wedge)은 위 문장과 표에 대응한다. 12개 obligation은 재증명하지 않았다.

$q$는 (P.1)--(P.3)에 없다. 구현 `_raw_predivision(mass, boundary)`와 경계 갱신도 `heredity`를 받지 않는다. (P.4)만 $q$를 갱신한다.

표시식에서 독립으로 다시 푼 $F_0$ 고정점 대수(인용 대조, obligation 재증명 아님): $q$-고정점은
$f(q)-q=(2q-1)\{s q(1-q)-\mu\}=0$이므로 $\{1/4,1/2,3/4\}$.
분열 가지 질량 이차식 $18m^2-5m-2=0$의 근은 $1/2,-2/9$. 양성 분열 질량은 $1/2$ 하나. 곱집합으로 6점, 양성 분열 3점.

## P-C2

출처 본문: 현재 $q$는 $(m,b,d)$와 Cartesian product로 분리되어 있어 genotype–phenotype coupling·자연선택으로 읽으면 안 된다.
총정리: “현재 식이 evolution을 증명한다”는 해석 반증, 이유 $q\to\mathrm{phenotype}\to\mathrm{descendant}$ 채널 없음.
$F_0$를 진화 정리로 올리지 않는다.

## P-C3 (구성, 구현 없음)

(P.6)에서 $\nu=0$이면 $E_{t+1}=E_t$. $E_0=1$이면 $E_t\equiv 1$.
출처 $F_0$는 $E$를 인자로 갖지 않는다. 호스트 항등식은

$$
\mathrm{step}(E=1)(Z)=F_0(Z)
$$

로 두는 인터페이스 규약이다. 같은 초기 $Z_0$에 대해 이산 궤적은 유일하다. 유한 유리 맵의 복사로 구현 가능하므로 구성 주장으로서 well-posed다. 생물 주장이 아니다. byte-identity와 허용오차 $10^{-15}$는 G-HOST 구현 게이트이지 이 항등식의 수학적 공백이 아니다.

## P-E1 / P-H1 / P-H2 독립 대수

(P.4)는 두 후보에서 불변이므로 $q$-고정점은 그대로 $\{1/4,1/2,3/4\}$다.
명목 $\rho,\delta,\lambda$에서 분열 고정점($m>0$, $\widetilde m=2m$)은

$$
b=\frac{2m}{1+2m}=\frac{1-r+\lambda+rm}{\lambda}.
$$

정리하면

$$
2r\,m^2+(2-r)m+\Bigl(\frac72-r\Bigr)=0,
\qquad
\Delta_r=9r^2-32r+4.
$$

명목 $\rho$ 채널에서는

$$
9\rho\,m^2+(9\delta-7\rho)m-2\delta=0.
$$

한 스텝 소멸 집합 $\{1+r(1-m)-\lambda(1-b)\le 0\}\cap[0,1]^2$의 면적은 $r=9/2$에서 $1/10$.
$q=1/2$이면 (P.7)·(P.8) 모두 $r$이 명목이므로 면적 $1/10\ge 1/20$. 출처 소멸 wedge를 구성이 깨지 않는다.

### 부호

$$
r(q)=\frac92\bigl(1+\kappa(2q-1)\bigr),\qquad
\rho(q)=\frac15\bigl(1+\kappa(2q-1)\bigr).
$$

$\kappa\in(0,1)$, $q\in[0,1]$이면 둘 다 양수.
$(\kappa,q)=(1,0)$에서만 $r=\rho=0$. 그 점은 $q$-고정점이 아니다. $q=0\mapsto 3/32$.
출처 정리 8의 $r=0$, $\rho=0$ ablation과 같은 모서리이나, 궤적이 그 모서리에 머물지 않는다.

$q$-고정점에서는 $\kappa\in(0,1]$ 전부 $r\ge 9/4>0$, $\rho\ge 1/10>0$.

### P-H1 — P0

$q=1/4$에서 $r=(9/2)(1-\kappa/2)$. 분열 질량이 $\theta_D/2=3/8$에 닿는 $\kappa$는

$$
\kappa=\frac{86}{315}\approx 0.2730,\qquad r=\frac{136}{35},\qquad (m,b)=\Bigl(\frac38,\frac37\Bigr).
$$

$\Delta_r=0$은 더 뒤($\kappa\approx 0.477$)다. 구속은 먼저 분열 게이트다.

등록값 $\kappa=1/2$: $r(1/4)=27/8$, $\Delta_r=-95/64<0$. 실근 없음.
등록값 $\kappa=1$: $r(1/4)=9/4$, $\Delta_r=-359/16<0$. 실근 없음.
추가: $\kappa=1$, $q=3/4$에서는 양성 분열 근이 있으나 $1+\mathrm{tr}+\det<0$이라 $2\times 2$ Jury–Schur 실패.

부모 명제 범위: “$r(q)$ 한 파라미터 변조가 모든 $\kappa\in(0,1]$에서 P-E1.2의 충분구성이다.”
반례 값: $\kappa=1/2$, $q=1/4$, $r=27/8$, $\Delta_r=-95/64$.
G-COUPLE 등록집합 $\{1/4,1/2,1\}$ 중 $\kappa=1/2,1$은 이 후보에서 즉시 불가.

$\kappa=1/4$는 아직 대수가 죽지 않는다. 정확 점
$(m,b,q)=(7/18,7/16,1/4)$, Jury $469/5760$, $12641/5760$, $331/384$ 모두 양수.
$q=3/4$ 쪽도 격자에서 선형 Schur는 통과. 비선형 LAS·basin·P-E1.3은 미검사.

### P-H2 — 즉시 불가는 아님

$q=1/4$에서 $m=3/8$이 되는 $\kappa$는 $86/87\approx 0.9885$, $\rho=44/435$.
$\kappa=1/4,1/2$ 모두 양성 분열 근 + 선형 Schur.

정확 점: $\kappa=1/2$, $q=1/4$, $(m,b)=(4/9,2/5)$, Jury $7/60$, $107/60$, $21/20$ 모두 양수.

$\kappa=1$, $q=1/4$: 유일한 양성 근 $m=(\sqrt{19}-1)/9$.
$19<(35/8)^2$이므로 $m<3/8$. 분열 가지에 없음. 끝점 제한이지, 열린 $\kappa$-상자 전체의 즉시 불가는 아니다.

$q=1/2$ 소멸 면적은 양쪽 후보 모두 $1/10$.

P-E1은 여전히 미완성이다. P-H1을 $\kappa\le 86/315$로 좁히거나, P-H2를 $\kappa\le 86/87$로 쓰거나, 문턱·누설·두 딸 생존 등 다른 채널이 남는다. 비선형 불변사각형·수축·$T=32$ $q_0$-부호 통계는 최소 보조정리로 남아 있다.

## P-E2

출처 $C_{\mathrm{strict}}=G\land D\land R\land H\land V\land A\land M$.
$A$는 세대마다 실험자가 extrusion·lysis·reset을 하지 않음.
P-E1이 살아도 외부 호출자 `step`에만 전진하는 맵은 experimenter-free lineage가 아니다.
결합은 $V$ 후보이지 $A$의 함의가 아니다. 접지하지 않는다.

## 무차원

등록 기호 $m,b,q,E,r,\lambda,\rho,\delta,s,\mu,\eta,\theta_D,K,\kappa,\nu$와 확률·면적·개수비는 모두 무차원.
$t$는 이산 틱, $d$는 지시함수, $\phi,E_\star\in[0,1]$, $T=32$는 걸음 수.
(P.1)--(P.8)에 $\exp/\log$ 없음. $q_{\mathrm{ext}}$, $\Omega_b$, occupancy 삼분율은 식에 없음.
`3_부트스트랩.md`의 $q_{\mathrm{ext}}\mapsto\Omega_b$는 우주론 branch 공리로 남고 여기 쓰이지 않는다.

## 숨은 공리·자유도

1. P-A0--P-A2 설계 공리.
2. selected-daughter, 이산 시간, chemostat $E\equiv 1$ ($\nu=0$).
3. 입방체 clip과 $\theta_D$ 하이브리드 분기.
4. 호스트 `step(E)`는 $E=1$에서 $F_0$와 같다는 인터페이스 규약. $E$는 $F_0$에 들어오지 않는다.
5. 명목 유리 모수는 모델 선택.
6. (P.7)·(P.8)의 선형 $2q-1$ 변조는 모델 선택. $\lambda,\theta_D$, 두 딸 생존 등 다른 채널이 같은 $\kappa$ 자격을 가진다.
7. 면적은 $(m,b)$ 단위제곱의 르베그 측도.
8. 선형 Jury는 비선형 LAS가 아니다.
9. 이미 있는 결합 toy(`10_원시생명_결합유전선택정리.md`)는 이 계약의 $F_\kappa$가 아니다. PREDECESSOR 없음.

## P0 / P1 / P2

### P0

1. (해소, 전칭 부모) P-H1이 모든 $\kappa\in(0,1]$에서 P-E1.2의 충분구성이라는 주장.
   반례: $\kappa=1/2$, $q=1/4$, $r=27/8$, $\Delta_r=-95/64<0$. 계약 수정 후 이 부모는
   활성 문장에 없다. 남은 가설은 $I_r=(0,86/315)$. 대수를 다시 풀지 않았다.

P-E1 자체(어떤 $F_\kappa$의 존재)에 대한 P0는 없다.

### P1

1. P-H2는 $\kappa=1$에서 $q=1/4$ 분열 고정점이 없다. 허용 상한 $\kappa\le 86/87$.
2. $(1,0)$에서 $r=\rho=0$. $q$-고정점은 아니나 $\kappa=1$ 폐구간 균일 양성은 거짓.
3. 살아 있는 $(\kappa,q)$ 격자에서도 비선형 LAS, 재귀 basin, P-E1.3($T=32$ 후손·basin 지표)은 공백. 최소 보조정리: $F_\kappa$의 불변사각형+수축, 그리고 $q_0-1/2$ 부호에 대한 등록 통계.
4. P-C3 byte-identity는 구현 게이트.
5. P-E1의 “box 안의 모든 $\kappa\in(0,1]$” 문장은 $\kappa$-구간을 상자로 제한할 수 있어,  surviving route는 $\kappa$ 상한을 명시해야 한다.

### P2

1. (P.7)·(P.8)이 기호 $r,\rho$를 함수와 상수로 겹쳐 쓴다.
2. “LAS only at $q\in\{1/4,3/4\}$”는 점의 성질을 $q$ 값에 붙인 약기. 출처 한 줄과 같음.
3. 출처 한 줄은 소멸을 “양부피 basin”이라 하고 정리 6은 면적 $1/10$. 계약은 양면적으로 맞춰 두었다.

## 재현

```
python _workspace/ce/agi-universe-sim-life-climb-20260814/artifacts/verify_coupling_algebra.py
```

산출: `_workspace/ce/agi-universe-sim-life-climb-20260814/artifacts/verify_coupling_algebra.txt`

생산 모듈을 import하지 않는다. 출처 12 obligation 재실행은 하지 않았다.
