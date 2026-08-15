# 05o. CE Residual Cosmology Forward Model

## 0. 범위와 출처

이 문서는 CE 성분비를 현재 우주의 경계조건으로 놓았을 때, 평탄 FLRW
배경과 선형 성장 관측량을 계산하는 조건부 forward model을 정리한다.

수치 판본과 Claim ID는
[우주론 판본·주장 원장](../3_상수/00_우주론_원장.md)을 따른다.

- `[공리: 물리 사상]` CE의 $q_{\rm ext}$와 성분 분해 경험식을 오늘의
  $(\Omega_b,\Omega_{\rm DM},\Omega_\Lambda)$ 경계조건으로 읽는다.
- `[공리: 모델 선택]` 평탄 FLRW, CPL 암흑에너지, GR 성장 한계를 채택한다.
- `[공리: 외부 입력]` $H_0$, $r_d$, $\sigma_{8,0}$는 독립 자료에서
  공급한다.
- `[정의]` 거리, BAO 압축량과 공분산 통계량의 계산 규약을 고정한다.
- `[산출]` 위 전제와 정의를 대입하면 $H(z)$, 거리와 선형 성장
  관측량이 결정된다.
- `[경험식]` 수정중력 결합과 같은 자료 기반 scale 최적화는 정리나
  사전 예측으로 사용하지 않는다.
- `[미완성]` CE 내부의 재결합·sound-horizon 유도, 완전한 CMB/SN/BAO
  likelihood, 입자 암흑물질과 검출기 응답은 이 문서에서 닫히지 않는다.

이 구분에서 실행 코드와 수치 회귀는 수식 구현을 검사할 뿐, 물리 사상을
증명하지 않는다.

## 1. 경계조건과 외부 입력

### 1.1 CE 성분비의 사용

`[공리: 런타임 호환 경계]` 현재 코드가
`LEGACY_ROUNDED_RUNTIME_V1`로 읽는 성분비는

$$
\Omega_b=0.0487,\qquad
\Omega_{\mathrm{DM}}=0.2623,\qquad
\Omega_\Lambda=0.6891
$$

이다. 이 세 값의 원시 합은 $1.0001$이다.
$\Omega_b=q_{\rm ext}$의 동일시는 `C-B-LEGACY-01`의 과거 물리 사상이고,
성분비 $\Omega_{\rm DM}/\Omega_\Lambda$ 분할은 경험식이다. 따라서 이 세 수는
고정점 정리만의 무입력 산출이 아니다.

반올림 오차를 배경 방정식에 넣지 않기 위해

$$
\widehat\Omega_m
=
\frac{\Omega_b+\Omega_{\mathrm{DM}}}
{\Omega_b+\Omega_{\mathrm{DM}}+\Omega_\Lambda},
\qquad
\widehat\Omega_\Lambda
=
\frac{\Omega_\Lambda}
{\Omega_b+\Omega_{\mathrm{DM}}+\Omega_\Lambda}
$$

로 정규화한다.

### 1.2 계산에 필요한 독립 입력

`[공리: 외부 입력]`

| 입력 | 역할 |
|---|---|
| $H_0$ | 거리 단위와 시간 척도 |
| $r_d$ | BAO 표준자 눈금 |
| $\sigma_{8,0}$ | 선형 성장의 현재 정규화 |
| $T_{\rm CMB}$ | 재결합 adapter를 사용할 때의 열적 입력 |

`[공리: 모델 선택]` 기본 계산은

$$
w_0=-1,\qquad w_a=0,\qquad \mu(a)=1
$$

인 평탄 $\Lambda$CDM/GR 한계다. 이 선택은 CE에서 유도된 값이 아니다.

## 2. 배경 우주론

### 2.1 CPL 배경

`[공리: 모델 선택]` $a>0$, 무차원 $(w_0,w_a)$에 대해 CPL 식을

$$
w(a)=w_0+w_a(1-a),
$$

$$
F_{\mathrm{DE}}(a)
:=
\frac{\rho_{\mathrm{DE}}(a)}{\rho_{\mathrm{DE}}(1)}
=
a^{-3(1+w_0+w_a)}\exp\!\big(3w_a(a-1)\big)
$$

로 둔다. `[산출]` 평탄 FLRW의 보존 방정식을 적용하면

$$
E^2(a)
:=
\frac{H^2(a)}{H_0^2}
=
\widehat\Omega_m a^{-3}
+
\widehat\Omega_\Lambda F_{\mathrm{DE}}(a)
$$

이다. 지수의 인자는 모두 무차원이다. $(w_0=-1,w_a=0)$이면

$$
F_{\mathrm{DE}}(a)=1,
\qquad
E^2(a)=\widehat\Omega_m a^{-3}+\widehat\Omega_\Lambda
$$

가 된다.

### 2.2 거리와 BAO 압축량

`[정의]` 평탄 배경에서 사용할 거리와 BAO 압축량을

$$
D_L(z)
=
\frac{c}{H_0}(1+z)\int_0^z\frac{dz'}{E(z')},
$$

$$
D_M(z)=\frac{D_L(z)}{1+z},
\qquad
D_H(z)=\frac{c}{H(z)},
$$

$$
D_V(z)=\big[zD_M(z)^2D_H(z)\big]^{1/3}
$$

이다. BAO 비교량은 $(D_M/r_d,D_H/r_d,D_V/r_d)$이며, $r_d$는 앞 절의
외부 입력이다. `--rd-mode external`은 이 입력을 전달하는 계산 인터페이스일
뿐 독립적인 물리 예측이 아니다.

## 3. BAO 중립 데이터 감사

### 3.1 공분산 계산

`[정의]` 관측 벡터와 모델 벡터가 같은 순서로 주어지고 공분산 $C$가
대칭 양의 정부호일 때 BAO 통계량을

$$
\chi^2_{\mathrm{BAO}}
=
\Delta O^\top C^{-1}\Delta O,
\qquad
\Delta O_i=O_i^{\mathrm{model}}-O_i^{\mathrm{obs}}
$$

로 둔다. 대각 공분산에서는

$$
\chi^2_{\mathrm{BAO,diag}}
=
\sum_i
\left(
\frac{O_i^{\mathrm{model}}-O_i^{\mathrm{obs}}}{\sigma_i}
\right)^2
$$

로 줄어든다. 잔차벡터를 \(r:=\Delta O\)라 쓰고 각 성분의 대수적 기여를

$$
c_i=r_i(C^{-1}r)_i
$$

로 두면 다음 항등식이 성립한다.

`[정리]`

$$
\sum_i c_i
=
\sum_i r_i(C^{-1}r)_i
=
r^\top C^{-1}r
=
\chi^2.
$$

증명은 유한합과 행렬곱의 정의를 전개하면 끝난다.

### 3.2 자료 범위

`[공리: 외부 입력]` 로컬 registry는 `CobayaSampler/bao_data`의
`desi_bao_dr2` Gaussian BAO mean/covariance 파일에서 옮긴 다음 압축 자료를
사용한다.

| 이름 | 내용 | source file |
|---|---|---|
| `desi-dr2-bgs` | $z=0.295$의 $D_V/r_d$ 한 점 | `desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_mean/cov.txt` |
| `desi-dr2-all` | 13개 BGS/LRG/ELG/QSO/Ly$\alpha$ 압축 관측량과 $13\times13$ 공분산 | `desi_gaussian_bao_ALL_GCcomb_mean/cov.txt` |

원본 저장소의 commit과 취득 날짜는 현재 registry에 고정되어 있지 않다.
따라서 아래 수치는 로컬 배열의 재현 계산이며, 동결된 외부 likelihood에 대한
사전 예측으로 세지 않는다. 이 provenance 보강은 `[미완성]`이다.

### 3.3 수치 기록

`[산출]` 현재 로컬 13점 벡터와 공분산에 대한 계산값은 다음과 같다.
표의 두 입력 구성은 구현 민감도 비교용이며 활성 물리 패키지가 아니다.

| 계산 구성 | $\chi^2$ | 자유도 | upper-tail $p$ |
|---|---:|---:|---:|
| 외부 calibration을 사용한 회귀 벡터 | 37.100260857 | 13 | $3.9957326\times10^{-4}$ |
| Eisenstein--Hu 근사를 사용한 진단 벡터 | 40.468225544 | 13 | $1.16176098\times10^{-4}$ |

공분산 기여가 큰 좌표는 $z=0.934$의 $D_M/r_d$, $z=0.706$의
두 값 $(D_H/r_d,D_M/r_d)$, $z=0.510$의 $D_H/r_d$다. 이는 잔차의 위치를
기술할 뿐 CE 코어 전체의 진리 판정이 아니다.

### 3.4 공통 scale 진단

`[경험식]` BAO 벡터 $y$에 같은 자료로 하나의 scale $q$를 맞추면

$$
O_i(q)=q y_i,
\qquad
q_*
=
\frac{y^\top C^{-1}d}{y^\top C^{-1}y}
=0.986476933470
$$

를 얻는다. 해당 회귀 계산은

$$
\chi^2_{\mathrm{scale}}=12.608346862,
\qquad
\nu=12,
\qquad
p=0.398138
$$

이고

$$
\mathrm{AIC}=\chi^2+2k,
\qquad
\mathrm{BIC}=\chi^2+k\ln N
$$

를 쓰면 $\mathrm{AIC}=14.6083$, $\mathrm{BIC}=15.1733$이다. 이 값은
동일한 DESI 벡터로 얻은 사후 최적화이므로 CE의 입력값, 닫힘 조건 또는
예측으로 재사용하지 않는다. 독립 자료에서 scale을 먼저 고정한 뒤에만 새
관측 비교를 정의할 수 있다.

## 4. Sound-horizon 계산 도구의 경계

### 4.1 Eisenstein--Hu 근사

`[경험식]` 물리 밀도

$$
\omega_b=\Omega_bh^2,
\qquad
\omega_m=\Omega_mh^2
$$

와

$$
\omega_\gamma
=
2.469\times10^{-5}
\left(\frac{T_{\rm CMB}}{2.7255\,{\rm K}}\right)^4,
\qquad
\omega_r
=
\omega_\gamma\left(1+0.22710731766N_{\rm eff}\right)
$$

를 입력으로 받는 Eisenstein--Hu drag fit은

$$
z_d
=
\frac{1291\omega_m^{0.251}}
{1+0.659\omega_m^{0.828}}
\left(1+b_1\omega_b^{b_2}\right),
$$

$$
b_1
=
0.313\omega_m^{-0.419}
\left(1+0.607\omega_m^{0.674}\right),
\qquad
b_2=0.238\omega_m^{0.223}
$$

이다. $a_d=(1+z_d)^{-1}$이고,

$$
R_b(a)=\frac{3\omega_ba}{4\omega_\gamma},
\qquad
c_s(a)=\frac{c}{\sqrt{3(1+R_b(a))}}
$$

에서

$$
r_d
=
\int_0^{a_d}
\frac{c_s(a)}{a^2H(a)}\,da
$$

를 계산할 수 있다. 이 절은 표준 경험적 drag fit의 구현 규약일 뿐 CE 내부의
재결합 또는 $r_d$ 유도가 아니다.

### 4.2 외부 재결합 history adapter

`[정의]` 단위와 열 순서가 고정된 $x_e(z)=n_e/n_H$ 표에 대해 drag
optical depth를

$$
\tau_{\rm drag}(z)
=
\int_0^z
\frac{c\sigma_Tn_{H,0}x_e(z')(1+z')^2}
{H(z')R(z')}\,dz',
\qquad
R(z)=\frac{3\omega_b}{4\omega_\gamma(1+z)}
$$

로 정의한다. $\tau_{\rm drag}(z_d)=1$의 bracket 내부 해를 수치적으로 찾고
입력 파일은 raw-byte SHA-256, solver와 backend 버전, $Y_p$, cosmology,
단위와 column 규약을 함께 가져야 한다.

실제 CLASS/CAMB/HyRec export와의 교차검증 및 원자물리·중성미자 입력 회계는
`[미완성]`이다. 합성 history는 수치 적분기 검사에만 사용한다.

## 5. 선형 성장

`[공리: 모델 선택]` $x=\ln a$와 주어진 $H(a),\mu(a)$에 대해 선형
성장식으로

$$
\frac{d^2D}{dx^2}
+
\left(2+\frac{d\ln H}{d\ln a}\right)
\frac{dD}{dx}
-
\frac32\mu(a)\Omega_m(a)D
=0
$$

을 채택한다. 기본 검산은 GR 한계 $\mu(a)=1$을 쓴다.

`[경험식]` 잔류 sector의 수정중력 효과를 탐색하는 선택적 ansatz는

$$
\mu(a)
=
1-\epsilon_{\mathrm{grav}}
\frac{\Omega_{\mathrm{DE}}(a)}{\Omega_{\mathrm{DE}}(1)}
$$

이다. $\epsilon_{\mathrm{grav}}$의 미시적 작용 유도는 없으므로 이 식은
물리 예측이 아니다.

`[정의]` $D(1)=1$ 정규화와 외부 $\sigma_{8,0}$에 대해

$$
\sigma_8(z)=\sigma_{8,0}D(z),
\qquad
f\sigma_8(z)=\frac{d\ln D}{d\ln a}\sigma_8(z),
$$

$$
S_8(0)=\sigma_{8,0}\sqrt{\frac{\widehat\Omega_m}{0.3}}
$$

로 정의한다. 성장 방정식의 해를 대입한 수치는 `[산출]`이다.

## 6. 실행과 회귀 범위

기본 수식 구현은 다음 명령으로 확인한다.

```powershell
python examples\physics\ce_residual_forward_model.py --rd-mpc <external-rd>
uv run --extra dev python -m pytest tests\test_ce_residual_forward_model.py tests\test_recombination_drag_adapter.py tests\test_cosmology_ratio_audit.py -q
```

`[산출]` 코드 회귀가 확인하는 범위는 다음과 같다.

- $(w_0=-1,w_a=0)$에서 $F_{\rm DE}=1$.
- $E(0)=1$, 거리 정의와 BAO 압축량의 단위 일치.
- 대각·full covariance $\chi^2$의 대수적 일치와 상관항 응답.
- $D(1)=1$, 성장 단조성과 $f\sigma_8>0$인 기준 해.
- 외부 $r_d$와 경험적 drag 계산의 provenance 분리.
- 재결합 history의 hash, 단위, grid와 root bracket 검사.

이 목록은 수치 구현 범위이며 물리 사상이나 관측 적합도의 증명이 아니다.

## 7. 남은 문제

| 항목 | 출처 | 필요한 작업 |
|---|---|---|
| CE 내부 $r_d$ | `[미완성]` | 공변 초기우주 동역학, 재결합·원자물리와 $\theta_*$ 동시 계산 |
| 동결된 외부 likelihood | `[미완성]` | source commit·취득일, nuisance와 systematic, convention 교차검증 |
| 낮은 $S_8$ 자료 | `[미완성]` | baryonic feedback 또는 $\mu(a)$의 원리적 유도 |
| 입자 암흑물질 | `[미완성]` | $(m_\chi,\sigma_{\chi N})$, transfer function과 detector likelihood |
| 사전 관측 시험 | `[미완성]` | 독립 $H_0r_d$ calibration과 비교 절차를 자료 공개 전에 고정 |

## 8. 결론

`[산출]` 평탄 FLRW/CPL/GR 전제와 외부 scale을 앞 절의 정의에 대입하면

$$
\boxed{
(\widehat\Omega_m,\widehat\Omega_\Lambda;H_0,r_d,\sigma_{8,0})
\longmapsto
H(z),D_L(z),D(a),f\sigma_8(z),S_8(0)
}
$$

의 계산 사상은 정해진다. CE 성분비를 그 입력 경계조건으로 읽는 단계는
물리 사상이며, $r_d$의 CE 내부 결정과 독립 likelihood 예측은 아직 추가
구조가 필요하다. 같은 자료로 얻은 scale 최적화는 경험식 진단에만 속한다.
