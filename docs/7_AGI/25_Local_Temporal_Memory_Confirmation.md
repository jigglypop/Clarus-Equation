# Local temporal-memory confirmation

> 최종 상태: `preregistered computational gate PASS`
>
> AML310 exploratory: `h=1 4/4`, `h=6 4/4`
>
> untouched AML32 confirmatory: `h=1 7/7`, `h=6 7/7`

## 1. 실제로 증명한 명제

이 결과가 지지하는 명제는 다음처럼 좁다.

> 움직이는 C. elegans의 이 calcium-activity 데이터에서, 한 뉴런의
> \(t-1,t-2\) 측정값은 그 뉴런의 현재 측정값 \(x_i(t)\)만으로 만든
> 비선형 기준선을 조건으로 한 뒤에도 \(t+h\) 측정값에 대한 held-out
> 예측정보를 가진다. 이 결과는 \(h=1,6\)과 별도의 AML32 일곱 동물에서
> 재현된다.

고정한 예측식은

\[
\widehat x_i(t+h)
=\beta_0+g\!\left(x_i(t)\right)
 \beta_1x_i(t-1)+\beta_2x_i(t-2),
\]

\[
g(x)=[x,x^2,x^3,\tanh x]
\]

이다. current-only 기준선은 \(\beta_1=\beta_2=0\)인 같은 ridge
family다. 따라서 단순 선형 current 기준선이 약해서 생기는 이득만을
세지 않았다.

## 2. 증명 판정식

기록 \(r\), 뉴런 \(i\), horizon \(h\)에 대해

\[
d_{r,i,h}
=R^2_{r,i,h}(\mathrm{local})
-R^2_{r,i,h}(\mathrm{current\ nonlinear})
\]

로 두고, 동물별 효과를

\[
\Delta_{r,h}=\operatorname{median}_i d_{r,i,h}
\]

로 정의했다. \(t-1,t-2\) 열을 train/validation/test 각 블록 안에서
함께 원형 이동하고 **매번 모델을 다시 학습**한 19개 null을
\(\Delta^{(b)}_{r,h}\)라 하면

\[
p_{r,h}
=\frac{1+\sum_{b=1}^{19}
\mathbf 1[\Delta^{(b)}_{r,h}\ge\Delta_{r,h}]}{20}.
\]

기록 하나의 사전등록 통과 술어는

\[
G_{r,h}=
\mathbf 1\!\left[
\begin{array}{l}
n_{\rm target}\ge20,\\
\Delta_{r,h}>0.01,\\
\Pr_i(d_{r,i,h}>0)\ge0.8,\\
p_{r,h}\le0.05
\end{array}
\right].
\]

확인 패널의 전체 술어는

\[
G_{\rm panel}
=
\mathbf 1\!\left[\sum_{r=1}^{7}G_{r,1}\ge5\right]
\land
\mathbf 1\!\left[\sum_{r=1}^{7}G_{r,6}\ge5\right].
\]

규칙은 AML32 activity를 평가하기 전에
`local_memory_aml32_preregistration.json`에 고정했다.

## 3. 누수 및 약한 대조군 방지

- 시간순 60/20/20 분할과 5-sample embargo를 썼다.
- ridge는 validation에서만 고르고 test는 변환·학습·선택에 쓰지 않았다.
- eligible target은 train 구간의 결측률과 분산으로만 정했다.
- acquisition gap을 가로지르는 lag/target window는 제외했다.
- null은 과거 열의 자기상관과 두 lag 사이 관계를 보존하고, 각 null
  feature에서 모델과 ridge를 다시 맞췄다.
- test block을 변조해도 fitted-model SHA-256이 변하지 않는 회귀시험을
  통과했다.
- 합성 AR(2)는 통과하고, 진짜 AR(1) current-state process는 실패했다.
- 복제 단위는 뉴런이 아니라 독립 기록/동물이다.

## 4. 탐색 패널: AML310

| horizon | 기록별 median \(\Delta R^2\) | 기록 통과 |
|---|---:|---:|
| \(h=1\) | 0.0293, 0.0226, 0.0604, 0.0379 | 4/4 |
| \(h=6\) | 0.2298, 0.1952, 0.3085, 0.1879 | 4/4 |

모든 기록의 positive-target fraction은 \(0.901\) 이상이고 null rank
\(p=0.05\)였다. 이 결과를 본 뒤에도 AML32 기준은 바꾸지 않았다.

## 5. untouched 확인 패널: AML32

### \(h=1\)

| recording | targets | current \(R^2\) | local \(R^2\) | \(\Delta R^2\) | positive | null \(p\) | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20170610_105634 | 107 | 0.9759 | 0.9994 | 0.0230 | 0.991 | 0.05 | PASS |
| 20170613_134800 | 118 | 0.9771 | 0.9994 | 0.0223 | 1.000 | 0.05 | PASS |
| 20170424_105620 | 110 | 0.9677 | 0.9990 | 0.0309 | 1.000 | 0.05 | PASS |
| 20180709_100433 | 135 | 0.9865 | 0.9996 | 0.0131 | 1.000 | 0.05 | PASS |
| 20200309_151024 | 121 | 0.9553 | 0.9986 | 0.0427 | 1.000 | 0.05 | PASS |
| 20200309_153839 | 131 | 0.9539 | 0.9981 | 0.0425 | 0.992 | 0.05 | PASS |
| 20200309_162140 | 134 | 0.9613 | 0.9988 | 0.0376 | 1.000 | 0.05 | PASS |

사전등록 요구치는 5/7이고 관측값은 **7/7**이다.

### \(h=6\)

| recording | targets | current \(R^2\) | local \(R^2\) | \(\Delta R^2\) | positive | null \(p\) | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20170610_105634 | 107 | 0.4875 | 0.6844 | 0.1949 | 0.991 | 0.05 | PASS |
| 20170613_134800 | 118 | 0.4778 | 0.6882 | 0.2098 | 1.000 | 0.05 | PASS |
| 20170424_105620 | 110 | 0.3673 | 0.5881 | 0.2135 | 0.991 | 0.05 | PASS |
| 20180709_100433 | 135 | 0.6862 | 0.8007 | 0.1153 | 1.000 | 0.05 | PASS |
| 20200309_151024 | 121 | 0.2405 | 0.4951 | 0.2388 | 0.950 | 0.05 | PASS |
| 20200309_153839 | 131 | 0.1745 | 0.4220 | 0.2561 | 0.969 | 0.05 | PASS |
| 20200309_162140 | 134 | 0.2735 | 0.5383 | 0.2430 | 0.993 | 0.05 | PASS |

사전등록 요구치는 5/7이고 관측값은 다시 **7/7**이다.

## 6. 독립 계산 검증

verifier는 결과 파일의 `gate_passed`를 신뢰하지 않고 다음을 다시
계산한다.

1. 사전등록 implementation SHA-256 일치
2. AML32 archive SHA-256 일치
3. 정확히 사전등록한 일곱 recording인지 확인
4. 각 기록의 네 조건을 원시 수치에서 재계산
5. horizon별 pass count와 5/7 조건 재계산
6. \(h=1,6\) 동시 통과 재계산

결과는

```text
proof_passed = true
errors = []
h=1: 7/7, required 5
h=6: 7/7, required 5
```

이다. threshold 변조와 implementation hash 변조를 거부하는 시험도
통과했다.

## 7. 무엇이 증명됐고 무엇은 아직 아닌가

| 명제 | 판정 |
|---|---|
| 고정된 코드와 artifact가 사전등록 술어를 만족 | **Exact computational PASS** |
| AML32 measured trace에 current를 넘는 aligned local-history 예측정보 존재 | **Confirmatory support, 7/7 at both horizons** |
| 이 정보가 calcium indicator/전처리 평활화가 아닌 세포 내부 기억기작 | 미증명 |
| 뉴런 간 population cloud가 local history 위에 추가 정보 제공 | 반증, 0/4 |
| anonymous activity에서 directed effective graph가 local보다 우수 | 반증, 0/4 |
| diffusion이 linear/persistence보다 우수 | 반증, 최대 1/4 |
| 뉴런 자체가 category-theoretic monad/CloudCell | 미증명이며 현재 자료로 식별 불가 |
| 이 결과가 AGI architecture를 직접 입증 | 미증명 |

특히 \(p=0.05\)는 19개 고정 phase-null이 주는 최소 해상도다. 이는
“모든 null보다 관측 정렬이 컸다”는 뜻이지, 생물학적 모집단에서 정확한
확률이 0.05라는 뜻은 아니다. 또한 calcium indicator kinetics나 기존
signal processing도 local history를 만들 수 있다. 따라서 정당한 결론은

\[
\boxed{
\text{measured neuron activity is predictively stateful over time}
}
\]

까지다. 현재 결과로

\[
\text{neuron}=\text{coded monadic CloudCell}
\]

을 쓰는 것은 증거 범위를 넘는다.

## 8. AGI 쪽에서 남는 의미

AGI 설계에 가져갈 수 있는 것은 존재론적 동일시가 아니라 설계 제약이다.

\[
\text{unit state}_{t+1}
=F(\text{unit state}_{t:t-2},\ \text{input}_t)
\]

처럼 각 unit에 짧은 local state를 두는 가정은 실제 데이터와 합치한다.
반대로 이번 자료는 dense population cloud, learned directed graph,
nonlinear diffusion을 local state 위에 반드시 얹어야 한다는 근거를
주지 않는다. 따라서 현재 우선순위는 **local recurrent state를 기본으로
두고, cross-unit 구조는 별도 데이터에서 증명될 때만 추가하는 것**이다.
