# 18. 분포적 refinement와 rigging map

이 장은 refinement의 정확한 작은 모형 하나를 닫는다. 유한 차원 truncation들이 Hilbert 노름에서는
서로 가까워지지 않아도, 유한 지지 검정 벡터와의 pairing은 안정될 수 있음을 보인다. 그 안정한
pairing에서 rank-one rigging map을 만들면 null direction을 나눈 물리 completion이
$\mathbb C$가 된다. 이것은 4차원 양자중력의 연속극한이 아니라, distributional route가 실제로
어떤 수학적 대상을 요구하는지 보이는 직접계 모형이다.

이 구분은 강한 Hilbert-norm 수렴을 refinement의 정의로 삼을 때 중요하다. 그 요구는 너무 강하면
topological theory로 굳을 수 있다는 문제가 있다. [Bruno et al. (2026)](https://arxiv.org/abs/2603.16999)의
Theorems 5.1--5.2와 Eqs. (5.3), (5.5), (5.6), (6.1)는 이 위험과 distributional/rigging-map
경로를 분리한다. 아래에서는 먼저 zero-extension 직접계를 만들고, 그 뒤 노름 실패와 pairing의
안정을 같이 계산한 다음, rigging pairing과 quotient를 만든다. 마지막으로 이 모형이 아직
말하지 못하는 물리를 경계로 남긴다.

## 18.1 refinement 공간과 무차원 branch phase

각 level $N\ge1$에서 Hilbert 공간을 $\mathcal H_N=\mathbb C^N$으로 둔다. $M\ge N$이면
refinement map $\iota_N^M$는 뒤에 $M-N$개의 $0$을 붙이는 zero extension이다. 그러므로

$$
\|\iota_N^M\varphi\|^2_{\mathcal H_M}=\|\varphi\|^2_{\mathcal H_N}
$$

이고, $\iota_M^K\iota_N^M=\iota_N^K$다. 이 exact isometry와 합성 법칙이 cylindrical
direct system의 출발점이다. 검정 공간은 유한 지지 수열
$\mathcal D=c_{00}\subset\ell^2$로 두고, 그 대수적 dual을 $\mathcal D'$로 쓴다.

무차원 수 $s$를 고정해 각 level에

$$
\Omega_s^{(N)}=(e^{ins})_{n=1}^{N}
$$

를 둔다. 이 $s$는 [17장](17_곡률_단일가지_가우스_템플릿.md)의 $S_{\rm curv}$를 값으로 받을
수 있다. 그 연결은 $s\leftarrow S_{\rm curv}$라는 **선언된 value map**일 뿐이다. 이 직접계가
기하적 refinement에서 $S_{\rm curv}$를 유도했다는 뜻은 아니다.

## 18.2 노름에서는 실패하고 pairing에서는 안정되는 열

각 성분의 절댓값이 $1$이므로 truncation 노름은

$$
\|\Omega_s^{(N)}\|^2=N.
$$

더 큰 level $M$에서 이전 벡터를 zero extension하면 새로 생긴 $M-N$개의 성분만 남는다. 따라서

$$
\left\|\Omega_s^{(M)}-\iota_N^M\Omega_s^{(N)}\right\|^2=M-N.
$$

특히 $M-N$을 고정한 양수로 두어도 이 차이는 사라지지 않으므로, 이 열은 direct-limit Hilbert
노름에서 Cauchy 열이 아니다. 이 사실은 강한 수렴을 이 모형에 요구할 수 없다는 정확한 계산이다.

그러나 $\varphi\in c_{00}$의 지지가 $\{1,\ldots,K\}$ 안에 있으면 $N\ge K$에서

$$
\langle\Omega_s^{(N)},\varphi\rangle
=\sum_{n=1}^{K}e^{-ins}\varphi_n
$$

으로 더 이상 $N$에 의존하지 않는다. 이 eventual constancy가 $\Omega_s\in
\mathcal D'\setminus\ell^2$를 정의한다. 벡터가 $\ell^2$에 들어가지 못해도 모든 유한 지지
검정에는 유한하고 일관된 값을 주는 이유가 여기 있다.

## 18.3 rigging pairing과 물리 completion

앞 절의 안정값을 선형 functional

$$
L_s(\varphi)=\sum_{n\ge1}e^{-ins}\varphi_n
$$

로 정의한다. 합은 $\varphi$의 유한 지지 때문에 유한하다. 이때 rigging pairing은

$$
P_s(\varphi,\psi)=\overline{L_s(\varphi)}L_s(\psi)
$$

이다. $L_s$는 선형이고, $P_s$는 첫 번째 슬롯에서 anti-linear, 두 번째 슬롯에서 linear다.
또

$$
P_s(\psi,\varphi)=\overline{P_s(\varphi,\psi)},\qquad
P_s(\varphi,\varphi)=|L_s(\varphi)|^2\ge0
$$

이므로 Hermitian이고 positive semidefinite다. zero extension은 유한 지지 계수를 바꾸지 않으므로

$$
P_s(\iota_N^M\varphi,\iota_N^M\psi)=P_s(\varphi,\psi)
$$

가 정확히 성립한다. 이것이 이 모형에서 말하는 cylindrical consistency다. 이 개념의 일반적인
refinement 맥락은 [Bahr (2014)](https://arxiv.org/abs/1407.7746)와
[Dittrich (2012)](https://arxiv.org/abs/1205.6127)에서 확인할 수 있다.

$L_s$가 $0$인 벡터들을 $\ker L_s$라 하자. $L_s$는 예컨대
$e^{is}(1,0,0,\ldots)$을 $1$로 보내므로 $\mathcal D\to\mathbb C$로 전사한다. 따라서

$$
\mathcal D/\ker L_s\simeq\mathbb C
$$

이고 quotient는 codimension $1$이다. 이 quotient에 $P_s$가 주는 norm을 넣어 completion하면
여전히 $\mathbb C$다. 이것은 하나의 rank-one 물리 공간을 얻는 정확한 대수 결론이다.

## 18.4 normalization을 바꾸면 consistency가 사라지는 대조군

분포적 제한은 아무 normalized truncation에도 자동으로 붙지 않는다. level $N$에서

$$
u_N=\frac{\Omega_s^{(N)}}{\sqrt N}
$$

를 만들고 첫 basis probe $e_1$을 쓰면 그 rank-one pairing은

$$
|\langle u_N,e_1\rangle|^2=\frac1N.
$$

같은 probe를 level $M$에서 계산하면 $1/M$이다. $N\ne M$이면 값이 달라져 cylindrical
consistency가 깨진다. 이 음성 대조군은 “노름을 $1$로 맞추면 더 좋다”는 직감이 distributional
rigging map에는 맞지 않을 수 있음을 정확히 보인다.

이 모형은 Bruno et al.의 Theorem 5.1이 문제 삼는 강한 Hilbert-space convergence 가정을
쓰지 않는다. 그렇다고 non-TQFT임을 증명하지는 않는다. geometric refinement, EPRL amplitude,
cutoff renormalization, constraint algebra 또는 Ward identity, Einstein--Hilbert dominance, 정확히
두 helicity도 여기서 얻지 못한다. [DonàㆍFrisoniㆍWilson-Ewing (2022)](https://arxiv.org/abs/2206.14755)은
refinement renormalization의 수치적 맥락을 제공하지만, 이 모형의 물리적 연속극한을 증명하지
않는다.

## 18.5 재현 범위

직접계, distributional functional, rigging pairing과 음성 대조군은
[distributional_rigging_map.py](../../examples/physics/distributional_rigging_map.py)와
[test_distributional_rigging_map.py](../../tests/test_distributional_rigging_map.py)에 있다.

```powershell
.codex/hooks/python.cmd pytest tests/test_distributional_rigging_map.py -q
```

원장에 기록된 focused 결과는 `25 passed`, source parse는 `422 PASS`다. 이 회귀는 유한 direct
system의 항등식만 검사한다. 비위상적 geometric refinement, EPRL state sum, Einstein 방정식의
지배나 two-DOF IR은 검사하거나 증명하지 않는다.

그 two-DOF의 정확한 수용 기준은 [19장](19_선형화_Einstein_두_편광_수용_정리.md)에 분리해 둔다.
그 장은 supplied linearized Einstein 모형의 quotient만 증명하며, 이 직접계가 그 모형을 유도했다는
주장은 하지 않는다.
