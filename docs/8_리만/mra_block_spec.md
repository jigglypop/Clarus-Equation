# Mellin–Riemann Attention Block: canonical score와 self-adjoint 조건

## 0. 범위

이 사양은 세 주장을 분리한다.

1. 검증된 zeta-zero ordinate를 deterministic frequency bank로 사용하는 것
2. explicit-formula에서 영감을 받은 **방향성 score**를 쓰는 것
3. bidirectional kernel을 실제 self-adjoint operator로 만드는 것

1은 유한한 공학 설계이고 2는 inductive bias다. 어느 것도 Riemann
hypothesis, GUE conjecture, Hilbert–Pólya conjecture의 증명이나 구현이
아니다.

## 1. zero 자료의 정확한 지위

유한 table에 대해 \(\rho_k=\tfrac12+i\gamma_k\)가 실제 영점임을 수치적으로
검증했다면 그 유한 항을 쓰는 데 전체 RH는 필요하지 않다. 반대로 검증
범위를 넘어 모든 비자명 영점이 critical line 위에 있다고 말할 때만 RH가
추가 가정이다.

또한 다음을 구분한다.

- RH: 비자명 영점의 실수부에 관한 conjecture
- Montgomery pair correlation 및 GUE spacing: 정규화된 고영점 간격의
  통계에 관한 더 강한 conjectural/empirical 진술
- Hilbert–Pólya: ordinate를 어떤 self-adjoint operator의 spectrum으로
  실현하려는 conjecture

RH는 GUE를 함의하지 않는다. 저장된 \(\gamma_k\)를 attention frequency로
쓴다고 해서 Hilbert–Pólya operator가 구성되지도 않는다.

검증 table 밖의 ordinate가 필요하면 Riemann–von Mangoldt counting
equation을 수치적으로 역해 얻은 값을 approximate frequency라고 표시한다.
\(2\pi n/\log n\)만으로 만든 수를 실제 zeta zero라고 기록하지 않는다.

현재 backend는 raw ordinate가 아니라

\[
\nu_k\equiv\frac{\gamma_k}{\gamma_0}
\]

를 사용해 첫 frequency를 1로 정규화한다. 따라서 benchmark coefficient

\[
a_k\equiv\frac1{\tfrac12+i\nu_k}
\]

는 \(1/(\tfrac12+i\gamma_k)\)라는 zeta explicit-formula amplitude가
아니다. 이 정규화는 engineering choice다. 아래 canonical 구현 score는
\((\nu_k,a_k)\)를 쓰고, raw-number-theory reference가 필요할 때만
\(\nu_k=\gamma_k\)로 둔다.

## 2. 하나의 위치·부호 convention

모든 문서와 구현은

\[
\tau_i=\log(1+i),\qquad
\Delta_{ij}=\tau_i-\tau_j,\qquad
x_{ij}=e^{-\Delta_{ij}}=\frac{1+j}{1+i}
\]

를 쓴다. 복소 채널은

\[
z_i^{(k)}=q_i^{2k}+iq_i^{2k+1},\qquad
y_j^{(k)}=k_j^{2k}+ik_j^{2k+1}
\]

이고 normalized design amplitude는

\[
a_k=\frac{1}{\tfrac12+i\nu_k}.
\]

canonical directed score는

\[
\boxed{
D_{ij}
=\sqrt{x_{ij}}\sum_{k=0}^{K-1}
a_k e^{-i\nu_k\Delta_{ij}}
z_i^{(k)}\overline{y_j^{(k)}}.
}
\]

이다. raw mode \(\nu_k=\gamma_k\)에서만 \(x=x_{ij}\)로 선택한
explicit-formula 항
\(\sqrt{x}\,e^{i\gamma\log x}/(\tfrac12+i\gamma)\)와 coefficient까지
같다. normalized benchmark는 부호와 함수형만 차용한다.

factorization은

\[
\widetilde z_i^{(k)}
=\frac{e^{-i\nu_k\tau_i}}{\sqrt{1+i}}z_i^{(k)},
\qquad
\widetilde y_j^{(k)}
=\sqrt{1+j}\,e^{-i\nu_k\tau_j}y_j^{(k)}
\]

로 두어

\[
D_{ij}=\sum_k a_k\widetilde z_i^{(k)}
\overline{\widetilde y_j^{(k)}}.
\]

실수 logit은 \(L_{ij}=\operatorname{Re}D_{ij}/\sqrt{d_h}\)다.
이 convention은 기존 PyTorch 구현의 negative rotation과
\(\sqrt{(1+j)/(1+i)}\) multiplicative decay에 맞는다.

## 3. 방향성 score는 Hermitian이 아니다

\(W_q=W_k\)로 묶더라도 일반적으로

\[
D_{ji}\ne\overline{D_{ij}}.
\]

이유는 두 가지다.

- \(\sqrt{x_{ji}}=1/\sqrt{x_{ij}}\)인 방향성 prefactor
- \(a_k\)가 복소수인 normalized design amplitude

따라서 tied projection만으로 self-adjointness가 보장된다는 과거 정리는
거짓이다.

## 4. bidirectional self-adjoint 정리

causal mask가 없는 bidirectional 모드에서 directed complex matrix를 먼저
계산하고

\[
\boxed{H=\frac12(D+D^\dagger)}
\]

로 정의한다.

**정리.** 유한 입력과 실수 \(\nu_k\)에 대해 \(H^\dagger=H\)다.

**증명.**

\[
H^\dagger
=\frac12(D^\dagger+D)=H.
\]

복소 구현 대신 실수 logit만 저장하면

\[
H_R=\frac12(L+L^\mathsf T)
\]

이고 \(H_R^\mathsf T=H_R\)다. 이 명제에는 RH나 GUE가 필요 없다.

## 5. softmax 이후의 정확한 operator

대칭 logit에도 표준 row softmax를 적용한

\[
P_{ij}
=\frac{e^{H_{R,ij}}}{\sum_\ell e^{H_{R,i\ell}}}
\]

는 일반적으로 \(P^\mathsf T\ne P\)다. 따라서 “대칭 score이므로 attention
operator도 Hermitian”이라고 말할 수 없다.

완전한 self-adjoint operator가 필요하면

\[
K_{ij}=e^{H_{R,ij}},\qquad
d_i=\sum_jK_{ij},\qquad D_d=\operatorname{diag}(d_1,\ldots,d_n),
\]

\[
\boxed{A_{\rm sym}=D_d^{-1/2}KD_d^{-1/2}}
\]

를 쓴다. \(K^\mathsf T=K\)이므로
\(A_{\rm sym}^\mathsf T=A_{\rm sym}\)가 exact다.

표준 row-normalized \(P=D_d^{-1}K\)를 유지하면 ordinary Euclidean
inner product에서는 대칭이 아니지만

\[
\langle u,Pv\rangle_{D_d}
=u^\mathsf TD_dPv
=u^\mathsf TKv
=\langle Pu,v\rangle_{D_d}
\]

이므로 \(D_d\)-weighted inner product에서 self-adjoint다.

## 6. causal mode와의 비양립성

strict lower-triangular causal mask는 \(i<j\) edge를 제거하고 역방향 edge는
남기므로 비자명한 symmetric kernel과 양립하지 않는다. causal decoder는
2절의 directed \(L\)을 사용하고 Hermitian claim을 하지 않는다.
bidirectional encoder만 4–5절의 self-adjoint mode를 사용할 수 있다.

mask 전에 score를 symmetrize한 뒤 causal mask를 씌우는 현재 구현은 future
value를 직접 섞지는 않더라도 최종 operator가 self-adjoint라는 증거가
아니다. exact self-adjoint backend는 \(A_{\rm sym}\) normalization을 별도로
구현해야 한다.

## 7. amplitude와 통계 해석

\[
|a_k|=\frac1{\sqrt{1/4+\nu_k^2}}
\]

는 normalized deterministic high-frequency damping이다. 이를 zeta
explicit-formula coefficient, GUE weight 또는 Hilbert–Pólya spectral
measure라고 부르지 않는다. 실제 GUE 진단은 unfolded
spacing, finite-size null distribution, multiple-testing rule을 따로
사전등록해야 한다.

## 8. acceptance tests

canonical 구현은 최소한 다음을 통과해야 한다.

1. \(x_{ij}=(1+j)/(1+i)\)와 phase sign의 hand calculation
2. factorized score와 direct score의 수치 동일성
3. bidirectional \(H-H^\dagger\) residual
4. \(A_{\rm sym}-A_{\rm sym}^\mathsf T\) residual
5. row normalization \(P\)의 \(D_dP=P^\mathsf TD_d\) detailed balance
6. causal mode에서 미래 token 변화가 과거 출력에 미치는 영향 0
7. verified zero table과 approximate frequency의 provenance 분리

## 9. 판정

| 명제 | 판정 |
|---|---|
| directed Mellin score의 부호·감쇠 convention | Fixed |
| tied \(W_q=W_k\)만으로 Hermitian | Refuted |
| \(H=(D+D^\dagger)/2\)의 Hermiticity | Exact |
| \(A_{\rm sym}\)의 self-adjointness | Exact |
| causal attention의 nontrivial Hermiticity | Incompatible |
| finite verified zero bank 사용 | Engineering choice |
| RH, GUE, Hilbert–Pólya의 증명 또는 구현 | Not claimed |
