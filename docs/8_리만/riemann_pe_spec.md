# Log-Mellin positional encoding: Riemann-zero frequency 사양

## 0. 이름과 범위

이 모듈은

\[
\theta_k(p)=\nu_k\log(1+p),
\qquad \nu_k:=\frac{\gamma_k}{\gamma_0}
\]

인 rotary positional encoding이다. 복소 로그의 multi-sheet analytic
continuation을 계산하지 않으므로 수학적으로 “Riemann surface 위의
attention”이라고 부르지 않는다. Riemann zeta zero ordinate를 frequency
bank로 선택한 **Log-Mellin rotary encoding**이 정확한 설명이다.

## 1. zero 자료

유한 목록의 \(\gamma_k\)는 출처와 검증 정밀도를 가진 table에서 읽는다.
각 listed zero가 critical line 위에 있음을 독립 검증한 범위에서는 전체
Riemann hypothesis를 가정할 필요가 없다.

검증 범위를 넘는 항은 approximate frequency로 저장하고 실제 zero와
구분한다. RH는 모든 비자명 영점에 관한 추가 conjecture이며, GUE spacing은
RH와 별개의 더 강한 통계 conjecture다. 둘을 모듈의 보장조건으로 쓰지 않는다.

현재 backend가 회전과 amplitude에 실제로 넣는 값은 raw ordinate가 아니라

\[
\nu_k=\frac{\gamma_k}{\gamma_0},\qquad
a_k=\frac{1}{\tfrac12+i\nu_k}
\]

이다. 따라서 \(a_k\)는
\(1/(\tfrac12+i\gamma_k)\)인 zeta explicit-formula coefficient가 아니라
정규화된 design weight다. raw-number-theory reference mode를 따로 만들 때만
\(\nu_k=\gamma_k\)로 둔다.

## 2. canonical 좌표와 회전

모든 관련 문서는

\[
\tau_p=\log(1+p),\qquad
\Delta_{ij}=\tau_i-\tau_j
\]

를 사용한다. 구현의 complex channel을
\(z=q_{2k}+iq_{2k+1}\)로 읽을 때 negative-rotation convention은

\[
\widetilde z_p^{(k)}
=e^{-i\nu_k\tau_p}z_p^{(k)}.
\]

실수 \(2\times2\) 행렬로는

\[
\begin{pmatrix}
\widetilde z_{\rm Re}\\
\widetilde z_{\rm Im}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta&\sin\theta\\
-\sin\theta&\cos\theta
\end{pmatrix}
\begin{pmatrix}
z_{\rm Re}\\
z_{\rm Im}
\end{pmatrix}.
\]

따라서 query/key product의 phase는

\[
e^{-i\nu_k(\tau_i-\tau_j)}
=e^{-i\nu_k\Delta_{ij}}.
\]

이 부호는
[mra_block_spec.md](mra_block_spec.md)의 canonical directed score와 같다.

## 3. 불변성의 정확한 표현

\[
\Delta_{ij}
=\log\frac{1+i}{1+j}
\]

는 \(i-j\)만의 함수가 아니므로 ordinary translation invariance가 없다.
대신 큰 양의 위치를 동시에 scale할 때

\[
\log\frac{1+\lambda i}{1+\lambda j}
\longrightarrow\log\frac{i}{j}
\]

라는 asymptotic ratio-coordinate 성질이 있다. 이것을 sequence-length
extrapolation 보장으로 확대하지 않는다. aliasing과 성능은 empirical
benchmark로 판정한다.

## 4. 선택적 wrap counter

\[
\sigma(p,k)
=\left\lfloor\frac{\nu_k\tau_p}{2\pi}\right\rfloor
\]

를 별도 feature로 저장할 수 있다. 대칭 bias를 쓰려면

\[
b_{ij}^{\rm wrap}
=-\lambda_\sigma\frac1K
\sum_k|\sigma(i,k)-\sigma(j,k)|
\]

로 둔다. 이 counter는 phase collision을 구분하는 공학 feature일 뿐
complex logarithm의 branch structure나 zeta function의 analytic
continuation을 구현하지 않는다.

## 5. score convention

단순 rotary score는

\[
L_{ij}^{\rm rotary}
=\frac{\operatorname{Re}
\sum_k\widetilde q_i^{(k)}
\overline{\widetilde k_j^{(k)}}}{\sqrt{d_h}}
+b_{ij}^{\rm wrap}.
\]

정규화된 complex weight와 explicit-formula-inspired prefactor를 함께 쓰는
현재 MRA score는

\[
x_{ij}=\frac{1+j}{1+i},\qquad
D_{ij}
=\sqrt{x_{ij}}\sum_k
a_k e^{-i\nu_k\Delta_{ij}}
q_i^{(k)}\overline{k_j^{(k)}}.
\]

두 식의 phase sign과 \(\Delta_{ij}\) 정의는 동일하다. 이 식을 raw zeta
explicit-formula 항과 coefficient까지 같다고 해석해서는 안 된다.

## 6. Hermitian 조건

rotary matrix 하나가 orthogonal/unitary라는 사실은 attention score matrix가
Hermitian이라는 뜻이 아니다. 독립 \(W_q,W_k\), 방향성 prefactor, complex
amplitude, causal mask 중 하나만 있어도 그 결론은 나오지 않는다.

bidirectional self-adjoint mode는

\[
H=\frac12(D+D^\dagger),\qquad
K_{ij}=\exp(\operatorname{Re}H_{ij}),\qquad
A_{\rm sym}=D_d^{-1/2}KD_d^{-1/2},
\quad (D_d)_{ii}=\sum_jK_{ij}
\]

로 별도 구성한다. 자세한 정리와 증명은
[mra_block_spec.md](mra_block_spec.md) 4–6절을 따른다. causal decoder에는
Hermitian claim을 적용하지 않는다.

## 7. 학습 파라미터

\(\gamma_k\) 자체는 provenance가 고정된 buffer이고 \(\nu_k\)는 그로부터
결정되는 정규화 frequency다. head별 scale이나
\(\lambda_\sigma\)를 학습하면 그 값은 더 이상 zeta에서 정해진 0-parameter
구조가 아니다. 학습 가능한 hyperparameter로 명시하고 ablation해야 한다.

## 8. backend acceptance

PyTorch, Rust, CUDA backend는 다음을 동일하게 구현해야 한다.

1. \(\tau_p=\log(1+p)\)
2. \(\nu_k=\gamma_k/\gamma_0\)와 \(a_k=(\tfrac12+i\nu_k)^{-1}\)
3. negative-rotation 행렬
4. \(\Delta_{ij}=\tau_i-\tau_j\)
5. verified/approximate frequency provenance
6. 동일 dtype에서 forward와 gradient tolerance
7. causal future-leakage test

현재 backend가 이 사양의 self-adjoint normalization을 구현하지 않았다면
단순 score symmetrization만으로 “Hermitian attention 구현 완료”라고
기록하지 않는다.

## 9. 참고

- Riemann, zeta explicit formula
- Titchmarsh, The Theory of the Riemann Zeta-Function
- Odlyzko, high-zero computations and spacing data
- Montgomery, pair-correlation conjecture
- RoFormer, rotary positional encoding
