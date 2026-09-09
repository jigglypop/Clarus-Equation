# 20. 게이지 보존 Fierz--Pauli 격자 refinement

이 장은 이미 공급한 자유 Fierz--Pauli/linearized Einstein symbol에 central difference를 넣었을 때,
낮은 운동량 창에서 gauge 항등식과 두 편광 quotient가 보존됨을 보인다. 또한 그 lattice symbol이
고정한 compact 운동량 상자에서 continuum symbol로 균등 수렴함을 명시적 상계로 증명한다. 이
결과는 spin-foam에서 격자 작용을 유도하거나 interacting continuum gravity를 만드는 결과가
아니다.

이 구분이 필요한 이유는 격자가 gauge freedom을 쉽게 깨뜨리고, central sine symbol은 전역에
doubler zero도 갖기 때문이다. 먼저 모든 운동량과 간격을 무차원화하고, 그 다음 gauge-nullㆍ
Bianchiㆍself-adjoint 구조를 대수적으로 확인한다. 이어 compact-window 오차 상계와 null ray의
두 편광을 보인 뒤, 전역 doubler와 남은 물리 의무를 분리한다.

## 20.1 무차원 격자 symbol을 고정한다

물리 운동량 $p$와 lattice spacing $a$에 기준 길이 $L_{\rm ref}$를 붙여

$$
q=L_{\rm ref}p,\qquad \bar a=\frac{a}{L_{\rm ref}}
$$

로 둔다. $q$와 $\bar a$는 모두 무차원이다. central difference는 각 성분을

$$
\widehat q_i=\frac{\sin(\bar a q_i)}{\bar a}
$$

로 바꾼다. 여기서 $K(q)$는 공급한 $10\times10$ 자유 linearized Einstein symbol이고,
$G(q)$는 gauge direction, $D(q)$는 linearized Bianchi divergence map이다. $K$, 이
discretization, 그리고 아래 compact window는 기하적 refinement가 도출한 것이 아니라 모형
입력이다.

각 spacing에서 같은 다항식 $K$에 $q$ 대신 $\widehat q$를 넣으면

$$
K(\widehat q)G(\widehat q)=0,\qquad
D(\widehat q)K(\widehat q)=0
$$

가 정확히 성립한다. 첫 식은 gauge 방향이 작용의 영방향이라는 뜻이고, 둘째 식은 Bianchi
항등식이다. 대칭 tensor component의 off-diagonal 중복을 세는 weight

$$
W_{\mu\nu}=(1+\mathbf1_{\mu\ne\nu})\eta_{\mu\mu}\eta_{\nu\nu}
$$

를 대각 행렬로 배열하면, weighted symbol은

$$
WK(\widehat q)=\bigl(WK(\widehat q)\bigr)^{\mathsf T}
$$

를 만족한다. 그러므로 이 자유 quadratic action은 선언한 성분 내적에서 self-adjoint다.

## 20.2 compact 창에서의 정확한 오차 상계

이제 $|q_i|\le B$와 $\bar aB<\pi/2$를 가정한다. sine의 Taylor remainder는

$$
\left|\frac{\sin(\bar a q_i)}{\bar a}-q_i\right|
\le\frac{\bar a^2|q_i|^3}{6}
\le\delta,\qquad
\delta=\frac{\bar a^2B^3}{6}
$$

를 준다. $K$의 각 entry는 운동량의 이차 다항식이다. 구현한 $10\times10$ symbol에서 항을
직접 세면 한 entry의 차이는

$$
|K_{AB}(\widehat q)-K_{AB}(q)|\le13B\delta
$$

이고, $100$개 entry의 제곱합에는 다음의 안전한 상계가 따른다.

$$
\begin{aligned}
\|K(\widehat q)-K(q)\|_F
&\le10\,(13B\delta) && \text{$10\times10$ entrywise 상계}\\
&=130B\delta && \text{계수 정리}\\
&\le\frac{130}{6}\bar a^2B^4 && \text{sine cubic bound 대입}.
\end{aligned}
$$

여기서 $B$를 먼저 고정해야 한다. 그때 $\bar a\to0$이면 우변은 $O(\bar a^2)$로 $0$에
수렴하므로, 자유 symbol은 이 compact 저운동량 창에서 균등 수렴한다. $B$를 lattice spacing과
함께 무한대로 보내는 전역 수렴은 이 계산이 말하는 범위 밖이다.

## 20.3 null ray에서 두 편광이 남는 이유

[19장](19_선형화_Einstein_두_편광_수용_정리.md)의 null 방향

$$
q=(\omega,0,0,\omega),\qquad0<\omega\le B
$$

을 넣으면

$$
\widehat q=
\frac{\sin(\bar a\omega)}{\bar a\omega}\,q.
$$

$\bar aB<\pi/2$에서는 이 scalar factor가 양수이므로 $\widehat q$는 영이 아니며 같은 null
ray 위에 있다. harmonic constraint와 residual gauge map은 이 방향의 nonzero scalar 배수에
대해 rank를 바꾸지 않는다. 따라서 harmonic rank $4$, gauge rank $4$이고

$$
10-4-4=2
$$

라는 quotient가 유한 spacing에서도 유지된다. 이는 자유 low-momentum symbol이 [19장](19_선형화_Einstein_두_편광_수용_정리.md)의
수용 기준과 양립한다는 뜻이다.

## 20.4 전역 doubler와 문헌의 범위

central sine symbol은 전역적으로 안전하지 않다. 한 성분에

$$
q_i=\frac{\pi}{\bar a}
$$

를 넣으면 $\widehat q_i=0$이 된다. 원래 운동량이 영이 아닌데 lattice symbol은 영운동량처럼
읽어 추가 zero, 곧 doubler를 만든다. 그래서 이 장의 결론은 $\bar aB<\pi/2$인 compact 창에만
적용하며, global lattice doubler를 배제하지 않는다.

이 장의 좁은 자유ㆍ선형화 맥락은 [Dittrich--Freidel--Speziale (2007)](https://doi.org/10.1103/PhysRevD.76.104020),
[Höhn (2015)](https://doi.org/10.1103/PhysRevD.91.124034),
[Bahr--Dittrich (2009a)](https://arxiv.org/abs/0905.1670),
[Bahr--Dittrich (2009b)](https://arxiv.org/abs/0909.5688),
[Bahr--Dittrich (2010/11)](https://arxiv.org/abs/1011.3667),
[Bahr--Dittrich (2011)](https://arxiv.org/abs/1101.4775)에 한정한다. 이 문헌들은 이 저장소의
symbol이나 상계의 provenance가 아니며, 여기서 spin-foam 유도나 curved/nonlinear constraint
closure를 얻지 않는다.

interacting renormalized limit, CE에서 Einstein--Hilbert 항이 지배한다는 증명도 남아 있다.
따라서 이 장의 uniform bound와 두 편광 보존을 4차원 quantum gravity의 연속극한으로 읽을 수
없다.

## 20.5 재현 범위

무차원 central symbol, gaugeㆍBianchi 항등식, compact 오차 상계와 doubler 대조군은
[lattice_fierz_pauli_refinement.py](../../examples/physics/lattice_fierz_pauli_refinement.py)와
[test_lattice_fierz_pauli_refinement.py](../../tests/test_lattice_fierz_pauli_refinement.py)에 있다.

```powershell
.codex/hooks/python.cmd pytest tests/test_lattice_fierz_pauli_refinement.py -q
```

원장에 기록된 focused 결과는 `25 passed`, source parse는 `428 PASS`다. 이 회귀는 선언한 자유
격자 family의 대수와 상계만 검사한다. spin-foam 유도, curved/nonlinear constraints, interacting
renormalized limit, CE$\to$Einstein--Hilbert dominance는 검사하거나 증명하지 않는다.

다음 [21장](21_두_미분_spin2_유일성.md)은 이 supplied 자유 symbol을 미시에서 유도하지 않은 채,
선언한 두 미분 quadratic ansatz 안에서 Fierz--Pauli 계수 ray가 왜 유일해지는지를 설명한다.
