# 19. 선형화 Einstein 두 편광 수용 정리

이 장은 supplied massless Fierz--Pauli, 곧 linearized Einstein--Hilbert 모형에서 물리 편광이
정확히 두 개임을 보인다. 대칭 텐서의 열 성분에서 시작해 harmonic 조건과 남는 gauge freedom을
차례로 나누면 plus와 cross 두 transverse-traceless 대표만 남는다. 이 결과는 미시 CE 이론이나
refinement가 Einstein 작용을 만들었다는 결과가 아니다. 그런 유도물이 통과해야 할 IR 수용
기준이다.

왜 이 기준이 필요한지는 “성분 수”와 “관측 가능한 자유도 수”가 다르기 때문이다. 먼저 null
파동의 성분과 trace reversal을 정의한다. 다음으로 정확한 선형 map의 rank를 세어
$10\to6\to2$ quotient를 증명한다. 이어 massive Fierz--Pauli의 다섯 mode를 음성 대조군으로
두고, 마지막에 아직 미시 이론에 남아 있는 의무를 적는다.

## 19.1 파동과 gauge rule을 고정한다

Minkowski metric은 $\eta_{\mu\nu}=\operatorname{diag}(-1,1,1,1)$로 둔다. 주파수 $\omega>0$는
차원을 가지지만, null 방향은 그것으로 나눈

$$
\frac{k^\mu}{\omega}=(1,0,0,1),\qquad
\eta_{\mu\nu}k^\mu k^\nu=0
$$

로 고정한다. 따라서 이 장의 rank 계산은 에너지 단위에 의존하지 않는다. 평면파의 polarization
$\varepsilon_{\mu\nu}=\varepsilon_{\nu\mu}$는 대칭이므로 $4(4+1)/2=10$개 성분을 가진다.

trace와 trace-reversed tensor를

$$
\varepsilon=\eta^{\mu\nu}\varepsilon_{\mu\nu},\qquad
\bar\varepsilon_{\mu\nu}=
\varepsilon_{\mu\nu}-\frac12\eta_{\mu\nu}\varepsilon
$$

로 정의한다. supplied massless linearized Einstein 방정식의 harmonic condition은

$$
k^\mu\bar\varepsilon_{\mu\nu}=0
$$

이고, 같은 null momentum에서 남는 gauge 변환은

$$
\delta\varepsilon_{\mu\nu}=k_\mu\xi_\nu+k_\nu\xi_\mu
$$

이다. 여기서 $\xi_\nu$는 네 성분 gauge parameter다. action, 배경, gauge rule은 이 장의
입력이며 앞 장들의 미시 kernel에서 유도하지 않았다.

## 19.2 정확한 $10\to6\to2$ quotient

열 개 대칭 성분을 열벡터로 놓으면 harmonic condition은 $4\times10$ 유리수 행렬이다. 위의
null 방향에서 row reduction은 rank가 $4$임을 준다. 따라서 해공간 $K$의 차원은

$$
\dim K=10-4=6.
$$

residual gauge map은 네 $\xi_\nu$를 열 개 성분으로 보내는 $10\times4$ 행렬이다. 이 행렬의
rank도 $4$이고, harmonic constraint 행렬과 곱하면 영행렬이다. 그러므로 gauge image $G$는
$K$ 안에 든다.

plus와 cross 대표를 공간 $x$-$y$ 평면에서

$$
\varepsilon^{(+)}_{11}=1,\quad
\varepsilon^{(+)}_{22}=-1,\qquad
\varepsilon^{(\times)}_{12}=
\varepsilon^{(\times)}_{21}=1
$$

로 두고 나머지 성분은 $0$으로 둔다. 둘은 trace가 $0$이고 $k^\mu\varepsilon_{\mu\nu}=0$이므로
harmonic kernel에 든다. gauge의 네 열과 이 두 열을 함께 놓은 행렬의 rank는 $6$이다. 따라서
그 여섯 열은 $K$를 모두 span하고, TT 두 열은 gauge image와 겹치지 않는다.

**증명.** harmonic map의 rank가 $4$이므로 rank-nullity로 $\dim K=6$이다. residual gauge
map의 rank가 $4$이고 그 image가 $K$에 포함된다. plus/cross 두 열도 $K$에 속하며, 네 gauge
열과 합친 rank가 $6$이다. 따라서

$$
\begin{aligned}
\dim(K/G)&=\dim K-\dim G && \text{$G\subset K$}\\
&=6-4 && \text{두 exact rank}\\
&=2.
\end{aligned}
$$

또한 plus/cross가 quotient의 두 독립 대표이므로 물리 편광 수는 정확히 $2$다. $\square$

인문학적 언어로 말하면, 열 개 성분은 파동을 적는 좌표의 수다. harmonic 조건은 그 좌표들
사이의 네 관계를 강제한다. 남은 여섯 값 중 네 값은 좌표계를 조금 바꿔도 같은 파동을 다른
방식으로 적은 것이다. 그 중복을 지우고 나면, 검출기가 구분할 수 있는 늘어남/줄어듦의 두
무늬가 plus와 cross로 남는다. 이 비유는 quotient의 증명을 대체하지 않고 그 의미만 설명한다.

gauge invariance와 locality가 tree-level GR amplitude를 강하게 제한하는 맥락은
[Arkani-HamedㆍHuangㆍHuang (2016)](https://arxiv.org/abs/1612.06342)을 따른다. 이 장은 그
일반 amplitude 논증을 재증명하지 않으며, 고정한 null 방향에서의 exact linear algebra만 쓴다.

## 19.3 massive 대조군과 잘못된 산술

massive Fierz--Pauli rest frame에서는 transversality $\varepsilon_{0\nu}=0$의 네 조건과
trace 조건 하나가 서로 독립이다. 열 성분 열 개에 대한 결합 map의 rank는 $5$이므로

$$
10-5=5
$$

개의 massive polarization이 남는다. 이것이 massless quotient의 두 편광과 같지 않다는 음성
대조군이다.

두 편광에 독립 scalar 하나를 숫자로 더해 $2+1=3$이라고 쓸 수는 있다. 이 등식은 bookkeeping일
뿐이며, scalar action, 결합, 안정성, CE와의 연결을 만들지 않는다. 고차 quadratic gravity에서
extra scalar와 massive spin-2 ghost가 나올 수 있다는 맥락은
[Stelle (1995)](https://arxiv.org/abs/hep-th/9509142)를 따른다. 이 인용은 특정 CE 미시 모형에
그 mode들이 있다는 증거가 아니다.

## 19.4 이 수용 정리가 아직 받지 못한 입력

이 장의 두 편광은 supplied action의 결과다. 따라서 실제 승격에는 microscopic/refinement
kernel이 이 작용을 도출해야 한다. nonlinear diffeomorphism과 constraint algebra, refinement
전역에서의 uniform Ward identity, 원치 않는 pole과 residue의 배제, CE에서 Einstein--Hilbert
항이 지배한다는 증명도 각각 필요하다. 이 중 하나라도 없으면 $10\to6\to2$ 계산을 미시
continuum gravity의 증명으로 읽을 수 없다.

## 19.5 재현 범위

정확 유리수 rank 계산과 massive 음성 대조군은
[linearized_spin2_acceptance.py](../../examples/physics/linearized_spin2_acceptance.py)와
[test_linearized_spin2_acceptance.py](../../tests/test_linearized_spin2_acceptance.py)에 있다.

```powershell
.codex/hooks/python.cmd pytest tests/test_linearized_spin2_acceptance.py -q
```

원장에 기록된 focused 결과는 `8 passed`, source parse는 `424 PASS`다. 이 회귀는 supplied
linearized model의 성분ㆍrankㆍquotient만 검사한다. microscopic kernel, nonlinear symmetry,
Ward identity, pole/residue, Einstein--Hilbert dominance는 검사하거나 증명하지 않는다.

[20장](20_게이지_보존_Fierz_Pauli_격자_refinement.md)은 같은 supplied 자유 symbol을
central-difference 격자로 바꾼 저운동량 refinement 모형에서 이 quotient가 유지되는지를 확인한다.
그 격자 모형이 이 action을 유도했다는 뜻은 아니다.
