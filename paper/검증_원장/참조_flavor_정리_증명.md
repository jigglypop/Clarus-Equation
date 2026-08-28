# CE 핵심 정리 증명 — flavor

<!-- 출처: paper/검증_원장/참조_핵심_정리_증명.md — §6,16 이동 | 이동일: 2026-08-23 (MULTIREPO_PLAN.md P2-6) -->

이 문서는 CE 문서에서 반복 사용하는 순수 수학 정리 중 flavor 도메인이
주로 소비하는 것을 보존한다. 물리적 동일시나 관측 비교는 포함하지 않는다.
정리 번호는 원파일 `참조_핵심_정리_증명.md`의 번호를 유지한다.

<a id="koide-angle"></a>

## 6. Koide 조건과 평균축 각도

**[정리]** $x=(x_1,x_2,x_3)\in(0,\infty)^3$와 정규화된 민주축

$$
u:=\frac1{\sqrt3}(1,1,1)
$$

를 두고

$$
Q:=\frac{x_1^2+x_2^2+x_3^2}{(x_1+x_2+x_3)^2}
$$

라 하자. 그러면 $Q=2/3$인 것과 $x$와 $u$ 사이의 각이
$\pi/4$인 것은 동치다.

**증명.** 두 벡터 사이의 각을 $\vartheta$라 하면 양의 성분 때문에
$0\leq\vartheta<\pi/2$이고

$$
\cos^2\vartheta
=\frac{(x_1+x_2+x_3)^2}{3(x_1^2+x_2^2+x_3^2)}
=\frac1{3Q}.
$$

따라서 $Q=2/3\iff\cos^2\vartheta=1/2\iff\vartheta=\pi/4$다.
$\square$

질량에 적용할 때는 $x_i=\sqrt{m_i}$라는 별도 정의를 사용한다. 실제
질량이 이 조건을 만족하는 동역학은 이 정리의 내용이 아니다.

<a id="flavor-realization"></a>

## 16. 임의의 unitary 혼합을 실현하는 질량행렬

**[정리]** $D_u,D_d$를 무차원인 음이 아닌 Yukawa 고유값의 대각행렬,
$V$를 임의의 $3\times3$ unitary 행렬이라 하자. Dirac Yukawa 행렬을

$$
Y_u=D_u,\qquad Y_d=VD_d
$$

로 선택하면 그 left diagonalization 행렬들의 상대적 혼합은 $V$다.
또한 음이 아닌 대각행렬 $D_\nu$와 unitary $U$에 대해

$$
M_\nu=U^*D_\nu U^\dagger
$$

는 복소 대칭 Majorana 질량행렬이고 $U^TM_\nu U=D_\nu$를 만족한다.

**증명.** $Y_u=I D_u I^\dagger$,
$Y_d=V D_d I^\dagger$는 singular-value decomposition이므로
$U_{uL}=I$, $U_{dL}=V$를 택할 수 있고
$U_{uL}^\dagger U_{dL}=V$다. Majorana 경우 $M_\nu^T=M_\nu$이고

$$
U^TM_\nu U=(U^TU^*)D_\nu(U^\dagger U)=D_\nu.
$$

$\square$

이는 CKM·PMNS 값을 일관된 장론 안에 실현할 수 있다는 존재구성이다. 그
값을 CE 코어에서 선택하는 동역학이나 매개변수 축소를 제공하지는 않는다.
quark 질량을 입력할 때는
$D_{u,d}=\sqrt2\,\operatorname{diag}(m_{u,d})/v$로 바꾼 뒤 이 정리를
적용한다.
