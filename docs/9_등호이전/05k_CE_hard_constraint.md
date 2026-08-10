# 05k. Hard Constraint와 유한모드 분율

## 0. 범위

Hard constraint는 measurable event로 prior를 조건화하는 연산이다. 이
문서는 조건부 측도의 존재, Brownian kinetic event의 영측도 no-go,
유한 Gaussian 모드의 분율과 layer-cake 항등식만 보존한다.

## 1. 조건부 측도의 존재

**[정의]** \(X\)를 Polish space, \(\mu\in\mathcal P(X)\),
\(\mathcal I:X\to[0,\infty]\)를 measurable한 무차원 functional이라
하자. 무차원 threshold \(c\)에 대해
\[
A_c:=\{x:\mathcal I(x)<c\}
\]
라 둔다.

**[정리]** Hard-constrained probability
\[
\mu_c(B):=\frac{\mu(B\cap A_c)}{\mu(A_c)}
\]
는 정확히 \(\mu(A_c)>0\)일 때 정의된다.

이는 정의의 필요충분조건이다. 물리 작용 \(S_E\)를 쓰려면 먼저
\(\mathcal I=S_E/\hbar\) 또는 \(S_E/S_*\)로 무차원화하고,
threshold도 \(c=S_{\rm th}/\hbar\) 또는 \(S_{\rm th}/S_*\)로 쓴다.

## 2. Brownian kinetic constraint no-go

**[공리: 모형]** \(X=C^0_{x_i,x_f}([0,1];\mathbb R^d)\),
\(\mu\)를 variance parameter \(\sigma>0\)인 Brownian bridge law라
하고
\[
\mathcal I_{\rm kin}(\gamma)
=
\begin{cases}
\displaystyle\int_0^1|\dot\gamma(t)|^pdt,
&\gamma\in W^{1,p}_{x_i,x_f},\\
+\infty,&\text{그 밖},
\end{cases}
\qquad p\geq1
\]
로 둔다.

**[정리]** 모든 유한 \(c\)에 대해
\[
\mu\{\mathcal I_{\rm kin}<c\}=0.
\]
따라서 이 event로 Brownian bridge를 조건화할 수 없다.

**증명.**
\[
\{\mathcal I_{\rm kin}<c\}\subset W^{1,p}_{x_i,x_f}
\]
이고 [05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md)
정리 2.2에 의해 Brownian bridge는 이 공간에 질량 0을 준다.
\(\square\)

이는 Brownian \(C^0\) route와 finite-kinetic-action Sobolev route를
같은 prior로 섞을 수 없다는 no-go다. 모든 continuum prior에 대한
보편 명제는 아니다.

## 3. 유한 독립 Gaussian 모드

**[공리: 모델 선택]** 독립 표준정규 변수
\(z_1,\dots,z_N\)과
\[
U_N=\frac12\sum_{j=1}^Nz_j^2
\]
를 택한다.

**[정리]**
\[
U_N\sim\operatorname{Gamma}\!\left(\frac N2,1\right).
\]
따라서 \(u>0\)일 때
\[
\mathbb P(U_N<u)
=
\frac{\gamma(N/2,u)}{\Gamma(N/2)}>0.
\]

**증명.** \(\sum_jz_j^2\)는 자유도 \(N\)인 chi-square 변수다. Gamma
density는 \(u>0\)에서 양수다. \(\square\)

### 3.1 고정 분율 threshold

**[정리]** \(q\in(0,1)\)를 고정하고 \(u_N\)이
\(\mathbb P(U_N<u_N)\to q\)를 만족한다고 하자. 그러면, 그리고 오직
그러할 때,
\[
u_N
=
\frac N2+z_q\sqrt{\frac N2}+o(\sqrt N),
\qquad
z_q=\Phi_{\rm normal}^{-1}(q).
\]

**증명.** 독립 변수 \((z_j^2-1)/2\)의 central limit theorem으로
\[
\frac{U_N-N/2}{\sqrt{N/2}}
\Rightarrow\mathcal N(0,1).
\]
연속이고 엄격히 증가하는 정규분포의 quantile convergence가 필요충분
조건을 준다. \(\square\)

\(q\)는 이 정리의 입력이다. 특정 \(q\)를 택하는 것은
**[공리: 모델 선택]** 또는 자료에서 역산하는 **[경험식]**이지, Gamma
law의 예측이 아니다.

### 3.2 개수 분율과 action 분율

**[정리]** \(k=N/2\)와 유한 \(u>0\)에 대해
\[
P_k(u):=\mathbb P(U_N<u)=\frac{\gamma(k,u)}{\Gamma(k)}
\]
이고, 잘린 action 분율은
\[
Q_k(u):=
\frac{\mathbb E[U_N\mathbf1_{\{U_N<u\}}]}{\mathbb EU_N}
=
\frac{\gamma(k+1,u)}{\Gamma(k+1)}.
\]
두 분율의 차이는
\[
P_k(u)-Q_k(u)
=
\frac{u^ke^{-u}}{\Gamma(k+1)}>0.
\]
\(u=k+O(\sqrt k)\)이면 이 차이는 \(O(k^{-1/2})\)다.

**증명.** Gamma density를 직접 적분한 뒤
\[
\gamma(k+1,u)=k\gamma(k,u)-u^ke^{-u}
\]
를 사용한다. 마지막 차수는 central window에서 Stirling 공식으로
따른다. \(\square\)

## 4. Threshold와 최소자 농축

**[정리]** \(\mathcal I:X\to\mathbb R\cup\{+\infty\}\)가
[05_CE_브리지.md](05_CE_브리지.md) 1절의 good-rate·recovery 조건을
만족하고 \(m=\min_X\mathcal I\)라 하자. \(c>m\)이면
\(\mu(A_c)>0\)이고 조건부 prior \(\mu_c\)도 같은 최소 높이에서 recovery
mass를 갖는다. 따라서
\[
\frac{e^{-\beta\mathcal I}\mu_c}
{\int e^{-\beta\mathcal I}\,d\mu_c}
\]
는 원래 최소집합 \(M=\{\mathcal I=m\}\)의 모든 열린 이웃에 농축한다.

**증명.** \(0<\delta<c-m\)이면
\[
\{\mathcal I<m+\delta\}\subset A_c.
\]
원래 recovery mass가 이 집합에 양의 질량을 주므로
\(\mu(A_c)>0\)이고 조건부 prior에서도 같은 집합의 질량이 양수다.
Good-rate 성질은 functional의 성질이므로 그대로다. 05 브리지의 농축
정리를 적용한다. \(\square\)

일반 measurable constraint \(A\)가 원래 최소집합을 제거하면 restricted
support 위의 새 최소집합을 사용해야 한다. 정확한 조건은
[05j_CE_supp_scaling_audit.md](05j_CE_supp_scaling_audit.md) 4절에
있다.

## 5. Smooth tilt와 threshold 분율

**[정리: layer cake]** \(\Phi:X\to[0,\infty]\)가 measurable이면
\[
\mathbb E_\mu e^{-\Phi}
=
\int_0^\infty e^{-t}\,\mu(\Phi\leq t)\,dt.
\]

**증명.** \(0\leq e^{-\Phi}\leq1\)에 Cavalieri identity를 적용하고
\(s=e^{-t}\)로 치환한다.
\[
\mathbb Ee^{-\Phi}
=\int_0^1\mu(e^{-\Phi}\geq s)\,ds
=\int_0^\infty e^{-t}\mu(\Phi\leq t)\,dt.
\quad\square
\]

따라서 smooth tilt의 평균은 한 threshold의 분율이 아니라 모든 threshold
분율의 가중 평균이다.

**[정리: mean-field bound]** \(0\leq\Phi\leq M<\infty\) almost surely면
\[
1
\leq
\frac{\mathbb Ee^{-\Phi}}{e^{-\mathbb E\Phi}}
\leq
\exp\!\left[
\frac{e^M}{2}\operatorname{Var}(\Phi)
\right].
\]

**증명.** 아래쪽은 Jensen 부등식이다. \(Y=\Phi-\mathbb E\Phi\)에
Taylor 정리를 쓰면
\[
\mathbb Ee^{-Y}
\leq
1+\frac{e^M}{2}\mathbb EY^2
\leq
\exp\!\left[
\frac{e^M}{2}\operatorname{Var}(\Phi)
\right].
\]
양변에 \(e^{-\mathbb E\Phi}\)를 곱한다. \(\square\)

이 bound를 특정 CE observable에 적용하려면 \(\Phi\)의 정의, 유계성,
분산과 observable map을 별도로 제시해야 한다.

## 6. 남은 물리 선택

- Gaussian 독립모드 prior와 mode 수 \(N\)
- threshold quantile \(q\)
- continuum에서 양의 질량을 갖는 hard event
- \(\Phi\)의 국소 작용·장·측정량 사상

이 항목들은 각각 **[공리]**, **[경험식]** 또는 **[미완성]**으로
분리하며, 한 분율을 다른 분율에 맞춘 사실을 독립 유도로 읽지 않는다.
