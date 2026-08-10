# 05b. \(\phi\) Kernel Catalog

## 0. 목표

05a는 잔류 측도에서 잔류장으로 내려가는 형식을 닫았다.

$$
\phi_\beta(x)=\int_{\Gamma_{\mathrm{ns}}}K_\phi(x,\gamma)\,\nu_{\mathrm{ns},\beta}(d\gamma)
$$

이 문서는 \(K_\phi\) 후보를 분류한다. 핵심은 다음이다.

> \(\phi\) pushforward의 적분 형식은 수학적으로 쉽다. 어려운 것은 어떤 커널을 어떤 readout으로 채택하는가다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| endpoint/occupation/curvature/embedding 커널 | `[정의]`; 적분 존재는 `[정리]` | measurable/integrable 조건 아래 닫힘 |
| raw/conditional 잔류 규약 | `[공리: 모델 선택]` | 질량 보존과 모양 보존 중 선택 |
| CE \(\Phi\)와 동일시 | `[미완성]` | 경로 작용의 Hessian readout을 별도 식별해야 함 |
| AGI residual channel | `[예측]` | 사전등록 ablation 검증 대상 |

## 1. 공통 세팅

경로 후보공간을 measurable space

$$
(\Gamma,\mathcal B_\Gamma)
$$

로 두고, readout 공간을

$$
(X,\mathcal B_X)
$$

로 둔다. 비선택 raw 잔류 측도는

$$
\nu_{\mathrm{ns},\beta}
=\mathbf 1_{\Gamma_{\mathrm{ns}}}\mu_\beta
$$

이다.

커널은

$$
K_\phi:X\times\Gamma\to\mathbb R^r
$$

이고, 각 \(x\)에 대해 \(K_\phi(x,\cdot)\in L^1(\nu_{\mathrm{ns},\beta})\)라고 가정한다.

그러면

$$
\phi_\beta(x)
=
\int_\Gamma K_\phi(x,\gamma)\,\nu_{\mathrm{ns},\beta}(d\gamma)
$$

는 잘 정의된다.

## 2. 네 가지 기본 커널

### 2.1 Endpoint kernel

경로의 끝점만 읽는다.

$$
K_{\mathrm{end}}(x,\gamma)
=
k(x,\gamma(T)).
$$

해석:

| 항목 | 의미 |
|---|---|
| 읽는 정보 | 비선택 경로의 도착점 |
| 잃는 정보 | 경로 중간 occupation, 곡률, 위상 |
| 적합한 상황 | outcome 후보, branch endpoint, token 후보 |

조건:

1. endpoint map \(e_T:\gamma\mapsto\gamma(T)\)가 measurable
2. \(k:X\times Y\to\mathbb R^r\)가 measurable
3. \(k(x,e_T(\gamma))\in L^1(\nu_{\mathrm{ns},\beta})\)

### 2.2 Occupation kernel

경로가 지나간 위치의 시간을 읽는다.

$$
K_{\mathrm{occ}}(x,\gamma)
=
\int_0^T k(x,\gamma(t))\,dt.
$$

해석:

| 항목 | 의미 |
|---|---|
| 읽는 정보 | 비선택 경로의 체류/점유 |
| 잃는 정보 | 국소 곡률, action Hessian |
| 적합한 상황 | 장 분포, 히트맵, worldline density |

조건:

1. evaluation map \((t,\gamma)\mapsto\gamma(t)\)가 product-measurable
2. \(k(x,\gamma(t))\)가 \(dt\otimes\nu_{\mathrm{ns},\beta}\)-적분 가능
3. Fubini/Tonelli를 쓰려면 비음수이거나 절대적분 가능

이때

$$
\phi_{\mathrm{occ}}(x)
=
\int_{\Gamma_{\mathrm{ns}}}
\int_0^T k(x,\gamma(t))\,dt\,\nu_{\mathrm{ns},\beta}(d\gamma)
$$

이고, 조건이 충분하면 순서를 바꿀 수 있다.

$$
=
\int_0^T
\int_{\Gamma_{\mathrm{ns}}}
k(x,\gamma(t))\,\nu_{\mathrm{ns},\beta}(d\gamma)\,dt.
$$

### 2.3 Curvature/Hessian kernel

경로 작용의 국소 이차 변화를 읽는다.

$$
K_{\mathrm{curv}}(x,\gamma)
=
\mathcal H_\gamma(x).
$$

대표 후보:

$$
\mathcal H_\gamma(x)
=
\frac{\delta^2S}{\delta\gamma^2}(x;\gamma)
$$

또는 finite-dimensional 근사에서

$$
\mathcal H_\gamma
=
\nabla^2 S(\gamma).
$$

해석:

| 항목 | 의미 |
|---|---|
| 읽는 정보 | 비선택 경로의 접힘 강도, 안정성, 곡률 |
| 잃는 정보 | endpoint 질량만의 단순 해석 |
| 적합한 상황 | CE \(\Phi\), effective residual field |

주의:

> 이 커널을 고른다고 해서 \(\phi=\Phi\)가 자동으로 성립하지 않는다.

닫히는 것은

$$
\phi_{\mathrm{curv}}(x)
=
\int_{\Gamma_{\mathrm{ns}}}
\mathcal H_\gamma(x)\,\nu_{\mathrm{ns},\beta}(d\gamma)
$$

가 정의된다는 사실이다. CE의 대문자 \(\Phi\)와 동일시하려면

$$
\Phi_{\mathrm{res}}(x)
\equiv
\phi_{\mathrm{curv}}(x)
$$

라는 별도 bridge readout을 채택해야 한다.

### 2.4 AGI embedding kernel

후보 trace 또는 token/action embedding을 residual channel로 보낸다.

유한 후보공간 \(A_t\)에서는

$$
K_{\mathrm{emb}}(i,a)
=
P h_a(i)
$$

이고

$$
\phi_{t+1}^{\mathrm{raw}}
=
\sum_{a\ne a_t^*}
\mu_{\beta,t}(a)P h_a
$$

이다.

해석:

| 항목 | 의미 |
|---|---|
| 읽는 정보 | 선택되지 않은 후보의 embedding 방향 |
| 잃는 정보 | 전체 후보 텍스트/경로의 세부 구조 |
| 적합한 상황 | LLM 후보분포, action 후보, hallucination review |

이 커널의 값은 유한합의 `[산출]`이다. 성능 향상은 사전등록 절차를 갖춘 `[예측]`으로만 시험한다.

## 3. Raw와 conditional의 차이

raw readout:

$$
\phi_{\mathrm{raw}}(x)
=
\int K_\phi(x,\gamma)\,\nu_{\mathrm{ns},\beta}(d\gamma)
$$

conditional readout:

$$
\phi_{\mathrm{cond}}(x)
=
\frac{
\int K_\phi(x,\gamma)\,\nu_{\mathrm{ns},\beta}(d\gamma)
}{
\nu_{\mathrm{ns},\beta}(\Gamma)
}
$$

단, 분모가 양수일 때만 정의한다.

| 규약 | 보존하는 것 | 사라지는 것 |
|---|---|---|
| raw | 비선택 총 질량 | shape만 비교하기 어려움 |
| conditional | 비선택 패턴의 모양 | 비선택 질량의 크기 |

따라서 에너지 저장량, 누적 억압량, 선택되지 않은 총 가능성을 읽으면 raw가 맞다. 패턴의 방향, trace의 모양, 재주입 방향을 읽으면 conditional이 맞다.

## 4. 선형성 정리

**정리 4.1**  
\(K_1,K_2\in L^1(\nu_{\mathrm{ns},\beta})\)이고 \(a,b\in\mathbb R\)이면

$$
\phi_{aK_1+bK_2}(x)
=
a\phi_{K_1}(x)+b\phi_{K_2}(x).
$$

**증명.**

적분의 선형성이다.

$$
\int(aK_1+bK_2)\,d\nu
=
a\int K_1\,d\nu+b\int K_2\,d\nu.
$$

\(\square\)

해석: endpoint와 curvature를 섞는 hybrid readout은 수학적으로 가능하다. 그러나 계수 \(a,b\)를 고르는 일은 `[공리: 모델 선택]`이다.

## 5. 유한 후보공간 닫힘

\(\Gamma=\{\gamma_1,\dots,\gamma_N\}\)이면 모든 커널은 행렬 또는 벡터 묶음으로 내려간다.

$$
\phi^{\mathrm{raw}}(x)
=
\sum_{\gamma_i\in\Gamma_{\mathrm{ns}}}
K_\phi(x,\gamma_i)\mu_\beta(\gamma_i).
$$

따라서 finite PreEq runtime에서는 아래가 모두 닫힌다.

| 대상 | 닫힘 |
|---|---|
| 후보분포 \(\mu_\beta\) | 유한 Gibbs 재가중 |
| 선택집합 \(\Gamma_*\) | 최소 에너지 argmin |
| raw residual | 제한측도 |
| conditional residual | 양질량일 때 정규화 |
| embedding \(\phi\) | 유한합 |

이 부분은 `reality_stone.clarus.pre_eq`와 `tests/test_pre_eq.py`로 코드 검증할 수 있다.

## 6. CE \(\Phi\)로 올리기 위한 체크리스트

curvature kernel을 CE \(\Phi\)와 연결하려면 아래 표를 채워야 한다.

| 항목 | 필요한 내용 | 판정 |
|---|---|---|
| 경로공간 \(\Gamma\) | 위상, measurable 구조 | `필수` |
| 작용 \(S[\gamma]\) | 정의역과 regularity | `필수` |
| Hessian \(\delta^2S/\delta\gamma^2\) | 존재 또는 약한 의미 | `필수` |
| 비선택 측도 | \(\nu_{\mathrm{ns},\beta}\) finite measure | `필수` |
| integrability | \(K_{\mathrm{curv}}\in L^1(\nu_{\mathrm{ns},\beta})\) | `필수` |
| readout 규약 | raw/conditional/hybrid 중 선택 | `[공리: 모델 선택]` |
| 물리 동일시 | \(\phi_{\mathrm{curv}}=\Phi_{\mathrm{res}}\) 여부 | `[미완성]`; 채택 시 `[공리: 물리 사상]` |

이 표가 비면 \(\phi\) pushforward는 형식만 닫힌다. 표가 채워지면 해당 가정 아래의 `[정리]`를 적용할 수 있다.

## 7. 다음 실험 연결

AGI toy gate에서는 embedding kernel이 가장 먼저 테스트 가능하다.

$$
E_{t+1}(a)
=
E_{\mathrm{base},t+1}(a)
-\alpha_\phi\langle h(a),\phi_t\rangle.
$$

최소 ablation:

| 조건 | 의미 |
|---|---|
| \(\alpha_\phi=0\) | 잔류 재주입 없음 |
| \(\alpha_\phi>0\) | 선택되지 않은 후보의 압축 재주입 |

성능이 좋아지면 embedding kernel bridge가 실험 축을 얻는다. 성능이 나빠지면 커널 \(P h_a\), raw/conditional 규약, decay \(\lambda_\phi\)를 바꿔야 한다.

## 8. 결론

이 문서에서 닫힌 것은 다음이다.

$$
\boxed{
\text{잔류 측도}
\xrightarrow{K_\phi}
\text{잔류장 readout}
}
$$

다만 어떤 \(K_\phi\)가 CE의 물리 \(\Phi\)인지, 또는 AGI runtime의 유효 residual channel인지는 아직 bridge와 실험의 문제다.
