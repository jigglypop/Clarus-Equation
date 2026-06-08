# 07a. Toy Runtime Gate

## 0. 목표

07장은 AGI runtime에서 후보분포와 잔류장 \(\phi\)를 연결했다. 이 문서는 그 구조를 작은 알고리즘으로 내린다.

핵심:

> 선택 전 분포를 버리지 말고, 선택되지 않은 질량을 압축해 다음 step의 에너지 또는 logits에 재주입한다.

현재 판정:

| 항목 | 판정 |
|---|---|
| 후보분포 재가중 | `Exact` |
| toy gate 알고리즘 | `Tooling` |
| 실제 LLM 성능 개선 | `Open/Experiment` |

## 1. 입력

한 step \(t\)에서 필요한 값:

| 기호 | 의미 |
|---|---|
| \(l_t(a)\) | 모델 logit |
| \(\mu_{0,t}(a)\) | 초기 후보분포 |
| \(E_t(a)\) | 문맥/위험/도구/기억 에너지 |
| \(\beta_t\) | 조건 강도 |
| \(h(a)\) | 후보 embedding 또는 trace vector |
| \(\phi_t\) | 현재 잔류장 |

초기 후보분포는

$$
\mu_{0,t}(a)=\operatorname{softmax}(l_t)(a)
$$

이다.

## 2. 에너지 합성

먼저 base energy를 여러 gate의 합으로 둔다.

$$
E^{\mathrm{base}}_t(a)
=
w_gE_{\mathrm{goal}}(a)
+w_rE_{\mathrm{risk}}(a)
+w_uE_{\mathrm{tool}}(a)
+w_mE_{\mathrm{memory}}(a)
$$

잔류장 재주입을 포함한 총 에너지는

$$
E_t(a)
=
E^{\mathrm{base}}_t(a)
-\alpha_\phi\langle h(a),\phi_t\rangle
$$

이다. \(\phi_t\)와 잘 맞는 후보는 에너지가 낮아진다.

## 3. PreEq 재가중

재가중된 분포:

$$
\mu_{\beta,t}(a)
=
\frac{e^{-\beta_tE_t(a)}\mu_{0,t}(a)}
{\sum_b e^{-\beta_tE_t(b)}\mu_{0,t}(b)}
$$

logit 형태:

$$
s_t(a)
=
l_t(a)-\beta_tE^{\mathrm{base}}_t(a)
+\beta_t\alpha_\phi\langle h(a),\phi_t\rangle
$$

$$
\mu_{\beta,t}=\operatorname{softmax}(s_t)
$$

## 4. Manifest 선택

가장 단순한 readout은 argmax다.

$$
a_t^*=\operatorname*{argmax}_a\mu_{\beta,t}(a)
$$

sampling runtime에서는

$$
a_t^*\sim\mu_{\beta,t}
$$

로 둔다.

## 5. 비선택 잔류 압축

비선택 후보공간:

$$
A_{\mathrm{ns},t}=A_t\setminus\{a_t^*\}
$$

raw residual:

$$
\nu_{\mathrm{ns},t}(a)
=
\mathbf1_{a\ne a_t^*}\mu_{\beta,t}(a)
$$

잔류 질량:

$$
q_t=\sum_{a\ne a_t^*}\mu_{\beta,t}(a)
=1-\mu_{\beta,t}(a_t^*)
$$

압축:

$$
\phi_{t+1}
=
\lambda_\phi\phi_t
+\eta_\phi
\sum_{a\ne a_t^*}\nu_{\mathrm{ns},t}(a)P h(a)
$$

여기서 \(P\)는 embedding을 residual channel로 보내는 projection이다.

## 6. Gate 지표

선택 안정성을 보기 위해 네 값을 추적한다.

| 지표 | 식 | 의미 |
|---|---|---|
| entropy | \(H_t=-\sum_a\mu_{\beta,t}(a)\log\mu_{\beta,t}(a)\) | 후보 모호함 |
| margin | \(\mu_{\beta,t}(a_1)-\mu_{\beta,t}(a_2)\) | 1등과 2등 차이 |
| residual mass | \(q_t\) | 버려지지 않은 비선택 질량 |
| risk energy | \(E_{\mathrm{risk}}(a_t^*)\) | 선택된 후보의 위험 |

review 조건 예:

$$
H_t>\tau_H
\quad\text{or}\quad
q_t>\tau_q
\quad\text{or}\quad
E_{\mathrm{risk}}(a_t^*)>\tau_r
$$

이면 바로 manifest하지 않고 review 상태로 보낸다.

## 7. Pseudocode

```text
input logits l, base energies E_base, embeddings h, residual phi

mu0 = softmax(l)
score[a] = l[a] - beta * E_base[a] + beta * alpha_phi * dot(h[a], phi)
mu = softmax(score)

a_star = readout(mu)

residual_mass = 1 - mu[a_star]
phi_next = lambda_phi * phi

for a in topk_except(mu, a_star):
    phi_next += eta_phi * mu[a] * P(h[a])

if entropy(mu) > tau_H or residual_mass > tau_q or E_risk[a_star] > tau_r:
    state = "review"
else:
    state = "manifest"

return a_star, phi_next, state
```

## 8. Toy 검증 항목

| 조건 | 기대 결과 |
|---|---|
| \(\beta=0\) | 기존 softmax와 동일 |
| \(E(a)=0\) | 에너지 gate 없음 |
| \(\beta\to\infty\) | 최소 에너지 후보로 농축 |
| \(\mu_{\beta,t}(a_t^*)=1\) | \(q_t=0\), 새 잔류 없음 |
| \(\lambda_\phi=0\) | 잔류가 한 step만 유지 |
| \(\alpha_\phi=0\) | 잔류 재주입 ablation |

## 9. Hallucination gate 버전

claim 후보 \(c\)에 대해

$$
E(c)=E_{\mathrm{evidence}}(c)+E_{\mathrm{contradiction}}(c)+E_{\mathrm{source}}(c)
$$

로 둔다.

accepted claim은 manifest된다. rejected claim은 완전히 삭제하지 않고 \(\phi\)에 저장한다.

$$
\phi_{t+1}^{\mathrm{claim}}
=
\lambda_\phi\phi_t
+\eta_\phi
\sum_{c\ne c^*}
\mu_{\beta,t}(c)P h(c)
$$

이렇게 하면 다음 답변에서 "방금 배제했던 후보"를 망각하지 않고, contradiction check의 배경장으로 쓸 수 있다.

## 10. 결론

toy gate의 최소 실험은 \(\alpha_\phi=0\)과 \(\alpha_\phi>0\)을 비교하는 ablation이다. 성능이 좋아지면 07장의 잔류장 bridge는 실험 축을 얻고, 나빠지면 \(\phi\) 압축 또는 재주입 커널을 바꿔야 한다.
