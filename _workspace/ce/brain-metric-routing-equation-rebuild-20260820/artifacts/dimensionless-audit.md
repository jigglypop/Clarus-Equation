# 무차원성 감사

Status: PASS

| 식 또는 코어 인자 | 차원 | 무차원? | 조건 |
|---|---|---|---|
| $\partial_a\log p$ | $[z_a]^{-1}$ | 아니오 | covector 성분의 올바른 단위 |
| $G_{ab}$ | $[z_a]^{-1}[z_b]^{-1}$ | 아니오 | metric 성분의 올바른 단위 |
| $dz^aG_{ab}dz^b$ | $1$ | 예 | line element의 핵심 무차원량 |
| $\lambda$ in $G+\lambda G_{\rm ref}$ | $1$ | 예 | $G_{\rm ref}$가 $G$와 같은 tensor/단위일 때 |
| generalized eigenvalue of $G_0^{-1}G_1$ | $1$ | 예 | 두 tensor가 비교 가능한 같은 공간에 있을 때 |
| $\log\rho_i$ in AIRM | $1$ | 예 | $\rho_i>0$인 무차원 generalized eigenvalue |
| $\log p_1-\log p_0$ | $1$ | 예 | 동일 target measure의 log-density ratio |
| $R$, $\Delta R_{\rm ctx}$ | nat/sample | 예 | 두 식 모두 test sample 평균으로 정규화 |
| $H,\ell,\delta$ | time 또는 bin | 아니오 | exp/log에 직접 넣지 않으며 기록 간 비교는 초로 환산 |

Gaussian 평균항은 $[\partial_a\mu]=[o][z_a]^{-1}$와 $[\Sigma^{-1}]=[o]^{-2}$이므로 $[z_a]^{-1}[z_b]^{-1}$이다. covariance 미분항도 같은 단위를 가진다.

차원 상태: **PASS**. 다만 무차원성은 식의 정합성만 보이며 생물학적 정당성이나 식별 가능성을 증명하지 않는다.

코드 검증:

- `tests/test_dimensionless.py`: `15 passed`.
- `reality_stone/python/reality_stone/clarus/dimensionless.py`: exit code `0`.
- 실행기: policy-blocked `uv`가 아니라 허용된 system CPython `3.11.9`, bytecode/cache 비활성화. 이 검사는 기존 CE dimensionless registry 회귀이며 새 신경식의 기호 감사는 위 표가 담당한다.
