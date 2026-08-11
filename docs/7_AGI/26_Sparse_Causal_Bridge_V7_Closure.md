# Sparse Causal Bridge V7 폐쇄 감사

> 날짜: 2026-08-11
> 범위: 4차원 합성 동역학의 단일 원점 H20 자유 롤아웃
> 정본 입력: `experiments/preregistration/sparse_causal_bridge_v7.json`
> 관측 기록: `artifacts/agi/sparse_causal_bridge_validation_v7.json`

이 문서는 AGI 성취를 보고하지 않는다. V7은 CE-AGI 전체가 아니라, 이미 실패한 V5 자유 롤아웃 부모에서 희소 성분의 추가 기여와 대조군 대비 예측 성능을 한 번 더 검사한 합성 브리지 폐쇄 실험이다. V7의 등록 조건은 결합식이므로 일부 조건의 충족만으로 경로 전체를 승격하지 않는다.

## 1. 정의역과 정의

**[정의]** 한 seed의 관측열은 무차원 4차원 상태 $x_t\in\mathbb R^4$이며, 모델은 $x_{0:80}$만 읽고 원점 $t=80$에서 H20 경로 $x_{81:100}$을 자유 롤아웃한다. H5는 같은 경로의 첫 5행인 진단량이며 게이트가 아니다.

**[정의]** 훈련 구간에서 고정한 좌표별 scale $s_j>0$에 대해 모델 $m$의 seed 단위 오차를

\[
E_m
=\sqrt{\frac1{20\cdot4}
\sum_{h=1}^{20}\sum_{j=1}^{4}
\left(\frac{\hat x^{(m)}_{80+h,j}-x_{80+h,j}}{s_j}\right)^2}
}
\]

로 둔다. 독립 단위는 trajectory row가 아니라 seed 하나다.

**[정의]** `sparse_consensus`는 sparse parent, stable adaptive dense, persistence의 세 전문가를 결합한다. `no_sparse_consensus`는 뒤의 두 전문가만 결합하고, `symmetric_dense_consensus`는 sparse parent 대신 같은 probe budget의 dense latent 전문가를 쓴다.

## 2. 전제와 사전 예측

**[공리: 모델 선택]** 각 consensus는 동일한 prefix, 동일한 정규화 오차, 동일한 inverse-root MSE 규칙으로 자기 가중치를 따로 맞춘다. validation 결과를 본 뒤 route, 전문가, seed 수 또는 임계값을 바꾸지 않는다.

**[공리: 외부 입력]** V4와 V5의 동결 코드·설정·아티팩트를 부모 기준으로 사용한다. V1--V5는 등록·코드·결과가 Git에 함께 처음 등장하므로 사전등록 시점을 독립적으로 인증할 수 없고, V6은 등록만 존재하며 실행 증거가 없다. V7은 구현 전에 등록을 잠갔다.

**[예측]** validation seed `77100..77195`의 96쌍에서 다음 조건을 모두 요구했다.

1. $L_{95}(E_{\mathrm{no\ sparse}}-E_{\mathrm{sparse}})>0$.
2. $L_{95}(E_{\mathrm{V5}}-E_{\mathrm{sparse}})>0$.
3. $L_{95}(E_{\mathrm{persistence}}-E_{\mathrm{sparse}})>0$.
4. adaptive dense 대비 paired log-ratio 상한이 $\log 1.05$ 이하이고, symmetric dense consensus 대비 상한이 $\log 1.02$ 이하일 것.
5. 모든 출력이 유한하고, 최대 동적 성분 pathwise Jacobian radius가 $0.98$ 이하이며, 미래 관측 read가 0일 것.

test seed `78100..78195`는 validation의 모든 결합 조건이 충족될 때만 열도록 잠갔다.

## 3. 무차원 감사

| 코어 항목 | 차원 | 정규화 | 판정 |
|---|---:|---|---|
| $(\hat x_j-x_j)/s_j$ | 0 | 훈련 구간 좌표 scale $s_j$ | 무차원 |
| $E_m$ | 0 | 정규화 잔차의 RMS | 무차원 |
| $E_a-E_b$ | 0 | 같은 정의의 오차 차이 | 무차원 |
| $\log(E_a/E_b)$ | 0 | 양의 무차원 오차 비 | 무차원 |
| consensus weight | 0 | 무차원 prefix MSE의 inverse-root | 무차원 |
| normalized-map Jacobian radius | 0 | 무차원 상태 사상의 고유값 크기 | 무차원 |

`tests/test_dimensionless.py` 10개가 통과했고 `dimensionless.py` checker가 정상 종료했다. 이 결과는 차원 정합성만 말하며 예측 성능이나 AGI 타당성을 보장하지 않는다.

## 4. validation 관측 비교

**[경험식]** 96개의 고정 validation seed에서 얻은 평균 normalized H20 path RMSE는 다음과 같다. 작을수록 좋다.

| 모델 | 평균 |
|---|---:|
| `symmetric_dense_consensus` | 0.563389 |
| `sparse_consensus` | 0.563406 |
| `v5_sparse_parent` | 0.556915 |
| `persistence` | 0.573898 |
| `no_sparse_consensus` | 0.583958 |
| `stable_adaptive_dense_prefix_free` | 0.642702 |

paired 비교는 다음과 같다.

| 등록 비교 | 평균 또는 기하평균 | 95% 구간 | 등록 관계 |
|---|---:|---:|---|
| sparse 기여: `no_sparse - sparse` | +0.020552 | [+0.008174, +0.032930] | 충족 |
| V5 부모 개선: `v5 - sparse` | -0.006491 | [-0.027431, +0.014449] | 미충족 |
| persistence 개선: `persistence - sparse` | +0.010492 | [-0.016487, +0.037470] | 미충족 |
| adaptive dense 대비 오차비 | 0.891979 | log-ratio [-0.159883, -0.068743] | 비열등 조건 충족 |
| symmetric dense 대비 오차비 | 0.999869 | log-ratio [-0.000579, +0.000318] | 비열등 조건 충족 |

미래 관측 read는 0이고 최대 관측 index는 80이었다. 모든 예측은 유한했지만 최대 동적 성분 pathwise Jacobian radius는 `1.114309`로 등록 상한 `0.98`을 넘었다. 이는 표본 경로의 안정성 검사이며 전역 수축 정리로 읽지 않는다.

**[산출]** 등록된 결합식에서 V5 부모 개선, persistence 개선, pathwise 안정성의 세 항이 거짓이므로 V7 validation 결합식은 거짓이다. 따라서 test 잠금 조건도 거짓이다.

**[경험식]** sparse 성분 제거 대비 paired 하한은 양수였다. 그러나 이는 결합 실험의 한 하위 관측량일 뿐이며, V7 route 성공·일반적 희소성 우위·인과 발견·world model 또는 AGI 효능으로 승격할 수 없다.

## 5. 폐쇄 범위

**[미완성]** locked test는 열리지 않았고 `artifacts/agi/sparse_causal_bridge_test_v7.json`도 생성되지 않았다. 따라서 test 일반화 증거는 0이다.

**[산출]** 사전등록 규칙에 따라 이 V7 route는 여기서 닫는다. 같은 validation을 본 뒤 임계값을 바꾸거나 두 번째 V7 route를 시험하지 않는다.

이 폐쇄가 부정하는 것은 이 합성 환경에서 등록한 symmetric consensus 경로의 결합 성공 주장이다. 모든 가능한 sparse 모델, SNN substrate, CE 코어 수학 또는 AGI 가능성 전체를 부정하는 보편 반례는 아니다. 반대로 어느 항목도 V7에서 지지되었다고 승격되지 않는다.

## 6. 재현 잠금과 자원

- registration raw SHA-256: `134ddaa793170b898649b79e11407c10f35d1468ba95701544a06905d9448c3e`
- registration canonical SHA-256: `2d1c06cb9259e52e435e28017b82d89924c4c305c0dc81b29beadf78ede13365`
- merged registration SHA-256: `3cfa4ddc9bb6ab04bb7b37403780ef2fd4a894d26e7c45c1c84e062434fb4259`
- implementation SHA-256: `7abf17f260f0046cb6eace7ed57e1115657c2dd4d32bd1024bc7c1940e910310`
- test definition SHA-256: `866e9e89274419b17e4b33a63df519c89565e6763480b7c8537b5f7b0ec88041`
- validation artifact SHA-256: `f172447ecc0d19ac206c6625bf5911805f28214bb5adc1d2a215c59dc3bc4e12`
- 계산량: 96 seeds, seed당 8 component rollouts, 총 768 rollouts, 외부 다운로드 0 byte

validation 재현 명령은 다음과 같다. 등록 조건이 충족되지 않으므로 종료 코드는 1이 정상적인 폐쇄 결과다.

```powershell
.\.venv\Scripts\python.exe examples\agi\reliability_rollout_bridge_gate.py --config experiments\preregistration\sparse_causal_bridge_v7.json --split validation
```

`--split test`는 실행하지 않는다.
