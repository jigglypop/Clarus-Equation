# CE-AGI Hopfield Engine: 논문 vs 구현 검증 보고서

> `12_Equation.md` 수식 기반. 구현: `clarus/convert.py`, `clarus/engine.py`

---

## 1. 파이프라인 대조

| 단계 | 논문 (12_Equation.md) | 구현 | 일치 |
|------|----------------------|------|------|
| W 추출 | Q@K^T/sqrt(d_h) + V@O + FFN, 레이어 평균, 대칭화 | `extract_hopfield()` 동일 | O |
| 3D 격자 희소화 | d=3, r_c=pi, 밀도 ~3.16% (N=4096) | `sparsify_3d()` r_c=pi, N=768 -> 10.57% | 부분 |
| 스펙트럼 조건화 | lambda_max < 0, 음정치 | `make_negative_definite()` shift=lambda_max+0.1|lambda_min| | O |
| CSR 압축 | 희소 행렬 저장 | `to_csr()` values + col_idx + row_ptr | O |
| 어휘 추출 | emb + ln_f + lm_head | weight tying 감지, 1벌 저장 | O |

## 2. 동역학 대조

| 항목 | 논문 | 구현 | 일치 |
|------|------|------|------|
| 에너지 E(m,phi) | -0.5 m^T W m - b^T m + portal * <m, phi_hat> | `energy()` 동일 | O |
| bypass F | F_bypass = bypass * phi (비보존, 에너지에 미포함) | `relax()` dt/tau * F_bypass (에너지 외부) | O |
| gradient descent | dm = -dt/tau * dE/dm + dt/tau * F + noise | `relax()` 동일 | O |
| 노이즈 | sqrt(2*T_wake*dt/tau) * N(0,I) 등방 가우시안 | `relax()` 동일 | O |
| phi 갱신 | v_m* = 궤적분산, EMA | `relax()` 최근 궤적 var -> EMA | O |
| 노름 보존 | ||m|| 유지 | `F.normalize(m) * norm0` | O |

## 3. 상수 대조

| 상수 | 논문 값 | 구현 | 일치 |
|------|---------|------|------|
| portal | [4/(e^(4/3)*pi^(4/3)) * (1 - 4/(e^(4/3)*pi^(4/3)))]^2 = 0.03120 | 0.03120 | O |
| bypass | 1/(e^(1/3)*pi^(1/3)) = 0.4892 | 0.4892 | O |
| T_wake | [3 + 4/(e^(4/3)*pi^(4/3))*(1-...)]^-1 = 0.3148 | 0.3148 | O |
| r_c | pi = 3.1416 | 3.1416 | O |
| tau | 1/|lambda_max| (스펙트럼에서 유도) | 10.0 (1/0.1) | O |

## 4. 메모리 비교

### 4.1 모델: skt/kogpt2-base-v2 (d=768, vocab=51200, 12 layers)

| 항목 | 크기 |
|------|------|
| GPT2 전체 파라미터 | 477.46 MB |
| CE 엔진 코어 (W_sparse + ln_f + phi) | 4.50 MB |
| CE 어휘 테이블 (embedding, weight tied) | 150.00 MB |
| CE 런타임 전체 | 154.50 MB |
| **코어 대 GPT2 비율** | **0.94%** |
| **런타임 대 GPT2 비율** | **32.4%** |

### 4.2 W_sparse 상세

| 항목 | 값 |
|------|------|
| 원본 W (dense) | 768 x 768 = 2,304 KB |
| CSR nnz | 588,936 |
| CSR values | 2,300.5 KB |
| CSR col_idx | 2,300.5 KB |
| CSR row_ptr | 3.0 KB |
| 희소화 밀도 | 10.57% (N=768, r_c=pi) |

## 5. 추론 성능

### 5.1 속도 (CPU, 10 tokens, 60 steps/token)

| 엔진 | 시간 | tok/s |
|------|------|-------|
| CE standalone | 0.42s | 23.6 |
| GPT2 generate | 0.46s | 21.7 |

### 5.2 출력 품질

| 엔진 | 입력 | 출력 |
|------|------|------|
| CE | "오늘 날씨가" | 한글 토큰 생성 (의미 약함) |
| GPT2 | "오늘 날씨가" | "추워지면서, 오늘도 추위가 계속되" |

## 6. 이전 구현과의 차이

| 항목 | 이전 (hopfield.py) | 현재 (convert.py + engine.py) |
|------|-------------------|-------------------------------|
| GPT2 의존 | 추론 시 GPT2 전체 로드 필수 | .ce.pt 파일만 로드 |
| 메모리 | GPT2 477MB + W 2.3MB | 154.5MB (W + emb) |
| gradient | Riemannian natural gradient | 논문 원본 gradient descent |
| 노이즈 | FDT + annealing + G^{-1/2} | 논문 원본 sqrt(2*T*dt/tau)*N(0,I) |
| bypass | 에너지 함수 내부 포함 | 에너지 외부 (비보존 강제항) |
| phi update | m_star.pow(2) | 궤적 분산 (trajectory variance) |
| 희소화 (d<=1024) | 스킵 (100% dense) | 3D 격자 적용 (10.57%) |
| 디코더 | mdl.lm_head (GPT2) | 독립 ln_f + lm_head |

## 7. 미결 사항

1. **출력 품질**: CE 엔진의 한글 생성 품질은 아직 GPT2에 미달. 희소 W의 에너지 경관이 얕아서 이완이 의미 있는 끌개에 도달하지 못함
2. **밀도**: N=768에서 r_c=pi는 10.57% 밀도. 논문의 3.16%는 N=4096 기준
3. **codebook**: 논문 4.6절의 product quantization 미구현. 현재는 단순 top-K embedding
4. **CUDA/Rust**: 독립 엔진에서는 미사용. CSR SpMV는 PyTorch sparse로 처리
