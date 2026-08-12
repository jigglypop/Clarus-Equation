# 30-implementation — agi-clarus-field-20260812

Status: COMPLETE

## 승인 범위

`20-audit.md` 수정 감사가 승인한 좁은 baseline만 구현했다. 구현 대상은 CF-1·CF-2와 수정된 CF-3의 형식 primitive이며, V14 route L 원형, R-A1 항상성 적응, R-A3 Poisson 해석, 전체 $p^*$ 자기수렴은 포함하지 않는다. 기존 `BrainRuntime` 기본 경로에도 연결하지 않았다.

## 구현

| 파일 | 변경 | 형식 경계 |
|---|---|---|
| `reality_stone/python/reality_stone/clarus/clarus_field.py` | 유한 대칭 연결 그래프, 정규화 라플라시안, 정확 상수소스 field step, 경성 gate, 단위공 write, phase readout, snapshot, 인증서 | 연구 primitive; 뇌·우주 동일성·AGI 효능 미주장 |
| `reality_stone/python/reality_stone/clarus/__init__.py` | 새 타입과 함수의 import-safe public export | 기존 optional import 관용 유지 |
| `tests/test_clarus_field.py` | 형식 불변식·반례 경계·public API 17개 회귀 | 과제 성능 채점 아님 |
| `examples/agi/clarus_field_demo.py` | 8-node ring에서 32 tick 결정적 예시 | 결과는 smoke output |
| `dimensionless_checker.py`, `tests/test_dimensionless.py` | prediction-error sigmoid와 $lambda\phi/R$ 등록·회귀 | 무차원성만 검사; 물리 타당성 아님 |
| `docs/7_AGI/1_AGI.md`, `12_Equation.md`, `18_CodeMap.md` | 현재 AGI 지위, 거짓 3-simplex 부모 정리 삭제, 구현 대응 기록 | 정리·공리·경험식·미완성 분리 |

## 구현 식과 공리 대응

D1의 readout은 다음으로 고정했다.

$$r(s_i)=\min(\lVert s_i\rVert_2,R).$$

$r$은 비음수이고 $R$로 유계이며 1-Lipschitz다. 그래프 field의 한 tick은 $A=\kappa L+\lambda I$에 대해 고유분해로 다음 정확해를 계산한다.

$$\phi^+=e^{-A\Delta t}\phi+A^{-1}(I-e^{-A\Delta t})r(s).$$

메모리 갱신은 외생 점수 $g_i\in[0,1]$를 먼저 경성화한다.

$$\hat g_i=g_i\mathbf 1[g_i>\theta_g],\qquad
s_i^+=(1-\hat g_i)s_i+\hat g_i\Pi_{B_1}(\tilde s_i).$$

닫힌 노드는 산술식을 평가하지 않고 기존 행을 그대로 복사하므로 NaN 곱이나 roundoff 없이 비트 단위 항등이다. CF-3은 drive가 i.i.d. 외생 입력의 공통 함수라는 A-E2R을 추가로 요구한다. 코드가 확률 과정의 외생성을 추론할 수 없으므로 `ClarusFieldCertificate.cf3_scope`에 조건부 범위를 노출한다.

prediction-error gate의 soft score는 다음 식이다.

$$g_i=\sigma\!\left(a\left\lVert\frac{x_i-\hat x_i}{x_0}\right\rVert_2^2+b\right).$$

$x_0>0$으로 나눈 오차, $a$, $b$는 무차원이다. 제곱노름이므로 부호 반전에 불변이고, 실제 latch 경계는 이후의 경성 gate가 담당한다. 구조 phase는 무차원 점수 $\lambda\phi_i/R$에 임계를 적용한다.

HRR circular convolution은 `bounded_hrr_bind`의 readout으로만 구현하고 단위공에 투영했다. 상태 의존 recurrent HRR write는 포함하지 않았다. V14 route L 원형은 1차원에서 $h^+=(1+gv)h$가 될 수 있고 sigmoid gate가 정확히 닫히지 않으므로 CF-1~3 인증을 상속하지 못한다.

## 설정값의 지위

기본값 $lambda=0.25$, $\kappa=1$, $\Delta t=1$, $R=1$, $\theta_g=0.5$, $\theta_s=0.25$는 모두 무차원 **구현 기본값**이다. 관측 또는 과제 점수에서 맞춘 값이 아니며 CE 상수의 산출로 주장하지 않는다. $p^*=(0.0487077,0.2623,0.6891)$은 config, loss, gate, phase 분류 어디에도 삽입하지 않았다.

## 불변식

- S1: 라플라시안은 $phi$에만 작용한다.
- S2: 메모리 write는 gate를 통해서만 들어오고 단위공에 투영된다.
- S3: 닫힌 gate는 정확한 항등이다.
- canonical `BrainRuntime` 5계층과 기본 STDP-off 경로는 변경하지 않았다.
- 기존 V10--V13 모델과 벤치 코드는 변경하지 않았다.

Status: COMPLETE
