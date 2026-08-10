# 인과 세계 시뮬레이터 실행 로드맵

> 기준일: 2026-08-10
>
> 상태 원장: 이 문서가 AGI 연구의 우선순위ㆍ게이트ㆍ중단 조건에 대한 단일 기준이다.
>
> 완료된 수학과 G1 결과: `26_Causal_World_Simulator.md`
>
> 최신 계산-뇌 대응 gate: `41_Sparse_Causal_Bridge_World_G9CB.md`

## 0. 주장 정책과 동결된 레거시 트랙

모든 주장은 `가설 -> 사전등록 측정 -> 기준선 비교 -> 통과/실패 -> 주장 등급 갱신` 순서로만 올린다. 합성 실험의 성공은 생물학적 증거나 AGI 달성을 뜻하지 않는다.

기존 CE 비율을 transformer에 강제 이식한 트랙은 우선순위에서 제외한다. KoGPT2 실험에서 자연 활성비 수렴은 나타나지 않았고, 강제 희소화는 perplexity를 악화시켰으며, 기존 수면 변형도 continual-learning 기준선을 이기지 못했다. 따라서 다음 항목은 별도 새 증거가 생길 때까지 동결한다.

- 우주 분율을 신경 활성 비율로 직접 동일시하는 주장
- transformer에 특정 활성비를 강제하는 실험
- 성능 게이트를 통과하지 않은 STDPㆍ수면ㆍ메타인지 모듈의 통합
- 합성 성공을 뇌 또는 의식의 증명으로 해석하는 주장

이전 참조가 사용하는 `G-S1`~`G-S5`는 동결된 SNN 검증 게이트를 뜻한다: 막전위와 spike time, 국소 STDP, 자연 활성분율, held-out 과제 성능, 연속학습 보존을 차례로 검증한다. 현재 모두 `미검증`이며 주 실행 경로를 막지 않는다.

## 1. 연구 목표와 현재 위치

작업 가설은 다음과 같다.

> 뇌는 국소 인과모델들이 감각ㆍ행동ㆍ기억을 통해 맞물리는 계층적 세계 시뮬레이터이며, 피질 주름과 연결 구조는 그 계산을 제약하고 안정화하는 물리적 형상일 수 있다.

이 문장은 세 개의 독립 트랙으로 나눈다.

| 트랙 | 질문 | 증거가 올라가는 순서 |
|---|---|---|
| A. 계산 | 국소 인과 세계모델이 실제로 예측ㆍ추론ㆍ계획하는가? | 합성 -> 시뮬레이션 -> 실제 센서 |
| B. 뇌 | 주름ㆍ연결ㆍ기능의 관계가 시간 선행성과 홀드아웃 예측력을 갖는가? | 공개 영상 -> 종단 자료 -> 생물 개입 |
| C. 통합ㆍ안전 | 모듈을 통합해도 안정성ㆍ감사 가능성ㆍ성능이 유지되는가? | sandbox -> 제한 행동 -> 실제 환경 |

현재 위치는 `G1--G8`과 `G9-S/G9-3D/G9-R/G9-H/G9-F/G9-B/G9-CB`의 저비용 단계까지 완료다. G9-CB V1은 두 true bridge를 정확히 찾았지만 bridge target의 직접 잠재교란 때문에 예측ㆍlesion gate에 실패했고 test를 열지 않았다. 범위를 잠근 V2는 새 seed의 locked test에서 `A→C`, `C→D`를 정확히 복원하고 common-cause `A--B`를 배제했다. OOD global RMSE는 local-only 대비 10.68%, dense observational 대비 56.20%, raw-correlation 대비 45.25% 낮았지만 가장 강한 observation-only predictive-gain 기준과는 사실상 동률이다. 이는 4-chart 합성 식별 gate이지 피질 연결이나 AGI의 증거가 아니다.

## 2. 의존성

```text
G1 선형 인과 gate [완료]
  -> G2 비선형 장기 rollout
       -> G3 가림ㆍ물체 영속성
            -> G4 능동 지각
                 -> G5 인과ㆍ조합 OOD
                      -> G6 자발적 chart/구조 성장
                           -> G7 기억ㆍ재생ㆍ계층 계획
                                -> G8 실제 센서ㆍ로봇
                                     -> G10 사회ㆍ언어

B1 발달 지도 -> B2 dHCP -> B3 HCP -> B4 유전/종단 -> B5 개입
                          \____________________________/
                                       |
                          G9 뇌 대응은 독립 판정
```

G9는 A 트랙의 성공으로 자동 통과하지 않는다. 계산 모델과 뇌 자료는 분석계획을 각각 고정한 뒤 공통 지표만 비교한다.

```text
G9-R/F/B 집단평균 기하 [저비용 단계 완료]
          -> G9-CB 합성 geometry-proposal/causal-selection 분리 [V2 완료]
               -> 실제 superficial-white tractography matched control [대기/SKIPPED_COST]
```

## 3. 게이트 원장

| Gate | 최소 결과 | 상태 | 다음 이동 조건 |
|---|---|---|---|
| G0 | 수학적 일관성ㆍ유한계 정리 | 완료 | 코드 대응 |
| G1 | 선형 상태복원ㆍ인과ㆍ계획 | 완료 | 재현 테스트 유지 |
| G2 | 비선형 1/5/20/100-step rollout | V1 완료 | 더 넓은 함수족ㆍ잡음에서 재검증 |
| G3 | 가림 중 상태ㆍ정체성ㆍ재등장 예측 | V1 완료 | 모호한 외형ㆍidentity 교란 추가 |
| G4 | 정보이득을 위한 감각 행동 | V3 완료 | 더 넓은 교란에서 margin 재검증 |
| G5 | 새 객체 수ㆍ속성ㆍ법칙ㆍ목표 | V6 완료 | 미지 basis에서 재검증 |
| G6 | chart 생성ㆍ전이ㆍ복구 | V3 완료 | 연속 regime에서 재검증 |
| G7 | 기억ㆍ재생ㆍ장기 계획 | V4 완료 | 실제 과제에서는 재검증 필요 |
| G8 | 실제 센서ㆍ제한 로봇 | G8-S V4ㆍG8-C V7ㆍG8-R V5 완료; 실장비 대기 | 다른 로봇/환경 holdout 또는 replay 폐루프 |
| G9 | 주름ㆍ연결ㆍ기능 대응 | 저비용 기하ㆍ합성 대응 완료; 생물 연결 미검증 | 기계적 성장모델 대비 홀드아웃 증분 |
| G9-S | 합성 differential-growth null | V3 완료 | 실제 MRI 없이 식별성만 확인; 공개 요약통계 연결 필요 |
| G9-3D | 방향성 성장 텐서 표면 | V2 완료 | OBJ 형상화; 실제 cortical mesh와 미대조 |
| G9-R | fsaverage 내재기하ㆍ경계보존 확산 | V4 locked testㆍV8 우반구 복제 완료 | 개인별ㆍ발달 자료 필요 |
| G9-H | connection/holonomy 수치 항등식 | V1 완료 | 독립 방향장 residual holonomy 필요 |
| G9-F | 기능ㆍ해부 경계 분리 | Yeo 실패ㆍDesikan 양성대조 완료 | 개인별 기능ㆍ연결 자료 필요 |
| G9-B | 희소 fold-contact 후보 | V1 실패 보존ㆍV2 양반구 완료 | tractography matched control 필요 |
| G9-CB | geometry proposal + intervention 방향화 | V1 validation 실패 보존ㆍV2 locked test 완료 | 직접 잠재교란 targetㆍ실제 연결은 미해결 |
| G10 | 타 에이전트ㆍ언어 인터페이스 | 대기 | 체계적 일반화와 믿음 추적 |
| G11 | 광범위 일반지능 평가 | 미정 | 독립 평가 전에는 AGI 명칭 금지 |

## 4. 트랙 A: 계산 실행계획

### A1. G2--G3: 비선형 물체 영속성

첫 환경은 외부 게임엔진 없이 재현 가능한 NumPy 2차원 다중물체 세계로 구현했다. 상태에는 물체별 위치ㆍ속도ㆍ반지름ㆍ질량ㆍ정체성이 있고, 관측에는 가림 마스크와 action이 포함된다. 충돌, 벽 반사, 마찰, 약한 비선형 위치 의존력이 존재한다.

필수 비교군은 `persistence`, `global linear`, `small monolithic nonlinear`, `local-chart nonlinear` 네 개다. 데이터 분할은 시간뿐 아니라 seed, 물체 수, 가림 길이, 물리 파라미터를 분리한다. 상세 수치와 실패 규칙은 27장에 고정한다.

산출물:

- `reality_stone/python/reality_stone/clarus/nonlinear_object_world.py` (완료)
- `examples/agi/nonlinear_object_permanence_gate.py` (완료)
- `tests/test_nonlinear_object_world.py` (완료)
- `artifacts/agi/nonlinear_object_permanence_report.json` (완료)
- 선택적 대용량 궤적은 Git에 넣지 않고 재생성 명령만 기록

### A2. G4: 능동 지각

G3 모델에 제한된 추가 센서를 구현했다. V1은 동일 공분산과 `3/5` seed 문제로 실패했고, V2는 질량 보정 후 validation을 통과했지만 locked test에서 다시 `3/5`로 실패했다. V3는 새 seed와 paired expected-cost 판정으로 바꾸어 test 20개에서 random 대비 비용 11.4% 감소, 95% 하한 양수로 통과했다. 상세 실패ㆍ수식 변경은 `28_Active_Perception_G4.md`에 보존한다.

### A3. G5: 인과ㆍ조합 OOD

객체 상태와 관계 그래프를 분리한다. 훈련에 없던 객체 수, 질량ㆍ마찰 조합, 도구, 인과 방향, 목표에서 시험한다. 영상 유사도만 좋아지고 개입 효과가 틀리면 실패다.

### A4. G6: 자발적 chart와 구조 성장

chart 수, 담당 영역, 전이함수, 결합 강도를 학습 대상으로 바꾼다. cocycle 위반과 holonomy/frustration으로 오염된 연결을 찾아 국소 복구한다. 모듈 추가ㆍ삭제는 검증 성능과 복잡도 페널티를 함께 개선할 때만 채택한다.

### A5. G7: 기억ㆍ재생ㆍ계층 계획

작업ㆍ일화ㆍ의미 기억을 분리하고 실패/고오차 사건을 오프라인 재생한다. 빠른 감각제어, 중간 객체사건, 느린 목표ㆍ법칙의 세 시간척도로 계획한다. 새 과제 적응과 과거 과제 보존을 동시에 측정한다.

### A6. G8ㆍG10: 실제 환경과 사회ㆍ언어

시뮬레이션을 통과한 모델만 제한된 실제 센서/로봇으로 옮긴다. 이후 다른 에이전트의 관측ㆍ목표ㆍ거짓 믿음을 추적하고, 언어는 세계모델의 질의ㆍ보고ㆍ계획 인터페이스로 결합한다. 언어모델 자체를 세계모델 성공의 대리변수로 사용하지 않는다.

## 5. 트랙 B: 뇌ㆍ주름 검증계획

### B1. 발달ㆍ유전 atlas

주요 고랑마다 출현 시기, 출생 후 변화, 좌우 비대칭, 집단 공통성, 개인차, 유전율, 기능영역 정렬, 백질 연결을 표준 표로 만든다. 개인 형태는 `인류 공통 발달 + 유전 + 비공유 환경 + 발달 잡음`의 계층모형으로 분석한다.

### B2. 공개 영상의 실행 순서

1. dHCP 신생아 구조 MRIㆍ확산 MRIㆍ휴지상태 fMRI로 곡률, 고랑 바닥, 연결 텐서, 기능 상관장을 계산한다. 공식 2차 공개 자료는 신생아 505명을 제공한다: <https://www.developingconnectome.org/data-release/second-data-release/information-registration-and-download/>
2. HCP 성인 자료로 V1/A1/S1/M1 같은 기능 앵커와 개인별 주름ㆍ연결 정렬을 시험한다: <https://www.humanconnectomeproject.org/data/>
3. ABCD는 통제 접근 승인 후 청소년 종단 변화에 사용한다: <https://abcdstudy.org/scientists/data-sharing/>
4. UK Biobank는 2026-08 현재 신규 신청 중단 상태이므로 즉시 의존성에서 제외한다: <https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/>

### B3. 현재 무료 검증의 경계

G9-R/F/B는 TemplateFlow 집단평균 표면에서 계산한 기하 파생치이고, G9-CB는
별도의 네 차트 합성 SCM이다. 전자는 실제 연결을 측정하지 않았고 후자는 실제
피질을 사용하지 않았으므로 서로를 검증한 것으로 합치지 않는다. 현재 연결은
“실제 표면에서 희소 후보가 있었다 → 그 후보/인과 선택의 역할을 합성계에서
분리해 보았다”는 연구 순서뿐이다.

### B4. 경쟁모형과 반증

- `M0`: 두께ㆍ면적 팽창ㆍ재료 특성의 기계적 좌굴
- `M1`: M0 + 백질 연결 방향/장력 대리변수
- `M2`: M1 + 기능 앵커ㆍ국소 위상결합ㆍfrustration

반드시 초기 성장ㆍ연결ㆍ기능으로 미래 곡률ㆍ고랑 위치를 예측한다. M2가 누수 없는 홀드아웃에서 M0/M1을 안정적으로 이기지 못하면 “세계모델 결합이 주름을 조직한다”는 강한 가설을 기각하거나 축소한다. 영상 신호가 살아남은 후에만 오가노이드나 주름뇌 동물 개입 협력을 검토한다.

## 6. 트랙 C: 통합ㆍ안전 계약

통합 순서는 `감각 -> 객체/사건 추정 -> 국소 인과 chart <-> 기억 -> 불확실성 -> 반사실 rollout -> 계획 -> 행동 -> 교정`이다.

실세계 행동 전 필수 조건:

- sandbox와 행동ㆍ시간ㆍ자원 예산
- OOD 또는 높은 불확실성에서 정지/추가 관측
- 목표모델과 세계모델의 분리
- 구조 변경 전 snapshot, 성능 하락 시 rollback
- 모든 예측ㆍ개입ㆍ행동의 감사 로그
- 외부 쓰기와 자기수정의 별도 승인

## 7. 공통 평가와 중단 조건

| 축 | 지표 |
|---|---|
| 예측 | rollout RMSE/NLL, horizon별 붕괴 |
| 상태 | 가림 복원, 정체성 보존, 재등장 위치 |
| 인과 | 개입효과 오차, 부호 정확도 |
| 계획 | 비용, regret, 성공률 |
| 일반화 | seedㆍ객체ㆍ법칙ㆍ목표 OOD |
| 불확실성 | calibration, 과신률, OOD 거부 |
| 기억 | 망각률, 재학습 속도 |
| 구조 | holonomy, frustration, chart 수 |
| 효율ㆍ안전 | 연산량, 행동 한도 위반, rollback |

다음 중 하나면 해당 주장을 승격하지 않는다.

- 강한 기준선을 반복 seed에서 이기지 못함
- one-step은 좋지만 장기 rollout이 기준선보다 먼저 붕괴
- 관측 적합은 좋지만 개입 효과가 틀림
- 데이터 누수, 사후 임계값 선택, seed 선택 효과
- 합성 결과를 실제 뇌나 일반지능의 증거로 과대해석

실패 결과도 보고서와 테스트로 남긴다. 구현을 버리는 대신 주장 등급을 낮추고 실패 원인을 다음 가설로 분리한다.

## 8. 실행 큐

| 순서 | 작업 | 완료 정의 |
|---|---|---|
| 1 | G2--G3 명세 고정 | 완료 |
| 2 | 환경ㆍoracle 구현 | 완료: seedㆍ가림ㆍ수치 테스트 |
| 3 | 기준선 구현 | 완료: persistenceㆍlinearㆍmonolithic |
| 4 | local-chart 모델 | 완료: 같은 관측ㆍaction 사용 |
| 5 | 홀드아웃 gate | 완료: IDㆍ장기가림ㆍ조합 OOD 각 5 seed |
| 6 | G4 능동 지각 | 완료: V1/V2 실패 보존, V3 locked test 통과 |
| 7 | G5 인과ㆍ조합 OOD | 완료: V1--V5 실패 보존, V6 locked test 통과 |
| 8 | 공개 뇌자료 파이프라인 | 부분 완료: TemplateFlow fsaverage 10k; 개인별 dMRI/fMRI 대기 |
| 9 | G6--G7 확장 | 완료: G6 V3, G7 V4 locked test 통과 |
| 10 | G8-S 결함주입 안전 sandbox | 완료: V1/V2 실패, V3 판정결함 감사, V4 강화 test 통과 |
| 11 | G8-C 결함 OOD calibration | 완료: V1--V6 한계 보존, V7 seed별 Wilson locked test 통과 |
| 12 | G8-R 실제 공개 센서 로그 | 완료: V1--V4 실패 보존, V5 시간순 locked test 통과 |
| 13 | G9-S 합성 주름ㆍ꼬임 식별 | 완료: V1/V2 실패 보존, V3 null+대안 locked test 통과 |
| 14 | G9-3D 리만 성장 표면 | 완료: V1 실패 보존, V2 회전-equivariant locked testㆍOBJ 통과 |
| 15 | G9-R 실제 fsaverage 기하 | 완료: V1/V2/V5--V7 실패 보존, V4 testㆍV8 우반구 통과 |
| 16 | G9-H connection holonomy | 완료: Gauss--Bonnetㆍ강체등변 수치 검증 |
| 17 | G9-F 기능/해부 경계 | 완료: Yeo null, Desikan 양성대조ㆍ분리 |
| 18 | G9-B 희소 fold bridge | 완료: V1 보편가설 실패, V2 양반구 sparse tail |
| 19 | G9-CB 희소 인과 bridge | 완료: V1 validation FAIL/test 미개봉, V2 validation+locked test PASS |

다음 저비용 계산 반증은 V1이 드러낸 bridge target 직접 잠재교란을 latent-state 추정 또는 환경불변 residual로 처리하는 별도 사전등록이다. 실제 뇌 연결 주장은 독립 superficial-white-matter tractography에서 G9-B strong 후보와 거리ㆍ곡률 matched control을 비교하기 전까지 보류하며, 접근ㆍ비용 한도를 넘으면 `SKIPPED_COST`로 남긴다.

## 9. 저장소와 산출물 정책

- 소스, 테스트, 사전등록, 작은 JSON 요약은 보존한다.
- 결과를 재현하는 데 필요한 작은 CSV는 보존할 수 있다.
- 대용량 trajectory, 렌더 프레임, checkpoint는 기본적으로 Git에서 제외하고 생성 명령과 hash만 남긴다.
- `__pycache__`, `.pytest_cache`, `.ruff_cache`, 임시 파일은 증거가 아니므로 삭제 가능하다.
- 실패 산출물은 불필요한 파일이 아니다. 해당 주장과 seedㆍ설정ㆍ코드 버전을 연결하는 보고서는 보존한다.
- 기존 모듈은 새 gate가 대체하고 회귀 테스트가 통과하기 전에는 삭제하지 않는다.

## 10. 다음 명령

G1 회귀 확인:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_causal_world_simulator.py tests/test_riemannian_gear.py tests/test_riemann_gear_certificate.py -q
.\.venv\Scripts\python.exe examples/agi/causal_world_simulator_gate.py
```

G2--G3 구현이 끝난 뒤 사용할 목표 명령:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nonlinear_object_world.py -q
.\.venv\Scripts\python.exe examples/agi/nonlinear_object_permanence_gate.py --config experiments/preregistration/nonlinear_object_permanence_v1.json
```

G9-CB 재현:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_sparse_causal_bridge.py -q --basetemp .tmp/g9cb-doc
.\.venv\Scripts\python.exe examples/agi/sparse_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v2.json --split validation
# 위 artifact가 동일 config SHA로 PASS한 경우에만
.\.venv\Scripts\python.exe examples/agi/sparse_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v2.json --split test
```
