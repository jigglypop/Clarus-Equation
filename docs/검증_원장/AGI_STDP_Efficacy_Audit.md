<!-- 도메인: ce-agi-runtime (멀티레포 이행 시 이관 대상, MULTIREPO_PLAN.md 참조) -->

# STDP 효능 감사

> 기준 checkout: 2026-07-30, `agi-runtime-diffusion-orchestration`
>
> 범위: F.14/F.14.2의 적격 흔적, critic 학습 게이트, 구조 투영, BrainRuntime 연결과 현재 효능 판정

## 현재 판정

STDP의 현재 지위는 다음 한 문장으로 고정한다.

> 적격 흔적과 critic 기반 학습 게이트의 인과 배선은 구현·회귀 검증되었지만, 현재 합성 다음 상태 예측 과제에서는 효능이 입증되지 않았고 held-out guard가 실패했으므로 기본값은 `stdp_enabled=False`로 유지한다.

세 층을 섞어 읽으면 안 된다.

| 층 | 판정 | 근거 |
|---|---|---|
| 구현 | 완료 | `stdp.py`, `runtime.py::_apply_runtime_stdp`, `agent.py::RuntimeAgent.step` |
| 인과 배선 | 회귀 테스트 통과 | 이전 tick의 critic이 다음 tick의 STDP gate를 구동하며 snapshot/restore도 tracker를 보존 |
| 효능 | `NO-EFFECT` | STDP on-off의 next-step prediction improvement 차이가 1 sigma를 넘지 않음 |
| 비회귀 guard | `FAIL` | 학습 후 frozen held-out probe의 prediction error가 허용치보다 증가 |
| 운영 기본값 | off | 효능과 guard를 모두 통과하기 전 자동 활성화 금지 |

## 재현 명령

```powershell
python -m pytest tests/test_stdp.py tests/test_runtime_contracts.py tests/test_agent.py tests/test_sleep.py tests/test_evidence.py -q
python examples/agi/stdp_diagnose.py
python examples/agi/stdp_efficacy_bench.py
```

2026-07-30 실행에서 첫 회귀 묶음은 `81 passed`였다. 보조 뇌 기능 묶음
(`test_neuromod.py`, `test_working_memory.py`, `test_layer_b.py`,
`test_consciousness.py`, `test_integration.py`)은 `49 passed`였다.

## 효능 A/B 결과

기본 벤치 설정은 `dim=48`, `steps=240`, `window=40`, `probes=32`,
seeds `1..7`이다. STDP off/on은 같은 초기 가중치, 입력열, 강제 WAKE
스케줄을 사용한다.

| 측정량 | on - off | 좋은 방향 | 판정 |
|---|---:|---|---|
| next-step prediction improvement | $-0.38330 \pm 0.49436$ | 양수 | `NO-EFFECT` |
| prediction-error slope | $+0.00172 \pm 0.00282$ | 음수 | 개선 없음 |
| held-out guard error | $+0.34797 \pm 0.06621$ | $0.02$ 이하 | `FAIL` |

STDP-on run은 seed마다 60회 갱신되었고 가중치 drift도 약 $5.56$에서
$5.65$로 명확했다. 따라서 실패 원인은 “업데이트가 전혀 일어나지 않음”이
아니다. 현재 규칙이 학습 스트림에 맞춰 가중치를 크게 움직이지만, 그 변화가
일관된 다음 상태 예측 개선이나 held-out 일반화로 이어지지 않는 것이 핵심이다.

## 진단 결과

`stdp_diagnose.py`의 기본 진단은 다음을 보였다.

| 진단 | 결과 | 해석 |
|---|---:|---|
| gate 양/음 횟수 | $39/21$ | gate가 상수나 dead signal은 아님 |
| critic의 weight 상대 민감도 | 약 $1.9\%$ | critic 자체는 $W$에 약하게만 제어됨 |
| 구조 투영 후 update 보존율 | 평균 $0.510$ | 투영이 update를 절반가량 남기며 완전히 지우지는 않음 |
| critic 기반 hyperparameter sweep | $+0.0196$에서 $+0.0399$ | critic의 $W$ 민감도가 낮아 효능 근거로 사용 불가 |

critic score는 일부 항이 소뇌 전방 모델에 의존하고 현재 벤치에서 novelty가
dead이므로, 효능 판정은 critic score가 아니라 $W$가 직접 제어하는
next-step prediction error를 사용한다. 따라서 위 hyperparameter sweep은
진단 기록일 뿐, 양의 평균만으로 STDP를 승격하는 근거가 아니다.

## 다음 루프

다음 승격은 하이퍼파라미터를 더 훑는 것이 아니라 목적함수와 구조를 분리하는
순서로 진행한다.

1. 학습용 recurrent prediction loss와 held-out probe를 동일한 정규화로 기록한다.
2. 구조 투영 전후의 update 방향과 held-out gradient proxy의 cosine을 측정한다.
3. critic derivative, bootstrap deviation, eligibility를 각각 ablation한다.
4. forced WAKE가 아닌 축약된 WAKE/NREM/REM schedule에서 mode별 결과를 분리한다.
5. 효능과 guard가 함께 통과할 때만 `stdp_enabled` 기본값 변경을 검토한다.

현재 벤치는 rate-state를 threshold해 spike event를 만드는 BrainRuntime 실험이다.
막전위, 명시적 spike time, refractory dynamics를 갖는 완전한 SNN substrate 검증과는
구분한다. 따라서 이 결과는 CE-AGI의 SNN 자연 수렴 주장을 증명하거나 반증하지 않는다.
