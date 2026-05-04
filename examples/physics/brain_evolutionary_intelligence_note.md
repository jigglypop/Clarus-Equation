# 원시 신경계, 안정 동역학, 지능의 발생에 대한 연구 정리

## 1. 지금까지 확인한 것

인간 데이터에서는 event-level로 세 축을 확인했다.

| 축 | 확인한 내용 |
|---|---|
| `ds000116` | 시각/청각 modality 특수 연산자 분리 |
| `ds000201 task` | 통증/촉각, 얼굴정서, 인지조절 특수 연산자 분리 |
| `ds000201 cognitive` | 주의/경계, 작업기억, 주관적 각성 특수 연산자 분리 |

수학 점검에서는 다음이 통과했다.

$$
\bar W < \min_{d\ne d'}B_{d,d'}
\qquad
\mathcal L_{\mathrm{matched}}
<
\min(
\mathcal L_{\mathrm{wrong}},
\mathcal L_{\mathrm{generic}}
)
$$

즉 인간 event-level에서는 특수 입력항 \(U^{(d)}\)를 둘 수 있는 필요조건이 닫혔다.

원시 신경계에서는 `C. elegans` connectome을 봤다.

| 검증 | 결과 |
|---|---|
| adult L1/L2/L3 weighted block | pass |
| chemical weighted | pass |
| binary graph | fail |
| electrical-only | fail |
| developmental stage 1-8 | 7/8 pass |

가장 중요한 발견:

$$
\boxed{
\text{원시 신경계의 층화 구조는 binary adjacency가 아니라 weighted chemical connectivity에 실려 있다.}
}
$$

## 2. 뇌는 알고리즘을 가진 컴퓨터인가

현재 해석은 아니다.

뇌가 내부에 명시적 알고리즘이나 수식을 저장하고 실행한다고 보는 것은 과하다.

```text
if danger:
    run_avoidance_algorithm()
```

이런 식이 아니다.

더 정확한 표현은 다음이다.

$$
\boxed{
\text{뇌는 수식을 아는 것이 아니라, 수식으로 표현 가능한 안정 동역학을 몸으로 구현한다.}
}
$$

새가 뉴턴 방정식을 알고 나는 것이 아니지만, 새의 비행이 뉴턴 방정식으로 설명되는 것과 같다.

뇌 방정식도 뇌 속 코드가 아니라 관측자가 쓰는 압축 표현이다.

$$
P_{n+1}
=
\Pi
\left[
(1-\rho)P^*
+\rho P_n
+\gamma\Delta_G P_n
+\sum_d a_{d,n}U^{(d)}
+H(Q-Q^*)
+F_{\mathrm{syn}}
+F_{\mathrm{slow}}
\right]
$$

이 식의 의미는 다음이다.

| 항 | 생물학적 의미 |
|---|---|
| \(\rho P_n\) | 현재 상태의 관성 |
| \((1-\rho)P^*\) | 기준 상태로 되돌아가려는 안정성 |
| \(\gamma\Delta_G P_n\) | weighted graph를 통한 상태 확산/결합 |
| \(U^{(d)}\) | 감각/통증/주의/작업기억 같은 domain 입력 |
| \(H(Q-Q^*)\) | 항상성, 각성, 몸 상태 |
| \(F_{\mathrm{syn}}\) | 학습/가소성 |
| \(F_{\mathrm{slow}}\) | 회복, 수면, 장기 조정 |

## 3. 원시 신경계는 왜 생겼나

원시 신경계의 첫 목적은 생각이 아니다.

가장 먼저 필요한 것은 다음 루프다.

```text
감지한다 -> 통합한다 -> 움직인다
```

즉:

```text
세계 상태를 행동으로 번역한다.
```

이 관점에서 L1/L2/L3는 최소 제어 구조다.

| 층 | 역할 |
|---|---|
| L1 | 외부/내부 자극 입력 |
| L2 | 감각 통합, relay, 상태 판정 |
| L3 | premotor, 행동 선택, 출력 준비 |

왜 weighted chemical 구조인가?

binary 연결만 있으면 “연결됨/안 됨”밖에 없다. 하지만 생존 행동에는 강도 조절이 필요하다.

| 필요 | weighted chemical synapse가 하는 일 |
|---|---|
| 위험 신호 우선순위 | 강한 회피 경로 |
| 먹이/냄새 추적 | taxis 경로 조절 |
| 같은 자극에 안정 반응 | 반복 가능한 weighted flow |
| 발달/경험 변화 | synaptic weight 재조정 |

그래서 원시 신경계의 핵심은 컴퓨터 알고리즘이 아니라 weighted chemical control system이다.

## 4. 그러면 신경계의 양만 늘리면 지능이 생기나

결론:

$$
\boxed{
\text{양은 필요조건일 수 있지만 충분조건은 아니다.}
}
$$

단순히 neuron 수나 synapse 수를 늘린다고 지능이 자동으로 생기지는 않는다.

왜냐하면 지능에는 적어도 다음 조건들이 필요하기 때문이다.

## 5. 지능 발생의 최소 조건

### 5.1 충분한 상태공간

노드 수와 연결 수가 너무 적으면 표현할 수 있는 상태가 적다.

$$
\dim(P)=3|V|
$$

따라서 양은 중요하다. 하지만 이것만으로는 부족하다.

### 5.2 비무작위 구조

무작위로 연결 수만 늘리면 안정 행동이 아니라 noise가 늘 수 있다.

필요한 것은 구조화된 그래프다.

$$
G
\ne
G_{\mathrm{random}}
$$

그리고 실제 검증 기준은 다음이어야 한다.

$$
\mathcal L_{\mathrm{structured}}
<
\mathcal L_{\mathrm{random}}
$$

### 5.3 domain별 특수 연산자

모든 입력을 하나로 처리하면 지능이 아니라 반사에 가깝다.

지능은 domain별 처리와 통합이 동시에 필요하다.

$$
U_n
=
\sum_d a_{d,n}U^{(d)}(z_n)
$$

시각, 촉각, 통증, 기억, 행동 선택, 사회 신호가 서로 다른 연산자를 가져야 한다.

### 5.4 recurrent loop

단순 feedforward는 반응은 만들 수 있지만, 내부 상태 유지가 약하다.

작업기억과 계획에는 recurrent loop가 필요하다.

$$
P_{n+1}
=
A_GP_n+U_n
$$

여기서 \(A_G\)가 단순 전달이 아니라 순환 구조를 가져야 한다.

### 5.5 안정성

너무 강한 recurrent loop는 폭주한다.

따라서 안정 조건이 필요하다.

$$
\max_k|\rho-\gamma\lambda_k|<1
$$

지능은 복잡성만으로 생기지 않는다. 복잡성과 안정성의 균형에서 생긴다.

### 5.6 가소성

학습이 없으면 복잡한 반사기계일 뿐이다.

필요한 항은 다음이다.

$$
F_{\mathrm{syn},n}
$$

즉 경험에 따라 graph weight나 domain operator가 바뀌어야 한다.

### 5.7 항상성

몸 상태가 없으면 행동 우선순위가 없다.

배고픔, 피로, 손상, 각성 같은 내부 상태가 지능의 방향을 정한다.

$$
H(Q_n-Q^*)
$$

이 항이 없으면 “무엇이 중요한가”가 없다.

### 5.8 memory/workspace

고등 지능에는 순간 반응을 넘는 상태 유지가 필요하다.

이것은 후기 진화에서 추가된 항일 수 있다.

$$
W_{\mathrm{workspace},n}
\quad\text{or}\quad
M_{\mathrm{episodic},n}
$$

## 6. 양만 늘린 경우의 실패 모드

신경계 양만 늘리면 다음 중 하나가 될 수 있다.

| 늘어난 것 | 가능한 결과 |
|---|---|
| random synapse | noise, 불안정성 |
| recurrent loop만 증가 | seizure-like 폭주 |
| sensory input만 증가 | 정보 과부하 |
| motor output만 증가 | 반응 다양성은 늘지만 계획은 없음 |
| chemical weight만 증가 | 특정 행동에 고착 |
| memory만 증가 | 행동 없는 내부 순환 |

따라서 지능은 양의 함수가 아니라 구조의 함수다.

더 정확히는:

$$
\text{Intelligence}
\sim
f(
|V|,
G_{\mathrm{weighted}},
\mathcal D,
A_G,
F_{\mathrm{syn}},
H,
W_{\mathrm{workspace}}
)
$$

## 7. 현재 가설

지능은 다음 순서로 올라왔을 가능성이 있다.

### 단계 1: 반응

```text
감각 -> 운동
```

### 단계 2: 선택

```text
여러 감각 -> 우선순위 -> 행동
```

### 단계 3: 상태

```text
현재 몸 상태 + 외부 자극 -> 행동
```

### 단계 4: 기억

```text
과거 상태 + 현재 자극 -> 행동
```

### 단계 5: 시뮬레이션

```text
가능한 미래 상태들을 내부에서 돌려봄
```

### 단계 6: 추상화

```text
감각-행동 루프를 기호/개념 수준으로 재사용
```

즉 고등 지능은 원시 감각-운동 제어가 완전히 다른 것으로 바뀐 것이 아니라, 내부 시뮬레이션과 memory/workspace가 붙으면서 확장된 것일 수 있다.

## 8. 현재 답

질문:

> 무작정 이 신경계의 양만 늘리면 지능이 생기나?

답:

$$
\boxed{
\text{아니다. 양만 늘리면 안 된다.}
}
$$

하지만:

$$
\boxed{
\text{weighted chemical graph, recurrent stability, domain specialization, plasticity, homeostasis, memory가 함께 커지면 지능으로 갈 수 있다.}
}
$$

즉 지능은 다음이 아니다.

```text
more neurons = intelligence
```

더 가까운 식은 다음이다.

```text
structured weighted control + stable recurrence + learnable memory + embodied goals = intelligence candidate
```

## 9. 다음 검증

다음에 볼 것은 단순 크기 증가가 아니라, 복잡성 증가가 어떤 항을 추가하는지다.

비교할 축:

| 생물 | 확인할 항 |
|---|---|
| C. elegans | weighted chemical L1/L2/L3 |
| Drosophila | 시각/후각/action selection, learning loop |
| zebrafish | 전뇌-중뇌-후뇌, arousal, whole-brain dynamics |
| mouse | cortex-thalamus-basal ganglia loop, memory |
| human | workspace, language, abstract domain |

검증 질문:

$$
\mathcal L_{\mathrm{base\ ancient}}
>
\mathcal L_{\mathrm{base+new\ loop}}
$$

새 loop나 memory 항을 넣었을 때만 holdout 손실이 줄어든다면, 그 항이 지능으로 올라가는 진화적 추가항이다.
