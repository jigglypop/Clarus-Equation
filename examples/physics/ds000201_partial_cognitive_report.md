# ds000201 인지/각성 부분 검증 보고서

## 범위

이번 검증은 `ds000201`에서 수면 단계 자체가 아니라, 행동 과제 기반의 인지/각성 축을 분리한 것이다.

사용한 범위:

- 데이터셋: OpenNeuro `ds000201`, snapshot `1.0.3`
- 피험자: `sub-9001` ~ `sub-9004`
- 세션: `ses-1`, `ses-2`
- 과제/파일:
  - `PVT`: 지속주의, 경계, 반응속도
  - `workingmemorytest`: 작업기억
  - `sleepiness`: 주관적 각성/KSS
- 사용하지 않은 파일: BOLD NIfTI, DWI, T1w

따라서 이번 결과도 신경활성 검증이 아니라 **event-level cognitive/arousal readiness gate**다.

## 검증 질문

전역 뇌 방정식의 과제 입력 항을 인지 기능별 특수 연산자로 나누면 다음처럼 쓸 수 있다.

$$
P_{n+1}
=
\Pi
\left[
(1-\rho)P^*
+\rho P_n
+\gamma\Delta_G P_n
+\mathcal U^{(c)}_n
\right]
$$

이번 부분 검증에서 기능 축 \(c\)는 다음 세 가지다.

$$
c\in
\{
\mathrm{vigilance},
\mathrm{working\ memory},
\mathrm{subjective\ arousal}
\}
$$

검증 기준:

$$
\mathcal L_{\mathrm{matched}}
<
\mathcal L_{\mathrm{wrong\;task}}
$$

그리고 generic prototype보다도 좋아야 한다.

$$
\mathcal L_{\mathrm{matched}}
<
\mathcal L_{\mathrm{generic}}
$$

## 과제별 의미

| 과제 | 해석한 특수 영역 | event-level 관측 신호 |
|---|---|---|
| `PVT` | 지속주의/경계 | response time, lapse fraction |
| `workingmemorytest` | 작업기억 | correct, no_grids, response time |
| `sleepiness` | 주관적 각성 | KSS rating, response time |

여기서 `sleepiness`는 수면 단계 분석이 아니라, 피험자가 scanner 안에서 보고한 주관적 각성 상태로 사용했다.

## 실행

```bash
python examples\physics\fetch_ds000201_cognitive_partial.py --subject-count 4
python examples\physics\ds000201_cognitive_gate.py --subject-count 4
```

확보한 작은 파일:

| 항목 | 값 |
|---|---:|
| 선택 피험자 | 4 |
| selected events | 22 |
| PVT files | 8 |
| sleepiness files | 8 |
| workingmemory files | 6 |
| failed | 0 |

작업기억 파일은 일부 피험자에게 없어서 8개가 아니라 6개만 사용했다.

## 결과

| 항목 | 값 |
|---|---:|
| 전체 case | 22 |
| PVT cases | 8 |
| PVT trials | 369 |
| workingmemory cases | 6 |
| workingmemory trials | 72 |
| sleepiness cases | 8 |
| sleepiness trials | 32 |

Subject holdout prototype gate:

| 항목 | 값 |
|---|---:|
| \(L_{\mathrm{matched}}\) | 0.01080994 |
| \(L_{\mathrm{wrong\;task}}\) | 0.33977652 |
| \(L_{\mathrm{generic}}\) | 0.19869305 |
| matched / wrong | 0.031815 |
| matched / generic | 0.054405 |
| holdout gate | pass |

## 해석

밝힌 것:

- `ds000201`은 수면뿐 아니라 지속주의, 작업기억, 주관적 각성 축도 작은 이벤트 파일만으로 분리할 수 있다.
- 한 피험자를 holdout으로 빼도, 같은 인지 과제 prototype이 wrong task prototype보다 잘 맞는다.
- 즉, 전역 뇌 방정식의 입력항은 감각/정서뿐 아니라 인지 기능 단위로도 분해 가능하다.

아직 밝히지 못한 것:

- frontoparietal network, thalamic/arousal network, motor network가 실제 BOLD에서 같은 방향으로 분리되는지.
- \(p_r=(x_a,x_s,x_b)\)를 region별로 만들었을 때도 matched cognitive operator가 이기는지.
- PVT/작업기억/각성 사이의 인과적 전이, 예를 들어 각성 저하가 작업기억 오류를 얼마나 설명하는지.
- 그래프 항 \(\gamma\Delta_G P_n\)이 flat baseline보다 나은지.

## 닫힌 결론

현재 부분 데이터로 닫을 수 있는 결론:

$$
\boxed{
\text{event-level cognitive/arousal readiness + subject-holdout prototype: passed}
}
$$

즉, 지금 검증 가능한 세 번째 축은 **주의/경계, 작업기억, 주관적 각성**이다.

이것도 아직 신경 방정식의 최종 검증은 아니며, BOLD 기반 \(p_r\) 검증으로 넘어가기 위한 구조 확인이다.
