# ds000201 부분 검증 보고서

## 범위

이번에는 수면 전이 자체가 아니라, `ds000201`의 과제 이벤트만 사용해 특수 영역별 연산자 검증 가능성을 확인했다.

사용한 범위:

- 데이터셋: OpenNeuro `ds000201`, snapshot `1.0.3`
- 피험자: `sub-9001`, `sub-9002`
- 세션: `ses-1`, `ses-2`
- 과제: `hands`, `faces`, `arrows`
- 사용 파일: `events.tsv`, root metadata
- 사용하지 않은 파일: BOLD NIfTI, DWI, T1w

따라서 이 결과는 신경활성 검증이 아니라 **event-level task-domain readiness gate**다.

## 검증 질문

전역 뇌 방정식의 과제 입력 항을 영역별 특수 연산자로 나누면 다음처럼 쓸 수 있다.

$$
P_{n+1}
=
\Pi
\left[
(1-\rho)P^*
+\rho P_n
+\gamma\Delta_G P_n
+\mathcal U^{(d)}_n
\right]
$$

여기서 영역 \(d\)는 이번 부분 검증에서 다음 세 가지로 제한했다.

$$
d\in
\{
\mathrm{hands/pain},
\mathrm{faces/emotion},
\mathrm{arrows/control}
\}
$$

검증 기준은 다음이다.

$$
\mathcal L_{\mathrm{matched\;task}}
<
\mathcal L_{\mathrm{wrong\;task}}
$$

추가로 generic prototype보다도 좋아야 한다.

$$
\mathcal L_{\mathrm{matched\;task}}
<
\mathcal L_{\mathrm{generic}}
$$

## 과제별 의미

| 과제 | 해석한 특수 영역 | event-level 관측 신호 |
|---|---|---|
| `hands` | 통증/촉각, 신체감각 | `Pain`, `No_Pain`, unpleasantness rating |
| `faces` | 얼굴정서, 사회적 시각처리 | happy/neutral/angry/fearful face, rating |
| `arrows` | 인지조절, 정서조절 | Maintain/Enhance/Suppress cue, regulation success |

이 세 과제는 서로 같은 감각 입력만 다른 것이 아니라, 요구하는 기능 축이 다르다.  
그래서 `ds000116`의 auditory/visual gate보다 더 넓은 특수영역 분해다.

## 실행

```bash
python examples\physics\fetch_ds000201_partial.py --subject-count 2
python examples\physics\ds000201_task_gate.py --subject-count 2
```

확보한 작은 파일:

| 항목 | 값 |
|---|---:|
| 선택 피험자 | 2 |
| small file 다운로드 | 19 files |
| 실패 파일 | 0 |
| 전체 manifest file | 11869 |
| 전체 BOLD 총량 | 133.344 GB |
| 선택 annex file | 50 |
| 선택 annex 총량 | 3955.02 MB |

## 결과

| 항목 | 값 |
|---|---:|
| 전체 case | 12 |
| missing | 0 |
| `hands` cases | 4 |
| `faces` cases | 4 |
| `arrows` cases | 4 |
| `hands` trials | 160 |
| `faces` trials | 960 |
| `arrows` trials | 1208 |

Subject holdout prototype gate:

| 항목 | 값 |
|---|---:|
| \(L_{\mathrm{matched}}\) | 0.01254154 |
| \(L_{\mathrm{wrong\;task}}\) | 0.56601281 |
| \(L_{\mathrm{generic}}\) | 0.22475054 |
| matched / wrong | 0.022158 |
| matched / generic | 0.055802 |
| holdout gate | pass |

## 해석

밝힌 것:

- `ds000201`에는 수면 외에도 통증/촉각, 얼굴정서, 인지조절 과제를 나눠 검증할 수 있는 이벤트 구조가 있다.
- 부분 표본 2명, 2세션만으로도 세 과제의 event-level 상태가 분리된다.
- 한 피험자에서 만든 task prototype이 다른 피험자의 같은 task를 wrong task보다 잘 맞춘다.
- 즉, 전역 방정식 하나만으로 끝나는 것이 아니라, 특수 영역 입력항 \(\mathcal U^{(d)}\)를 실제 데이터셋 구조에 맞춰 분해할 수 있다.

아직 밝히지 못한 것:

- 실제 BOLD에서 pain network, face network, control network가 같은 방향으로 움직이는지.
- region-resolved \(p_r=(x_a,x_s,x_b)\)를 만들었을 때도 matched task operator가 이기는지.
- \(\gamma\Delta_G P_n\) 그래프 항이 flat baseline보다 실제 예측력이 좋은지.
- 수면 조건과 과제 특수영역이 상호작용하는지.

## 닫힌 결론

현재 부분 데이터로 닫을 수 있는 결론은 다음이다.

$$
\boxed{
\text{event-level task-domain readiness + subject-holdout prototype: passed}
}
$$

즉, 수면 말고도 검증 가능한 축은 있다.  
이번에 확보된 축은 **통증/촉각, 얼굴정서, 인지조절**이다.

다만 이것은 아직 신경 방정식의 최종 검증이 아니라, BOLD 검증으로 넘어가기 위한 데이터 구조 검증이다.
