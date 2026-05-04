# ds000116 부분 검증 보고서

## 범위

이 보고서는 `ds000116` 전체를 내려받지 않고, 현재 로컬에 확보한 작은 파일만으로 닫은 1차 검증이다.

사용한 범위:

- 데이터셋: OpenNeuro `ds000116`, snapshot `00003`
- 피험자: `sub-01`, `sub-02`
- 과제: auditory oddball, visual oddball
- run: 각 과제 3 runs
- 사용 파일: `events.tsv`, root metadata
- 사용하지 않은 파일: BOLD NIfTI, EEG MAT, T1w

따라서 이 결과는 신경활성 방정식의 최종 검증이 아니다.  
현재 결론은 **시각/청각 특수 연산자 검증으로 넘어갈 데이터 구조가 부분 표본에서 성립한다**는 것이다.

## 검증 질문

전역 뇌 방정식의 과제 입력 항을 modality별로 나누면 다음 형태다.

$$
P_{n+1}
=
\Pi
\left[
(1-\rho)P^*
+\rho P_n
+\gamma\Delta_G P_n
+\mathcal U^{(\mu)}_n
\right]
$$

여기서 modality는 다음 두 가지로 제한했다.

$$
\mu\in\{\mathrm{auditory},\mathrm{visual}\}
$$

부분 검증의 핵심 부등식은 다음이다.

$$
\mathcal L_{\mathrm{matched}}
<
\mathcal L_{\mathrm{wrong\;modality}}
$$

즉 auditory run에는 auditory operator가, visual run에는 visual operator가 더 잘 맞아야 한다.

## 결과

실행 명령:

```bash
python examples\physics\fetch_ds000116_partial.py --subject-count 2
python examples\physics\ds000116_modality_gate.py --subject-count 2
```

확보된 작은 파일:

| 항목 | 값 |
|---|---:|
| 피험자 | 2 |
| 이벤트/메타 다운로드 | 18 files |
| 누락 파일 | 0 |
| 선택된 큰 annex 파일 | 16 files |
| 선택 annex 총량 | 368.84 MB |
| 전체 BOLD 총량 | 2.888 GB |

게이트 결과:

| 항목 | 값 |
|---|---:|
| 전체 case | 12 |
| auditory runs | 6 |
| visual runs | 6 |
| auditory stimuli | 731 |
| visual stimuli | 730 |
| auditory targets | 145 |
| visual targets | 141 |
| auditory mean RT | 0.351 s |
| visual mean RT | 0.382 s |
| \(L_{\mathrm{matched}}\) | 0.00000000 |
| \(L_{\mathrm{wrong\;modality}}\) | 0.07428359 |
| \(L_{\mathrm{generic}}\) | 0.01864522 |
| readiness gate | pass |

추가로, 단순 label matching이 아니라 subject holdout prototype gate도 실행했다.  
`sub-01`로 auditory/visual prototype을 만들면 `sub-02`를 예측하고, 반대로 `sub-02`로 만들면 `sub-01`을 예측한다.

| 항목 | 값 |
|---|---:|
| holdout \(L_{\mathrm{matched}}\) | 0.00168547 |
| holdout \(L_{\mathrm{wrong\;modality}}\) | 0.89155374 |
| holdout \(L_{\mathrm{generic}}\) | 0.22408516 |
| holdout matched/wrong | 0.001890 |
| holdout matched/generic | 0.007522 |
| holdout gate | pass |

## 해석

이 결과가 밝힌 것은 제한적이다.

밝힌 것:

- `ds000116`은 auditory/visual run이 명확히 분리되어 있다.
- target/standard 구조가 충분하다.
- 반응행동 row가 target과 대응된다.
- 부분 표본에서도 auditory와 visual을 구분하는 modality gate를 구성할 수 있다.
- 한 subject에서 얻은 modality prototype이 다른 subject의 event-level 상태를 wrong modality보다 잘 예측한다.
- 따라서 다음 단계의 neural proxy 검증 대상으로 적합하다.

아직 밝히지 못한 것:

- BOLD 또는 EEG에서 실제 visual cortex, auditory cortex가 예측대로 움직이는지.
- \(p_r=(x_a,x_s,x_b)\)를 region별로 안정적으로 만들 수 있는지.
- matched modality operator가 실제 신경 상태 \(P_{n+1}\)를 wrong operator보다 잘 예측하는지.
- graph term \(\gamma\Delta_G P_n\)이 flat model보다 나은지.

## 닫힌 결론

현재 부분 데이터로 닫을 수 있는 결론은 다음이다.

> `ds000116`의 `sub-01~02` 이벤트 데이터 기준, 시각/청각 특수 연산자 검증을 시작할 수 있는 데이터 구조는 확인되었다.  
> 단, 이것은 신경활성 검증이 아니라 modality-readiness gate 통과다.

따라서 이 단계의 판정은 다음이다.

$$
\boxed{
\text{event-level modality readiness + subject-holdout prototype: passed}
}
$$

신경 방정식 검증으로 격상하려면, 다음에는 같은 피험자 중 하나만 골라 BOLD 또는 EEG를 소량 확보해 region-level \(p_r\)를 만들어야 한다.
