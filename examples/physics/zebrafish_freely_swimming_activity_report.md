# Zebrafish freely swimming activity gate

Figshare freely-swimming zebrafish 자료의 작은 figure5/S8 chunk로 자유수영 상태 activity 구조를 점검했다.

이 chunk에는 정렬된 tail-behavior label이 없으므로 최종 activity->behavior gate는 아니다.

## 결과

- region count: 18
- green free/imm mean similarity: 0.476526
- red free/imm mean similarity: 0.011651
- free green recurrent/baseline: 0.154988
- free red recurrent/baseline: 0.157001
- pass: True

## 해석

- 자유수영에서도 region activity는 저차원 recurrent state로 평균 baseline보다 잘 닫힌다.
- free/imm similarity는 조건이 달라도 일부 region-level activity 구조가 보존되는지 보는 보조 지표다.
- 다음에는 같은 freely-swimming 자료에서 neural trace와 tail/stage tracking이 시간 정렬된 chunk를 찾아 activity->movement gate를 만들어야 한다.
