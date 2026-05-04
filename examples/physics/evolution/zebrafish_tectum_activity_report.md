# Zebrafish tectum spontaneous activity gate

공개 zebrafish optic tectum calcium activity 자료로 assembly 공유 구조와 저차원 recurrent 예측을 점검했다.

이 자료에는 행동이 없으므로 whole-brain behavior gate가 아니라 activity-only pilot이다.

## 결과

| fish | cells | time | assemblies | covered | coassembly/flat | coassembly p | recurrent/baseline | R2 | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| zf_20151104-f1 | 114 | 4245 | 5 | 77 | 0.681772 | 0.000500 | 0.447369 | 0.552631 | True |
| zf_20170215-f3 | 75 | 5660 | 4 | 64 | 0.572443 | 0.000500 | 0.330413 | 0.669587 | True |

## 해석

- 통과하면 connectome-only 단계를 넘어 실제 calcium activity에서 assembly/recurrent state 구조가 보인다는 뜻이다.
- assembly CSV는 셀별 단일 라벨이 아니라 assembly별 뉴런 인덱스 목록이며, 일부 뉴런은 여러 assembly에 겹친다.
- 따라서 검증 단위는 단일 라벨 블록이 아니라 두 뉴런이 assembly를 공유하는지 여부다.
- 행동 자료가 없으므로 stimulus-action 방정식은 아직 아니고, 척추동물 국소 회로의 폐쇄 동역학 후보만 검증한다.
- 다음은 whole-brain + behavior zebrafish 자료로 넘어가야 한다.
