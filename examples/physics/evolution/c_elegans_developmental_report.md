# C. elegans 발달 connectome 게이트

Witvliet dataset 1-8을 모두 읽어 L1/L2/L3 층화 구조가 발달 전반에서 유지되는지 점검했다.

## 질문

$$
\mathcal L_{\mathrm{chemical\ weighted\ L1/L2/L3}}
<
\mathcal L_{\mathrm{flat/random}}
$$

## 단계별 결과

| stage | synapses | chem block/flat | chem p | all block/flat | electrical block/flat | binary block/flat | lambda max | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1235.0 | 0.756479 | 0.057971 | 0.758016 | 0.891754 | 0.766957 | 210.529187 | False |
| 2 | 1791.0 | 0.737261 | 0.043978 | 0.734655 | 0.826484 | 0.737934 | 318.376645 | True |
| 3 | 1957.0 | 0.734236 | 0.041979 | 0.726537 | 0.778895 | 0.769426 | 346.999413 | True |
| 4 | 2697.0 | 0.714496 | 0.029485 | 0.712483 | 0.784445 | 0.800157 | 513.036590 | True |
| 5 | 3958.0 | 0.728187 | 0.030485 | 0.723991 | 0.759796 | 0.793939 | 666.960449 | True |
| 6 | 4113.0 | 0.717524 | 0.029985 | 0.718669 | 0.820706 | 0.846896 | 722.188214 | True |
| 7 | 6624.0 | 0.717431 | 0.028986 | 0.720519 | 0.816848 | 0.823661 | 1141.809166 | True |
| 8 | 7222.0 | 0.717370 | 0.028986 | 0.719382 | 0.830206 | 0.914439 | 1242.163296 | True |

## 요약

| 항목 | 값 |
|---|---:|
| stages | 8 |
| passed weighted chemical stages | 7 |
| mean chemical block/flat | 0.727873 |
| min chemical block/flat | 0.714496 |
| max chemical block/flat | 0.756479 |
| mean chemical permutation p | 0.036482 |
| Spearman stage vs chemical block/flat | -0.761905 |
| Spearman stage vs synapses | 1.000000 |
| Spearman stage vs lambda max | 1.000000 |

## 해석

- weighted chemical 구조가 여러 발달 단계에서 유지되면, 층화 그래프 문법은 성체에서 갑자기 생긴 것이 아니다.
- stage와 synapse/lambda max의 상관은 성장하면서 그래프 강도와 안정성 스케일이 어떻게 변하는지 보여준다.
- 이 검증은 구조 connectome만 본 것이며, 행동 동역학 전이 검증은 아니다.
