# C. elegans 발달 자극-행동 구조 게이트

Witvliet dataset 1-8 전체에서 L1 stimulus domain이 같은 L3 output domain으로 보존되는지 확인했다.

## 기준

$$
\mathrm{Flow}(L1_d\to L3_d)
>
\mathrm{Flow}(L1_d\to L3_{d'\ne d}),
\qquad
\mathrm{matched/wrong}>1.5
$$

## 단계별 결과

| stage | synapses | chem matched/wrong | chem p | chem pass | binary matched/wrong | binary pass |
|---:|---:|---:|---:|---|---:|---|
| 1 | 1235.0 | 3.186427 | 0.034393 | True | 1.446780 | False |
| 2 | 1791.0 | 2.600517 | 0.034393 | True | 1.264549 | False |
| 3 | 1957.0 | 3.302136 | 0.034393 | True | 1.380026 | False |
| 4 | 2697.0 | 3.464723 | 0.034393 | True | 1.328685 | False |
| 5 | 3958.0 | 3.297920 | 0.034393 | True | 1.208942 | False |
| 6 | 4113.0 | 2.995919 | 0.034393 | True | 1.193745 | False |
| 7 | 6624.0 | 3.428513 | 0.034393 | True | 1.040436 | False |
| 8 | 7222.0 | 3.431872 | 0.034393 | True | 1.113045 | False |

## 요약

| 항목 | 값 |
|---|---:|
| stages | 8 |
| passed chemical weighted stages | 8 |
| mean chemical matched/wrong | 3.213504 |
| min chemical matched/wrong | 2.600517 |
| max chemical matched/wrong | 3.464723 |
| mean p value | 0.034393 |
| Spearman stage vs matched/wrong | 0.476190 |
| Spearman stage vs synapses | 1.000000 |

## 해석

- 통과 단계가 많으면 자극-domain에서 output-domain으로 이어지는 구조 channel이 발달 전반에서 보존된다는 뜻이다.
- binary가 실패하고 weighted chemical이 통과하면, 행동 proxy도 단순 연결 유무가 아니라 synaptic weight에 실린다는 뜻이다.
- 이 검증은 실제 행동 trial이 아니라 구조 proxy다.
