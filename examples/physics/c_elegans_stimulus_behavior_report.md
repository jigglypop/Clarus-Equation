# C. elegans 자극-행동 구조 게이트

이 검증은 실제 행동 기록이 아니라 connectome 기반 proxy다. 질문은 L1 input domain이 L2 relay를 거쳐 같은 L3 premotor/integrative domain으로 더 강하게 흐르는가이다.

## 검증식

$$
\mathrm{Flow}(L1_d\to L3_d)
>
\mathrm{Flow}(L1_d\to L3_{d'\ne d})
$$

## 결과

| 항목 | 값 |
|---|---:|
| matched mean | 0.37828533 |
| wrong mean | 0.11022711 |
| matched / wrong | 3.431872 |
| permutation p | 0.034393 |
| effect threshold | 1.500000 |
| passed | True |

## domain flow

| route | combined flow | direct | two-step |
|---|---:|---:|---:|
| Anterior->Anterior | 0.40329661 | 0.42442357 | 0.38216966 |
| Anterior->Lateral | 0.24763359 | 0.16310845 | 0.33215873 |
| Anterior->Avoidance | 0.01528595 | 0.01280956 | 0.01776234 |
| Anterior->Taxis | 0.04885711 | 0.04269855 | 0.05501568 |
| Lateral->Anterior | 0.10947095 | 0.02564103 | 0.19330087 |
| Lateral->Lateral | 0.58360729 | 0.65811966 | 0.50909491 |
| Lateral->Avoidance | 0.05052657 | 0.04273504 | 0.05831809 |
| Lateral->Taxis | 0.04143545 | 0.00000000 | 0.08287091 |
| Avoidance->Anterior | 0.11056432 | 0.08068460 | 0.14044405 |
| Avoidance->Lateral | 0.24579962 | 0.20782396 | 0.28377527 |
| Avoidance->Avoidance | 0.30458385 | 0.38141809 | 0.22774962 |
| Avoidance->Taxis | 0.07716321 | 0.04645477 | 0.10787165 |
| Taxis->Anterior | 0.14174111 | 0.12020033 | 0.16328189 |
| Taxis->Lateral | 0.17264602 | 0.10434057 | 0.24095147 |
| Taxis->Avoidance | 0.06160138 | 0.05175292 | 0.07144985 |
| Taxis->Taxis | 0.22165357 | 0.23622705 | 0.20708010 |

## 해석

- 통과하면 C. elegans connectome 안에 자극 domain을 같은 행동-output domain으로 보존하는 구조 경로가 있다는 뜻이다.
- 실패하면 L1/L2/L3 층화는 있어도 행동 domain channel은 아직 connectome proxy만으로 닫히지 않는다는 뜻이다.
- 이 검증은 실제 행동 데이터가 아니므로 최종 행동 방정식 검증은 아니다.
