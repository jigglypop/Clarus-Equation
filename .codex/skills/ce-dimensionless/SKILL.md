---
name: ce-dimensionless
description: CE 코어에 들어가는 식의 무차원성을 감사한다. exp/log/고정점/확률 코어에 들어가는 인자가 무차원인지, 차원 있는 양이 기준 스케일로 정규화됐는지 검사하고 dimensionless checker로 검증한다. 새 식을 코어에 넣거나 "이 식 차원 맞나 / 무차원 게이트 통과하나" 류 요청에 사용.
---

# 무차원 게이트 감사

정본(`docs/참조/무차원_감사_수학.md`)의 단일 판정:

> **exp/log/고정점/확률 코어에 들어가는 값은 반드시 무차원이어야 한다.**

차원 있는 물리량은 먼저 기준 스케일로 나눠 무차원 비율로 만든 뒤에만 코어에 넣는다.

## 차원 벡터 규약
기저 `d=(M,L,T,Θ)`. 무차원 = `(0,0,0,0)`.

| 양 | 차원 벡터 | 코어 사용 규칙 |
|---|---|---|
| ε², Ω_b, α_s | (0,0,0,0) | 직접 사용 가능 |
| R (Ricci) | (0,-2,0,0) | R·L_c² 또는 R/R_c |
| m_φ | (1,0,0,0) | m_φ/m_p, m_φ/v_EW |
| H_0 | (0,0,-1,0) | H_0·t_Pl |

## 절차

뇌/AGI 식에서는 `.codex/harnesses/real_brain_equation_discovery_loop.md`의 상태식과 측정모형을 함께 감사한다. 시간, 전압, 전류, 발화율, 전도 지연, 시냅스 강도, 칼슘·형광과 sampling scale을 구분하고, 각 무차원화 기준을 출처·단위와 함께 기록한다. indicator kinetics·preprocessing·측정 단위를 신경 동역학의 무차원 자유 파라미터에 숨기지 않는다.

1. 검사할 식에서 exp/log/sin/확률/고정점에 들어가는 모든 인자를 추출한다.
2. 각 인자의 차원 벡터를 구해 `(0,0,0,0)`인지 확인한다. 아니면 어떤 기준 스케일로 정규화해야 하는지 제시한다.
3. 여러 양이 곱해진 경우 Buckingham-Pi로 무차원 조합인지 점검한다(차원 벡터 행렬의 영공간).
4. 코드로 검증한다:

```powershell
python -m pytest tests\test_dimensionless.py -q
python reality_stone\python\reality_stone\clarus\dimensionless.py
```

체커가 아직 다루지 않는 새 식이면 `dimensionless.py`/`dimensionless_checker.py`의 등록 방식을 읽고, 같은 패턴으로 식을 추가한 뒤 테스트를 다시 돌린다. (B10/Co5: 체커 완성은 미완 항목 — 누락 식을 채우는 것이 가치 있는 작업.)

## 출력 형식

```
식: <식>
| 코어 인자 | 차원 벡터 | 무차원? | 정규화 |
|---|---|---|---|
| ... | (a,b,c,d) | yes/no | /기준스케일 |

차원 상태: 무차원 / 차원 불일치(어느 인자가 차원을 가짐)
코드 검증: <test_dimensionless 결과>
```

주의: 무차원성은 **차원 정합일 뿐 물리적 정당성이 아니다.** 식이 옳다는 뜻으로 보고하지 마라.
