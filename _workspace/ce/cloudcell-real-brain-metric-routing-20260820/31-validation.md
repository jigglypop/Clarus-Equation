# Input-audit validation

Status: COMPLETE

## 실행

다음 focused command를 repository root에서 실행했다.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe' -B `
  '_workspace/ce/cloudcell-real-brain-metric-routing-20260820/artifacts/inspect_cloudcell_inputs.py' `
  --output '_workspace/ce/cloudcell-real-brain-metric-routing-20260820/artifacts/cloudcell-input-audit.json'
```

Exit code는 0이었다. JSON envelope의 검증값은 다음과 같다.

| 항목 | 값 |
|---|---:|
| schema | `clarus.cloudcell.input-audit.v1` |
| dataset | 3 |
| recording | 22 |
| GCaMP | 11 |
| GFP control | 11 |
| all recording checks | `true` |

세 archive hash는 markdown inventory와 기계 JSON에서 일치했다.

- AML18: `588d7666f4e8afebad1ab9b8483244a6de0303251d862425522c2b8dd78bbd82`
- AML32: `6b71a6ba1a5d2f1ef3bf9661e845e1e52634bae217fc0c2630a83fca07daed63`
- AML310: `144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a`

## 검증 경계

`PASS_INPUT_SCHEMA`는 형광/행동 필드와 시간 정렬 apparatus가 존재한다는 뜻이다.
아직 likelihood, Fisher tensor, predictive score, biological effect를 적합하거나
측정하지 않았다. 따라서 경험 GO, anatomical routing, causal routing 또는 metric
mediation의 증거가 아니다.
