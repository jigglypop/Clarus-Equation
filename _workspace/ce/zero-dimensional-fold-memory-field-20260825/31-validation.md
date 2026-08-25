# 31-validation — focused numerical and symbolic checks

Status: COMPLETE

## Command

```powershell
& '.codex\hooks\python.cmd' python '_workspace\ce\zero-dimensional-fold-memory-field-20260825\artifacts\verify_fold_memory_field.py'
```

## Result

등록된 27개 검사가 모두 참이었다. 핵심 carrier witness는 $n=8$, row sum $1$, $A=1.2$, $\mathcal B=1.2$이며

$$
\chi_*=0.2,
\qquad
\max\operatorname{Re}\operatorname{spec}J_*=-0.06666666666666665,
\qquad
A\sigma'(\chi_*)\rho(W)=0.8333333333333334<1
$$

를 냈다. positive-branch residual은 0이고 zero-history/zero-seed의 우변도 정확히 0이었다.

causal kernel witness는 declared $\tau=2.5$에 대해 spacetime 적분 $2.499999999999909$, residual $9.10\times10^{-14}$를 냈으며 cone 밖과 최소 delay 전의 값은 0이었다. Gaussian width를 절반으로 할 때 $L^2$ 항은 8배, gradient $L^2$ 항은 32배가 되어 각각 $\ell^{-3}$, $\ell^{-5}$ UV scaling을 재현했다. Derrick certificate는 first variation 0, second variation $-6$이었다.

이 결과는 선택한 유효모형의 대수·수치 일관성이다. quantum retention map, stress tensor 또는 dark-sector 관측 증거가 아니다.

## Canonical post-copy validation

canonical root `C:\dev\ce\ce-cosmo`, branch `main`, HEAD
`f78accbdd075454437e57ff39b6b6b0154088c10`에서 세 문서의 SHA-256을 staging과
3/3 대조했다. 각 문서의 display-math delimiter가 짝수이고 금지된 `\\[`/`\\]`
delimiter가 없음을 확인했으며, tracked 문서의 `git diff --check`에는 whitespace
오류가 없었다. CRLF가 향후 Git write에서 LF로 정규화된다는 비차단 경고만 있었다.

기존 우주론 subsystem의 집중 회귀는 다음 다섯 파일로 제한해 실행했다.

```powershell
$env:PYTHONPATH='C:\dev\ce\ce-cosmo\src;C:\dev\ce\ce-core\src'
& '.codex\hooks\python.cmd' pytest `
  'C:\dev\ce\ce-cosmo\tests\test_cosmology_registry.py' `
  'C:\dev\ce\ce-cosmo\tests\test_cosmology_ratio_audit.py' `
  'C:\dev\ce\ce-cosmo\tests\test_ce_residual_forward_model.py' `
  'C:\dev\ce\ce-cosmo\tests\test_cosmology_closure_gate.py' `
  'C:\dev\ce\ce-cosmo\tests\test_recombination_drag_adapter.py' -q
```

결과는 `58 passed in 1.06s`였다. branch에는 upstream이 없으며 기존 census 작업의
나머지 modified/untracked path는 그대로 보존했다.
