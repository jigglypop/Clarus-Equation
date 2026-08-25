# Focused validation

Status: COMPLETE

## Self-measurement verifier

```powershell
& '.codex\hooks\python.cmd' python `
  '_workspace\ce\dimensionless-self-measurement-time-20260825\artifacts\verify_self_measurement_time.py'
```

모든 검사가 통과했다. fixed-$Z$ channel의 $\theta_*=1.5$를
$N=1,2,5,100$으로 분할한 target residual은 각각

$$
0,quad 1.96\times10^{-17},quad
5.89\times10^{-17},quad 2.94\times10^{-16}
$$

였다. $|+\rangle$ 초기상태에서 $\|A\|_1=1$이고

$$
v(1.5)=0.111565080074215,
\qquad
L(1.5)=0.388434919925785.
$$

$N=1,2,5,100,1000$의 path-length refinement가 모두 같은 $L$을
$10^{-12}$ 이내에서 재현했다. $p=(0.8,0.2)$에서

$$
\overline C_I=0.293213034199730,
\qquad
C_{\rm self}=0.227788362921137,
\qquad
(1-e^{-1.5})\ln2=0.538485139155172
$$

이므로 유계성이 확인됐다. stationary diagonal state는 $L=0$인데도
$\overline C_I>0$인 완전 반례를 통과했다.

비가환 순서 반례의 Frobenius 차이는
$0.049497474683058$이었다. recoherence family는 $\theta$가
$0\to\infty\to0$으로 되돌아감을 재현했다. unitary cycle은 endpoint가 초기상태와
일치하면서 path length $3.141592601912349$를 내어 국소 변화가 시간의 화살을
보장하지 않음을 확인했다.

## Canonical document checks

세 canonical 문서의 SHA-256은 staging과 3/3 일치했다. 줄 하나가 `$$`인
display delimiter 수는 원장 102, 선택과 접힘 66, dark-energy derivation 40으로
모두 짝수였고, 금지된 `\[`/`\]`/`\(`/`\)` delimiter와 trailing whitespace는
없었다. `git diff --check`의 exit code는 0이었다. ledger와 한 기존 source file에
대한 CRLF-to-LF 예고만 있었고 whitespace error는 없었다.

## Focused cosmology regression

```powershell
$env:PYTHONPATH='C:\dev\ce\ce-cosmo\src;C:\dev\ce\ce-core\src'
& '.codex\hooks\python.cmd' pytest `
  'C:\dev\ce\ce-cosmo\tests\test_cosmology_registry.py' `
  'C:\dev\ce\ce-cosmo\tests\test_cosmology_ratio_audit.py' `
  'C:\dev\ce\ce-cosmo\tests\test_ce_residual_forward_model.py' `
  'C:\dev\ce\ce-cosmo\tests\test_cosmology_closure_gate.py' `
  'C:\dev\ce\ce-cosmo\tests\test_recombination_drag_adapter.py' -q
```

Result: `58 passed in 1.28s`, exit code `0`.

이 검증은 fixed dephasing 유효모형의 algebra, counterexamples와 문서 회귀를
확인한다. physical 0D sector, retention map, energy scale 또는 dark-sector
관측 증거를 제공하지 않는다.

