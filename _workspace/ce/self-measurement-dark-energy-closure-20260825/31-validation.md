# SMQ validation record

Status: COMPLETE

Stable staging snapshot에서 source compile과 focused validation을 다시
실행했다.

```text
.....                                                                    [100%]
5 passed in 2.56s
```

테스트는 shooting, Friedmann/background, fixed point, refinement, pinned BAO
asset과 profile을 검사한다. 수치 결과의 두 외부 matter boundary에서 얻은
사후 최적점은 다음과 같다.

| boundary | $\Lambda$ limit $\chi^2$ | finite-slope best | best $\chi^2$ |
|---|---:|---:|---:|
| DESI-target boundary | 10.691056 | $\lambda=0.93$ | 8.619095 |
| Planck-control boundary | 14.436420 | $\lambda=1.15$ | 9.196408 |

최적 slope가 외부 boundary에 따라 이동하며, amplitude $\rho_*$도 각
boundary에 맞추어 shooting했다. 따라서 이 결과는 사후 보정이지 0D/양자
기원의 예측이 아니다.

더 근본적으로, 위 모든 background와 smooth-growth observable은

$$
(\Theta,\rho_*)\sim(\Theta+\Delta,\rho_*e^\Delta)
$$

동치류에서 불변이다. 수치 테스트가 통과해도 절대 측정깊이 또는 0D
retention origin을 검증하지 못한다. full suite, CMB, LSS, SN 및 nonlinear
검증은 실행하거나 주장하지 않았다.

