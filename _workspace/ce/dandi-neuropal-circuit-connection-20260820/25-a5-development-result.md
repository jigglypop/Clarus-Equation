# A5 개발 판정

Status: STOP_A5_DEVELOPMENT

A5는 새 세 개체 모두 provenance·apparatus·rank gate를 통과했지만 primary gate에서 실패했다.

- A5 mean $\Delta=-38.2709121230$, positive $1/3$;
- no-deformation $0$;
- edge-shuffle $-0.3557882721$;
- time-shift $-1.3017098713$;
- state-shift $-15.7745794845$;
- identity-shuffle $-6.9570004854$;
- phase-randomized $-14.5992319400$.

개체별 A5 $\Delta$는 $-114.8217591$, $-0.0205724$, $+0.0295952$였다. 첫 개체의 $\beta_\Delta=0.43338$이 construction에서 적합됐으나, deformation feature RMS가 construction $1.0713$에서 test $5.7517$로 $5.37$배 커졌다. 둘째 개체도 $3.52$배 증가했고 셋째는 $0.17$배로 감소했다. 따라서 $s_\Delta$로 construction 표준화한 incremental direction은 worm 안에서도 시간 구간 사이에 안정적이지 않았다.

이 결과는 A5를 기각한다. `source-manifest-v4.json`의 confirmation 다섯 자산은 봉인한다. 같은 DANDI cohort 계열에서 threshold, clip, RMS, ridge 또는 horizon을 다시 조정하지 않는다. 현재 실자료가 지지하는 것은 NeuroPAL 좌표와 calcium activity로 상태 의존 graph feature를 계산할 수 있다는 apparatus 사실뿐이며, 그 feature가 fixed geometry나 matched null보다 미래 상태를 더 잘 예측한다는 주장은 지지되지 않는다.

