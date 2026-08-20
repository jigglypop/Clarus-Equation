# A4 개발 판정

Status: STOP_A4_DEVELOPMENT

A4의 새 세 개체는 모두 apparatus와 식별성 gate를 통과했지만 primary prediction gate에서 실패했다.

- A4 mean $\Delta=-0.0342255412$, positive $0/3$;
- fixed geometry $0$;
- edge-shuffle $+0.0040588113$;
- time-shift $-0.0090686783$;
- state-shift $-0.0190679967$;
- identity-shuffle $-0.0296543100$;
- phase-randomized $-0.0253293683$.

세 개체의 mean $h$는 약 $0.00189$--$0.00244$, mean 유효 edge 길이비는 $0.99880$--$0.99910$이었다. 즉 구성된 순간 수축은 평균 약 $0.09$--$0.12\%$로 매우 작았다. 그러나 A4는 전체 $L_tz$를 construction RMS로 정규화한 한 graph feature에 fixed geometry와 dynamic deformation을 함께 넣었다. 그 결과 동일한 $\beta_g$가 두 성분을 묶었고, 작은 deformation 방향도 단위 RMS로 확대됐다.

이 관찰은 A4를 구하지 않는다. `source-manifest-v3.json`의 confirmation 다섯 자산은 봉인하고 A4는 기각한다. 아직 열지 않은 별도 여덟 자산에서만, inherited fixed geometry $L_0$와 activity-induced deformation $\Delta L_t=L_t-L_0$의 계수를 분리한 새 A5를 시험할 수 있다.

