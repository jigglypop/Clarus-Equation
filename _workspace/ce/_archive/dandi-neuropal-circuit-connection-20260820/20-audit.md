# 구현 전 안정 snapshot 감사

Status: PASS_TO_DEVELOPMENT

Gate: PASS

- 000565 `<300 MB` 선택 실패는 endpoint 전에 schema failure로 보존됐다.
- 대체 source 000541의 8개 worm은 공식 content size만으로 선택됐다.
- 첫 000541 LINDI schema는 dNMF/raw calcium `936 x 177`, labels `177`, NeuroPAL masks `177`, sample rate `4 Hz`를 보인다.
- chemical stimuli는 butanone, NaCl, pentanedione 각 10초이며 stimulus 전 5초부터 후 20초까지 고정 제외한다.
- label·mask·trace row 수가 첫 worm에서 일치한다. blank label 하나는 결과와 무관하게 제외한다.
- A3는 observational time-reversal connection operator이며 causal-parent 지위가 아니다.
- calibration-only 뉴런별 median/MAD 역치가 $u_i=q_i r_i$를 통해 실제 operator 구성에 연결되며, $c_{ij},\Omega_{ij}$가 edge별 관측 강도를 보존한다.
- 독립적인 edge별 물리 지연 receipt는 없으므로 1-sample 관측 lag를 축삭 지연으로 해석하지 않는다.
- identity-shuffle과 construction-block phase-randomized control을 개발 판정 전에 구현했다.
- $L_c\succeq0$, $\Omega^T=-\Omega$, 모든 exp/log 인자 부재와 좌표 무차원화를 확인했다.
- development 3 worms와 confirmation 5 worms가 분리됐고 confirmation response는 아직 열지 않았다.
- 실패 뒤 허용된 식 수정은 spectral instability일 때의 operator normalization 한 번뿐이다.
