# BA-SRM2 최종 보고 — 고차원 방향은 맞지만 현재 small 입력은 막혔다

Status: COMPLETE

이번 재설정에서 4차원을 뇌의 고정 차원으로 보는 해석을 폐기했다. 실제 시냅스는
과거 spike와 전압ㆍ전류 이력, 자극 frequency와 recovery interval에 반응하는
연산자이므로 고차원 함수공간이 더 자연스럽다. 그러나 유한 관측으로 무한차원
전체의 양의 정부호 리만 계량을 식별할 수는 없다. 정확한 대상은 관측에 보이지 않는
kernel 방향을 나눈 finite observable quotient다.

local small DB에서 즉시 고차원화를 시도할 수 있을 것처럼 보였던
`stp_all_stimuli.pulse_amplitudes`는 사용할 수 없다. 공식 producer가 하나의 list를
12개 slot에 공유하여 모든 pulse를 같은 aggregate로 저장했다. train-only 구조
감사에서도 ex 1,324개와 in 2,613개 protocol record 전부에서 12 slot의 동일성이
재현됐다. 따라서 이를 12, 36 또는 48차원 pulse trajectory로 펼치면 데이터 차원이
아니라 중복 좌표를 만드는 셈이다.

이 반례는 식을 대충 만들었다는 뜻이 아니라, 결과를 열기 전에 측정 pipeline이
주장한 좌표를 실제로 보존하는지 검사한 것이다. 그 검사 덕분에 가짜 고차원
geometry, 인공 rank와 confirmation 오염을 막았다. train JSON은 slot 동일성
검사에만 열었고 amplitude 크기를 보고ㆍ적합ㆍ채점하지 않았다. confirmation
outcome은 열지 않았다.

함수공간 수학은 살아 있다. Fréchet 미분 가능한 반응연산자 $M$에 대해

$$
G_x(u,v)=\langle C^{-1/2}DM_xu,C^{-1/2}DM_xv\rangle
$$

는 response pullback PSD form이다. 유한 output이 $m$개면 rank는 $m$ 이하이므로
전체 무한차원 공간이 아니라 $T_x\mathcal H/\ker J_x$에서만 내적이 된다. 여러 점을
하나의 manifold로 잇기 위해서는 rank가 국소적으로 일정하다는 조건도 필요하다.

다음 실제 경로는 Allen-SynPhys medium event DB다. 이 파일은 10.36 GiB이며
row-level pulse identity, stimulus, recording과 fitted amplitude를 보존한다. 아직
내려받지 않아 SHA-256, integrity와 usable support는 미검증이다. medium을 잠근 뒤
과거 pulse history와 held-out future/protocol을 분리하고, FPCA/RKHS sieve 차원을
train-only로 선택해야 한다. 그때도 성공 주장은 finite observable quotient의 예측력에
한정되며 conductance, $Npq$, STDP, homeostasis, 기억 또는 AGI mechanism으로
승격하지 않는다.

최종 판정은 `BLOCKED_INPUT / TRAIN_STRUCTURE_CONTACT_ONLY /
CONFIRMATION_UNTOUCHED`다. 고차원 방향은 채택했지만,
현재 small source를 억지로 부풀린 실행은 기각했다.
