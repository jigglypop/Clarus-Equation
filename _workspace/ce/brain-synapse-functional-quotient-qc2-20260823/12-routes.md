# BA-SRM3 route register

Status: `ROUTE_A_ACTIVE / OTHERS_LOCKED`

| Route | 내용 | 현재 상태 | 승격 조건 |
|---|---|---|---|
| A | 동일 frozen train manifest + official sign-matched response QC | `ACTIVE_SUPPORT_GATE` | E/I별 160 slices, 16 MAD PASS |
| B | BA-SRM2 strict `stim_pulse.qc_pass=1` | `STOP_PRESERVED` | 재개 금지; 새 source가 있어야 별도 후보 |
| C | small DB 12-slot pulse vector | `REJECTED_PRODUCER_ALIAS_BUG` | medium/full event row만 허용 |
| D | full waveform Hilbert output | `NOT_ACQUIRED / NOT_THIS_CONTRACT` | full DB와 새 prereg 필요 |
| E | conductance/$Npq$/STDP/homeostasis | `UNOBSERVED` | joint physiological source 필요 |

Route A가 실패하면 QC, target, pulse horizon 또는 dimension을 바꾸어 같은 run을 계속하지 않는다.
