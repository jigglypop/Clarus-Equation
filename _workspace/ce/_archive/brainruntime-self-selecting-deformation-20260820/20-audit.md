# M4-R 구현 전 감사

Status: COMPLETE

Gate: REVISE (Revision 2 fold semantics re-audit pending)

Audited: 2026-08-20

## 안정 입력

| 파일 | SHA-256 |
|---|---|
| `00-contract.md` | `840f0138dcf3f50f34857e3032d56e2b752109b947e622bb5b0a2fd8009ed6e7` |
| `10-sources.md` | `d799079550f6c4c0f96763fea558ae82545310452b167c1d44e2f7779056a37f` |
| `11-math.md` | `0377c63b129fb19521a7b387f020b10f05eda085aacd3d58addd221385cfef00` |
| `12-routes.md` | `69b907460a0dad28e579cf69cde3a7550d8fc7d58b955e00737b549e57f7a618` |

## 판정

**[산출]** 선행 M1/T1의 증거 경로와 hash가 일치한다. 최초 M4-0의 공선 후보와 affine
held-out target 누수는 구현 전에 제거했다. 현재 M4-R은 cue/value-start tensor를 분리하고
terminal post error와 delayed pre trace의 $d\times d$ outer product를 사용한다.

**[산출]** 모든 score·cosine·temperature·fold variance와 install penalty는 무차원이다.
held-out `(1,1)`은 formula-discovery, candidate generation, score, update와 decoder calibration에
들어갈 수 없으며 frozen endpoint에서만 읽는다.

**[산출]** seed별 `min_control_advantage`는 실제 실행된 모든 matched control을 포함한다.
결측, nonfinite 또는 ambiguous write receipt는 seed fail이다. discovery, untouched validation,
sealed confirmation의 seed 집합은 서로 분리되어 있다.

## 구현 인가 범위

M4-R basic과 unconditional controls, focused source tests, formula-discovery seeds `97401..97408`
만 인가한다. fold는 max-scale 75% 또는 instability trigger가 machine receipt에서 성립할 때만
Revision 2로 열 수 있다. development-validation과 confirmation seed는 아직 봉인한다.

구현은 contract hash, candidate score/selection/applied delta, direction rank/cosine, held-out
row-count receipts, control endpoints, parity/store cutoff와 fold trigger를 기록해야 한다.
