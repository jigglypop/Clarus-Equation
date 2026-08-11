# Local–Cloud Transition Kernel V10

## 지위

이 문서는 V9의 약한 반복 셸을 성공으로 다시 해석하지 않는다. V9 메모리 벤치의 STOP은
그 구조에 대한 음성 결과로 유지한다. V10은 local/private recurrent state와 shared state의
조건부 결합을 명시적으로 구현한 별도 경로다.

## 형식 구조

[정의] local state $h_i$, shared state $c$, 정규화된 local 관측 $o_i$, shared 관측 $s$에
대하여 동기 전이를 다음처럼 둔다.

$$
h_i'=\tanh\left(\alpha h_i+g_o o_i+g_{CL}c+g_\times(h_i\odot c)\right),
$$

$$
c'=\tanh\left(\gamma c+g_s s+g_{LC}\operatorname{mean}_i h_i\right).
$$

[정리] 상태영역이 $[-1,1]$이고 다음 비음 행렬의 spectral radius가 1보다 작으면 이 전이는
가중 block-sup norm에서 수축한다.

$$
M=
\begin{pmatrix}
\alpha+g_\times & g_{CL}+g_\times\\
g_{LC} & \gamma
\end{pmatrix}.
$$

현재 고정 구현은

$$
M=\begin{pmatrix}0.82&0.26\\0.06&0.72\end{pmatrix},
\qquad q=0.9355555556<0.95
$$

를 사용한다. 이 정리는 bounded synchronous map의 안정성 정리이며 과제 효용이나 AGI를
증명하지 않는다.

## 등록 개발 결과

[경험식] 64개 fresh development seed에서 seed마다 256개 train episode와 256개 evaluation
episode를 사용했다. 모든 arm은 20개 recurrent scalar와 동일한 ridge 규칙을 사용했다.

| arm | mean accuracy |
|---|---:|
| full local–cloud | 0.6530151367 |
| local only | 0.5017089844 |
| cloud only | 0.5084838867 |
| no memory | 0.5060424805 |

[경험식] 각 seed에서 가장 강한 factorial control을 선택한 보수적 paired improvement는
`0.1284790039`, 95% seed-bootstrap interval은 `[0.1192001343, 0.1373291016]`이다.
factorial interaction은 `0.1488647461`, interval은 `[0.1389770508, 0.1586303711]`이다.

[경험식] full readout을 재학습하지 않고 전이만 훼손했을 때 cross-cut, local reset,
cloud reset의 paired loss는 각각 `0.1578979492`, `0.1552734375`, `0.0848999023`이었다.
세 interval의 하한은 모두 양수였다. duplicate seed, nonfinite output, label-state bypass,
arm mismatch는 모두 0이었다.

## 등록 confirmation 결과

[경험식] 개발 전에 예약한 별도 64개 seed로 코드·과제·threshold·bootstrap을 바꾸지 않고
confirmation을 한 번 실행했다. Full accuracy는 `0.6520996094`였고, 각 seed의 최강 대조군
대비 paired improvement는 `0.1387939453`, 95% interval은
`[0.1288436890, 0.1497192383]`이었다.

[경험식] confirmation factorial interaction은 `0.1530761719`, interval은
`[0.1411117554, 0.1654067993]`이었다. Cross-cut, decision local reset, decision cloud
reset의 paired loss는 각각 `0.1597900391`, `0.1578369141`, `0.0787963867`이고 세 하한은
모두 양수였다. 일곱 gate가 모두 통과했고 integrity counter는 모두 0이었다.

## 해석 경계

[미완성] 이 결과는 하나의 합성 conditional-binding 과제에서 확인된 메커니즘 결과다.
자연언어, 도구 사용, 장기 계획, 자기수정, OOD 전이, 임의의 learned recurrent comparator에
대한 우위는 아직 시험하지 않았다.

[미완성] 이 transition kernel을 CloudCell, 기저핵, 피질, 또는 뇌 전체와 동일시할 근거는
없다. SCC 수학은 recurrent 구조를 기술하는 형식 도구일 뿐, 이 개발 결과가 생물학적
SCC 중첩을 입증하지 않는다.

[삭제된 예측] 동일 raw observation을 받는 learned recurrent comparator와 noise/horizon
OOD에서도 V10 우위가 유지된다는 예측은 V11에서 실패했다. Elman-20과 GRU-20은 V10을 크게
앞섰고, compute-matched Elman-3도 모든 panel에서 V10보다 높았다. 자세한 음성 결과는
`30_Strong_Recurrent_OOD_V11.md`에 둔다.

## 재현 경로

- 구현: `reality_stone/python/reality_stone/clarus/local_cloud_kernel.py`
- 평가: `reality_stone/python/reality_stone/clarus/local_cloud_benchmark.py`
- 등록과 결과: `_workspace/ce/agi-v10-local-cloud-development-20260812/artifacts/`
- confirmation: `_workspace/ce/agi-v10-local-cloud-confirmation-20260812/artifacts/`
- 개발 결과 SHA-256:
  `CF6F304E1217E7CC446A9B5363C38F52D95261AB513E08F56CF6DF50DDE71302`
- confirmation 결과 SHA-256:
  `E348C6D18CF6D5C11BC287BD2899FDAC52DCA69DB6B3A2FA559D5AF88F8FD6F8`
