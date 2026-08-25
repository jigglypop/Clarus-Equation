# 자기측정 잔여량에서 암흑에너지로 가는 대안 경로

Status: COMPLETE

| route | energy assignment | dynamics | 판정 |
|---|---|---|---|
| R0 | $V=\rho_*e^{-\Theta}=\rho_*u$ | source-free canonical scalar | 채택: 안정한 최소 완성 |
| R1 | $V_L=\rho_*(1-e^{-\Theta})=\rho_*c$ | source-free canonical scalar | 기각: rest-data 역방향 가속 |
| R2 | $\rho_{\rm fold}\propto c$ | explicit reservoir/source $Q$ | 보류: 별도 interacting model 필요 |
| R3 | $\rho_{\rm fold}\propto c$ | phantom kinetic | 기각: ghost를 허용하는 다른 이론 |
| R4 | geometric readout | modified gravity/nonminimal coupling | 보류: 별도 작용과 screening 필요 |

## R0. 남은 비선택 잔여량

앞선 자기측정 semigroup에서 실제로 지수적으로 남는 양은

$$
u=e^{-\Theta}=1-c
$$

이다. 이를 공변 장의 퍼텐셜 fraction으로 옮겨

$$
V=\rho_*u=\rho_*e^{-\Theta}
$$

로 두면 positive canonical kinetic, positive curvature와 가속 fixed point를
동시에 얻는다. 이것은 “사용된 기회비용”의 에너지가 아니라 아직 선택되지
않은 경로의 **남은 가용도**가 중력에 읽힌다는 해석이다.

이 경로는 수학적으로 닫히지만 다음 화살표는 공리다.

$$
\text{operational record}
\xrightarrow{\mathcal R_\Theta}
\text{local field initial data}
\xrightarrow{\rho_*\ \text{normalization}}
\text{energy density}.
$$

## R1. 누적 기회비용의 literal 자율장

$c=1-e^{-\Theta}$를 그대로 $V_L/\rho_*$로 잡으면 퍼텐셜 경사가 양수라
정지한 장을 $\Theta$ 감소 방향으로 민다. 누적깊이가 계속 증가한다는 원래
해석과 동역학이 반대로 움직이므로 완전 반례다. 이 부모 경로는 활성 주장과
최종 유도에서 제거하고 실패 기록으로만 남긴다.

## R2. reservoir가 있는 누적 에너지

누적량이 실제 밀도로 계속 증가하려면 일반적으로

$$
\dot\rho_{\rm fold}+3H(1+w_{\rm fold})\rho_{\rm fold}=Q
$$

처럼 source $Q$가 필요하다. 그러면 에너지를 주는 reservoir의 stress와
반대 부호 continuity equation, covariant interaction action, 안정성을 함께
제시해야 한다. 현재 CE operational theorem은 $Q$를 결정하지 않으므로 이
경로는 미완성이다.

## R3. phantom 경로

kinetic sign을 뒤집으면 일부 증가 밀도를 흉내 낼 수 있지만 ghost를 도입한다.
이는 채택한 안정성 기준을 어기며 “같은 수식을 살리는 작은 수정”이 아니다.
따라서 본 run에서는 폐기한다.

## R4. 수정중력 경로

$F(\Theta)R$, higher derivative 또는 nonlocal memory term으로 0D fold를
geometry에 직접 읽힐 수 있다. 그러나 유효 Planck mass, tensor speed,
Ostrogradsky/degeneracy 조건과 국소중력 검사가 모두 새로 필요하다. 현재
자료로는 R0보다 주장 수가 많고 식별 가능성은 낮아 후속 독립 모형으로 둔다.

## 선택과 주장 상한

R0을 유일한 활성 경로로 채택한다. 그 이유는 기회비용과 상태흐름에서 이미
나온 지수 잔여량을 보존하면서, 가장 작은 공변 작용으로 background와
perturbative sign을 닫을 수 있기 때문이다. R1은 반례로 삭제하고, R2와 R4는
새 물리를 추가해야 하므로 이번 유도의 결과로 가장하지 않는다.

따라서 이번 연구가 도달할 수 있는 문장은 다음과 같다.

> 고정 self-measurement partition에서 남은 비선택 잔여량
> $u=e^{-\theta}$를 local scalar의 $e^{-\Theta}$에 대응시키는 retention 공리와
> 독립 scale $\rho_*$를 채택하면, 공변·보존·안정한 가속 quintessence 모형과
> 재현 가능한 배경/성장/BAO 산출을 구성할 수 있다.

이 문장은 microscopic creation, 암흑에너지 절대 크기, radiative naturalness,
암흑물질 clustering 또는 관측적 동일성의 증명이 아니다.
