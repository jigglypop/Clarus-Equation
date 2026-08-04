# Resonance 상관길이에서 웜홀 응력으로의 식별 루프

## 1. 검증 질문

CE 응용 문서에는

\[
\xi(Q)=Q\xi_0
\]

라는 공명 상관길이 ansatz가 있다. 웜홀 후보로 승격하려면 이것으로부터

\[
\langle T_{kk}(Q)\rangle_{\rm ren}<0,
\qquad
|T_{kk}(Q)|\propto Q^p
\]

의 부호와 exponent $p$를 유도해야 한다.

## 2. CE 문서에서 실제로 주어진 것

`docs/경로적분.md` 11.7절의 조건부 pole 형식은

\[
C(r)\sim Z_\phi\frac{e^{-r/\xi}}r,
\qquad
C(q)\sim\frac{Z_\phi}{q^2+m^2},
\qquad m=\hbar c/\xi
\]

이다. 그러나 같은 절은 isolated positive pole, residue, reflection
positivity, unitarity/LSZ를 CE+SM action에서 유도하지 않았다고 명시한다.
Q0 gate도 `spectral_density_derived=False`,
`stress_tensor_derived=False`다.

따라서 현재 입력은 $\xi$의 ansatz뿐이고 $Z_\phi(Q)$와 renormalized
coincident-limit prescription은 없다.

## 3. 비식별 반례

상관길이 scaling을 모두 동일하게 $\xi(Q)=Q\xi_0$로 고정하자. 응력의
질량차원만 흉내 낸 프록시

\[
\mathcal E_{\rm proxy}(Q)=\frac{Z(Q)}{\xi(Q)^d}
\]

를 사용하면, 임의의 exponent $p$에 대해

\[
Z(Q)=Q^{d+p}Z(1)
\]

을 택하여

\[
\frac{\mathcal E_{\rm proxy}(Q)}{\mathcal E_{\rm proxy}(1)}=Q^p
\]

를 만들 수 있다. 모든 반례는 같은 상관길이를 갖지만 응력 프록시 scaling은
서로 다르다. 이 프록시를 실제 $T_{kk}$라고 주장하는 것이 아니라,
$\xi(Q)$만으로 $T_{kk}(Q)$가 유일하게 정해지지 않는다는 반례다.

$d=4$, $Q=1.5038\times10^{14}$에서 실행한 세 family는 다음과 같다.

| 요청 $p$ | 필요한 residue gain | 프록시 gain |
|---:|---:|---:|
| 0 | $5.11\times10^{56}$ | 1 |
| 1 | $7.69\times10^{70}$ | $1.50\times10^{14}$ |
| 2 | $1.16\times10^{85}$ | $2.26\times10^{28}$ |

따라서 `상관길이 증가 => 음의 응력 증가`는 증명되지 않으며, residue와
spectral density 없이 $p$를 역산할 수 없다.

## 4. 현재 가능성 단계

공간접힘 전체 사슬의 단계는 다음처럼 구분한다.

| 단계 | 의미 | CE 현재 상태 |
|---|---|---|
| W0 | 수학적 경로/선택 논리 | 통과 |
| W1 | 주어진 웜홀 기하의 유한 shortcut | 통과 |
| W2 | 허용 가능한 renormalized 음의 $T_{\mu\nu}$ | 미도달 |
| W3 | self-consistent backreaction과 안정한 전역 해 | 미도달 |
| W4 | 입구 제작·분리·유지 | 미도달 |
| W5 | 실험·운용 가능한 장치 | 미도달 |

0D 선택기, 입구망 라우팅과 chronology interlock은 W0/W1의 제어 문제를
닫았다. 전체 물리 사슬은 가장 낮은 미통과 gate 때문에 **W1에서 정지**한다.
GJW/MMP 같은 외부 대조군은 특정 모형에서 W2/W3까지 가지만 CE를 W3로
승격하지 않는다.

## 5. 판정과 다음 루프

| 명제 | 판정 |
|---|---|
| $\xi(Q)$만으로 stress exponent $p$ 식별 | `REFUTED` |
| CE pole residue $Z_\phi(Q)$ 유도 | `OPEN` |
| CE spectral density 유도 | `OPEN` |
| CE renormalized $T_{kk}(Q)$ 부호·크기 유도 | `OPEN` |
| CE 공간접힘 전체 단계 | `W1 / KINEMATIC ONLY` |

다음 루프는 가정으로 $p$를 고르는 것이 아니라, Q0 작용을 실제 이차
연산자까지 닫아

\[
\Gamma^{(2)}_Q(q),\quad
\rho_Q(\mu^2),\quad
Z_Q,\quad
\frac{2}{\sqrt{-g}}\frac{\delta\Gamma_{\rm ren}}{\delta g^{\mu\nu}}
\]

을 순서대로 얻을 수 있는지 검사해야 한다. 현재 Q0 action bridge가
미완성이므로 첫 작업은 새 응력 계산이 아니라 누락된 action sector를
특정하는 것이다.

## 6. 실행

```powershell
uv run --extra dev python -m pytest tests/test_resonance_stress_identifiability.py -q
uv run python examples/physics/resonance_stress_identifiability_gate.py
```
