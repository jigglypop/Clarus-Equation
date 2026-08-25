# SMQ final report — 유효한 control, 실패한 origin bridge

Status: COMPLETE

## 결론

채택한 exponential action은 안정한 conditional quintessence EFT로 구현되고
focused 검증을 통과했다. 그러나 이 run의 더 강한 목표였던 “operational
측정깊이의 절대 원점이 암흑에너지 background에서 읽힌다”는 주장은 실패했다.

완전 반례는 정확한 shift--amplitude redundancy다.

$$
V(\Theta)=\rho_*e^{-\Theta},
$$

$$
(\Theta,\rho_*)\longmapsto(\Theta+\Delta,\rho_*e^\Delta).
$$

이 변환은 action, stress tensor, Klein--Gordon 식, background expansion,
smooth growth와 BAO shape를 모두 보존한다. 자유 $\rho_*$ shooting이 field
origin을 흡수하므로, 수치 적합은 standard exponential quintessence control의
검증이지 quantum/0D origin의 검증이 아니다.

literal 누적 기회비용

$$
V_L=\rho_*(1-e^{-\Theta})
$$

도 source-free rest data에서 $\ddot\Theta<0$을 주므로, 스스로 증가하는
positive-canonical cost field라는 경로는 폐기한다.

## 보존된 결과와 후속 경로

보존되는 좁은 결과는 canonical stress conservation, $c_s^2=1$, fixed-point
조건과 background control이다. 폐기되는 부모 주장은 absolute measurement
origin과 quantum-path derivation이다.

후속 run은 이 실패를 숨기지 않고
`../self-nonidentity-kinetic-dark-sector-20260825`에서 식을 바꾸었다. 새
경로는 물리적 zero-clock matching surface, tied
$1-e^{-\Gamma T}$ readout, 그리고 양의 initial canonical current
$\Pi_{\rm fold}$를 사용한다. 따라서 본 run은 그 후속식의 negative control로
동결한다.

