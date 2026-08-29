# 23. 두 편광 물리 pole 게이트

두 편광을 셌다고 해서 전파가 자동으로 건강해지는 것은 아니다. 이 장은 Fierz--Pauli symbol을 이미 받은 뒤, 실제 TT(transverse-traceless) 두 채널의 propagator에 pole이 몇 개인지를 확인한다. determinant가 두 번 영이 되는 사실과 각 물리 채널에 double pole이 있다는 말은 다르다는 점이 핵심이다.

축 운동량 $q=(\omega,0,0,k)$와 $q^2=-\omega^2+k^2$를 쓰고, plus/cross tensor를 Frobenius norm으로 정규화한다. $A\ne0$를 전체 계수, $W$를 대칭 tensor 성분 weight, $K$를 Fierz--Pauli equation symbol이라 하면 action Hessian은 $H=A\,W\,K$다.

## 23.1 TT 제한에서 보이는 것

정규화한 plus/cross 부분공간에만 제한하면 Hessian은

$$
H_{\rm TT}=Aq^2I_2.
$$

따라서 두 대각 channel의 inverse는 각각 $1/(Aq^2)$다. light cone $q^2=0$에서 **각** helicity가 simple pole 하나를 갖는다. 반면 determinant는 $(Aq^2)^2$이므로 zero의 차수는 $2$다. 이것은 plus와 cross라는 서로 다른 두 channel의 곱이지, 하나의 channel에 $1/(q^2)^2$가 생겼다는 뜻이 아니다.

이 계산은 [19장](19_선형화_Einstein_두_편광_수용_정리.md)의 두 대표가 실제로 독립 전파 채널이 되는지 확인하는 다음 관문이다. 선언한 two-derivative ansatz 안에서는 추가 **물리 TT** pole이 없다.

## 23.2 두 미분을 풀면 즉시 달라진다

만일 kernel이 $Aq^2(1+\beta q^2)$이고 $\beta\ne0$라면, $q^2=0$ 외에

$$
q^2=-\frac1\beta
$$

에서도 root가 생긴다. 이 한 줄이 왜 21장의 두 미분 가정을 생략할 수 없는지 보여 준다. 고차 quadratic gravity에서의 추가 mode 논의는 [Stelle (1995)](https://arxiv.org/abs/hep-th/9509142)를 따른다. 이것은 CE에 추가 pole이 있다는 판정이 아니다.

## 23.3 이 gate가 아직 검사하지 않는 것

$A$의 부호와 positive residue, full gauge-fixed propagator, higher-derivative/nonlocal correction의 배제, 그리고 실제 microscopic refinement spectrum은 이 제한된 계산으로 결정되지 않는다. 따라서 ‘두 simple pole’은 supplied Fierz--Pauli의 TT gate를 통과했다는 뜻이지, 완성된 quantum-gravity unitarity 증명이 아니다. 무질량 spin-2 consistency의 조건부 맥락은 [Deser (2004)](https://arxiv.org/abs/gr-qc/0411023)와 [Rodina (2016)](https://arxiv.org/abs/1612.06342)를 따른다.

## 23.4 재현 범위

TT projection, channel별 simple pole, determinant 차수 대조, 4미분 반례는 [massless_spin2_physical_pole_gate.py](../../examples/physics/massless_spin2_physical_pole_gate.py)와 [대응 테스트](../../tests/test_massless_spin2_physical_pole_gate.py)에 있다. 원장의 focused 결과는 `20 passed`다.
