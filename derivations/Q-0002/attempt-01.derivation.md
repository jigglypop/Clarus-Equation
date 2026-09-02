---
question: Q-0002
attempt: 1
claim: "유한 이상 시계 둘+큐비트 제약 모형에서 (i) O^{(1)}_{1⊗O_S}=O^{(2)}_{1⊗O_S} on H_phys ⇔ [O_S,H_S]=0 이고 (ii) Φ_12=(√N R_2)(√N R_1)^{-1}는 V_1→V_2 유니터리이며 O^{(1)}_A=O^{(2)}_{Φ_12 A Φ_12^†} 가 H_phys 위 연산자 항등식이다"
assumptions:
  - "시계 C_i=C^N, H_i=diag(-M..M), N=2M+1 (유한·이산·정수 스펙트럼)"
  - "S=C^2, H_S=diag(ε0,ε1), ε_s∈Z (정수 간격)"
  - "제약은 H=H_1+H_2+H_S 하나뿐이며 H_phys=ker H ⊂ H_kin 에 유도 내적을 쓴다 (유한차원이라 Π는 정직한 직교사영, group averaging 불필요)"
  - "환원은 τ=0 조건화 R_iΨ=⟨τ=0|_iΨ 로 정의한다 (S3에 의해 다른 τ는 유니터리 차이)"
  - "(i)의 ⇒ 방향: 겹침 가정 — ε_s≠ε_{s'} 인 쌍마다 -k2-ε_s, -k2-ε_{s'} 가 모두 [-M,M]에 드는 k2가 존재 (N=5, ε=(0,1)에서는 k2=0)"
  - "Φ_12 의 정규화는 후보 문구 N R_2 R_1^{-1} 이 아니라 (√N R_2)(√N R_1)^{-1}=R_2 R_1^{-1} 이다 (문구 그대로는 인자 N이 남는다; S5.2)"
symbols:
  k2: integer
  e0: integer
  e1: integer
  tau: real
  ar: real
  ai: real
  br: real
  bi: real
  N: positive integer
verify:
  - type: identity
    lhs: "(-k2-e1)-(-k2-e0)"
    rhs: "e0-e1"
  - type: identity
    lhs: "sin((-k2-e0)*tau)+sin((k2+e0)*tau)"
    rhs: "0"
  - type: identity
    lhs: "cos((-k2-e0)*tau)-cos((k2+e0)*tau)"
    rhs: "0"
  - type: identity
    lhs: "((ar*br+ai*bi)+(br*ar+bi*ai))/N"
    rhs: "2*(ar*br+ai*bi)/N"
  - type: numeric
    expr: "N*(1/sqrt(N))**2 - 1"
    tol: 1e-9
---

# Q-0002 attempt-01 — 후보 K1: 관측자별 관계적 관측량의 불일치 조건과 frame-change Φ_12

기계 검사: 프론트매터 verify 블록(스칼라 항등식 5개, sympy 없음 → symbolic skipped)과
`verify/Q-0002/attempt-01/check_k1.py`(행렬 수준, numpy, 씨앗 20260902, TOL=1e-10, SEP=1e-3).
행렬 수준 주장 (S4)(S5)의 실질 증거는 후자다.

## 기호와 설정 (S0)

$$ \mathcal C_i=\mathbb C^N,\quad H_i|k\rangle=k|k\rangle,\ k\in\{-M,\dots,M\},\ N=2M+1 $$  (S0.1) 시계 정의
$$ |\tau\rangle=N^{-1/2}\sum_k e^{-ik\tau}|k\rangle,\quad \tau_n=2\pi n/N,\quad \langle\tau_n|\tau_m\rangle=\delta_{nm} $$  (S0.2) 공변 시계 상태, 이산 푸리에 직교기저
$$ \langle k|\tau=0\rangle=N^{-1/2}\ \ \forall k $$  (S0.3) (S0.2)에 τ=0 대입
$$ H=H_1\otimes1\otimes1+1\otimes H_2\otimes1+1\otimes1\otimes H_S,\quad H_S=\mathrm{diag}(\epsilon_0,\epsilon_1) $$  (S0.4) 제약 연산자, 곱기저에서 대각
$$ \Pi=\sum_{k_1+k_2+\epsilon_s=0}|k_1k_2s\rangle\langle k_1k_2s| $$  (S0.5) ker H 위 직교사영

## (S1) H_phys 의 기저

$$ \mathcal H_{\rm phys}=\ker H=\mathrm{span}\{|k_1,k_2,s\rangle:\ k_1+k_2+\epsilon_s=0\} $$  (S1.1) H가 곱기저에서 대각이므로 핵은 고윳값 0 기저벡터의 span
$$ I_1:=\{(k_2,s):\ -k_2-\epsilon_s\in[-M,M]\},\qquad \mathcal B=\{|{-k_2-\epsilon_s},\,k_2,\,s\rangle:(k_2,s)\in I_1\} $$  (S1.2) 기저를 (k2,s)로 라벨링, k1은 제약으로 결정
$$ I_2:=\{(k_1,s):\ -k_1-\epsilon_s\in[-M,M]\},\qquad \mathcal B=\{|k_1,\,{-k_1-\epsilon_s},\,s\rangle:(k_1,s)\in I_2\} $$  (S1.3) 같은 기저를 (k1,s)로 재라벨링
$$ \dim\mathcal H_{\rm phys}=\sum_s\#\{k_2: -k_2-\epsilon_s\in[-M,M]\}=\sum_s (N-|\epsilon_s|)\ \ (|\epsilon_s|\le N) $$  (S1.4) 라벨 개수 세기
$$ N=5,\ \epsilon=(0,1):\quad \dim\mathcal H_{\rm phys}=5+4=9 $$  (S1.5) (S1.4)에 대입
$$ \langle\Psi|\Psi'\rangle_{\rm phys}:=\langle\Psi|\Psi'\rangle_{\rm kin} $$  (S1.6) 유한차원이므로 유도 내적을 물리 내적으로 채택 (가정)

## (S2) R_i 와 O^{(i)}_A 의 행렬 표현

$$ R_1\Psi:=\langle\tau{=}0|_1\Psi=N^{-1/2}\sum_{k_1}\langle k_1|_1\Psi\ \in\ \mathcal C_2\otimes\mathcal S $$  (S2.1) 환원 정의, (S0.3) 사용
$$ R_1|k_1,k_2,s\rangle=N^{-1/2}|k_2,s\rangle,\qquad R_2|k_1,k_2,s\rangle=N^{-1/2}|k_1,s\rangle $$  (S2.2) 기저 작용
$$ V_1:=\sqrt N\,R_1\big|_{\mathcal H_{\rm phys}}:\ |{-k_2-\epsilon_s},k_2,s\rangle\mapsto|k_2,s\rangle,\qquad \mathcal V_1:=\mathrm{ran}\,V_1=\mathrm{span}\{|k_2,s\rangle:(k_2,s)\in I_1\} $$  (S2.3) (S1.2)의 라벨 일대일 → V_1 은 H_phys→𝒱_1 등거리 동형
$$ V_2:=\sqrt N\,R_2\big|_{\mathcal H_{\rm phys}}:\ |k_1,{-k_1-\epsilon_s},s\rangle\mapsto|k_1,s\rangle,\qquad \mathcal V_2:=\mathrm{span}\{|k_1,s\rangle:(k_1,s)\in I_2\} $$  (S2.4) (S1.3)으로 같은 논증
$$ V_i^\dagger V_i=1_{\mathcal H_{\rm phys}},\qquad V_iV_i^\dagger=P_{\mathcal V_i} $$  (S2.5) 정규직교기저 사이 전단사의 등거리성
$$ O^{(1)}_A:=N\,\Pi\,(|0\rangle\langle0|_1\otimes A)\,\Pi,\quad A\in\mathrm{End}(\mathcal C_2\otimes\mathcal S) $$  (S2.6) 관계적 관측량 정의
$$ \langle k_1k_2s|O^{(1)}_A|k_1'k_2's'\rangle=N\,\langle k_1|0\rangle\langle0|k_1'\rangle\,\langle k_2s|A|k_2's'\rangle=\langle k_2s|A|k_2's'\rangle\quad(\text{두 상태 모두 물리적}) $$  (S2.7) (S0.3)으로 N·N^{-1/2}·N^{-1/2}=1
$$ O^{(1)}_A\big|_{\mathcal H_{\rm phys}}=V_1^\dagger A V_1,\qquad O^{(2)}_A\big|_{\mathcal H_{\rm phys}}=V_2^\dagger A V_2 $$  (S2.8) (S2.7)을 (S2.3)(S2.4)로 다시 씀
$$ \omega_\Psi(O^{(i)}_A)=\langle\Psi|V_i^\dagger AV_i|\Psi\rangle=\langle\sqrt N R_i\Psi|A|\sqrt N R_i\Psi\rangle $$  (S2.9) (S2.8)의 기댓값

## (S3) 조건화 상태의 시간 발전

$$ \langle\tau|_1=N^{-1/2}\sum_k e^{ik\tau}\langle k|_1=\langle0|_1\,e^{iH_1\tau} $$  (S3.1) (S0.2)의 켤레, H_1 대각
$$ \Psi\in\mathcal H_{\rm phys}\ \Rightarrow\ H_1\Psi=-(H_2+H_S)\Psi $$  (S3.2) HΨ=0 정리
$$ e^{iH_1\tau}\Psi=e^{-i(H_2+H_S)\tau}\Psi $$  (S3.3) 곱기저 성분별로 e^{ik_1\tau}=e^{-i(k_2+\epsilon_s)\tau} (verify 2,3번 항등식)
$$ \langle\tau|_1\Psi=\langle0|_1e^{-i(H_2+H_S)\tau}\Psi=e^{-i(H_2+H_S)\tau}\langle0|_1\Psi $$  (S3.4) e^{-i(H_2+H_S)τ}는 C_2⊗S 에만 작용하므로 부분 bra와 교환
$$ R_1(\tau)\Psi=e^{-i(H_2+H_S)\tau}\,R_1\Psi,\qquad \tau\in\{2\pi n/N\} $$  (S3.5) 결론: 조건화 상태는 H_2+H_S 로 슈뢰딩거 발전 (유한 Page–Wootters). τ=0 선택은 유니터리 차이뿐

## (S4) 주장 (i): 일치 ⇔ 보존 전하

$$ [O_S,H_S]_{ss'}=(O_S)_{ss'}(\epsilon_{s'}-\epsilon_s) $$  (S4.1) H_S 대각
$$ \langle k_1k_2s|O^{(1)}_{1\otimes O_S}|k_1'k_2's'\rangle=\delta_{k_2k_2'}(O_S)_{ss'},\qquad \langle k_1k_2s|O^{(2)}_{1\otimes O_S}|k_1'k_2's'\rangle=\delta_{k_1k_1'}(O_S)_{ss'} $$  (S4.2) (S2.7)에 A=1⊗O_S 대입, 물리 기저쌍에서
$$ (\Leftarrow)\ [O_S,H_S]=0,\ (O_S)_{ss'}\neq0\ \Rightarrow\ \epsilon_s=\epsilon_{s'}\ \Rightarrow\ (k_2=k_2'\Leftrightarrow k_1=-k_2-\epsilon_s=-k_2'-\epsilon_{s'}=k_1') $$  (S4.3) (S4.1)과 제약 k_i 결정
$$ (\Leftarrow)\ \delta_{k_2k_2'}(O_S)_{ss'}=\delta_{k_1k_1'}(O_S)_{ss'}\ \ \forall\ \text{물리 기저쌍}\ \Rightarrow\ O^{(1)}_{1\otimes O_S}=O^{(2)}_{1\otimes O_S}\ \text{on}\ \mathcal H_{\rm phys} $$  (S4.4) (S4.3)을 (S4.2)에 적용
$$ (\Rightarrow)\ \text{가정: }(O_S)_{ss'}\neq0,\ \epsilon_s\neq\epsilon_{s'};\ \text{겹침 가정으로 }k_2\text{ 택해 }k_1:=-k_2-\epsilon_s,\ k_1':=-k_2-\epsilon_{s'}\in[-M,M] $$  (S4.5) 반례 상태쌍 선택
$$ \langle k_1k_2s|O^{(1)}_{1\otimes O_S}|k_1'k_2s'\rangle=(O_S)_{ss'}\neq0,\qquad \langle k_1k_2s|O^{(2)}_{1\otimes O_S}|k_1'k_2s'\rangle=\delta_{k_1k_1'}(O_S)_{ss'}=0\ (k_1-k_1'=\epsilon_{s'}-\epsilon_s\neq0) $$  (S4.6) (S4.2)에 대입 → 두 연산자가 다름, (⇒) 증명 끝
$$ \Psi=a|k_1,k_2,0\rangle+b|k_1',k_2,1\rangle,\quad k_1'=k_1+\epsilon_0-\epsilon_1,\quad |a|^2+|b|^2=1 $$  (S4.7) 명시 반례 상태 (verify 1번 항등식: k_1'-k_1=ε_0-ε_1)
$$ \sqrt N R_1\Psi=a|k_2,0\rangle+b|k_2,1\rangle,\qquad \sqrt N R_2\Psi=a|k_1,0\rangle+b|k_1',1\rangle $$  (S4.8) (S2.2)
$$ \omega_\Psi(O^{(1)}_{1\otimes\sigma_x})=\bar a b+\bar b a=2\,\mathrm{Re}(\bar a b),\qquad \langle R_1\Psi|\sigma_x|R_1\Psi\rangle=2\,\mathrm{Re}(\bar a b)/N $$  (S4.9) (S2.9); 조건화 상태(√N 없이)의 값이 후보 문구의 2Re(a*b)/N 이고 관계적 관측량 값은 그 N배 (verify 4번 항등식)
$$ \omega_\Psi(O^{(2)}_{1\otimes\sigma_x})=\bar a b\,\langle k_1|k_1'\rangle+\bar b a\,\langle k_1'|k_1\rangle=0\quad(\epsilon_0\neq\epsilon_1\Rightarrow k_1\neq k_1') $$  (S4.10) (S2.9), 시계 1 에너지 라벨이 달라 간섭항 소거
$$ \omega_\Psi(O^{(1)}_{1\otimes\sigma_z})=|a|^2-|b|^2=\omega_\Psi(O^{(2)}_{1\otimes\sigma_z}) $$  (S4.11) 보존 전하 σ_z 는 일치 (S4.4의 예)
$$ \epsilon_0=\epsilon_1\ \Rightarrow\ k_1'=k_1\ \Rightarrow\ \omega_\Psi(O^{(2)}_{1\otimes\sigma_x})=2\,\mathrm{Re}(\bar ab)=\omega_\Psi(O^{(1)}_{1\otimes\sigma_x}) $$  (S4.12) 음성대조: 축퇴 H_S 에서는 σ_x 도 일치

## (S5) Φ_12 의 유니터리성과 주장 (ii)

$$ \Phi_{12}:=V_2V_1^\dagger=(\sqrt N R_2)(\sqrt N R_1)^{-1}:\ \mathcal V_1\to\mathcal V_2,\qquad \Phi_{12}|k_2,s\rangle=|{-k_2-\epsilon_s},s\rangle\ ((k_2,s)\in I_1),\quad \Phi_{12}|_{\mathcal V_1^\perp}=0 $$  (S5.1) 정의: (S2.3)(S2.4)의 합성, 에너지기저 작용
$$ N R_2R_1^{-1}=N\cdot N^{-1/2}(\sqrt N R_2)\cdot N^{1/2}(\sqrt N R_1)^{-1}=N\,\Phi_{12} $$  (S5.2) 후보 문구의 정규화 오류 확인: 올바른 식은 Φ_12=R_2R_1^{-1}
$$ \Phi_{12}^\dagger\Phi_{12}=V_1V_2^\dagger V_2V_1^\dagger=V_1V_1^\dagger=P_{\mathcal V_1},\qquad \Phi_{12}\Phi_{12}^\dagger=V_2V_2^\dagger=P_{\mathcal V_2} $$  (S5.3) (S2.5) → 부분등거리, 𝒱_1→𝒱_2 유니터리
$$ \sqrt N R_2\Psi=V_2\Psi=V_2V_1^\dagger V_1\Psi=\Phi_{12}\sqrt N R_1\Psi\quad\forall\Psi\in\mathcal H_{\rm phys} $$  (S5.4) (S2.5) V_1^†V_1=1 삽입: Φ_12 는 조건화 상태를 보존(옮김)
$$ O^{(2)}_{\Phi_{12}A\Phi_{12}^\dagger}\big|_{\mathcal H_{\rm phys}}=V_2^\dagger\Phi_{12}A\Phi_{12}^\dagger V_2=V_2^\dagger V_2V_1^\dagger AV_1V_2^\dagger V_2=V_1^\dagger AV_1=O^{(1)}_A\big|_{\mathcal H_{\rm phys}} $$  (S5.5) (S2.8)(S5.1)(S2.5): 연산자 항등식 (기댓값보다 강함)
$$ \omega_\Psi(O^{(1)}_A)=\omega_\Psi(O^{(2)}_{\Phi_{12}A\Phi_{12}^\dagger})\quad\forall\Psi\in\mathcal H_{\rm phys},\ \forall A $$  (S5.6) (S5.5)의 기댓값 — 주장 (ii)
$$ \Phi_{12}^\dagger(1\otimes O_S)\Phi_{12}|k_2,s\rangle=\sum_{s'}(O_S)_{s's}\,|k_2+\epsilon_s-\epsilon_{s'},\,s'\rangle $$  (S5.7) (S5.1) 두 번 적용: frame change 는 S의 비보존 성분을 시계 에너지 이동 ε_s-ε_{s'} 와 짝지음
$$ \Phi_{12}^\dagger(1\otimes O_S)\Phi_{12}=P_{\mathcal V_1}(1\otimes O_S)P_{\mathcal V_1}\ \Leftrightarrow\ (O_S)_{s's}(\epsilon_s-\epsilon_{s'})=0\ \forall s,s'\ \Leftrightarrow\ [O_S,H_S]=0 $$  (S5.8) (S5.7)에서 이동이 0 ⇔ 교환: (i)가 (ii)의 따름정리로 재유도됨

## (S6) 가정 목록과 증명하지 않은 것

$$ \text{증명한 것: 유한 정수 스펙트럼 모형에서 (S4.4)(S4.6)(S5.3)(S5.5)(S5.6)(S5.8)} $$  (S6.1) 범위 확정
$$ \text{증명하지 않은 것 1: 연속 이상 시계 } (H_i=\hat p,\ \tau\in\mathbb R)\text{ — } \Pi\text{가 사영이 아니고 group averaging·rigged 내적이 필요} $$  (S6.2) 비범위
$$ \text{증명하지 않은 것 2: 36장 M3(양성·Hadamard·foliation 독립)의 통과 — 본 모형은 0+0차원 유한계} $$  (S6.3) 비범위
$$ \text{증명하지 않은 것 3: 중력·제약 대수(hypersurface deformation) — 제약이 아벨 하나뿐} $$  (S6.4) 비범위
$$ \text{증명하지 않은 것 4: Q-0002 (v) 비선택 에너지 장부 } E_{\rm total}=E_{\rm seen}+E_{\rm unseen}\text{ — (S5.7)의 에너지 이동이 그 출발점이나 본 attempt 범위 밖} $$  (S6.5) 비범위, 주차
$$ \text{증명하지 않은 것 5: (i)의 ⇒ 방향은 겹침 가정(S4.5) 없이는 거짓일 수 있음 } (|\epsilon_s-\epsilon_{s'}|>2M\text{이면 두 관측량 모두 해당 블록에서 0)} $$  (S6.6) 가정 의존성 명시
