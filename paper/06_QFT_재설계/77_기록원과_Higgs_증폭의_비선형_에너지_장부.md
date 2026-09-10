# 77. 기록원과 Higgs 증폭의 비선형 에너지 장부

## 77.1 계산 전 등록: CE-UR4-D3

**[후보: 같은 작용의 자율적인 비선형 증폭]** 76장의 선형 Higgs
섭동을 실제 고전장 좌표로 되돌리고 기록원과 함께 진화시킨다.
배경에 외력·감쇠·열원이나 새 검출기 결합을 더하지 않는다.
고전적 증폭 seed를 양자 진공 fluctuation 또는 실제 확정 사건으로
동일시하지 않는다.

1. 원래 82차원 작용에서 $\Sigma_Y,\Sigma_3,X_{\alpha Y},
   X_{\alpha3},q$의 다섯 복소 좌표를 추출한다.
   $H_4=q/\sqrt2$, $\bar H_4=-q/\sqrt2$이고 다른 Higgs 성분은 0이다.
   두 Cartan 방향과 한 기록 질량 종류의 불변성, 정준 운동항,
   $D=0$ 및 gauge current 0을 전체 tensor와 독립 다항식으로 확인한다.
2. 기존 $V=g_5=\lambda_\Sigma=\lambda_R=\lambda_H=1$,
   $m_0=10^{-4}$를 유지한다. 공통 기준 $m_{\rm ref}=m_-$,
   $\xi_{\rm ref}=0.1$,
   $n_0=2\xi_{\rm ref}^2m_{\rm ref}^2/\kappa$로 고정한다.
   두 기록 종류 모두 $A_\alpha=\sqrt{n_0/(2m_\alpha)}$,
   $S(0)=A_\alpha e^{i\theta}$, $\dot S(0)=-im_\alpha S(0)$를 쓴다.
   이는 같은 선도 입자 수 밀도의 정규화이며 질량별 새 피팅이 아니다.
3. Higgs seed는 공통 $q(0)=\eta A_{\rm ref}$, $\dot q(0)=0$다.
   $\eta=0.01,0.1$ 및 $\theta=0,\pi/4$의 네 준비점을 두 질량에서
   모두 계산한다. seed 0의 불변 해도 대조한다.
   무거운 세 좌표의 초기값은 light 좌표를 고정한 potential의
   국소 정상점 중 원래 진공에 연결된 것으로 정하고, 초기 속도는
   그 해의 접공간에서 light 속도를 따른다. 이후에는 이 좌표들도
   완전한 동역학으로 진화하며 매 순간 다시 최소화하지 않는다.
4. 동일한 물리적 구간 $m_{\rm ref}t\in[0,60]$에서 전체
   $\rho=\sum_i(|\dot z_i|^2+|F_i|^2)$와
   $p=\sum_i(|\dot z_i|^2-|F_i|^2)$를 계산한다.
   $F_i$ 좌표별로 정의한 양의 부분 에너지의 반대칭 교환 current와
   전체 에너지 보존을 독립 검산한다. 이 분할을 유일한 상호작용
   에너지 배분이나 독립 입자 에너지라고 부르지 않는다.
5. 사전 지정 pointer 진단은 $|q|^2/A_{\rm ref}^2=1$의 첫 통과,
   마지막 구간 $m_{\rm ref}t\in[50,60]$에서 그 문턱의 유지·재통과다.
   peak, late-window 최소·최대와 Higgs 부분 에너지 비율을 모두 기록한다.
   원래 기록의 자유 운동식 잔차와 heavy record 성분도 추적한다.
   source phase나 seed를 골라 실패를 숨기지 않는다.
6. 기호·tensor·gauge·current 허용치는 정규화된 $10^{-8}$,
   에너지 보존과 독립 적분 비교는 $10^{-6}$를 우선 요구한다.
   필요한 수치 가속은 별도 실행 환경에서 사용하고 원래 모델 코드는
   바꾸지 않는다. 시간 간격 수렴과 다른 적분 방법으로 확인하며,
   오차가 이를 만족하지 않으면 정밀한 pointer 판정을 유보한다.

닫힌 균일 증폭기가 기록을 유지하지 못하면 그 적용 범위를 축소해
보존한다. 다음 준비·경계 분기는 유한 source와 밖으로 나가는 방사장을
같은 상태공간에 유지하는 D4로 둔다. 밖으로 나간 진폭·에너지를
삭제하거나 임의 감쇠항으로 대신하지 않는다는 조건을 먼저 고정한다.
이번 균일 계산으로 그 공간적 장치를 검증했다고 주장하지 않는다.

사전 등록을 해시로 고정한 뒤 수치 계산한다. 새 관측 입력·피팅은 없고
scientific_success=false, full_joint_rmse=null을 유지한다.

## 77.2 검산 결과

**[유도·독립 구현 검산]** 아래 다섯 장의 부분공간은 원래 고전
bosonic 작용 아래 닫힌다. 횡방향 양자 요동의 안정성이나 전체
양자 instrument까지 닫혔다는 뜻은 아니다. 사전 등록 SHA256은
`e9b621150bfc4b1a0d71e6054a038ecb71eb95ce0dedae8d10588e400ab72133`이다.

좌표를 $(z,t,s,u,q)=(\Sigma_Y,\Sigma_3,X_{\alpha Y},X_{\alpha3},q)$,
$a=1/(2\sqrt{15})$, $b=3a$, $d=1/2$로 둔다.
원래 진공에서 이동한 superpotential은 고정된 $V=1$ 단위에서

$$
W=-\frac{z^2}{2}-\frac{5t^2}{2}
  +\frac{m s^2}{2}+\frac{(m-4)u^2}{2}
  +\frac{a z^3}{3}+3azt^2+azs^2+3azu^2+6atsu
  -\frac{(bz+dt)q^2}{2}.
$$

여기서 $m$은 기존 $m_+$ 또는 $m_-$다. $V$를 복원하면 이차항의
계수는 질량 차원 1, 장은 차원 1, $W$는 차원 3이다. 세제곱 결합은
무차원이다. 음의 holomorphic 이차계수를 음의 물리적 질량제곱으로
바꾸어 읽지 않는다.

$$
\begin{aligned}
F_z&=-z+az^2+3at^2+as^2+3au^2-bq^2/2,\\
F_t&=-5t+6azt+6asu-dq^2/2,\\
F_s&=(m+2az)s+6atu,\\
F_u&=(m-4+6az)u+6ats,\\
F_q&=-(bz+dt)q,\\
\ddot z_i&=-\sum_j W_{ji}^*F_j.
\end{aligned}
$$

전체 82성분 좌표로 가는 정준 embedding은 실수이며
$\mathcal B^\dagger\mathcal B=I_5$다. 두 Cartan adjoint와 같은 크기의
$H_4=-\bar H_4$가 $D=0$을 만든다. 일반 복소 좌표와 속도에서도
전체 gauge current가 0이고, $F$와 그 gradient가 embedding 밖으로
나가지 않음을 직접 검사했다. $u,t$는 seed가 자라면 반응하므로
이를 0으로 고정한 두 singlet 계산을 사용하지 않았다.

검산은 원래 82성분 tensor의 진공 이동식, 위 다항식의 SymPy 미분,
별도 작성한 빠른 운동식, potential 유한차분을 대조한다. 정준 운동항,
불변성, gauge current, 에너지 교환과 초기 정상점용 실수 Hessian을
포함한 최대 정규화 오차는 $4.912\times10^{-13}$이다.

## 77.3 전체 보존량과 부분 에너지 교환

**[정의·유도]** 균일 고전장과 $A_\mu=0$에서

$$
E_i=|\dot z_i|^2+|F_i|^2,\qquad
\rho=\sum_iE_i,\qquad
p=\sum_i\bigl(|\dot z_i|^2-|F_i|^2\bigr).
$$

각 $E_i$는 양수지만 상호작용을 $F_i$ index에 배정한 분할이다.
독립 입자 에너지나 유일한 sector stress라고 부르지 않는다.
그 시간 변화는

$$
\dot E_i=\sum_j J_{ij},\qquad
J_{ij}=2\operatorname{Re}\!left(
F_i^*W_{ij}\dot z_j-F_j^*W_{ji}\dot z_i\right),\qquad
J_{ji}=-J_{ij}
$$

이므로 $\dot\rho=0$이다. $J$는 자연단위에서 차원 5,
$\rho,p$는 차원 4다. 압력은 따로 보존되는 수가 아니다.
$(z,t)$, $(s,u)$, $q$의 부분 에너지 사이에 교환이 일어나도
전체 에너지는 동일한 국소 작용의 시간 병진 대칭으로 보존된다.
이 계산은 공급된 평탄 배경의 stress 장부이며 Einstein 방정식이나
공통 중력 작용을 새로 유도한 것이 아니다.

초기 heavy 좌표는 light 좌표를 고정한 potential 정상점을 진공에서
연속 추적해 정한다. 실수 heavy/light Hessian을 $H_{HH},H_{HL}$로
쓰면 초기 속도는

$$
v_H=-H_{HH}^{-1}H_{HL}v_L
$$

이다. 이것은 초기 조건에만 사용한다. 적분 중에는 heavy 좌표를
다시 최소화하지 않고 위의 다섯 운동식을 그대로 푼다.
질량별 에너지 밀도는 같은 선도 수 밀도에서도 약 $m_\alpha n_0$로
다르다. 이를 맞추려고 seed나 결합을 재조정하지 않는다.

## 77.4 공통 준비 조건의 비선형 결과

장시간 적분과 독립 방법 비교를 진행 중이다. 사전 지정 여덟 준비점의
수치 수렴을 확인한 뒤 이 절에 결과와 적용 범위를 기록한다.
