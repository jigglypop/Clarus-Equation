# CE-CR1: 공간 기록의 누적, 유한 준비 에너지와 조건부 결합 하한

2026-09-19. 기준 main: `ac06cc8f1a8508d36d550b2ed8709c02c84d397c` (CE-RS1 게시).

## 0. 이번에 전진한 것

기존 세 복소 매개장의 공간 운동항을 유지한 평탄 3+1차원 Gaussian 가지에서, 서로 다른 힉스 크기를 구별하는 상태 겹침의 부피밀도를 계산했다. 조건부 진공의 구별도는 공간 모드에 누적된다. 그 UV 적분은 유한하지만, 같은 질량을 순간적으로 바꿀 때의 실제 여기 에너지는 로그 발산한다. 일반적인 매끄러운 준비의 충분조건을 증명하고, 정확히 풀리는 tanh 이력과 독립 ODE로 검산했다.

지정한 부피와 이상적 판별오류 한계에서는 비영 kappa의 하한이 유일하다. 이는 자원 제약 아래의 조건부 선택이지 자연상수의 예측이 아니다. 특히 epsilon=0이어도 이 포털은 정보를 남긴다. 따라서 기록 요구만으로 CE 고유 분할의 비영 값을 선택한다는 주장은 성립하지 않는다.

27개 검사를 실행했다. 관측 피팅, 실제 힉스 질량 예측, 단일 결과 발생, 비가역한 국소 중복 기록, 자율적인 전체 힉스–환경 진화의 완결을 주장하지 않는다.

## 1. 출발식과 새 준비조건

원전은 CE-RS1의 source/CE_input.txt 및 기존 88–89장의 포털이다.

$$X(u)=(s_0+\kappa u)I+Y,\qquad u=H^\dagger H,$$
$$x_j(u)=s_0+\kappa u+(-2\epsilon,\epsilon,\epsilon)_j.$$

공통 epsilon, theta=pi, s0>2epsilon>0, kappa>=0와 표준 양의 공간 운동항을 유지한다. 비교할 두 원천 ua,ub는 외부로 지정하고, 질량제곱 x_aj,x_bj는 양수다. hbar=c=1이다. k는 공간 운동량, 부피 V의 차원은 mass^-3이다. 이 공간차원, 절대척도와 준비는 유도된 값이 아니다.

이 연구의 정적 비교는 조건부 진공 또는 단열 준비의 극한이다. CE-RS1의 uref에서 순간적으로 분기한 유한 조화모드 이력과 같은 준비라고 하지 않는다. 에너지 검사는 ua에서 ub로 이어지는 지정한 이력의 진단이다. 실제 힉스 운동방정식에서 그 이력을 생성한 것은 아니다.

## 2. 같은 공간 장에서 얻은 상태 겹침

주기적 유한 상자의 공통 운동량 기저에서 omega_aj(k)=sqrt(k²+x_aj), omega_bj(k)=sqrt(k²+x_bj)다. 실수 조화모드 두 진공의 겹침은 [2sqrt(omega_a omega_b)/(omega_a+omega_b)]^(1/2)다. 복소모드는 두 실수 성분이므로 한 운동량·한 복소모드의 겹침은 이 값의 제곱이다. 실제 실수장의 k와 -k 중복은 전체 실수 고유모드 수와 동일하게 세며 복소 성분을 한 번만 두 배로 센다.

따라서 유한 상자의 정확한 곱은

$$-\ln|\nu_V|=\sum_{j,\mathbf k}\ln\frac{\omega_{aj}+\omega_{bj}}{2\sqrt{\omega_{aj}\omega_{bj}}}.$$

큰 부피에서

$$\boxed{-\frac1V\ln|\nu_V|\longrightarrow\gamma_{ab}
=\frac1{2\pi^2}\sum_j\int_0^\infty k^2dk\,
\ln\cosh\left[\frac14\ln\frac{k^2+x_{aj}}{k^2+x_{bj}}\right].}$$

유한 V에서 연속 적분을 곱의 정확한 대체라고 하지 않는다. 상태 준비가 끝난 뒤 각 조건부 Hamiltonian이 일정하고 상태가 그 진공이면 겹침의 절댓값은 시간에 따라 변하지 않는다. 반대로 공통 조건으로 단열적으로 되돌리면 구별 정보를 지울 수 있다. 이것은 불가역성의 증명이 아니다. 또한 전체 환경을 판별할 수 있다는 것과 여러 국소 관찰자에게 독립 기록이 생겼다는 것은 다르다.

## 3. 수렴·작은 대비와 같은 스펙트럼의 정보 기하

적분함수 안의 log cosh는 k->infinity에서 (x_b-x_a)²/(32 k⁴)+O(k^-6)다. 양의 질량 때문에 IR도 유한하다. log cosh(r)<=r²/2와 log 차이 상계에서, cutoff Lambda 이후의 엄밀한 보수적 꼬리 상계는

$$0\le\gamma-\gamma_\Lambda\le\sum_j\frac{(x_{bj}-x_{aj})^2}{64\pi^2\Lambda}.$$

평균 x_j를 고정한 대칭 대비 delta_s=kappa(ub-ua)에서

$$\boxed{\gamma=\frac{\delta_s^2}{256\pi}\sum_j\frac1{\sqrt{x_j}}+O(\delta_s^4).}$$

이는 integral d³k/(2pi)³ (k²+m²)^(-2)=1/(8pi m)를 사용한다. 대응하는 순수 진공 계열의 quantum Fisher information 밀도는 kappa² sum(1/m_j)/(32pi)다. 기존 유효작용의 리만 운동계량과 이 상태 구별 계량을 동일한 수치라고 부르지 않는다. 둘은 같은 스펙트럼으로 계산하지만 서로 다른 물리적 미분·주파수 가중을 갖는다.

## 4. 비영 결합의 조건부 하한과 남는 비유일성

동일한 사전확률의 두 순수 조건부 상태를 이상적으로 전체 판별할 때 최소 오류는

$$p_{err}^{opt}=\frac{1-\sqrt{1-|\nu_V|^2}}2.$$

부피의 선도 연속 근사에서는 오류 p 이하를 만족할 조건이

$$\boxed{V\gamma\ge-\tfrac12\ln[4p(1-p)]}$$

이다. 실제 검출기가 이 최적 판별을 구현한다는 가정은 하지 않는다. 이 한계는 실제 장치의 충분조건이 아니라 주어진 최적 전역 판별 모형의 기준이다.

ub>ua>=0와 A=k²+s0+(-2epsilon,epsilon,epsilon)_j>0에서

$$\partial_\kappa\ln\frac{A+\kappa ub}{A+\kappa ua}
=\frac{(ub-ua)A}{(A+\kappa ub)(A+\kappa ua)}>0.$$

따라서 gamma(kappa)는 연속·엄격 증가하며 gamma(0)=0이다. 이 문서의 ua,ub>0에서는 큰 kappa에서 k를 sqrt(kappa)로 재척도화하면 gamma가 kappa^(3/2)에 비례해 증가한다. 유한한 V와 0<p<1/2를 지정하면 kappa_min이 정확히 하나 존재한다. 에너지가 kappa에 단조 증가하는 기존 후보에서도, 이 판별 제약을 추가한 허용집합의 최소는 0이 아니라 하한 kappa_min이다.

그러나 V와 p를 바꾸면 그 값도 바뀐다. 기록해야 한다는 제약 자체도 추가 선택조건이다. 여기에서 우주의 kappa를 예측했다고 하지 않는다. 더 강한 음성대조로 epsilon=0에서도 gamma>0다. 기록 형성만으로 양의 CE 분할이 필수라는 명제는 반례를 갖는다.

## 5. 순간 전환은 유한 겹침과 달리 무한 여기 에너지를 요구한다

한 실수모드의 갑작스러운 질량 전환에서

$$n_k=\frac{(\omega_b-\omega_a)^2}{4\omega_a\omega_b}.$$

세 복소장의 최종 진공보다 높은 여기 에너지밀도는

$$\Delta e_{exc}=\frac1{\pi^2}\sum_j\int_0^\Lambda k^2dk\,\omega_{bj}n_{jk}.$$

큰 운동량에서

$$\boxed{\Delta e_{exc}(\Lambda)=\frac{\sum_j(x_{bj}-x_{aj})^2}{16\pi^2}\ln\Lambda+O(1).}$$

겹침 적분이 유한하다는 사실만으로 그 상태를 순간적으로 유한한 일로 준비할 수는 없다. 이 발산은 최종 진공에 대한 실제 여기 에너지다. 절대 진공에너지 정합과 혼동하지 않는다. 원래 유한 조절자의 순간 전환 계산은 여전히 유한한 조절자 문제로 남으며, 그 계산을 무한 cutoff로 그대로 승격하는 처방이 실패한다.

## 6. 일반적인 매끄러운 준비의 충분조건

양의 x(t)가 두 양의 상수로 접근하고, dot x가 끝점에서 0이며 ||dot x||_1, ||ddot x||_1, ||dot x||_2가 유한하다고 하자. omega=sqrt(k²+x), Theta=integral omega dt, g=dot omega/(2omega)=dot x/(4omega²)로 두면 정확한 Bogoliubov 계수는

$$\dot\alpha=g e^{2i\Theta}\beta,\qquad \dot\beta=g e^{-2i\Theta}\alpha.$$

초기 alpha=1,beta=0에서 |alpha|+|beta|<=exp(integral |g|)이다. beta의 적분에서 exp(-2iTheta)를 한 번 부분적분하면, k>0에 대해

$$\boxed{|\beta_k|\le e^{\|\dot x\|_1/(4k^2)}
\left[\frac{\|\ddot x\|_1}{8k^3}+\frac{7\|\dot x\|_2^2}{32k^5}\right].}$$

첫 항은 d[dot x/(8omega³)]/dt, 두 번째는 omega의 미분과 dot alpha를 함께 상계한 결과다. 끝점항은 0이다. 따라서 beta=O(k^-3)이고 3차원 여기 에너지의 UV 적분 k³|beta|²는 수렴한다. 양의 최소 질량 간극은 IR 유한성을 보장한다. 이 충분조건은 tanh 모양 하나에 의존하지 않는다.

## 7. 정확한 tanh 진단과 일의 장부

수치 대조에는

$$x(t)=\tfrac12(x_a+x_b)+\tfrac12(x_b-x_a)\tanh(t/\tau),\qquad\tau>0$$

를 사용한다. 해석적인 점유수는

$$n_k=\frac{\sinh^2[\pi\tau(\omega_b-\omega_a)/2]}{\sinh(\pi\tau\omega_a)\sinh(\pi\tau\omega_b)}.$$

이는 알려진 자유장 quench 해다. 여기의 새 계산은 같은 CE 질량쌍과 정해진 준비 시간에 이를 적용하고 독립적으로 검산한 것이다. t=-16tau에서 진공 모드함수를 준비하고 ddot f+(k²+x(t))f=0을 DOP853 및 RK45로 적분한다. 점유수는 (omega_b f-i dot f)/sqrt(2omega_b)의 절댓값 제곱으로 안정하게 추출한다.

한 실수모드의 E=(|dot f|²+omega²|f|²)/2는 dot E=dot x |f|²/2를 만족한다. 복소모드는 두 배다. 힉스 원천에 대응하는 work를 적분해 에너지 변화와 대조했다. 외부에서 처방한 이력의 일이 포함되므로 진공에서 공짜 에너지를 만든 것이 아니다. 전체 힉스 되먹임을 자율적으로 풀었다는 주장도 하지 않는다. 공간 연속계의 절대 진공 일에는 별도 재규격화가 필요하며, 여기서 유한하게 계산한 연속량은 최종 진공보다 높은 여기 에너지다.

동일 tanh 이력의 가장 느린 k=0 모드에 대해 단열 진단값 max |dot omega|/omega²도 정확히 구한다. c=(xa+xb)/2, d=(xb-xa)/2, y*=-3d/[2c+sqrt(4c²-3d²)]이면

$$\eta_{max}=\frac{|x_b-x_a|(1-y_*^2)}{4\tau(c+dy_*)^{3/2}}.$$

이는 허용 여기량의 정확한 오차 상계가 아니라 단열성의 진단 기준이다. 해당 기준을 작게 유지할 조건은 간극이 0으로 접근할 때 필요한 시간이 커짐을 보여준다. 유한한 tau를 지정하지 않고 자연의 분할 상수를 정하지는 못한다.

## 8. 실제 계산

s0=.5, epsilon=.15, kappa=1, ua=.2, ub=.8에서 xa=(.4,.85,.85), xb=(1,1.45,1.45)다.

- gamma=0.0013825453194253562, quadrature 내부 추정오차 약 1.7e-17.
- bulk 최적 판별오류 10%, 1%, 0.01%에 필요한 부피 추정은 각각 369.4820, 1167.7470, 2829.6165. 모두 내부 mass^-3 단위이며 실제 장치 크기나 관측 정확도가 아니다.
- cutoff 8에서 연속 밀도 0.00117101448245. 주기적 상자 L=8,16,32의 직접 합은 0.00117814760352, 0.00117122177401, 0.00117099698898로 접근한다.
- 순간 전환 cutoff 10,30,100,300의 여기 에너지밀도는 .01421677,.02168962,.02991914,.03743234로 계속 증가한다.
- 부드러운 tau=.1,.25,.5,1,2의 여기 에너지는 .00872550,.00346076,.000911326,.0000707585,.000000565108이다.
- 독립 ODE 점유수는 해석식과 약 8.9e-14 이내, 검사한 모드의 최대 work 장부 잔차 약 2.5e-11 이내다. 유한 시간 끝점과 수치 적분 오차이지 관측 오차가 아니다.

조건부 자원 예: V=1000, 최적 오류=.01를 외부로 지정하면 epsilon=.15에서 kappa_min=1.0937639337354188. 별도 절단면 kappa=1,tau=3,eta_max<=.1에서는 epsilon<=0.15878607369460138이다. 두 값은 같은 최적점으로 혼합하지 않는다. 무차원 진단 기준을 맞추기 위한 수치근은 물리상수 피팅이 아니라 명시한 부등식 경계 계산이다.

kappa=.25,.5,1,2가 각각 다른 부피에서 같은 판별기준을 만족하므로 비유일성을 남긴다. epsilon=0에서도 gamma=0.0013525966007023969다. 따라서 정보 기록을 요구하는 것만으로 CE 관계 분할의 미시적 기원을 해결하지 않았다.

## 9. 판정과 재현

새로운 것은 정적 진공 구별도의 공간 누적, 유한 UV 꼬리 상계, 일반 매끄러운 전환의 유한 여기 에너지 충분조건, 같은 질량의 정확한 일 장부, 자원에 조건부인 양의 결합 하한이다. free kappa/epsilon 최소화가 자연상수를 선택하지 못한다는 CE-RS1 판정은 번복하지 않는다. 필요한 것은 원래 관계 작용이 자원·준비·분할 보호 조건을 선택하는 추가 기원이지 관측 숫자에 맞춘 새 퍼텐셜이 아니다.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_continuum.py --output results.json
```

독립 Gaussian 적분, finite-box 합, 적분 꼬리, 작은 대비, 질량 재척도화, Bogoliubov ODE, 일/에너지 장부, 반례와 자원 경계까지 27개 검사를 실행했다. 원래 RS1은 이전 게시에서 37개를 재실행했으며 이번 수치와 합쳐 관측 성공률로 계산하지 않는다.

## 출처

프로젝트: ../ce_record_selection_20260919/REPORT_ko.md 및 source/CE_input.txt, 기준 main의 88–89장.

표준 자유장 vacuum squeezing: Y. Zhou, H. Liu, J. Evslin, arXiv:1909.13497, https://arxiv.org/abs/1909.13497 .

정확한 smooth quench와 excess energy: S. R. Das, Old and new scaling laws in quantum quench, PTEP 2016 12C107, https://doi.org/10.1093/ptep/ptw146 (특히 §§5–6의 Bogoliubov 계수와 excess-energy 식).

진공 환경의 기록과 국소 중복성의 구분: R. Blume-Kohout, W. H. Zurek, arXiv:0704.3615, https://arxiv.org/abs/0704.3615 . 여기의 정적 전역 overlap만으로 그 논문의 국소 중복성까지 달성했다고 하지 않는다.
