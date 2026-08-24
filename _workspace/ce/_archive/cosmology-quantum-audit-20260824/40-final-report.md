# CE 우주론–양자역학 연결 점검

Status: COMPLETE

## 초록

CE의 중심 서사는 환경이 가능한 성분 중 하나를 강제하는 **끼임**, 선택되지
않은 성분을 잔류 측도로 보존하려는 **접힘**, 그 잔류를 암흑물질·암흑에너지로
읽으려는 **암흑 표현**의 세 단계다. 이번 감사에서 Poisson 고정점과 Lambert
$W$ 해, 지정 입력에서의 수치, 무차원성은 조건부 수학으로 닫혀 있음을
확인했다. 반면 끼임을 실제 양자 instrument와 단일 시행 결과로 만드는 단계,
접힌 측도를 국소 공변장과 보존 stress tensor로 만드는 단계, 그 장을 오늘의
세 우주 성분과 관측량으로 보내는 단계는 아직 닫히지 않았다. 정본 문서는 이
경계를 대체로 `[공리]`와 `[미완성]`으로 정확히 표시하며, 표본 문서에서 지위
부풀림은 발견하지 못했다. 따라서 현재 CE는 일관된 조건부 수학과 명시적 물리
연구 프로그램을 갖지만, 양자 측정이나 암흑 부문의 완성된 이론 또는 독립
관측 예측으로 판정할 수 없다.

## 1. 무엇을 점검했는가

점검 대상은 수치 사슬만이 아니라 다음 물리 서사 전체였다.

$$
\text{끼임: 환경 선택}
\longrightarrow
\text{접힘: 비선택 성분의 잔류}
\longrightarrow
\text{암흑 표현: 우주론 readout}.
$$

각 연결을 네 층으로 나누었다. 첫째는 위 세 문장으로 이루어진 동기 서사와
물리 공리다. 둘째는 Poisson 분지과정, measurable pushforward, 지정 FLRW
모형처럼 전제를 명시하면 성립하는 조건부 정리다. 셋째는
$D=D_{\rm eff}$, $q_{\rm ext}\mapsto\Omega_b$, 평탄 closure처럼 모형이
채택한 공리다. 넷째는 quantum instrument, 국소 공변 작용, 종별 current와
관측 forward likelihood처럼 아직 비어 있는 다리다. 관측 중심값과의 근접은
이 네 층의 지위를 바꾸지 않는다.

## 2. 닫힌 수학

독립적인 자손 수가 평균 $D>1$의 Poisson 분포를 따를 때 최소 소멸확률은

$$
q=\exp[-D(1-q)]
$$

의 낮은 고정점이다. $z=-De^{-D}$라 두면 두 실근은

$$
q_{\rm ext}=-\frac{1}{D}W_0(z),
\qquad
1=-\frac{1}{D}W_{-1}(z)
$$

로 표현된다. $D>1$에서 낮은 근은 $(0,1/D)$에 유일하고
$Dq_{\rm ext}<1$이므로 고정점 반복에 대해 국소적으로 안정하다. 이는
Poisson genealogy의 정리이며 양자 상태나 우주 물질에 대한 정리는 아니다.

현재 정확 입력 사슬

$$
\alpha_s=0.11789,
\qquad
\sin^2\theta_W=4\alpha_s^{4/3},
\qquad
\delta=\sin^2\theta_W(1-\sin^2\theta_W),
\qquad
D_{\rm eff}=3+\delta
$$

을 그대로 사용하면

$$
D_{\rm eff}=3.1777584234099736,
\qquad
q_{\rm ext}=0.048646719644028225,
\qquad
Dq_{\rm ext}=0.15458752312007412
$$

를 얻는다. 독립 bisection과 Lambert $W$ 계산은
$6.94\times10^{-18}$ 이내에서 일치했고 binary64 방정식 잔차는 0으로
반올림되었다. 지수와 Lambert $W$의 인자, $D$, $q$와 $\Omega_i$는 모두
무차원이다. 다만 $\alpha_s$의 scale·scheme과
$4\alpha_s^{4/3}$ 관계는 물리 유도가 아니라 지정 입력과 경험 관계다.

지정한 추가 비율 $R=\alpha_sD$를 채택하면

$$
\Omega_c=(1-q)\frac{R}{1+R},
\qquad
\Omega_\Lambda=\frac{1-q}{1+R}
$$

라는 분할은 합이 1인 대수적 산출을 준다. 그러나 $R$을 바꾸면 같은 $q$에서
연속적으로 다른 분할을 얻는다. 따라서 고정점만으로
$\Omega_c$와 $\Omega_\Lambda$가 결정되지 않는다는 비식별성이 명확하다.

## 3. 끼임: 양자역학에서 남은 다리

양자 instrument는 outcome마다 completely positive map을 주고, 그 합이
trace-preserving이어야 한다. 이 자료가 outcome 확률과 조건부 사후 상태를
함께 정한다. Lindblad/GKSL 생성자나 Kossakowski 행렬은 주어진
Markovian open-system 모형의 시간 진화를 기술할 수 있지만, 어느
instrument와 unravelling을 자연이 택하는지 또는 한 시행에서 어느 결과가
발생하는지를 혼자 결정하지 않는다.

현재 `quantum_jump_bridge.py`의 검사는 공급된 Hamiltonian, jump operator와
population sector 아래에서 고전적 닫힌 block이 존재하는지를 조건부로
검사한다. coherent Hamiltonian이나 collective jump는 population closure를
깨는 반례가 되며, 닫힌 Markov chain이 존재해도 birth 해석, reset,
independent increments와 genealogy 독립성을 추가하지 않으면 Poisson
분지과정이 되지 않는다. 따라서 Born prior의 조건부 정리와 실제 측정
동역학을 구분한 현재 문서 지위가 맞다.

이 다리를 닫으려면 미시 작용, system–environment 분할, bath와 coupling,
초기상태, 근사 한계, outcome algebra를 먼저 고정하고 그로부터 CPTP
instrument와 count process를 유도해야 한다. complete positivity, Born
일관성, basis 견고성, Markov/secular 근사와 독립 genealogy가 반증 기준이
되어야 한다.

## 4. 접힘: 잔류 측도에서 물리장까지

비선택 subprobability $\nu_{{\rm ns},\beta}$와 measurable integrable
kernel이 주어지면

$$
\phi_\beta(x)=
\int_{\Gamma_{\rm ns}}K_\phi(x,\gamma)
\nu_{{\rm ns},\beta}(d\gamma)
$$

는 well-defined pushforward가 될 수 있다. 이것은 측도론적으로 닫힌
결과다. 그러나 임의의 전역 path functional에 의존하는 kernel도 같은 적분을
정의할 수 있으므로, 적분의 존재는 locality나 diffeomorphism covariance를
뜻하지 않는다. 조건부 정규화한 잔류 분포는 원래의 총 비선택 질량 정보도
제거한다.

독립 물리장으로 읽으려면 공변 작용 $S[g,\phi,\ldots]$, 장방정식,
renormalization과

$$
T_{\mu\nu}
=-\frac{2}{\sqrt{-g}}\frac{\delta S}{\delta g^{\mu\nu}}
$$

가 필요하다. 이어 Ward identity, visible sector와의 교환 current, 전체
stress-energy 보존과 안정성을 보여야 한다. 현재 pushforward는 이 자료를
제공하지 않으므로 접힘의 수학적 보존과 물리적 에너지 보존을 동일시할 수
없다.

## 5. 암흑 표현: 우주론에서 남은 다리

과거 직접 사상 $q_{\rm ext}\mapsto\Omega_b$는 현재 원장에서 역사적
경계모형의 `[공리]`로 보존된다. conditioned tree에서 얻는
$Dq_{\rm ext}$와 같은 전이 면의 $\Omega_m=1/D$를 독립적으로 채택하면
$\Omega_b=q_{\rm ext}$가 되는 것은 곱셈 항등식이다. 이 항등식은 conditioned
node를 baryon current로 바꾸는 작용이나 두 전제가 같은 공변 전이 면에서
성립하는 이유를 제공하지 않는다.

필요한 물리 닫힘은 다음과 같다.

1. 비선택 sector의 공변 장과 total stress tensor를 정한다.
2. conditioned node를 보존 species current로 바꾸는 전이를 유도한다.
3. 유일한 전이 hypersurface와 freeze law, radiation·heat·wall을 포함한
   전체 보존을 보인다.
4. 배경뿐 아니라 섭동, sound speed, anisotropic stress, 초기조건과
   Einstein–Boltzmann 진화를 정한다.
5. CMB·BAO·초신성·lensing·성장 observable, nuisance parameter,
   covariance와 likelihood를 자료를 보기 전에 고정한다.

이 과정 없이 하나의 density ratio를 여러 관측량의 동시 예측으로 확장할 수
없다. 현재 forward-model 문서는 평탄 FLRW/CPL/GR과 외부 scale 아래의
조건부 계산기로는 유효하지만, 그 방정식상태와 scale을 CE가 유도했다는
증거는 아니다.

## 6. 관측 비교

Planck 2018의 base flat $\Lambda$CDM 결합은
$H_0=67.4\pm0.5\ {\rm km\,s^{-1}\,Mpc^{-1}}$와
$\Omega_m=0.315\pm0.007$을, DESI DR1 BAO의 flat $\Lambda$CDM 분석은
$\Omega_m=0.295\pm0.015$를 보고한다. DESI full-shape+BAO+CMB 결합은
$\Omega_m=0.3056\pm0.0049$를 준다. 이 값들은 모두 dataset, likelihood와
우주론 모형에 의존하는 posterior이며 서로 독립적인 이론 무관 상수가 아니다.

CE 수치가 일부 posterior 중심값에 가깝다는 것은 계산 재현성과 조건부
호환성을 보여 줄 수 있다. 그러나 현재 density map은 이미 알려진 우주 성분을
목표로 구성되었고 독립 confirmatory holdout이 없으므로, 그 근접을 사전
예측이나 물리 bridge의 증거로 세지 않는다. 특히
$\omega_b=\Omega_bh^2$와 $\Omega_b$를 혼동하지 않고, 서로 다른 posterior의
성분을 한 tuple로 조합하지 않아야 한다.

## 7. 판정과 우선순위

형식 감사의 판정은 `PASS`다. 이는 정본 표본에서 수학, 공리, 산출과 미완성
다리의 지위가 실제 근거와 일치했고 P0 오류나 status inflation을 찾지
못했다는 뜻이다. 물리 이론 완결 판정은 별개이며 다음 네 P1이 남아 있다.

| 우선순위 | 미완성 항목 | 닫힘 조건 |
|---|---|---|
| P1 | quantum instrument·Born outcome·unravelling·Poisson genealogy | 미시 작용에서 instrument와 count law 유도 |
| P1 | pushforward에서 국소 공변장·stress tensor로의 연결 | 공변 EFT, metric variation, Ward identity와 안정성 |
| P1 | $q$에서 $\Omega_b,\Omega_{\rm DM},\Omega_\Lambda$로의 물리 readout | species current, transition surface, 보존과 비식별성 해소 |
| P1 | 독립 관측 시험 | 섭동·관측 연산자·likelihood·holdout 사전 고정 |

세 작업 경로는 순차적이다. 먼저 미시 instrument와 count process를 닫고,
다음으로 공변 residual EFT를 세우며, 마지막으로 종별 current와 관측 forward
model을 연결해야 한다. 뒤 단계의 수치 근접은 앞 단계의 누락을 대체하지
않는다.

## 8. 재현과 출처

독립 계산은
`_workspace/ce/cosmology-quantum-audit-20260824/artifacts/verify_math_lane.py`
에 있다. 실행 명령은 다음과 같다.

` .codex/hooks/python.cmd python _workspace/ce/cosmology-quantum-audit-20260824/artifacts/verify_math_lane.py `

집중 회귀는 다음 명령으로 실행했고 `19 passed in 4.56s`를 얻었다.

` .codex/hooks/python.cmd pytest tests/test_cosmology_registry.py tests/test_quantum_jump_bridge.py -q `

양자 instrument는 Blume-Kohout 외(2021), Markovian generator는
Lindblad(1976), open-system 근사 조건은 Nathan과 Rudner(2020)를 확인했다.
우주론 배경·보존·섭동 요구는 Einstein Online과 Mukhanov, Feldman,
Brandenberger(1992)를, 관측 비교는 Planck 2018 VI와 DESI 2024 VI/VII를
확인했다. 정확한 링크, 판본과 source fact/inference 구분은
`10-sources.md`에 기록했다.
