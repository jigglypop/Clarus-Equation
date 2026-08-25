# 10-sources — 정보 기회비용, 열역학, 영향함수 근거

Status: COMPLETE

최종 접근일: 2026-08-25

## 1. 1차 출처 판정표

| 주장 | 1차 출처 | 확립하는 범위 | 확립하지 않는 것 |
|---|---|---|---|
| 논리적 비가역성과 열 발생 | R. Landauer, “Irreversibility and Heat Generation in the Computing Process,” *IBM Journal of Research and Development* 5 (1961), [DOI 10.1147/rd.53.0183](https://doi.org/10.1147/rd.53.0183) | 지정된 물리 memory의 logically irreversible reset과 최소 열비용의 연결 | 모든 미실현 가능성이 실제 저장 에너지라는 주장 |
| 측정과 소거 비용의 protocol dependence | T. Sagawa and M. Ueda, “Minimal Energy Cost for Thermodynamic Information Processing: Measurement and Information Erasure,” *Physical Review Letters* 102 (2009), [DOI 10.1103/PhysRevLett.102.250602](https://doi.org/10.1103/PhysRevLett.102.250602) | measurement와 erasure의 최소 work를 memory free energy와 information으로 구분 | outcome surprisal이 protocol 없이 고유 energy라는 주장 |
| 비평형 quantum state의 thermodynamic resource | F. G. S. L. Brandão, M. Horodecki, J. Oppenheim, J. M. Renes, and R. W. Spekkens, “Resource Theory of Quantum States Out of Thermal Equilibrium,” *Physical Review Letters* 111 (2013), [DOI 10.1103/PhysRevLett.111.250404](https://doi.org/10.1103/PhysRevLett.111.250404) | thermal operation 아래 비평형 자유에너지·상태변환 제약 | 임의의 비선택 branch가 선택 branch에서 중력원이라는 주장 |
| small quantum system의 work/free-energy 경계 | M. Horodecki and J. Oppenheim, “Fundamental Limitations for Quantum and Nanoscale Thermodynamics,” *Nature Communications* 4 (2013), [DOI 10.1038/ncomms3059](https://doi.org/10.1038/ncomms3059) | bath와 허용 operation을 고정한 work extraction/formation free energies | bath·Hamiltonian·operation 없이 정보량을 energy로 바꾸는 규칙 |
| 환경 경로를 적분한 influence functional | R. P. Feynman and F. L. Vernon Jr., “The Theory of a General Quantum System Interacting with a Linear Dissipative System,” *Annals of Physics* 24 (1963), [DOI 10.1016/0003-4916(63)90068-X](https://doi.org/10.1016/0003-4916(63)90068-X) | 환경 자유도를 적분해 system 변수의 influence functional, dissipation과 noise를 만드는 형식 | “선택되지 않은 outcome”의 확률을 양의 local energy density로 자동 변환하는 규칙 |
| curved-spacetime open-system effective action | E. Calzetta and B. L. Hu, “Closed-Time-Path Functional Formalism in Curved Spacetime: Application to Cosmological Back-Reaction Problems,” *Physical Review D* 35 (1987), [DOI 10.1103/PhysRevD.35.495](https://doi.org/10.1103/PhysRevD.35.495) | closed-time-path effective action으로 curved-spacetime backreaction을 다루는 구조 | CE의 특정 opportunity functional 또는 암흑 abundance |

## 2. 출처가 허용하는 계층

문헌이 허용하는 안전한 순서는

$$
\text{dimensionless information}
\longrightarrow
\text{specified thermodynamic work/free energy}
\longrightarrow
\text{effective action}
\longrightarrow
\text{metric stress}
$$

다. 첫 화살표에는 $H$, $T$, bath와 protocol이 필요하고, 두 번째 화살표에는
local/covariant completion과 독립 energy-density scale이 필요하다. 따라서
$-\ln p$, $k_BT[-\ln p]$, $k_BT D(\rho\|\gamma_T)$와
$T_{\mu\nu}$는 같은 타입이 아니다.

## 3. 출처 경계

1. Landauer principle은 erasure의 물리적 구현에 관한 제한이지 discarded
   histories의 ontology가 아니다.
2. relative entropy free energy는 Hamiltonian, Gibbs reference와 operation
   class에 의존한다.
3. Lorentzian influence functional은 일반적으로 복소수이며 그 imaginary part는
   noise/decoherence를 담을 수 있다. 이를 양의 energy density로 읽지 않는다.
4. effective action의 metric variation이 있어야 stress를 정의할 수 있고, 외부
   apparatus나 reservoir가 있으면 그 stress까지 합쳐 conservation을 닫아야 한다.
