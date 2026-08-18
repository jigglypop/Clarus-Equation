# TDCOSMO covariance 재현성 노트 **[미완성]**

현재 checkout에는 이 문서가 전제로 삼았던
$\texttt{examples/physics/h0_readout/}$ 코드와 입력 bundle이 없다.
따라서 TDCOSMO/SLACS covariance ingest, factor-role 분해와 cross-channel
결론은 로컬에서 재현되지 않는다.

## 시간지연 렌즈의 조건부 정리

**[정의]** thin-lens 근사에서 두 영상의 시간지연을

$$
\Delta t_{ij}=\frac{D_{\Delta t}}{c}
\Delta\phi_{ij},\qquad
D_{\Delta t}=(1+z_d)\frac{D_dD_s}{D_{ds}}
$$

로 쓴다.

**[정리]** dimensionless background cosmology와 redshift를 고정하면
$D_{\Delta t}\propto H_0^{-1}$다.

**[정리: mass-sheet degeneracy]** lens 변환

$$
\kappa_\lambda(\theta)
=\lambda\kappa(\theta)+(1-\lambda)
$$

는 영상 위치와 상대 확대 구조를 보존하면서
$\Delta\phi_{ij}\mapsto\lambda\Delta\phi_{ij}$로 보낸다. 같은 관측
$\Delta t_{ij}$를 맞추면

$$
D_{\Delta t}\mapsto\frac{D_{\Delta t}}{\lambda},
\qquad
H_0\mapsto\lambda H_0.
$$

따라서 stellar kinematics, source-size 정보 또는 환경 convergence
같은 추가 자료 없이 lens imaging과 time delay만으로 $H_0$를 고유하게
정할 수 없다. 이 no-go는 CE readout에도 그대로 적용된다.

## 누락 의존성

- $\texttt{h0_fisher_io_examples/tdcosmo_slacs_covariance.json}$
- 원본 HDF5 또는 posterior chain과 그 버전·checksum
- $\texttt{h0_tdcosmo_hdf5_to_json.py}$ 및 변수 변환 규약
- $\texttt{h0_tdcosmo_notebook_factor_extract_gate.py}$
- $\texttt{h0_fisher_matrix_io_gate.py}$와
  $\texttt{h0_fisher_io_full_suite.py}$
- lens별 nuisance parameter, covariance 라벨과 fiducial model 설명

## 재개 조건

원자료에서 JSON까지의 변환을 독립적으로 재생하고, 변수 순서·단위·대칭성·
양의 정부호를 확인한 뒤 동일 likelihood에서 비교해야 한다. Source-role
분해는 $H_0$ 결과를 보기 전에 고정하고 ablation과 민감도 분석을 함께
제시한다. 이 조건이 충족되기 전 관측 audit은 **[미완성]**이다.
