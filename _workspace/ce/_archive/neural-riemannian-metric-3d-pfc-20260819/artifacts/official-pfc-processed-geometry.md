# 공식 원숭이 PFC 처리자료의 3D 계량 후보 분석

Status: `PFC_FEASIBILITY_ONLY`

## 입력

- 저자 공식 코드 저장소: `https://github.com/m-j-wojcik/pfc_learning.git`
- 분석 커밋: `48ada8054940f6a7ac26e8e83d150357a9f249d2`
- 원자료 저장소: https://doi.org/10.5061/dryad.c2fqz61kb
- 연계 논문: https://doi.org/10.1038/s41593-026-02333-w
- Nature 공식 Source Data Fig. 2: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-026-02333-w/MediaObjects/41593_2026_2333_MOESM3_ESM.xlsx
- 대상: 실험 1의 실제 macaque PFC 녹화에서 산출된 Nature Source Data와 저자 공식 Git 캐시
- 분석 좌표: colour, shape, XOR selectivity의 3차원 공간

입력 SHA-256:

- `Source_Data_Fig_2.xlsx`: `f2555c030da4f96f0a0c6d46450a146a2d24650479933633dda50158efb0ea5c`
- `selectivity_coefficients_exp1_140_1504stages.pickle`: `a5c0b1ad9b6f0b533449b3983b553b49fbeb12fb084e19843433f627f528bfac`
- `exp1_decoding_collocked_50_150_4stages.pickle`: `11e429d2c207ef2ddb6ee6e080ef671bec115057564c7f319a849202dd7e8206`
- `exp1_decoding_shapelocked_100_150_4stages.pickle`: `6539e4d510792c35531d9da4f7cab8963b72f7abac7c4b7735c46b6e10f6b4c2`

Nature XLSX와 저자 Git 캐시 교차대조:

- Stage 1/4 selectivity 최대 절대 오차: `5.551e-17`
- 4단계 decoding 최대 절대 오차: `0.000e+00`
- 4단계 dimensionality 최대 절대 오차: `0.000e+00`

## 3D SPD 후보

각 단계의 뉴런 selectivity 벡터를 $s_n=(s_{colour},s_{shape},s_{XOR})$로 두고, 기술 통계로

$$
C_k=\operatorname{Cov}(s_n\mid k),\qquad g_k=C_k^{-1}
$$

를 계산했다. 이는 selectivity chart의 **단계별 상수 SPD 후보**이며, 위치 의존 field나 곡률 측정이 아니다.

| 단계 | 데이터 출처 | 뉴런 | $\lambda(C)$ | 유효차원 | $\mathrm{tr}(C)$ | cond$(C)$ | stage 1 대비 AIRM | shape 비율 |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | Nature XLSX + author cache | 114 | `[0.0026658, 0.0028082, 0.0051394]` | 2.8576 | 0.010613 | 1.928 | 0.0000 | 0.0% |
| 2 | author cache | 83 | `[0.0023217, 0.0045186, 0.0048019]` | 2.8695 | 0.011642 | 2.068 | 0.5882 | 93.0% |
| 3 | author cache | 88 | `[0.0032553, 0.003656, 0.0086359]` | 2.7030 | 0.015547 | 2.653 | 0.9216 | 62.1% |
| 4 | Nature XLSX + author cache | 91 | `[0.0015992, 0.0021198, 0.0062483]` | 2.4982 | 0.009967 | 3.907 | 1.1365 | 90.8% |

Stage 1에서 4까지 AIRM 변형은 `1.136529`이고, 제곱거리의 `90.8%`가 공통 scale이 아닌 anisotropic shape 변화다.

Stage 1 covariance:

`[0.00385375, -0.000296812, -0.00119727]<br>[-0.000296812, 0.00288724, 0.000299822]<br>[-0.00119727, 0.000299822, 0.00387242]`

Stage 4 covariance:

`[0.00191539, 0.000269165, 0.00015272]<br>[0.000269165, 0.00183988, 0.000358671]<br>[0.00015272, 0.000358671, 0.00621205]`

Stage 1 inverse-covariance metric candidate $g_1$:

`[288.265, 20.5441, 87.535]<br>[20.5441, 350.623, -20.7952]<br>[87.535, -20.7952, 286.91]`

Stage 4 inverse-covariance metric candidate $g_4$:

`[533.511, -76.3526, -8.70767]<br>[-76.3526, 560.629, -30.4925]<br>[-8.70767, -30.4925, 162.952]`

## 공식 decoding과의 방향 일치

| 관측량 | Stage 1 | Stage 4 | 변화 |
|---|---:|---:|---:|
| colour decoding | 0.7278 | 0.5945 | -0.1333 |
| shape decoding | 0.6801 | 0.5388 | -0.1413 |
| width decoding | 0.9404 | 0.6975 | -0.2429 |
| XOR decoding | 0.6122 | 0.6505 | +0.0384 |
| shattering dimensionality score | 0.6757 | 0.5850 | -0.0908 |
| covariance effective rank | 2.8576 | 2.4982 | -0.3594 |

실측 처리자료에서는 학습 후 width와 전체 shattering score가 감소하고 XOR decoding은 증가했다. 동시에 3D selectivity covariance의 유효차원이 `2.8576 -> 2.4982`로 감소했다. 따라서 '학습에 따라 PFC 표현 기하가 저차원·과제 선택적으로 재편된다'는 방향과 일치한다.

## 반드시 남는 경계

- 이것은 목업이나 합성 데이터가 아니라 저자 공개 저장소의 실제 macaque PFC 파생 자료다.
- 하지만 원시 trial trajectory를 이번 계산에 다시 적합한 것은 아니다. 저자 캐시의 두 fold는 정확히 동일하며 held-out 검증으로 셀 수 없다.
- 단계마다 다른 뉴런을 합친 pseudopopulation이므로 동일 뉴런의 종단 $\Delta g$가 아니다.
- `cell_loc`의 연속 좌표, 피질 주름/두께, 구조 연결 $W^s$가 없으므로 물리적 3D cortical-ribbon metric이나 $\Delta W^s\to\Delta g$를 판정하지 못한다.
- $g_k$는 단계별 상수 행렬이어서 이 chart에서는 곡률이 0이다. 이 결과로 비영 곡률을 주장할 수 없다.

## 판정

공식 실측 PFC 자료는 **학습에 따른 3D 표현 기하의 anisotropic deformation과 저차원화**를 지지한다. 현재 자료가 지지하지 않는 것은 그 deformation의 원인이 구조 연결 변화라는 주장과, 뇌가 비평탄한 3D Riemann field를 직접 구현한다는 강한 주장이다.
