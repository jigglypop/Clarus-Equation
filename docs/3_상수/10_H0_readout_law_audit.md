# H0 readout law audit

이 문서는 현재 \(H_0\) readout law의 수식, 검증 상태, 그리고 앞으로 계속 진행해도 되는지를 평가하기 위한 audit이다.

## 1. 핵심 수식

기본 CE 상태량은

$$
D_{\rm eff}=3+\delta,
\qquad
x=e^{-D_{\rm eff}(1-x)},
\qquad
\sigma=1-x.
$$

수치값은

$$
D_{\rm eff}=3.17775842,
\qquad
x=0.04864672,
\qquad
\sigma=0.95135328,
\qquad
\delta\sigma=0.16911106.
$$

e-fold 수는

$$
N_e=\frac32D_{\rm eff}N_{\rm gauge}=57.19965162.
$$

late-time de Sitter horizon entropy는

$$
S_{\rm dS}=\frac{\pi}{(H_0t_{\rm Pl})^2}
$$

로 읽는다.

CE horizon readout의 기본 불변량 후보는

$$
I_H=\log S_{\rm dS}+\pi\delta\sigma,
\qquad
I_{\rm phase}=\frac{\pi^2}{2}N_e.
$$

따라서 global readout은

$$
\boxed{
\log S_{\rm low}
=
\frac{\pi^2}{2}N_e-\pi\delta\sigma
}
$$

이고,

$$
\boxed{
H_0^{\rm low}=67.247245\,{\rm km\,s^{-1}Mpc^{-1}}.
}
$$

high local endpoint readout은

$$
\boxed{
\log S_{\rm high}
=
\frac{\pi^2}{2}N_e-\pi\delta\sigma-\delta\sigma
}
$$

이고,

$$
\boxed{
H_0^{\rm high}=73.180689\,{\rm km\,s^{-1}Mpc^{-1}}.
}
$$

두 branch를 하나로 쓰면

$$
\boxed{
\log S(q)
=
\frac{\pi^2}{2}N_e-\pi\delta\sigma-q\delta\sigma.
}
$$

## 2. Readout selector

초기에는 \(q\)를 데이터에서 역산했다.

$$
q_{\rm req}
=
\frac{\log S_{\rm global}-\log S_{\rm obs}}{\delta\sigma}.
$$

그 결과 CMB는 \(q\simeq0\), Cepheid/SN은 \(q\simeq1\), BAO/TRGB/JAGB/GW/lens는 중간값을 요구했다.

이를 topology 값으로 압축하면

$$
\boxed{
q_{\rm topo}=\frac{L}{L+G}
}
$$

이다. 여기서 \(L\)은 local endpoint closure weight, \(G\)는 global ruler/horizon closure weight다.

이를 다시 graph conductance로 쓰면

$$
C_{\rm path}
=
\frac{\prod_{e\in p}r_e}{|p|},
\qquad
q_{\rm graph}=\frac{C_L}{C_L+C_G}.
$$

Fisher matrix가 주어진 경우 가장 정제된 selector는

$$
\boxed{
r_{ij}
=
\frac{|F_{ij}|}{\sqrt{F_{ii}F_{jj}}},
\qquad
q_F=\frac{C_L(F)}{C_L(F)+C_G(F)}.
}
$$

최종 data-facing readout law 후보는

$$
\boxed{
\log S
=
\frac{\pi^2}{2}N_e-\pi\delta\sigma-q_F\delta\sigma.
}
$$

## 3. 현재 검증 요약

| 단계 | 결과 | 지위 |
|---|---:|---|
| low branch | \(H_0=67.247245\) | Planck/CMB 계열 통과 |
| high branch | \(H_0=73.180689\) | SH0ES/megamaser/local 계열 통과 |
| \(q_{\rm req}\) ordering | \(\chi^2/{\rm dof}=0.220/6\) | 강한 selector 신호 |
| prospective external channels | \(\chi^2/{\rm dof}=0.462/5\) | lens/maser/GW 통과 |
| graph selector | \(\chi^2/{\rm dof}=0.402/10\) | schematic graph 통과 |
| Fisher selector | \(\chi^2/{\rm dof}=0.199/6\) | schematic Fisher 통과 |
| Fisher/covariance IO suite | PASS | 실제 matrix 입력 준비 완료 |

## 4. 닫힌 것과 닫히지 않은 것

닫힌 것:

1. \(H_0\) low/high branch를 하나의 무차원 식으로 쓸 수 있다.
2. high branch correction은 새 상수가 아니라 기존 결핍 밀도 \(\delta\sigma\)다.
3. \(q\)는 자유 parameter에서 topology ratio, graph conductance, Fisher conductance로 내려왔다.
4. 현재 IO 계층은 Fisher/covariance/CSV/labelled CSV를 받을 수 있고 full suite를 통과한다.

아직 닫히지 않은 것:

1. \(\log S\)가 왜 정확히 primordial phase-area count \((\pi^2/2)N_e\)를 읽는지의 물리 유도.
2. \(\pi\delta\sigma\)와 \(\delta\sigma\)가 각각 global horizon integral과 local endpoint defect로 들어가는 정식 경로적분 유도.
3. 실제 공개 likelihood covariance에서 \(q_F\)를 계산한 결과의 채널 간 반복성. TDCOSMO 첫 투입은 `11_TDCOSMO_real_covariance_audit.md`에 기록되어 있으나, 아직 여러 독립 관측군으로 일반화된 것은 아니다.
4. \(L,G\) 또는 \(C_L,C_G\)의 graph 노드 정의가 관측 pipeline마다 고유하게 정해지는지.
5. 같은 파라미터가 likelihood closure에 따라 local endpoint 또는 global closure로 바뀌는 role map \(R\)을 외부 분석 규약에서 독립적으로 정할 수 있는지.

## 5. 반증 조건

다음 중 하나가 확인되면 현재 readout law는 깨진다.

1. 실제 covariance/Fisher matrix로 계산한 \(q_F\)가 channel class와 무관하게 모두 비슷한 값으로 수렴한다.
2. CMB/Planck 계열의 실제 \(q_F\)가 \(q\simeq1\)에 가깝게 나온다.
3. SH0ES/local geometric 계열의 실제 \(q_F\)가 \(q\simeq0\)에 가깝게 나온다.
4. TDCOSMO처럼 같은 관측군에서 model closure를 바꿀 때 \(q_F\)가 변하지 않는다.
5. 실제 covariance를 넣었을 때 \(H_0(q_F)\)의 pull이 여러 독립 채널에서 \(3\sigma\) 이상 체계적으로 벗어난다.

## 6. 계속 진행해도 되는가?

판정:

$$
\boxed{
\text{계속 진행해도 된다. 단, 다음 진행은 새 수식 추가가 아니라 실제 covariance와 role map 반증이어야 한다.}
}
$$

이유는 다음과 같다.

1. 수식 구조는 현재 충분히 압축되었다.
2. \(q\)는 더 이상 임의 fitting parameter가 아니다.
3. IO/validation/batch/full-suite가 준비되었다.
4. TDCOSMO 계열에서는 실제 covariance 투입이 시작되었다.
5. 남은 핵심은 이론 내부 조작이 아니라 외부 covariance와 관측 pipeline별 role map으로 반증하는 것이다.

따라서 다음 우선순위는 다음이다.

1. TDCOSMO 결과는 `11_TDCOSMO_real_covariance_audit.md`를 정본으로 삼는다.
2. 다음 독립 채널은 GW standard siren, BAO, SH0ES/Pantheon+, Planck covariance 중 하나로 잡는다.
3. 각 채널의 source/version, likelihood closure, role map \(R\), conductance mode를 manifest에 기록한다.
4. 같은 \(F\)에서 \(R\)을 바꿔도 결론이 버티는지, 같은 \(R\)에서 독립 covariance를 바꿔도 branch ordering이 유지되는지 본다.
5. \(q_F\), \(H_0(q_F)\), pull뿐 아니라 role-map 선택의 사유를 함께 문서에 추가한다.

추천 다음 채널은 GW standard siren 또는 Planck/CMB covariance다. 이유는 TDCOSMO와 관측 topology가 달라서, \(q_F=q_F(F,R)\) 규칙이 lensing 특수 규칙인지 더 빨리 반증할 수 있기 때문이다.

## 7. 첫 실제 소스 스카우트

첫 실제 covariance 후보로 TDCOSMO IV public hierarchy analysis repository를 등록했다. 이 절은 스카우트 기록이며, 실제 posterior covariance 투입 결과와 source-aware role rule은 후속 문서 `11_TDCOSMO_real_covariance_audit.md`를 우선한다.

```text
https://github.com/TDCOSMO/hierarchy_analysis_2020_public.git
```

고정 commit:

```text
6c293af582c398a5c9de60a51cb0c44432a3c598
```

소스 등록 파일:

```text
examples/physics/h0_readout/h0_real_covariance_targets.json
```

스카우트 게이트:

```bash
python examples/physics/h0_readout/h0_real_source_scout_gate.py
```

결과:

| target | remote HEAD | status |
|---|---|---|
| TDCOSMO hierarchy analysis 2020 public | `6c293af582c3` | PASS |

후보 파일:

- `JointAnalysis/tdcosmo_ifu_chain_slope_log_scatter.h5`
- `JointAnalysis/tdcosmo_slacs_chain_slope_log_scatter.h5`
- `JointAnalysis/tdcosmo_slacs_ifu_chainifu_separate_slope_log_scatter.h5`
- `TDCOSMO_sample/tdcosmo_chain_alpha_free.h5`
- `TDCOSMO_sample/tdcosmo_chain_alpha_free_om.h5`
- `TDCOSMO_sample/tdcosmo_chain_alpha_fixed_om.h5`

현재 환경 제한:

```text
h5py = missing
```

따라서 다음 실제 계산을 하려면 HDF5 reader가 필요하다. 선택지는 두 가지다.

1. `h5py`를 설치하고 HDF5 chain에서 covariance/Fisher block을 추출한다.
2. TDCOSMO repo 안의 CSV/텍스트 likelihood product를 먼저 찾아 labelled CSV adapter로 처리한다.

현재 추천은 1번이다. TDCOSMO의 핵심 산출물은 HDF5 chain이므로, 실제 \(q_F\) 계산으로 가려면 chain covariance extractor가 필요하다.

