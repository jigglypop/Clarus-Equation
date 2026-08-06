# 핵융합 flavor-aligned 직접 scalar 후보 루프

코드: `reality_stone/python/reality_stone/clarus/fusion_flavor_aligned_loop.py`  
실행: `examples/physics/fusion_flavor_aligned_gate.py`  
테스트: `tests/test_fusion_flavor_aligned_loop.py`

## 1. 결과

최신 canonical 질량 \(29.6991596174\) MeV에서 10 keV D--T 반응률을 1% 높이는
직접 결합은

\[
g_N=0.0174469513,
\qquad
g_Dg_T=6g_N^2=1.82637665\times10^{-3}
\]

이다. 이번 반복에서는 이 저에너지 해를 gauge-invariant flavor-aligned
\(u,d,s\) scalar 연산자와 5 TeV vector-like-quark(VLQ) 예시로 올렸다.
표시된 실제 Yukawa coupling은 perturbative이고 중앙 rare-kaon 곡선도
후보를 허용한다. 다만 full SMEFT--WET RG matching과 29.6991596174 MeV 질량의
radiative-stability gate는 아직 닫히지 않았다.

그러나 neutron--Pb 상한은 30 MeV에서 외삽한 중앙값과 후보 사이의 여유가
0.547%뿐이고, rare-kaon (uds) 곡선은 원 논문이 최대 order-one-decade NLO
변화를 명시한다. 따라서 판정은

```text
CLOSEST_CONDITIONAL_CANDIDATE_NOT_CONSTRAINT_CLEARED
```

이다. 수학적 1%와 gauge-invariant UV 후보의 존재는 확인했지만, 완전한 UV
closure와 실험 제약 gate는 아직 통과하지 않았다.

## 2. quark에서 D/T까지의 정확 매칭

사용한 gauge-invariant 저에너지 방향은

\[
\mathcal L\supset-\frac{\phi}{f_\phi}
\sum_{q=u,d,s}y_q\,\bar Q_LHq_R+\mathrm{h.c.}
\]

이다. 전기약 대칭 깨짐 뒤 \(g_q/m_q=1/f_\phi\)인 flavor alignment를 준다.
[Broggini et al.](https://arxiv.org/abs/2509.03486)의 scalar matrix element
계수를 사용하면

\[
g_p=\frac{0.154\ \mathrm{GeV}}{f_\phi},
\qquad
g_n=\frac{0.158\ \mathrm{GeV}}{f_\phi}.
\]

D와 T의 charge product를 기존 1% 해에 정확히 맞추면

\[
(g_p+g_n)(g_p+2g_n)=6g_N^2
\]

이고,

```text
f_phi                         8.960475410 GeV
g_p                           0.01718658809
g_n                           0.01763299298
relative product residual     < 2e-15
```

을 얻는다. 따라서 universal (A_DA_T=6) 근사를 nonuniversal (p,n) 결합으로
바꾸어도 필요한 D/T 퍼텐셜 세기는 그대로 재현된다.

## 3. perturbative VLQ 예시

[Delaunay et al.](https://arxiv.org/abs/2501.16477)은 flavor-aligned light scalar의
구체적인 VLQ UV 예시와 invisible rare-meson 제약을 제시한다. (M=5) TeV에서

\[
\kappa=\frac{M}{f_\phi}=558.0061,
\qquad
\frac{\kappa v}{M}=\frac{v}{f_\phi}=27.4539.
\]

큰 \(\kappa^\phi\) 자체만으로 nonperturbative라고 판정하면 안 된다. 이것은
단일 Lagrangian coupling이 아니라 논문의 matching
\(\kappa_q^\phi=-\lambda_Qy_Q/y_q\)으로 정의되는 유효 계수비다.
\(\lambda_Q=-1\), \(y_Q=\kappa^\phi y_q\)로
잡으면 실제 새 Yukawa는

```text
y_U     0.0069290
y_D     0.0149808
y_S     0.2996165
max theta_L  0.0104236
```

이다. 모두 perturbative다. 다만 5 TeV threshold 아래 29.6991596174 MeV CP-even scalar
질량의 naturalness 보호기구, full RG running, 유도 Higgs portal은 이
예시만으로 공급되지 않는다. 따라서 표시 coupling의 perturbativity와
full UV-action gate를 별도 Boolean으로 기록한다.

## 4. neutron--Pb 경계

Broggini et al. Appendix의 기존 neutron-scattering 요약은 equal proton/neutron
coupling에

\[
g_N\lesssim2\times10^{-5}\left(\frac{m_\phi}{\mathrm{MeV}}\right)^2
\]

를 준다. 29.6991596174 MeV로 단순 외삽하면 0.01764080이다. flavor-aligned 후보를
Pb nucleus와 neutron projectile에 맞추어

\[
g_{\rm Pb,eff}=
\sqrt{g_n\frac{82g_p+126g_n}{208}}
=0.01754478
\]

로 비교하면 `bound/candidate - 1`로 정의한 중앙 여유는 0.5473%다.

이 값은 `PASS`가 아니다.

- 원 논문의 nuclear signal 범위는 약 6.05 MeV 이하이고 식도
  \(\lesssim\) 근사다.
- 대표 Pb 자료의 \(q_{\max}\simeq13.2\) MeV이면
  \(q^2/m_\phi^2\simeq0.19754\)다.
- contact-limit 질량제곱 scaling의 보정 scale이 0.547% 여유보다 훨씬 크다.
- strong phase cancellation과 nuclear form factor nuisance를 포함한 29.6991596174 MeV
  differential likelihood가 없다.
- \(g_n\) 하나만 equal-coupling 중앙 상한과 비교하면 약 0.0443% 아래지만, 이
  외삽 proxy는 mass-specific likelihood가 아니다.

따라서 mass-specific Pb 각분포 재분석 없이는 허용도 배제도 주장하지 않는다.

## 5. invisible kaon 경계

Delaunay et al. Fig. 4의 invisible \(uds\) 중앙곡선을 로그축에서 digitize한
`29.65 MeV, 182`는 구형 PDF 판독의 **frozen historical sample**이다. 최신 canonical
질량의 tabulated likelihood가 아니므로 그 점을 새 질량으로 문자열 치환하지 않는다.
단순 인접질량 proxy로만 비교하면 후보의 27.4539는 182보다 6.629배 낮다. 최신
canonical 질량에서의 별도 Figure 2 선분 보간과 그 한계는
[여유 강건성 루프](FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md)에 기록한다.

그러나 논문은 이 \(uds\) bound가 partial NLO 식에 의존하며 correction이 최대
한 order of magnitude일 수 있다고 명시한다. 보수적으로 중앙 bound를 10으로
나눈 18.2는 후보 27.4539보다 낮다. 따라서 필요한 것은

- full \(O(p^4)\) weak-ChPT amplitude,
- low-energy-constant covariance,
- E949/NA62 mass-bin efficiency와 likelihood

이다. 중앙 plot만 보고 95% CL 통과를 선언하지 않는다.

## 6. invisible decay는 mechanism이지 joint constraint가 아니다

예시로

\[
\mathcal L\supset-y_\chi\phi\bar\chi\chi,
\quad m_\chi=5\ \mathrm{MeV},\quad y_\chi=10^{-4}
\]

를 두면

\[
\Gamma(\phi\to\chi\bar\chi)
=\frac{y_\chi^2m_\phi}{8\pi}
\left(1-\frac{4m_\chi^2}{m_\phi^2}\right)^{3/2}
=9.86541\times10^{-9}\ \mathrm{MeV},
\]

\[
\tau=6.67191\times10^{-14}\ \mathrm{s},
\qquad c\tau=20.0019\ \mu\mathrm m
\]

이다. 즉 prompt invisible decay mechanism은 쉽게 만들 수 있다. 하지만 total
branching fraction, cosmology, supernova, direct detection을 함께 넣은 likelihood는
아직 없으므로 dark-sector gate는 `False`다.

## 7. 남은 최소 입력

후보를 물리적 CE fusion branch로 승격하려면 네 묶음이 동시에 필요하다.

1. 29.6991596174 MeV에서 form factor와 strong phase nuisance를 가진 Pb differential
   neutron likelihood,
2. full \(O(p^4)\) weak-ChPT \(uds\) amplitude와 NA62/E949 likelihood,
3. invisible sector의 cosmology/SN/direct-detection joint likelihood,
4. VLQ에서 WET까지의 RG matching과 29.6991596174 MeV scalar mass의 radiative stability.

그 전까지는 UV가 있다는 사실과 실험을 통과했다는 사실을 분리한다.
Finite-size와 최신 NA62 그림 보간을 포함한 다음 반복은
[FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md](FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md)에
기록한다.

## 8. 실행

```bash
uv run python examples/physics/fusion_flavor_aligned_gate.py
uv run --extra dev python -m pytest tests/test_fusion_flavor_aligned_loop.py -q
```

현재 wrapper의 legacy 기본값을 바꾸지 않고 최신 snapshot을 재계산한 실제 명령은 다음과
같다. `DEFAULT_SCALAR_MASS_MEV`와 이 모듈의 등록질량을 같은 값으로 주입하고 두 cache를
비운 뒤 기존 solver를 그대로 호출했다.

```powershell
.\.venv\Scripts\python.exe -c "import reality_stone.clarus.fusion_equation_iteration_loop as fe; import reality_stone.clarus.fusion_flavor_aligned_loop as fa; m=29.69915961743591; fe.DEFAULT_SCALAR_MASS_MEV=m; fa.REGISTERED_SCALAR_MASS_MEV=m; fe.current_fusion_equation_iteration_report.cache_clear(); fa.current_fusion_flavor_aligned_report.cache_clear(); r=fa.current_fusion_flavor_aligned_report(); print(r.operator, r.neutron_constraint, r.invisible_completion, r.physical_ce_fusion_branch_accepted)"
```

핵심 출력은 `g_N=0.01744695128447136`,
`Q_D Q_T=0.0018263766547363012`, `f_phi=8.960475410167874 GeV`,
`neutron central margin=0.0054730007095653566`이며 모든 physical gate는 계속 `False`다.
