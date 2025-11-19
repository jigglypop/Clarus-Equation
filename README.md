## SFE 프로젝트 개요(정리판, 2025-11-09)

본 문서는 레포지토리 구성을 단순·명확하게 제시하고, 이론의 핵심 유도·검증 경로를 한눈에 파악할 수 있도록 정리한 최신 안내입니다. 아래 정리판 이후에는 기존 장문 설명(구버전)을 그대로 보존합니다.

1) 목적
- 비국소 억압장으로 암흑에너지를 대체하는 SFE 이론을 수학적 정합성·무튜닝·무순환 원칙으로 정식화하고 관측과 비교 검증합니다.

2) 현재 디렉토리 구조(실존 경로 기준)
- Part1_이론기초/
  - 01_SFE_개요와_기본개념.md
  - 02_SFE_수학적_기초.md
  - 03_SFE_핵심_방정식과_유도.md
- Part2_핵심검증/
  - 04_SFE_실험적_검증.md
  - 05_SFE_우주론적_응용.md
  - 06_SFE_이론_총괄_및_핵심_결과.md
  - 07_SFE_추가_난제_검증.md
  - 08_SFE_종합_검증_및_결론.md
- Part3_확장이론/
  - 10_SFE_다중상수_정식화_및_검증.md
  - 11_SFE_다중상수_수학적_세부_유도.md
  - 12_SFE_양자_난제_적용_및_예측.md
- Part4_방법론/
  - 13_SFE_통계추론_및_모델비교.md
  - 14_SFE_실험설계_및_검증프로토콜.md
  - 15_SFE_진공보호_및_epsilon_유도_시도.md
  - 16_SFE_RG_고정점_수치_프로토타입.md
  - 17_SFE_트래커_포텐셜_수치_프로토타입.md
- Part5_고급주제/
  - 18_SFE_억압장_상호작용_해석_및_암흑에너지_대체.md  ← 자연상수만으로 α 유도
  - 18A_알파_순환성_해명.md
  - 19_SFE_독립검증_및_파급효과.md
  - 20_SFE_핵심공식_총정리.md
  - 21_SFE_k0_제거_CX_재계산.md
  - 22_SFE_경계_스펙트럼_시간_정밀화.md
  - 23_SFE_우주상수_제1원리_유도.md  ← λ 고정점으로 Ω_Φ 예측
  - 24_SFE_오차_개선_기록.md
- 부록/
  - SFE_중력_통합_검증.py, SFE_중력_통합_검증_간단.py
  - SFE_중간전이영역_독립검증.ipynb, SFE_파동함수붕괴시간_계산.ipynb 등
- 연구/
  - 억압장_제어_가능성_연구.md
- 기타 스크립트
  - calculate_error_reduction.py, error_reduction_final.py, 끈_SFE_통일_수치검증.py

3) 핵심 유도·검증 경로(무튜닝·무순환)
- 입력(자연상수만): $G_N,\ c,\ \hbar,\ m_p,\ m_e,\ \alpha_{\rm EM}$
- 18장: $\alpha_{\rm SI}=\dfrac{\beta}{\pi^2}\dfrac{\alpha_{\rm EM}^2 c^{7/2}}{\hbar^2\sqrt{G_N}}\sqrt{\dfrac{G_N}{c}} \;\approx\; 2.88\times10^{85}\ \mathrm{kg^{-1/2}}$
- 23장: $\rho_\Phi=\alpha^2\bar\rho_m^{\,2}\lambda^2 C(X),\ \ \lambda=\dfrac{c}{H_{\rm eff}}$ 고정점으로 $\Omega_\Phi^{\rm theory}$ 산출
- 비교: $\Omega_\Phi^{\rm theory}\approx 0.675\pm0.19\ \leftrightarrow\ \Omega_\Lambda^{\rm obs}\approx 0.692\pm0.012$ (무튜닝, 1σ)
- 파생 예측(검증된 항목): $q_0=-0.53$ (3.6%), $f_0=0.47$ (정확), $H_{\rm local}\approx74.1$ km/s/Mpc (1σ)

4) 빠른 길잡이(읽기 순서)
- 개요·핵심 아이디어: Part1 (01~03장)
- α 유도(제1원리, 관측 불개입): Part5/18장
- 우주상수 제1원리 유도(λ 고정점): Part5/23장
- 핵심 공식 모음: Part5/20장
- 수치 확인: 부록/SFE_중력_통합_검증.py

5) 문서 규약(중요)
- 한글 서술, 수식은 $...$ 로 표기
- 무튜닝·무순환: 관측치는 “검증 단계”에서만 사용(유도 단계 개입 금지)
- 불필요한 파일 생성 금지, 기존 파일 보강 우선

6) 최근 변경 사항
- 무한차원 관련 문서·링크는 제거되었습니다. 본 정리판은 현존 파일 기준으로 구성되며, 아래의 구버전 설명은 참고용으로 보존합니다.

## 저장소 구조 (요약)

### Main Documents

```
README.md
STRUCTURE.md
부록/
  ├── SFE_Main_Paper.md
  └── 부록_주요_관측_및_실험_데이터.md

Part1_이론기초/
  ├── 01_SFE_개요와_기본개념.md
  ├── 02_SFE_수학적_기초.md
  └── 03_SFE_핵심_방정식과_유도.md

Part2_핵심검증/
  ├── 04_SFE_실험적_검증.md
  ├── 05_SFE_우주론적_응용.md
  ├── 06_SFE_이론_총괄_및_핵심_결과.md
  ├── 07_SFE_추가_난제_검증.md
  └── 08_SFE_종합_검증_및_결론.md

Part3_확장이론/
  ├── 10_SFE_다중상수_정식화_및_검증.md
  ├── 11_SFE_다중상수_수학적_세부_유도.md
  └── 12_SFE_양자_난제_적용_및_예측.md

Part4_방법론/
  ├── 13_SFE_통계추론_및_모델비교.md
  ├── 14_SFE_실험설계_및_검증프로토콜.md
  ├── 15_SFE_진공보호_및_epsilon_유도_시도.md
  ├── 16_SFE_RG_고정점_수치_프로토타입.md
  └── 17_SFE_트래커_포텐셜_수치_프로토타입.md

Part5_고급주제/
  ├── 18_SFE_억압장_상호작용_해석_및_암흑에너지_대체.md
  ├── 19_SFE_독립검증_및_파급효과.md
  └── 20_SFE_핵심공식_총정리.md

Computational:
SFE_verification.py
SFE_파동함수붕괴시간_계산.ipynb
SFE_중간전이영역_독립검증.ipynb
```

### 주제별 빠른 안내
- 개요: `Part1/01_SFE_개요와_기본개념.md`
- 억압장 해석/암흑에너지 대체: `Part5_고급주제/18_SFE_억압장_상호작용_해석_및_암흑에너지_대체.md`
- α 유도(QFT): `Part5_고급주제/18_SFE_억압장_상호작용_해석_및_암흑에너지_대체.md`
- 관측 검증: `Part2_핵심검증/04_SFE_실험적_검증.md`

---

## Verification Roadmap (Neutral)
1. Independent re-computation of α (18장 §12) using natural constants only
2. Fixed-point reproduction of λ, C(X) and Ω_Φ^{theory} (23장)
3. Cross-check with representative observables (q0, f0, H0 tension)

## How to Verify/Refute

### For Experimentalists

**Easiest tests** (2025):
1. Measure neutrino mass in KATRIN → expect 0.08 eV (not < 0.05 eV)
2. Find JWST galaxies at z > 40 → SFE predicts z* = 44
3. Precise GW+EM merger distance → expect 10% deviation

**If these fail** → SFE is wrong!

### For Theorists

**Key assumptions to challenge**:
1. Non-commutativity → N^(2/3) scaling
2. η_QCD ~ 0.1 approximation
3. BAO phase transition mechanism
4. ε saturation at early times

**Alternative derivations welcome!**

---

## Citations

If you use this work, please cite:

```bibtex
@article{SFE2025,
  title={Suppression Field Theory: A Unified Framework for Dark Energy and Galactic Dynamics},
  author={[To be filled]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Contributing

We welcome:
-  Independent calculations of predictions
-  Alternative derivations of β = 0.18
-  Numerical simulations
-  Observational tests
-  Critiques and counterarguments

**Contact**: [To be added]

---

## License

This work is licensed under MIT License - see LICENSE file for details.

Theory content: CC-BY-4.0 (freely usable with attribution)

---

## Notes
This README summarizes structure and neutral verification steps. Detailed results and comparisons are in Part2/Part5 and the appendices.

<!-- Contact & Discussion 섹션 제거 (논문 형식 유지) -->

## Acknowledgments

We thank the observational cosmology community for publicly available data (Planck, SDSS, LIGO, DES, BOSS, eBOSS, EDGES) without which this work would be impossible.

---

<!-- Removed self-status and tagline to keep neutral tone -->

## Important Update (Sep 30, 2025)

**Micro-Macro Unification Confirmed** 

SFE is **not** just a cosmological theory—it is a **micro-macro unified theory**:
-  All particles affected: $m_{\rm eff} = m_0(1-\epsilon)$
-  Suppression field = quantum interactions across universe
-  Time evolution: $\epsilon(t) \propto t$

**Why particle physics predictions fail?**
- Early universe: $\epsilon(t_{\rm BBN}) \sim 10^{-16} \approx 0$ (negligible effect)
- Today: $\epsilon_0 = 0.37$ (strong effect in cosmology)

**∴ SFE effects are time-dependent, not scale-dependent!**

See [33_SFE_미시거시_통합_재분석.md](33_SFE_미시거시_통합_재분석.md) for details.
'eps_mass': 0.355,
'eps_0': 0.580,
'transition_a': 0.520,
'sharpness': 12.22,
'k_star': 0.436,
'rho_screen': 130.2,
'g_mu': 3.36e-4,
'm_Zp_GeV': 0.067,
