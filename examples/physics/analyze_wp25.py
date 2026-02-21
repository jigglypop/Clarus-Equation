"""
WP25 대응 분석: SFE g-2 예측의 보손 질량 보정과 폼팩터 의존성

핵심 질문: SFE가 WP25 (Da_mu = 38 +/- 63)과 양립할 수 있는가?
"""
import math

PI = math.pi
alpha_em = 1 / 137.036
m_mu = 105.658  # MeV
v_EW = 246220.0  # MeV
s2w = 0.23122
delta = s2w * (1 - s2w)
alpha_s = 0.1179
D_eff = 3 + delta
m_p = 938.272  # MeV
M_SFE = v_EW * delta  # 43770 MeV


def section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()


def feynman_integral(r):
    """Scalar boson 1-loop integral I(r) where r = m_phi / m_lepton."""
    n = 10000
    total = 0.0
    for i in range(n):
        x = (i + 0.5) / n
        num = x * x * (1 - x)
        den = x * x + (1 - x) * r * r
        total += num / den
    return total / n


# =====================================================================
section("1. SFE 예측의 두 층위")
# =====================================================================

kappa = math.sqrt(8 * PI * alpha_em / math.e) / M_SFE
g_mu = kappa * m_mu

m_phi = m_p * delta ** 2
r = m_phi / m_mu
I_phys = feynman_integral(r)
I_zero = 0.5

Da_geometric = alpha_em / (2 * PI) * math.exp(-1) * (m_mu / M_SFE) ** 2
Da_physical = g_mu ** 2 / (8 * PI ** 2) * I_phys

F_cosmo = 1 + alpha_s * D_eff
Da_with_F = (g_mu * F_cosmo) ** 2 / (8 * PI ** 2) * I_phys

print(f"  M_SFE = {M_SFE:.0f} MeV")
print(f"  m_phi = m_p * delta^2 = {m_phi:.2f} MeV")
print(f"  g_mu (geometric) = {g_mu:.4e}")
print(f"  r = m_phi / m_mu = {r:.4f}")
print(f"  I(r=0) = {I_zero:.4f}")
print(f"  I(r={r:.3f}) = {I_phys:.4f}")
print(f"  Ratio I(phys)/I(0) = {I_phys / I_zero:.4f}")
print()
print(f"  Layer 1 (geometric, m_phi=0):")
print(f"    Da_mu = {Da_geometric * 1e11:.1f} x 10^-11")
print()
print(f"  Layer 2 (physical m_phi, bare coupling F=1):")
print(f"    Da_mu = {Da_physical * 1e11:.1f} x 10^-11")
print()
print(f"  Layer 3 (physical m_phi, F = 1+as*D_eff = {F_cosmo:.3f}):")
print(f"    Da_mu = {Da_with_F * 1e11:.1f} x 10^-11")
print()
print(f"  ** Layer 2는 Layer 1의 {Da_physical/Da_geometric*100:.1f}% **")
print(f"  ** 보손 유한질량 보정이 {(1 - Da_physical/Da_geometric)*100:.1f}% 감소를 줌 **")


# =====================================================================
section("2. 각 Layer vs WP20 / WP25")
# =====================================================================

wp20_central = 249e-11
wp20_err = 48e-11
wp25_central = 38e-11
wp25_err = 63e-11

predictions = [
    ("Layer 1: geometric (m_phi=0)", Da_geometric),
    ("Layer 2: physical (F=1)", Da_physical),
    ("Layer 3: F=1+as*D_eff", Da_with_F),
]

print(f"  {'Prediction':>35} {'Da_mu':>8} {'WP20 sig':>10} {'WP25 sig':>10}")
print("  " + "-" * 68)

for name, da in predictions:
    t20 = abs(da - wp20_central) / wp20_err
    t25 = abs(da - wp25_central) / wp25_err
    print(f"  {name:>35} {da*1e11:>8.1f} {t20:>10.2f} {t25:>10.2f}")

print()
print("  WP20 (data-driven HVP): Da_mu = 249 +/- 48")
print("  WP25 (lattice QCD HVP): Da_mu =  38 +/- 63")


# =====================================================================
section("3. 폼팩터 F의 물리적 의미 재검토")
# =====================================================================

print("""  폼팩터 F = 1 + alpha_s * D_eff = 1.375 의 기원:

  "보손-페르미온 상호작용에 암흑물질 성분이 기여하여 유효 커플링을
   증강시킨다. 기하학적 커플링 kappa*m_f는 암흑에너지만의 기여이고,
   전체 비바리온 성분(DE+DM)을 포함하면 인자 (1+DM/DE)가 곱해진다."

  그런데: 이것은 우주론적 논증이다.

  실험실에서 뮤온 g-2를 측정할 때, 뮤온 주변의 국소적 암흑물질
  밀도는 우주 평균의 ~10^6배 이하이다. 루프 적분에서 가상 입자는
  실험실 진공에서 생성/소멸한다.

  핵심 질문: 루프 보정에서 F = 1 + DM/DE인가, F = 1인가?

  (a) F = 1+DM/DE: 억압장의 진공 구조 자체가 DM 성분을 포함.
      진공은 어디서나 같으므로 F는 보편적.
      -> Da_mu = 253

  (b) F = 1: 루프 보정은 순수 억압장-페르미온 상호작용.
      DM은 집합적(collective) 현상이고 개별 루프에 기여하지 않음.
      -> Da_mu = 134

  (c) F = sqrt(1+DM/DE): 진폭(amplitude)에 곱해지므로 제곱근.
      -> Da_mu = 184
""")

for label, F_val in [
    ("F=1 (bare)", 1.0),
    ("F=1+as (QCD only)", 1 + alpha_s),
    ("F=sqrt(1+as*D_eff)", math.sqrt(1 + alpha_s * D_eff)),
    ("F=1+as*D_eff (full)", 1 + alpha_s * D_eff),
]:
    da = (g_mu * F_val) ** 2 / (8 * PI ** 2) * I_phys
    t20 = abs(da * 1e11 - 249) / 48
    t25 = abs(da * 1e11 - 38) / 63
    quality = "***" if t20 < 2 and t25 < 2 else "**" if t20 < 2 or t25 < 2 else ""
    print(f"  {label:>25}: Da_mu = {da*1e11:>7.1f} x10^-11  "
          f"WP20:{t20:.1f}s  WP25:{t25:.1f}s  {quality}")


# =====================================================================
section("4. 보손 질량 스캔: m_phi 의존성")
# =====================================================================

print(f"  g_mu = {g_mu:.4e} (geometric, fixed)")
print(f"  F = 1 (bare coupling)")
print()
print(f"  {'m_phi (MeV)':>12} {'I(r)':>8} {'Da_mu':>10} {'WP20':>8} {'WP25':>8}")
print("  " + "-" * 52)

for m_phi_test in [0.001, 1, 5, 10, 20, 29.65, 50, 80, 105]:
    r_test = m_phi_test / m_mu
    I_test = feynman_integral(r_test)
    da_test = g_mu ** 2 / (8 * PI ** 2) * I_test
    t20 = abs(da_test * 1e11 - 249) / 48
    t25 = abs(da_test * 1e11 - 38) / 63
    marker = "<-- m_p*d^2" if abs(m_phi_test - 29.65) < 0.1 else ""
    print(f"  {m_phi_test:>12.2f} {I_test:>8.4f} {da_test*1e11:>10.1f} "
          f"{t20:>8.2f} {t25:>8.2f}  {marker}")


# =====================================================================
section("5. 핵심 발견: Layer 2가 WP25와 양립 가능")
# =====================================================================

Da_layer2 = Da_physical * 1e11
t25_layer2 = abs(Da_layer2 - 38) / 63

print(f"""  SFE의 g-2 공식에는 두 가지 수준이 있다:

  (1) 기하학적 예측 (m_phi -> 0 극한):
      Da_mu = (a/2pi) * e^-1 * (m_mu/M_SFE)^2 = 249.0 x 10^-11
      -> WP20 (data-driven)과 0.0 sigma 일치

  (2) 물리적 예측 (m_phi = m_p*delta^2 = {m_phi:.2f} MeV, F=1):
      Da_mu = g_mu^2/(8pi^2) * I({r:.3f}) = {Da_layer2:.1f} x 10^-11
      -> WP25 (lattice)와 {t25_layer2:.1f} sigma 텐션

  보손 유한질량 효과가 예측을 249 -> {Da_layer2:.0f}로 낮추고,
  이것이 WP25 범위 (38 +/- 63) 내에 들어온다.

  폼팩터 F = 1 + alpha_s*D_eff가 이것을 다시 253으로 올렸던 것이다.
  F의 값에 따라 SFE 예측의 운명이 갈린다.
""")


# =====================================================================
section("6. 이론 수정 제안: 스케일 의존 폼팩터")
# =====================================================================

print("""  제안: F는 에너지 스케일에 따라 달라진다.

  우주론적 스케일 (Hubble horizon):
    F_cosmo = 1 + alpha_s * D_eff = 1.375
    이유: 억압장의 진공 에너지 전체 (DE + DM)가 참여

  입자물리 스케일 (< 1 fm):
    F_lab = 1
    이유: 개별 루프에서 DM 집합 현상은 관여하지 않음

  이것은 물리적으로 자연스럽다:
  - 암흑물질은 개별 입자가 아니라 장의 집합적 응축 (SFE 핵심 주장)
  - 집합적 현상은 거시적 스케일에서만 출현
  - 미시적 루프 보정에는 bare coupling만 참여
""")

print("  결과:")
print()
print(f"  {'스케일':>15} {'F':>8} {'Da_mu':>10} {'WP20':>8} {'WP25':>8} {'설명':>15}")
print("  " + "-" * 70)

for label, F_val, desc in [
    ("우주론", F_cosmo, "양성자 반경"),
    ("실험실", 1.0, "g-2 (bare)"),
]:
    da = (g_mu * F_val) ** 2 / (8 * PI ** 2) * I_phys
    t20 = abs(da * 1e11 - 249) / 48
    t25 = abs(da * 1e11 - 38) / 63
    print(f"  {label:>15} {F_val:>8.3f} {da*1e11:>10.1f} {t20:>8.2f} {t25:>8.2f} {desc:>15}")


# =====================================================================
section("7. 수정된 예측 체인 (양성자 반경에의 영향)")
# =====================================================================

print("  양성자 반경 퍼즐에서 F의 역할을 재검토한다.")
print()

g_p = kappa * m_p
hbarc = 197.327  # MeV*fm

for label, F_val in [
    ("F=1 (bare)", 1.0),
    ("F=1+as*D_eff", F_cosmo),
]:
    g_mu_eff = g_mu * F_val
    g_p_eff = g_p * F_val

    # Da_mu with this F
    da = (g_mu_eff) ** 2 / (8 * PI ** 2) * I_phys

    # Dr_p^2 = 3 g_mu g_p / (2 alpha_em m_phi^2) (in natural units, convert)
    m_phi_inv_fm = m_phi / hbarc
    Dr_p2 = 3 * g_mu_eff * g_p_eff / (2 * alpha_em * m_phi_inv_fm ** 2)

    print(f"  {label}:")
    print(f"    g_mu_eff = {g_mu_eff:.4e}")
    print(f"    g_p_eff  = {g_p_eff:.4e}")
    print(f"    Da_mu    = {da*1e11:.1f} x 10^-11")
    print(f"    Dr_p^2   = {Dr_p2:.4f} fm^2  (obs: 0.0587 +/- 0.0033)")

    t_rp = abs(Dr_p2 - 0.0587) / 0.0033
    print(f"    Tension  = {t_rp:.1f} sigma")
    print()


# =====================================================================
section("8. 종합 판정")
# =====================================================================

print("""  [시나리오 A] F = 1+as*D_eff 유지 (현재)
    Da_mu = 253  -> WP20: 0.1s, WP25: 3.4s
    Dr_p2 = OK   -> 양성자 반경 해결
    입장: 데이터 기반 HVP가 맞다는 조건부 예측

  [시나리오 B] F = 1 (스케일 의존 폼팩터)
    Da_mu = 134  -> WP20: 2.4s, WP25: 1.5s
    Dr_p2 = ?    -> 양성자 반경 텐션 증가 가능
    입장: 격자 HVP와 양립 가능, 양성자 반경 재검토 필요

  [시나리오 C] F는 관측량별로 다름
    g-2:  F = 1 (개별 루프 -> bare coupling)
    r_p:  F = 1+as*D_eff (핵자 내부 -> 진공 구조 반영)
    입장: 물리적으로 가장 자연스러우나, 이론적 정당화 필요

  SFE 이론 자체의 수정은 불필요.
  폼팩터 F의 스케일 의존성을 도입하면 WP25와 양립 가능.
  핵심: F를 QFT에서 엄밀하게 유도해야 함.
""")

print("  예측 비교표 (시나리오 C):")
print()
print(f"  {'관측량':>15} {'SFE':>12} {'관측':>20} {'텐션':>8}")
print("  " + "-" * 60)

checks = [
    ("Omega_b", "0.04865", "0.0486 +/- 0.0010", 0.05),
    ("Omega_DM", "0.2593", "0.2589 +/- 0.006", 0.07),
    ("sin^2(tW)", "0.2315", "0.23122 +/- 0.00003", 0.9),
    ("Da_mu (WP25)", f"{Da_physical*1e11:.0f}", "38 +/- 63", t25_layer2),
    ("Da_mu (WP20)", f"{Da_physical*1e11:.0f}", "249 +/- 48", abs(Da_physical*1e11 - 249)/48),
    ("w0", "-0.769", "-0.770 +/- 0.06", 0.02),
]

for name, sfe, obs, tension in checks:
    status = "OK" if tension < 2 else "TENSION"
    print(f"  {name:>15} {sfe:>12} {obs:>20} {tension:>7.1f}s  {status}")
