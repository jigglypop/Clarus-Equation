"""
CE 중성미자 질량: dimensional 정합 + Casimir 멱지수 + 내부 스케일 m_τ

공식: m_nu_l = (delta^4 / [(16*pi^2)^2 * 32*pi^3 * (1+R)]) * m_l^((d+2)/(d^2-1)) * m_tau^((d^2-d-3)/(d^2-1))
       d=3 대입 → m_l^(5/8) * m_tau^(3/8)

  - (16*pi^2)^2 : 2-loop Weinberg 억압
  - 32*pi^3 = 2*pi * 16*pi^2 : Majorana 위상 통과의 1-loop 위상공간
  - 1/(1+R) : DE 분율 (Phi 매개 질량은 DE 모드만)
  - m_l^((d+2)/(d^2-1)) : 외부 ν chiral flip mass insertion (d=3 → 5/8)
  - m_tau^((d^2-d-3)/(d^2-1)) : 내부 heaviest charged lepton dominance (d=3 → 3/8)
    (SU(d) adjoint Casimir d^2-1 가 멱 분배의 분모, top-Yukawa-like dominance)

결과: m_3 = 51.94 meV (2.8%), m_2 = 8.90 meV (2.8%), m_1 = 0.32 meV.
      Δm²_21 = 7.87e-5 eV² (+4.5%), Δm²_31 = 2.70e-3 eV² (+10.0%).
"""
import math

alpha_s = 0.11789
sin2_tW = 4 * alpha_s**(4/3)
cos2_tW = 1 - sin2_tW
delta = sin2_tW * cos2_tW
D = 3 + delta
d = 3

# 멱지수: SU(d) adjoint Casimir 기반 분배
EXP_OUT = (d + 2) / (d**2 - 1)             # = 5/8 at d=3
EXP_IN = 1.0 - EXP_OUT                     # = 3/8 at d=3

# 하전 렙톤 질량 (MeV)
m_e = 0.51100
m_mu = 105.658
m_tau = 1776.86

# 관측 중성미자 질량 (meV) - NH minimal
m1_obs = 0.0
m2_obs = 8.66        # sqrt(dm21^2)
m3_obs = 50.5        # sqrt(dm31^2 + m1^2)
dm21_sq_obs = 7.53e-5  # eV^2
dm31_sq_obs = 2.453e-3 # eV^2
sum_planck = 120.0   # meV upper bound

print("=" * 78)
print("CE 중성미자 질량: m_l^(5/8) * m_tau^(3/8) (d=3, SU(d) Casimir 멱 분배)")
print("=" * 78)

# -----------------------------------------------------------------------
# I. 무차원 prefactor
# -----------------------------------------------------------------------
loop2 = (16 * math.pi**2)**2
phase_majorana = 32 * math.pi**3
R = 0.38063  # 3계층 바리온 관성
prefactor = delta**4 / (loop2 * phase_majorana * (1 + R))

print(f"\nalpha_s         = {alpha_s}")
print(f"delta           = {delta:.6f}  (= sin^2 tW * cos^2 tW)")
print(f"delta^4         = {delta**4:.6e}")
print(f"(16 pi^2)^2     = {loop2:.4f}")
print(f"32 pi^3         = {phase_majorana:.4f}")
print(f"1+R             = {1+R:.5f}   (R={R}, 3계층 바리온 관성)")
print(f"prefactor       = {prefactor:.6e}    [dimensionless / mass^0]")
print(f"\n멱지수 (d=3, SU(d) adjoint Casimir d^2-1=8):")
print(f"  외부 m_l 멱  = (d+2)/(d^2-1)     = {EXP_OUT:.4f}  (= 5/8 at d=3)")
print(f"  내부 m_tau 멱 = (d^2-d-3)/(d^2-1) = {EXP_IN:.4f}  (= 3/8 at d=3)")
print(f"  합계         = {EXP_OUT+EXP_IN:.4f}  (질량 1제곱)")

# -----------------------------------------------------------------------
# II. 세대별 예측
# -----------------------------------------------------------------------
def m_nu(ml, mtau=m_tau):
    """Returns m_nu in meV given charged lepton mass in MeV."""
    return prefactor * (ml ** EXP_OUT) * (mtau ** EXP_IN) * 1e9  # MeV -> meV

m_nu_e = m_nu(m_e)
m_nu_mu = m_nu(m_mu)
m_nu_tau = m_nu(m_tau)
sum_nu = m_nu_e + m_nu_mu + m_nu_tau

print("\n" + "=" * 78)
print("II. 세대별 예측")
print("=" * 78)
print(f"  m_nu_e   = m_1 = {m_nu_e:8.4f} meV  (관측 m_1 < ~1 meV, 양립)")
print(f"  m_nu_mu  = m_2 = {m_nu_mu:8.4f} meV  (관측 m_2 = {m2_obs} meV, "
      f"{abs(m_nu_mu-m2_obs)/m2_obs*100:5.2f}%)")
print(f"  m_nu_tau = m_3 = {m_nu_tau:8.4f} meV  (관측 m_3 = {m3_obs} meV, "
      f"{abs(m_nu_tau-m3_obs)/m3_obs*100:5.2f}%)")
print(f"  sum            = {sum_nu:8.4f} meV  "
      f"(Planck < {sum_planck} meV: {'양립' if sum_nu < sum_planck else 'X'})")

# 질량 제곱차
dm21_sq = (m_nu_mu * 1e-3)**2 - (m_nu_e * 1e-3)**2  # eV^2
dm31_sq = (m_nu_tau * 1e-3)**2 - (m_nu_e * 1e-3)**2

print(f"\n질량 제곱차 (직접 도출):")
print(f"  dm21^2 = {dm21_sq:.4e} eV^2  (관측 {dm21_sq_obs:.3e}, "
      f"{(dm21_sq-dm21_sq_obs)/dm21_sq_obs*100:+5.1f}%)")
print(f"  dm31^2 = {dm31_sq:.4e} eV^2  (관측 {dm31_sq_obs:.3e}, "
      f"{(dm31_sq-dm31_sq_obs)/dm31_sq_obs*100:+5.1f}%)")
print(f"  ratio dm31^2/dm21^2 = {dm31_sq/dm21_sq:5.1f}  "
      f"(관측 {dm31_sq_obs/dm21_sq_obs:.1f})")

# -----------------------------------------------------------------------
# III. 비율 검증
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("III. 질량비 (멱 분배의 직접 검증)")
print("=" * 78)

ratio_pred = (m_mu / m_tau) ** EXP_OUT
ratio_obs = m2_obs / m3_obs
print(f"  m2/m3 pred = (m_mu/m_tau)^(5/8) = {ratio_pred:.4f}")
print(f"  m2/m3 obs                       = {ratio_obs:.4f}")
print(f"  차이: {abs(ratio_pred-ratio_obs)/ratio_obs*100:.2f}%")

ratio_em = (m_e / m_mu) ** EXP_OUT
print(f"\n  m1/m2 pred = (m_e/m_mu)^(5/8) = {ratio_em:.5f}")

# -----------------------------------------------------------------------
# IV. 멱지수 5/8 의 첫원리 의미
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("IV. 5/8 멱지수의 SU(d) Casimir 기원 (d=3)")
print("=" * 78)
print(f"""
2-loop Weinberg 다이어그램 (외부 ν 두 개, 내부 charged lepton + Phi 두 개):

  외부 ν chiral flip:       1 power of m_l  (Yukawa Y_l 한 번)
  내부 lepton mass insert:  m_l^? × m_tau^?  (Barr-Zee cascade)

CE의 SU(d) flavor 구조에서 멱 분배는 adjoint Casimir로 결정:
  외부 멱 = (d+2)/(d^2-1) = 5/8   (d=3)
  내부 멱 = (d^2-d-3)/(d^2-1) = 3/8   (d=3)
  합계   = (d^2-1)/(d^2-1) = 1     (질량 1제곱 보존)

(d^2-1)은 SU(d)의 adjoint 표현 차원 = 8 (d=3).
이는 |V_ub| 도출과 동일한 SU(d) 기하학.

내부 스케일이 m_tau인 이유:
  - 2-loop Weinberg 의 internal lepton sum이 heaviest mass에 의해 dominant
  - CKM W-loop 에서 top quark 가 dominant 한 것과 같은 구조
  - Yukawa 결합 Y_l 가 m_l/v_EW 비례이므로 (m_l)^2 / v_EW^2 에서 m_tau 가 우세
""")

# -----------------------------------------------------------------------
# V. 비교: 기존 m_l^1 vs 신 m_l^(5/8) m_tau^(3/8)
# -----------------------------------------------------------------------
print("=" * 78)
print("V. 기존 m_l^1 공식과의 비교")
print("=" * 78)

prefactor_old = delta**4 / (loop2 * phase_majorana * (1 + R))
m_old = lambda ml: prefactor_old * ml * 1e9
m1_old, m2_old, m3_old = m_old(m_e), m_old(m_mu), m_old(m_tau)

print(f"{'세대':6s} {'m_l (MeV)':>11s} {'기존 m_l^1':>13s} {'신 m_l^(5/8)':>16s} "
      f"{'관측':>8s} {'기존 잔차':>11s} {'신 잔차':>10s}")
print("-" * 78)
for name, ml, mo, mn, obs in [
    ("e/1", m_e, m1_old, m_nu_e, m1_obs if m1_obs > 0 else 1.0),
    ("mu/2", m_mu, m2_old, m_nu_mu, m2_obs),
    ("tau/3", m_tau, m3_old, m_nu_tau, m3_obs),
]:
    d_old = abs(mo - obs) / obs * 100
    d_new = abs(mn - obs) / obs * 100
    print(f"{name:6s} {ml:>11.3f} {mo:>13.4f} {mn:>16.4f} {obs:>8.2f}"
          f" {d_old:>10.2f}% {d_new:>9.2f}%")

print(f"\n총 합 비교:")
print(f"  기존: {m1_old+m2_old+m3_old:.2f} meV")
print(f"  신:   {sum_nu:.2f} meV")
print(f"  관측: < {sum_planck} meV (Planck)")

# -----------------------------------------------------------------------
# VI. 최종 정본
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("VI. CE 중성미자 질량 공식 (최종 정본)")
print("=" * 78)
print(f"""
m_nu_l = (delta^4 / [(16*pi^2)^2 * 32*pi^3 * (1+R)])
       * m_l^((d+2)/(d^2-1)) * m_tau^((d^2-d-3)/(d^2-1))

d = 3 (Hodge 자기쌍대) 대입:
  m_nu_l = (delta^4 / [(16*pi^2)^2 * 32*pi^3 * (1+R)]) * m_l^(5/8) * m_tau^(3/8)

자유 매개변수: 0개

예측:
  m_1 = {m_nu_e:.3f} meV  (관측 < 1 meV, 양립)
  m_2 = {m_nu_mu:.3f} meV  (관측 8.66 meV, {abs(m_nu_mu-m2_obs)/m2_obs*100:.1f}%)
  m_3 = {m_nu_tau:.3f} meV  (관측 50.5 meV, {abs(m_nu_tau-m3_obs)/m3_obs*100:.1f}%)
  sum = {sum_nu:.2f} meV   (Planck < 120 meV)
  dm21^2 = {dm21_sq:.3e} eV^2 (관측 7.53e-5, {(dm21_sq-dm21_sq_obs)/dm21_sq_obs*100:+.1f}%)
  dm31^2 = {dm31_sq:.3e} eV^2 (관측 2.453e-3, {(dm31_sq-dm31_sq_obs)/dm31_sq_obs*100:+.1f}%)

계층: 정상(NH)        -> JUNO ~2026 검증
유형: Majorana       -> nEXO/LEGEND ~2030 검증
""")
