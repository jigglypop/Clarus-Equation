"""
CE 비섭동적 R: 바리온 관성의 3계층 구조

바리온(eps^2)이 잔여 차원(delta)에만 피드백하는 것이 아니라,
3층 건물 전체에 피드백한다면?

3층 = 강한 핵력 (SU(3), d=3)
2층 = 약한 핵력 (SU(2), d=2)  
1층 = 전자기력 (U(1), d=1)
잔여 = delta (스칼라장 Phi)
"""
import math

alpha_s = 0.11789
sin2_tW = 4 * alpha_s**(4/3)
cos2_tW = 1 - sin2_tW
delta = sin2_tW * cos2_tW
D = 3 + delta
d = 3

def bootstrap(D_val):
    x = 0.5
    for _ in range(5000):
        x = math.exp(-(1-x)*D_val)
    return x

eps2 = bootstrap(D)
sigma = 1 - eps2

OL_p = 0.68470
OD_p = 0.26070
Ob_p = 0.04930
R_p = OD_p / OL_p

print("=" * 72)
print("바리온 관성의 3계층 구조")
print("=" * 72)

# =====================================================================
# I. 각 층의 기여 분해
# =====================================================================
print("\n" + "=" * 72)
print("I. D_eff의 계층 분해")
print("=" * 72)

# D_eff = 3 + delta = 1 + 2 + delta (U(1) + SU(2) + delta 기여)
# 아니, 더 정확하게:
# D_eff = d + delta 에서 d = 3 은 SU(3)xSU(2)xU(1)의 하강 분할 {3,2,1}
# 각 층의 차원 기여:
# 3층 (SU(3)): 3 색전하 -> 차원 기여 = ?
# 2층 (SU(2)): 2 약한 상태 -> 차원 기여 = ?
# 1층 (U(1)): 1 전하 -> 차원 기여 = ?
#
# 하지만 D_eff = 3 + delta 에서 "3"은 공간 차원이지 게이지 층이 아님.
# d = 3 = N_c (색 전하 = 공간 차원). 이것이 CE의 핵심 동일성.
#
# QCD 요동의 관점:
# R = alpha_s * D_eff = alpha_s * (3 + delta)
# 3 = 3개의 공간 차원 = 3개의 색 자유도
# delta = 잔여 차원 = 전자약 혼합
#
# 바리온 피드백: 바리온은 3개의 쿼크로 구성.
# 각 쿼크는 3색 중 하나를 가짐.
# 바리온의 피드백은 색 구조를 통해 전달됨.

print(f"""
D_eff = d + delta = {d} + {delta:.5f} = {D:.5f}

d = 3 의 내부 구조 (하강 분할 {{3,2,1}}):
  3층: SU(3) 강한 핵력 - 3가지 색
  2층: SU(2) 약한 핵력 - 2가지 상태  
  1층: U(1) 전자기력  - 1가지 전하
  잔여: delta = {delta:.5f} - 스칼라장 Phi

QCD 요동 R = alpha_s * D_eff 에서:
  3층 기여: alpha_s * 3 = {alpha_s*3:.5f}  (전체의 {3/D*100:.1f}%)
  잔여 기여: alpha_s * delta = {alpha_s*delta:.5f}  (전체의 {delta/D*100:.1f}%)
""")

# =====================================================================
# II. 3계층 바리온 피드백
# =====================================================================
print("=" * 72)
print("II. 바리온 피드백의 3계층 구조")
print("=" * 72)

# 바리온(eps^2)이 각 층에 피드백한다면:
# 
# 바리온 = 양성자/중성자 = 3개의 쿼크
# 쿼크는 어떤 층에 참여하는가?
#   3층 (SU(3)): 색전하를 가짐 -> 강하게 참여
#   2층 (SU(2)): 약한 이소스핀 가짐 -> 약하게 참여
#   1층 (U(1)): 전하 가짐 -> 약하게 참여
#   잔여 (delta): 히그스 포탈 -> 간접 참여
#
# 각 층의 결합 세기:
#   3층: alpha_s (강한 결합)
#   2층: alpha_w = alpha_em / sin^2(tW) (약한 결합)
#   1층: alpha_em (전자기 결합)
#   잔여: delta^2 = lambda_HP (히그스 포탈)

alpha_em = 1/129.0  # at M_Z scale
alpha_w = alpha_em / sin2_tW
alpha_1 = alpha_em / cos2_tW  # U(1) hypercharge
lambda_HP = delta**2

print(f"각 층의 결합 상수:")
print(f"  3층 SU(3): alpha_s  = {alpha_s:.5f}")
print(f"  2층 SU(2): alpha_w  = {alpha_w:.5f}")
print(f"  1층 U(1):  alpha_1  = {alpha_1:.5f}")
print(f"  잔여 Phi:  lambda_HP = delta^2 = {lambda_HP:.5f}")

# 바리온 피드백: 각 층에서 바리온이 "관성"으로 되먹임
# 피드백 세기 = eps^2 * (그 층의 결합/전체 결합)
# 전체 결합 = alpha_total = 1/(2*pi)
alpha_total = 1/(2*math.pi)

print(f"\n바리온 피드백 세기 (각 층):")
print(f"  3층: eps^2 * alpha_s/alpha_total  = {eps2:.5f} * {alpha_s/alpha_total:.4f} = {eps2*alpha_s/alpha_total:.6f}")
print(f"  2층: eps^2 * alpha_w/alpha_total  = {eps2:.5f} * {alpha_w/alpha_total:.4f} = {eps2*alpha_w/alpha_total:.6f}")
print(f"  1층: eps^2 * alpha_1/alpha_total  = {eps2:.5f} * {alpha_1/alpha_total:.4f} = {eps2*alpha_1/alpha_total:.6f}")
print(f"  잔여: eps^2 * delta               = {eps2:.5f} * {delta:.5f}   = {eps2*delta:.6f}")

# =====================================================================
# III. R 계산: 각 층별 바리온 피드백 포함
# =====================================================================
print("\n" + "=" * 72)
print("III. 3계층 바리온 피드백을 포함한 R")
print("=" * 72)

# 방법 1: 각 차원이 자기 층의 피드백을 받는다
# d=3 차원은 SU(3) 색 구조. 
# 하강 분할 {3,2,1}에서:
# 차원 1: U(1)에 속함 -> 피드백 = eps^2 * alpha_1/alpha_total
# 차원 2: SU(2)에 속함 -> 피드백 = eps^2 * alpha_w/alpha_total
# 차원 3: SU(3)에 속함 -> 피드백 = eps^2 * alpha_s/alpha_total
# 잔여 delta: Phi에 속함 -> 피드백 = eps^2 * delta

# 더 단순하게: 각 "차원"이 동일한 피드백을 받는다면
# 피드백 = eps^2 * 그 차원의 결합 세기

# 구현 1: 하강 분할별
fb_1 = eps2 * alpha_1 / alpha_total
fb_2 = eps2 * alpha_w / alpha_total  
fb_3 = eps2 * alpha_s / alpha_total
fb_delta = eps2 * delta

R_3layer = alpha_s * (1*(1+fb_1) + 1*(1+fb_2) + 1*(1+fb_3) + delta*(1+fb_delta))

print(f"하강 분할 {{3,2,1}} 바리온 피드백:")
print(f"  1층 (U(1)):  1 * (1 + {fb_1:.6f}) = {1+fb_1:.6f}")
print(f"  2층 (SU(2)): 1 * (1 + {fb_2:.6f}) = {1+fb_2:.6f}")
print(f"  3층 (SU(3)): 1 * (1 + {fb_3:.6f}) = {1+fb_3:.6f}")
print(f"  잔여 (Phi):  {delta:.5f} * (1 + {fb_delta:.6f}) = {delta*(1+fb_delta):.6f}")
print(f"  합계 D_eff(fb) = {(1+fb_1)+(1+fb_2)+(1+fb_3)+delta*(1+fb_delta):.6f}")
print(f"  R = alpha_s * D_eff(fb) = {R_3layer:.6f}")

def show(name, R_val):
    ol = sigma/(1+R_val)
    od = sigma*R_val/(1+R_val)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    dt = math.sqrt(d1**2+d2**2)
    print(f"  {name}: OL={ol:.5f}({d1:.3f}%), OD={od:.5f}({d2:.3f}%), tot={dt:.3f}%")
    return ol, od

show("3계층 피드백", R_3layer)

# 구현 2: 각 차원이 동일한 "자기 층의 결합"에 비례하는 피드백
# 하강 분할 {3,2,1}에서 각 차원 k의 결합 alpha_k
# 차원 1 -> N_1 = 1 -> alpha_1 (hypercharge)
# 차원 2 -> N_2 = 2 -> alpha_w
# 차원 3 -> N_3 = 3 -> alpha_s
# 하지만 SU(N)은 N개의 차원을 "한꺼번에" 담당한다

# 더 자연스럽게: 각 게이지 그룹이 자기 차원 수만큼 기여
# SU(3): 3차원 기여, 피드백 = eps^2 * alpha_s
# SU(2): 2차원 기여, 피드백 = eps^2 * alpha_w
# U(1):  1차원 기여, 피드백 = eps^2 * alpha_em
# Phi:   delta 기여, 피드백 = eps^2 * delta

print(f"\n게이지 그룹별 바리온 피드백 (차원 x 결합):")
R_gauge = alpha_s * (
    3 * (1 + eps2 * alpha_s) +
    # 아니, d=3 은 공간 차원이지 게이지 차원이 아님
    # R = alpha_s * D_eff 에서 alpha_s는 이미 QCD 결합
    # d=3의 의미: "3개의 공간 방향으로 QCD 요동이 전파"
    # 각 방향에서 바리온 피드백 = eps^2 * (그 방향의 유효 결합)
    0  # placeholder
)

# 핵심 재고: d=3은 공간 차원이다.
# 하강 분할은 게이지 구조이다.
# 바리온 피드백은 어느 쪽을 통해 전달되나?
#
# 바리온 = QCD bound state -> QCD(강한 핵력)를 통해 피드백
# 바리온 = 전하를 가짐 -> 전자기력을 통해서도 피드백
# 바리온 = 약한 이소스핀 -> 약한 핵력을 통해서도 피드백
# 바리온 = 질량을 가짐 -> 히그스 포탈을 통해서도 피드백

# 각 "힘"의 기여는 결합 상수의 크기에 비례:
# 전체 피드백 = eps^2 * (alpha_s + alpha_w + alpha_em + lambda_HP)
total_coupling = alpha_s + alpha_w + alpha_em + lambda_HP
fb_total_per_dim = eps2 * total_coupling

print(f"전체 결합 합: alpha_s + alpha_w + alpha_em + lambda_HP")
print(f"  = {alpha_s:.5f} + {alpha_w:.5f} + {alpha_em:.5f} + {lambda_HP:.5f}")
print(f"  = {total_coupling:.5f}")
print(f"피드백/차원 = eps^2 * total = {fb_total_per_dim:.6f}")

R_total_fb = alpha_s * D * (1 + fb_total_per_dim)
show("전체 결합 피드백", R_total_fb)

# 또는: 1/(2*pi) = alpha_total 이 전체 결합이므로
fb_via_alpha_total = eps2 * alpha_total
R_alpha_total = alpha_s * D * (1 + fb_via_alpha_total)
print(f"\nalpha_total 피드백 = eps^2 * alpha_total = eps^2/(2*pi)")
print(f"  = {eps2:.5f} * {alpha_total:.5f} = {fb_via_alpha_total:.6f}")
show("alpha_total 피드백", R_alpha_total)

# =====================================================================
# IV. 가장 자연스러운 형태: eps^2 * sin^2(theta_W)
# =====================================================================
print("\n" + "=" * 72)
print("IV. 물리적으로 가장 자연스러운 피드백 경로")
print("=" * 72)

# 바리온 피드백의 경로를 다시 생각:
# 
# 바리온 -> QCD 진공 요동 -> DM/DE 분할
# 
# 이 과정에서 전자약 혼합이 개입한다.
# 왜? delta = sin^2(tW) * cos^2(tW)가 D_eff를 결정하므로.
# 
# 바리온이 QCD 진공에 피드백하는 경로:
# (1) 직접: 바리온 -> QCD -> R. 세기 = eps^2 자체
# (2) 전자약 경유: 바리온 -> EW 혼합 -> QCD -> R. 세기 = eps^2 * delta
# (3) 히그스 경유: 바리온 -> Higgs -> Phi -> R. 세기 = eps^2 * lambda_HP = eps^2*delta^2
#
# CE에서 이미 쓰고 있는 것: (2) eps^2 * delta

# 그런데 "3계층이 다 관성이 있다"는 질문의 핵심:
# 바리온은 3개의 힘 모두에 참여한다.
# 각 힘을 통한 피드백:
#   강한 핵력: eps^2 * alpha_s = {eps2*alpha_s}
#   약한 핵력: eps^2 * alpha_w = {eps2*alpha_w}
#   전자기력:  eps^2 * alpha_em = {eps2*alpha_em}
# 합: eps^2 * (alpha_s + alpha_w + alpha_em)

fb_3forces = eps2 * (alpha_s + alpha_w + alpha_em)
R_3forces = alpha_s * D * (1 + fb_3forces)

print(f"3가지 힘을 통한 피드백:")
print(f"  강한 핵력: eps^2 * alpha_s  = {eps2*alpha_s:.6f}")
print(f"  약한 핵력: eps^2 * alpha_w  = {eps2*alpha_w:.6f}")
print(f"  전자기력:  eps^2 * alpha_em = {eps2*alpha_em:.6f}")
print(f"  합:                          = {fb_3forces:.6f}")
print(f"  cf. eps^2 * delta            = {eps2*delta:.6f}")

show("3-force 피드백", R_3forces)

# 비교: 여러 피드백 형태
print(f"\n" + "=" * 72)
print("V. 전체 비교")
print("=" * 72)

candidates = [
    ("Planck 2020", R_p),
    ("LO (no feedback)", alpha_s * D),
    ("eps^2*delta (delta only)", alpha_s*D*(1+eps2*delta)),
    ("eps^2*alpha_s (QCD only)", alpha_s*D*(1+eps2*alpha_s)),
    ("eps^2*alpha_total (all/2pi)", alpha_s*D*(1+eps2*alpha_total)),
    ("eps^2*sin^2(tW)", alpha_s*D*(1+eps2*sin2_tW)),
    ("eps^2*(as+aw+aem)", R_3forces),
    ("eps^2*(as+aw+aem+lHP)", R_total_fb),
    ("eps^2*D_eff/D_eff (=eps^2)", alpha_s*D*(1+eps2)),
    ("eps^2*(3forces)/D", alpha_s*D*(1+fb_3forces/D)),
    ("NLO resum", alpha_s*D/(1-alpha_s*D/(4*math.pi))),
]

print(f"{'Method':>45} {'R':>9} {'OL%':>8} {'OD%':>8} {'tot%':>8}")
print("-" * 80)
for name, R_val in candidates:
    ol = sigma/(1+R_val)
    od = sigma*R_val/(1+R_val)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    dt = math.sqrt(d1**2+d2**2)
    print(f"{name:>45} {R_val:9.5f} {d1:7.3f}% {d2:7.3f}% {dt:7.3f}%")

# =====================================================================
# VI. sin^2(tW)의 역할
# =====================================================================
print(f"\n" + "=" * 72)
print("VI. delta vs sin^2(tW) vs alpha_total")
print("=" * 72)

# delta = sin^2(tW) * cos^2(tW) = sin^2(tW) - sin^4(tW)
# delta 는 "전자약 혼합의 세기"
# sin^2(tW) 는 "약한 혼합각"
# alpha_total = 1/(2pi) 는 "전체 결합"

# delta가 최적인 이유:
# delta = sin^2(tW) * cos^2(tW) = (1/4)*sin^2(2*tW)
# 이것은 SU(2)xU(1) -> U(1)_em 깨짐에서의 "혼합 진폭"
# 바리온 피드백이 전자약 혼합 진폭에 비례하는 것은 자연스럽다.
# 왜? 바리온의 QCD 요동 -> 전자약 혼합을 통해 -> DM/DE 분할에 전달

sin2_2tW = 4 * sin2_tW * cos2_tW  # = 4*delta
print(f"delta = sin^2(tW)*cos^2(tW)  = {delta:.5f}")
print(f"sin^2(2*tW) = 4*delta        = {sin2_2tW:.5f}")
print(f"sin^2(tW)                     = {sin2_tW:.5f}")
print(f"alpha_total = 1/(2pi)         = {alpha_total:.5f}")
print(f"alpha_s + alpha_w + alpha_em  = {alpha_s+alpha_w+alpha_em:.5f}")

# alpha_s + alpha_w + alpha_em vs alpha_total?
print(f"\nalpha_s + alpha_w + alpha_em = {alpha_s+alpha_w+alpha_em:.5f}")
print(f"alpha_total = 1/(2pi)         = {alpha_total:.5f}")
print(f"차이: {(alpha_s+alpha_w+alpha_em)/alpha_total:.4f} * alpha_total")

# =====================================================================
# VII. 결론: 3계층 관성의 의미
# =====================================================================
print(f"\n" + "=" * 72)
print("VII. 결론")
print("=" * 72)

print(f"""
질문: "3계층이 다 관성이 있지 않나?"

답: 맞다. 하지만 CE에서 이미 delta가 그 역할을 한다.

delta = sin^2(theta_W) * cos^2(theta_W)
      = 전자약 혼합 진폭의 제곱
      = SU(2) x U(1) 깨짐의 "혼합 세기"

delta는 3계층의 결합을 모두 반영하는 양이다:
  sin^2(tW) = 4*alpha_s^(4/3)  <- alpha_s에서 유도
  cos^2(tW) = 1 - sin^2(tW)    <- 나머지
  delta = 이 둘의 곱           <- 두 세계의 "교차"

바리온 피드백 = eps^2 * delta 는 사실상:
  "바리온이 전자약 혼합 진폭을 통해 QCD 진공에 되먹임"
  = "바리온이 3계층 전체를 관통하는 혼합을 통해 되먹임"

구체적으로:
  3층 (SU(3)): alpha_s가 delta를 만드는 데 기여 (sin^2(tW) = 4*alpha_s^(4/3))
  2층 (SU(2)): sin^2(tW)가 SU(2) 혼합각
  1층 (U(1)):  cos^2(tW)가 U(1) 혼합각
  
  delta = sin^2(tW) * cos^2(tW)
        = "2층의 세기" x "1층의 세기"
        = "약력과 전자기력의 교차점"

  그리고 sin^2(tW) 자체가 alpha_s에서 유도되므로,
  delta 안에 3층(강한 핵력)도 내재되어 있다.

따라서 eps^2 * delta 는 이미 "3계층 전체의 관성"이다.

수치 확인:
  eps^2 * delta         = {eps2*delta:.6f}   -> OD 0.08%
  eps^2 * (as+aw+aem)   = {fb_3forces:.6f}   -> OD {abs(sigma*R_3forces/(1+R_3forces)-OD_p)/OD_p*100:.2f}%
  eps^2 * alpha_total   = {fb_via_alpha_total:.6f}   -> OD {abs(sigma*R_alpha_total/(1+R_alpha_total)-OD_p)/OD_p*100:.2f}%

delta가 최적인 이유: delta는 "혼합 진폭"이지 "결합 상수의 합"이 아니다.
피드백은 각 힘이 독립적으로 작용하는 것이 아니라,
3계층이 하나로 혼합된 지점(전자약 혼합)을 통해 전달된다.

비유: 3층 건물의 관성은 각 층의 무게를 더하는 것이 아니라,
      층과 층을 잇는 접합부(혼합각)의 강도로 결정된다.
""")
