import numpy as np
import matplotlib.pyplot as plt

"""
CE Condensed Matter Analogy Check
==================================
이 스크립트는 CE 이론의 질량 억압 효과(m_eff = m0 * (1 - epsilon))를
응집물질물리학의 폴라론(Polaron) 유효 질량 모델과 비교 검증합니다.

1. Fröhlich Polaron Model (Large Polaron)
   m_pol = m_band * (1 + alpha/6 + ...) -> 질량 증가 모델
   
   * CE는 '질량 감소(Suppression)'를 주장하므로, 
     이는 전형적인 폴라론 모델과는 부호가 반대입니다.
     즉, CE는 입자가 매질로부터 에너지를 '빌려오는' (Negative coupling)
     특수한 형태의 상호작용을 시사합니다.

2. 검증 목표:
   실험적으로 관측된 '변칙적 질량 감소' 사례(예: 특정 메타물질이나 액체 헬륨 내의 불순물)가
   CE 예측식과 정량적으로 매칭될 수 있는지 확인합니다.
"""

def ce_effective_mass(m0, epsilon):
    """CE 예측: m_eff = m0 * (1 - epsilon)"""
    return m0 * (1 - epsilon)

def frohlich_polaron_mass(m0, alpha):
    """Fröhlich 예측: m_eff = m0 * (1 + alpha/6) (alpha < 1)"""
    return m0 * (1 + alpha/6.0)

def check_compatibility():
    print("[CE vs Condensed Matter Models]")
    
    # 1. CE epsilon 범위 (0 ~ 0.5)
    eps_range = np.linspace(0, 0.5, 50)
    m_sfe = ce_effective_mass(1.0, eps_range)
    
    # 2. Polaron alpha (일반적으로 양수)
    # 하지만 CE 효과를 내려면 alpha가 '음수'여야 함.
    # Effective 'negative' coupling constant alpha_eff 유도
    # 1 - epsilon = 1 + alpha_eff/6  => alpha_eff = -6 * epsilon
    
    alpha_eff = -6 * eps_range
    
    print(f"CE epsilon=0.37 (Dark Energy) corresponds to Polaron alpha_eff = {-6*0.37:.2f}")
    print("-> 이는 매우 강한 '반발적(Repulsive)' 상호작용을 의미함.")
    print("-> 일반적인 전자-포논 상호작용(인력)과는 정반대 메커니즘 필요.")
    
    # 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(eps_range, m_sfe, label='CE Theory (Mass Suppression)', color='blue', linewidth=2)
    plt.plot(eps_range, ce_effective_mass(1.0, 0) * np.ones_like(eps_range), 'k--', label='Bare Mass (m0)')
    
    plt.title("Effective Mass: CE vs Standard Models")
    plt.xlabel("Coupling Strength (Epsilon / |Alpha|)")
    plt.ylabel("Effective Mass Ratio (m_eff / m0)")
    plt.grid(True)
    plt.legend()
    plt.annotate('Epsilon ~ 0.37\n(Cosmic Value)', xy=(0.37, 0.63), xytext=(0.2, 0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig('examples/ce_vs_polaron.png')
    print("Graph saved to examples/ce_vs_polaron.png")

if __name__ == "__main__":
    check_compatibility()

