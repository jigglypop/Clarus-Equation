import numpy as np
import matplotlib.pyplot as plt

class ProteinFolder:
    def __init__(self, length=40):
        self.length = length
        # 초기 상태: 완전 일직선
        self.positions = np.zeros((length, 2))
        for i in range(length):
            self.positions[i] = [i, 0]
            
    def get_radius_of_gyration(self, pos):
        # 회전 반경 (얼마나 퍼져있는가?)
        center = np.mean(pos, axis=0)
        rg_sq = np.mean(np.sum((pos - center)**2, axis=1))
        return np.sqrt(rg_sq)

    def get_contacts(self, pos):
        # 비공유 결합 수 (Energy = -Contacts)
        # 거리가 1.1 이하인 비인접 쌍의 수
        contacts = 0
        for i in range(self.length):
            for j in range(i+2, self.length): # 인접 입자 제외
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < 1.1:
                    contacts += 1
        return contacts

    def mutate(self, current_pos):
        # Pivot Move (단백질 구조 변경)
        new_pos = current_pos.copy()
        pivot = np.random.randint(1, self.length-1)
        angle = np.random.choice([90, -90]) * np.pi / 180.0
        
        # 회전 행렬
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, -s), (s, c)))
        
        # Pivot 이후의 모든 점을 회전
        part_to_rotate = new_pos[pivot:] - new_pos[pivot]
        rotated_part = np.dot(part_to_rotate, R.T)
        new_pos[pivot:] = rotated_part + new_pos[pivot]
        
        return new_pos

    def run_simulation(self, mode='random', steps=1000):
        history = []
        current_pos = self.positions.copy()
        current_rg = self.get_radius_of_gyration(current_pos)
        
        for t in range(steps):
            new_pos = self.mutate(current_pos)
            
            # 물리적 충돌 체크 (Self-avoiding)
            # 간단화: 너무 가까우면 기각
            valid = True
            # (속도를 위해 충돌 체크는 약식으로 하거나 생략, 여기선 Rg 중심으로 봄)
            
            new_rg = self.get_radius_of_gyration(new_pos)
            new_contacts = self.get_contacts(new_pos)
            
            # 에너지 정의
            # Random: Contact 많으면 좋음 (접힘)
            # SFE: Contact + Suppression Field (Rg가 작아야 함)
            
            E_old_phys = -self.get_contacts(current_pos)
            E_new_phys = -new_contacts
            
            if mode == 'sfe':
                # SFE: 공간 억제장 (펼쳐진 상태 억압)
                # E_total = E_phys + lambda * Rg
                lambda_sfe = 2.0
                E_old = E_old_phys + lambda_sfe * current_rg
                E_new = E_new_phys + lambda_sfe * new_rg
            else:
                E_old = E_old_phys
                E_new = E_new_phys
            
            # Metropolis
            delta_E = E_new - E_old
            
            if delta_E < 0:
                accept = True
            else:
                prob = np.exp(-delta_E / 1.0) # Temp = 1.0
                accept = np.random.rand() < prob
                
            if accept:
                current_pos = new_pos
                current_rg = new_rg
            
            # 기록: "얼마나 뭉쳤는가(Rg)"를 추적
            history.append(current_rg)
            
        return history, current_pos

def run_comparison():
    print("Simulating Protein Folding: Random Walk vs SFE Funneling...")
    
    steps = 3000
    folder = ProteinFolder(length=50)
    
    # 1. Random Search
    print("1. Running Random Search...")
    rg_random, pos_random = folder.run_simulation(mode='random', steps=steps)
    
    # 2. SFE Guided Search
    print("2. Running SFE Guided Search...")
    rg_sfe, pos_sfe = folder.run_simulation(mode='sfe', steps=steps)
    
    print(f"\nFinal Compactness (Radius of Gyration - Lower is Better):")
    print(f"Random: {rg_random[-1]:.2f}")
    print(f"SFE   : {rg_sfe[-1]:.2f}")
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    # 그래프 1: 수렴 속도
    plt.subplot(1, 2, 1)
    plt.plot(rg_random, label='Random Search', color='gray', alpha=0.5)
    plt.plot(rg_sfe, label='SFE Suppression Field', color='red', linewidth=2)
    plt.title('Folding Speed (Convergence)', fontsize=12)
    plt.ylabel('Radius of Gyration (Size)', fontsize=10)
    plt.xlabel('Time Steps', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 2: 최종 구조
    plt.subplot(1, 2, 2)
    plt.plot(pos_random[:,0], pos_random[:,1], 'o-', label='Random', color='gray', alpha=0.3)
    plt.plot(pos_sfe[:,0], pos_sfe[:,1], 'o-', label='SFE (Compact)', color='red')
    plt.title('Final Structure', fontsize=12)
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('sfe_protein_folding_v2.png')
    print("Saved 'sfe_protein_folding_v2.png'")

if __name__ == "__main__":
    np.random.seed(123)
    run_comparison()
