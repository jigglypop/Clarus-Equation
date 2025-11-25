import pandas as pd

def compare_improvements():
    print("=== SFE 성능 개선 정량 분석 ===\n")
    
    # 1. SCQE (Part8 기하학적 제어) 효과 분석
    try:
        df_base = pd.read_csv('sfe_core/geometric_output.csv')
        df_scqe = pd.read_csv('sfe_core/scqe_result.csv')
        
        # 마지막 스텝의 Suppression Factor (생존율) 비교
        final_base = df_base['Suppression'].iloc[-1]
        final_scqe = df_scqe['Suppression'].iloc[-1]
        
        scqe_gain = (final_scqe - final_base) / final_base * 100.0
        
        print(f"[1] 기하학적 제어(SCQE) 효과 (이론적 한계 돌파)")
        print(f"  - 기존 방식 생존율: {final_base:.4f}")
        print(f"  - SCQE 방식 생존율: {final_scqe:.4f}")
        print(f"  => 개선율: +{scqe_gain:.2f}% (생존 시간 연장)")
        
    except Exception as e:
        print(f"SCQE 데이터 분석 실패: {e}")

    print("\n" + "-"*40 + "\n")

    # 2. IBM Fez 시뮬레이션 (펄스 최적화) 효과 분석
    try:
        df_fez = pd.read_csv('examples/results/ibm_fez_sweep.csv')
        
        # Error Rate = 1.0 - Score (Clipped 0~1)
        df_fez['Error_UDD'] = (1.0 - df_fez['UDD_Score'].clip(0, 1))
        df_fez['Error_SFE'] = (1.0 - df_fez['SFE_Score'].clip(0, 1))
        
        # 유효한 구간(UDD가 완전히 망가지지 않은 구간)에서의 평균 개선율
        valid_mask = df_fez['Error_UDD'] < 0.99
        df_valid = df_fez[valid_mask].copy()
        
        if len(df_valid) > 0:
            df_valid['Reduction'] = (df_valid['Error_UDD'] - df_valid['Error_SFE']) / df_valid['Error_UDD'] * 100.0
            avg_red = df_valid['Reduction'].mean()
            max_red = df_valid['Reduction'].max()
            
            print(f"[2] IBM Fez 시뮬레이션 (실용적 오류 감소)")
            print(f"  - 평균 오류 감소율: {avg_red:.2f}%")
            print(f"  - 최대 오류 감소율: {max_red:.2f}% (특정 노이즈 구간)")
            
            # 극한 환경(High Noise > 0.2)에서의 생존 여부
            high_noise = df_fez[df_fez['NoiseAmp'] >= 0.2]
            udd_survived = (high_noise['UDD_Score'] > 0.1).sum()
            sfe_survived = (high_noise['SFE_Score'] > 0.1).sum()
            
            print(f"  - 극한 환경(Noise>=0.2) 생존 케이스: UDD {udd_survived}건 vs SFE {sfe_survived}건")
            
    except Exception as e:
        print(f"Fez 데이터 분석 실패: {e}")

if __name__ == "__main__":
    compare_improvements()

