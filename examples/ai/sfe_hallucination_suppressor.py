import numpy as np
import matplotlib.pyplot as plt

# CE-LLM Simulation: "Reality Stone"
# 가상의 임베딩 공간에서 토큰의 궤적(Curvature)을 분석

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class RealityStoneEngine:
    """
    CE 기반 환각 억제 엔진 (v2: 다중 스케일 곡률 + 적응형 억제)
    
    특징:
    - 1차 곡률 (방향 변화) + 2차 곡률 (가속도/굴곡) 동시 고려
    - 문맥 윈도우: 최근 N개 토큰 궤적 기반
    - 적응형 lambda: 곡률 분포에 따라 동적 조절
    - 소프트 억제: 시그모이드 기반 부드러운 전이
    """
    def __init__(self, dimension=128, lambda_param=5.0, curvature_threshold=0.5,
                 use_second_order=True, adaptive_lambda=True, soft_suppression=True):
        self.dim = dimension
        self.context_history = []  # 문맥 히스토리 (최근 N개)
        self.max_history = 5
        self.curvature_threshold = curvature_threshold
        self.lambda_param = lambda_param
        self.use_second_order = use_second_order
        self.adaptive_lambda = adaptive_lambda
        self.soft_suppression = soft_suppression
        
    def add_context(self, vec):
        """문맥 히스토리에 벡터 추가"""
        self.context_history.append(vec.copy())
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)
    
    def compute_first_order_curvature(self, prev_vec, curr_vec, next_vec):
        """1차 곡률: 방향 변화 (1 - cos theta)"""
        v1 = curr_vec - prev_vec
        v2 = next_vec - curr_vec
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-9 or norm_v2 < 1e-9:
            return 0.0
        
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return 1.0 - cos_theta
    
    def compute_second_order_curvature(self, prev_prev_vec, prev_vec, curr_vec, next_vec):
        """2차 곡률: 곡률의 변화율 (굴곡/가속도)"""
        # 1차 미분 (속도)
        v1 = prev_vec - prev_prev_vec
        v2 = curr_vec - prev_vec
        v3 = next_vec - curr_vec
        
        # 2차 미분 (가속도)
        a1 = v2 - v1
        a2 = v3 - v2
        
        norm_a1 = np.linalg.norm(a1)
        norm_a2 = np.linalg.norm(a2)
        
        if norm_a1 < 1e-9 or norm_a2 < 1e-9:
            return 0.0
        
        # 가속도 방향 변화
        cos_theta = np.dot(a1, a2) / (norm_a1 * norm_a2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # 가속도 크기 변화도 고려
        accel_magnitude = (norm_a1 + norm_a2) / 2.0
        return (1.0 - cos_theta) * min(1.0, accel_magnitude)
    
    def compute_trajectory_smoothness(self, next_vec):
        """궤적 전체의 부드러움 점수 (낮을수록 좋음)"""
        if len(self.context_history) < 2:
            return 0.0
        
        # 전체 궤적에 next_vec 추가했을 때의 총 곡률
        trajectory = self.context_history + [next_vec]
        total_curvature = 0.0
        
        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i-1]
            v2 = trajectory[i+1] - trajectory[i]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 1e-9 and norm_v2 > 1e-9:
                cos_t = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_t = np.clip(cos_t, -1.0, 1.0)
                total_curvature += 1.0 - cos_t
        
        return total_curvature / max(1, len(trajectory) - 2)
    
    def compute_combined_curvature(self, prev_vec, curr_vec, next_vec, prev_prev_vec=None):
        """통합 곡률: 1차 + 2차 + 궤적 부드러움"""
        # 1차 곡률 (가중치 1.0)
        k1 = self.compute_first_order_curvature(prev_vec, curr_vec, next_vec)
        
        # 2차 곡률 (가중치 0.5)
        k2 = 0.0
        if self.use_second_order and prev_prev_vec is not None:
            k2 = self.compute_second_order_curvature(prev_prev_vec, prev_vec, curr_vec, next_vec)
        
        # 궤적 부드러움 (가중치 0.3)
        k_traj = self.compute_trajectory_smoothness(next_vec)
        
        # 가중 합산
        combined = k1 * 1.0 + k2 * 0.5 + k_traj * 0.3
        return combined, k1, k2, k_traj
    
    def sigmoid_suppression(self, curvature, threshold, steepness=10.0):
        """소프트 억제: 시그모이드 기반 부드러운 전이"""
        x = steepness * (curvature - threshold)
        return 1.0 / (1.0 + np.exp(-x))
    
    def compute_adaptive_lambda(self, curvatures):
        """적응형 lambda: 곡률 분포 기반 동적 조절"""
        if len(curvatures) == 0:
            return self.lambda_param
        
        mean_curv = np.mean(curvatures)
        std_curv = np.std(curvatures) + 1e-9
        
        # 곡률 분포가 넓으면 (환각 후보가 많으면) lambda 증가
        # 분포가 좁으면 (대부분 비슷하면) lambda 감소
        adaptive_factor = 1.0 + std_curv / (mean_curv + 1e-9)
        return self.lambda_param * adaptive_factor

    def apply_suppression(self, logits, candidates_vecs, prev_vec, curr_vec, prev_prev_vec=None):
        """
        환각 억제 적용
        
        Args:
            logits: [batch_size, vocab_size] 원본 로짓
            candidates_vecs: [vocab_size, dim] 후보 토큰 임베딩
            prev_vec: 이전 문맥 벡터
            curr_vec: 현재 문맥 벡터
            prev_prev_vec: 이전이전 문맥 벡터 (2차 곡률용, optional)
        
        Returns:
            억제된 로짓
        """
        batch_size, vocab_size = logits.shape
        suppression_field = np.zeros(vocab_size)
        curvatures = []
        details = []
        
        # 1단계: 모든 후보의 곡률 계산
        for i in range(vocab_size):
            combined, k1, k2, k_traj = self.compute_combined_curvature(
                prev_vec, curr_vec, candidates_vecs[i], prev_prev_vec
            )
            curvatures.append(combined)
            details.append((k1, k2, k_traj))
        
        curvatures = np.array(curvatures)
        
        # 2단계: 적응형 lambda 계산
        effective_lambda = self.lambda_param
        if self.adaptive_lambda:
            effective_lambda = self.compute_adaptive_lambda(curvatures)
        
        # 3단계: 억제량 계산
        for i in range(vocab_size):
            r = curvatures[i]
            
            if self.soft_suppression:
                # 소프트 억제: 시그모이드 기반
                suppression_weight = self.sigmoid_suppression(r, self.curvature_threshold)
                suppression_field[i] = effective_lambda * suppression_weight * r
            else:
                # 하드 억제: threshold 초과분만
                excess = max(0.0, r - self.curvature_threshold)
                suppression_field[i] = effective_lambda * (excess ** 2)
        
        return logits - suppression_field
    
    def get_curvature_report(self, candidates_vecs, prev_vec, curr_vec, prev_prev_vec=None):
        """상세 곡률 리포트 생성"""
        report = []
        for i in range(len(candidates_vecs)):
            combined, k1, k2, k_traj = self.compute_combined_curvature(
                prev_vec, curr_vec, candidates_vecs[i], prev_prev_vec
            )
            report.append({
                'index': i,
                'combined': combined,
                'first_order': k1,
                'second_order': k2,
                'trajectory': k_traj,
            })
        return report


def apply_reality_stone(logits, embedding_matrix, prev_vec, curr_vec, 
                        prev_prev_vec=None, lambda_param=5.0, curvature_threshold=0.5,
                        use_second_order=True, adaptive_lambda=True, soft_suppression=True):
    """
    CE Reality Stone 환각 억제 적용 (v2)
    
    Args:
        logits: [batch_size, vocab_size] 원본 로짓
        embedding_matrix: [vocab_size, dim] 토큰 임베딩 행렬
        prev_vec: 이전 문맥 벡터
        curr_vec: 현재 문맥 벡터
        prev_prev_vec: 이전이전 문맥 벡터 (2차 곡률용)
        lambda_param: 억제 강도
        curvature_threshold: 곡률 임계값
        use_second_order: 2차 곡률 사용 여부
        adaptive_lambda: 적응형 lambda 사용 여부
        soft_suppression: 소프트 억제 사용 여부
    
    Returns:
        억제된 로짓
    """
    dim = embedding_matrix.shape[1]
    engine = RealityStoneEngine(
        dimension=dim, 
        lambda_param=lambda_param, 
        curvature_threshold=curvature_threshold,
        use_second_order=use_second_order,
        adaptive_lambda=adaptive_lambda,
        soft_suppression=soft_suppression,
    )
    return engine.apply_suppression(logits, embedding_matrix, prev_vec, curr_vec, prev_prev_vec)

def simulate_hallucination_suppression():
    print("=" * 60)
    print("CE Reality Stone Engine v2 - Hallucination Suppression Demo")
    print("=" * 60)
    
    # 가상의 단어장 (Vocab)
    vocab = ["Logical", "Therefore", "Thus", "True", "Fact",
             "Suddenly", "Alien", "Chocolate", "Fly", "Blue"]
    
    # 임베딩 공간 설정 (3D for visualization)
    embeddings = np.zeros((10, 3))
    # 논리 그룹: +X 방향 (문맥 흐름 유지)
    embeddings[0] = np.array([2.0, 0.2, 0.0])   # Logical
    embeddings[1] = np.array([1.9, 0.3, 0.0])   # Therefore
    embeddings[2] = np.array([2.1, 0.1, 0.0])   # Thus
    embeddings[3] = np.array([1.8, 0.25, 0.0])  # True
    embeddings[4] = np.array([1.95, 0.15, 0.0]) # Fact
    # 환각 그룹: Y,Z 방향으로 튐
    embeddings[5] = np.array([1.0, 1.5, 0.0])   # Suddenly
    embeddings[6] = np.array([0.5, 0.5, 1.5])   # Alien
    embeddings[7] = np.array([1.0, -1.0, 0.5])  # Chocolate
    embeddings[8] = np.array([0.0, 1.0, 1.0])   # Fly
    embeddings[9] = np.array([1.0, 0.0, 2.0])   # Blue
    
    # 문맥 히스토리: 3개 토큰의 궤적 (일관된 +X 방향)
    prev_prev_vec = np.array([-0.5, 0.0, 0.0])
    prev_vec = np.array([0.0, 0.0, 0.0])
    curr_vec = np.array([1.0, 0.1, 0.0])
    
    # 원본 로짓: 환각 토큰(Suddenly, Alien)이 비정상적으로 높음
    raw_logits = np.array([[2.0, 2.5, 1.0, 1.0, 1.0, 5.0, 4.0, 1.0, 1.0, 1.0]])
    
    print("\n[1] Before CE Suppression (Standard LLM)")
    print("-" * 40)
    probs = softmax(raw_logits)[0]
    top_idx = np.argmax(probs)
    print(f"  Selected: '{vocab[top_idx]}' (Prob: {probs[top_idx]:.4f})")
    print(f"  Status: HALLUCINATION - Model chose context-breaking token")
    
    # v1: 기본 억제 (1차 곡률만, 하드 threshold)
    print("\n[2] CE v1: Basic Suppression (1st-order only)")
    print("-" * 40)
    suppressed_v1 = apply_reality_stone(
        raw_logits, embeddings, prev_vec, curr_vec,
        prev_prev_vec=None,
        lambda_param=10.0,
        curvature_threshold=0.3,
        use_second_order=False,
        adaptive_lambda=False,
        soft_suppression=False,
    )
    probs_v1 = softmax(suppressed_v1)[0]
    top_v1 = np.argmax(probs_v1)
    print(f"  Selected: '{vocab[top_v1]}' (Prob: {probs_v1[top_v1]:.4f})")
    
    # v2: 정교한 억제 (2차 곡률 + 적응형 + 소프트)
    print("\n[3] CE v2: Advanced Suppression (2nd-order + adaptive + soft)")
    print("-" * 40)
    suppressed_v2 = apply_reality_stone(
        raw_logits, embeddings, prev_vec, curr_vec,
        prev_prev_vec=prev_prev_vec,
        lambda_param=10.0,
        curvature_threshold=0.3,
        use_second_order=True,
        adaptive_lambda=True,
        soft_suppression=True,
    )
    probs_v2 = softmax(suppressed_v2)[0]
    top_v2 = np.argmax(probs_v2)
    print(f"  Selected: '{vocab[top_v2]}' (Prob: {probs_v2[top_v2]:.4f})")
    
    # 상세 곡률 리포트
    print("\n[4] Detailed Curvature Analysis")
    print("-" * 40)
    engine = RealityStoneEngine(
        dimension=3,
        use_second_order=True,
        adaptive_lambda=True,
        soft_suppression=True,
    )
    report = engine.get_curvature_report(embeddings, prev_vec, curr_vec, prev_prev_vec)
    
    print(f"  {'Token':<12} {'Combined':>10} {'1st-Order':>10} {'2nd-Order':>10} {'Status':<15}")
    print("  " + "-" * 58)
    for i, r in enumerate(report):
        status = "LOGICAL" if r['combined'] < 0.5 else "HALLUCINATION"
        print(f"  {vocab[i]:<12} {r['combined']:>10.4f} {r['first_order']:>10.4f} {r['second_order']:>10.4f} {status:<15}")
    
    # 확률 비교표
    print("\n[5] Probability Comparison")
    print("-" * 40)
    print(f"  {'Token':<12} {'Original':>10} {'CE v1':>10} {'CE v2':>10} {'Suppression':>12}")
    print("  " + "-" * 54)
    for i, word in enumerate(vocab):
        suppression = probs[i] - probs_v2[i]
        print(f"  {word:<12} {probs[i]:>10.4f} {probs_v1[i]:>10.4f} {probs_v2[i]:>10.4f} {suppression:>+12.4f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 왼쪽: 확률 비교
    x = np.arange(len(vocab))
    width = 0.25
    axes[0].bar(x - width, probs, width, label='Original (Hallucinating)', color='red', alpha=0.6)
    axes[0].bar(x, probs_v1, width, label='CE v1 (Basic)', color='orange', alpha=0.7)
    axes[0].bar(x + width, probs_v2, width, label='CE v2 (Advanced)', color='green', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(vocab, rotation=45, ha='right')
    axes[0].set_ylabel('Token Probability')
    axes[0].set_title('Hallucination Suppression Comparison')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 오른쪽: 곡률 분석
    curvatures = [r['combined'] for r in report]
    colors = ['green' if c < 0.5 else 'red' for c in curvatures]
    axes[1].barh(vocab, curvatures, color=colors, alpha=0.7)
    axes[1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[1].set_xlabel('Combined Curvature')
    axes[1].set_title('Curvature Analysis (Green=Logical, Red=Hallucination)')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ce_llm_suppression_v2.png', dpi=150)
    print(f"\n  Saved: 'ce_llm_suppression_v2.png'")
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Original selection:  '{vocab[top_idx]}' (HALLUCINATION)")
    print(f"  CE v1 selection:    '{vocab[top_v1]}'")
    print(f"  CE v2 selection:    '{vocab[top_v2]}'")
    halluc_suppression = (probs[5] + probs[6]) - (probs_v2[5] + probs_v2[6])
    print(f"  Hallucination tokens (Suddenly+Alien) suppressed by: {halluc_suppression:.4f} ({halluc_suppression/((probs[5]+probs[6])+1e-9)*100:.1f}%)")

if __name__ == "__main__":
    simulate_hallucination_suppression()

