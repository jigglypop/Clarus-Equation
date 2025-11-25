use super::manifold::{Manifold, SuppressionManifold};
use std::sync::Arc;

/// Part8 통합 이론을 구현하는 기하학적 엔진
/// x_new = e^(-R(x)) * exp_x(-eta * nabla_g Phi)
pub struct GeometricEngine {
    pub manifold: Arc<SuppressionManifold>,
    pub eta: f64, // 학습률 or 확산 계수
}

impl GeometricEngine {
    pub fn new(dim: usize) -> Self {
        // 테스트용 가상 필드: 원점에서 멀어질수록 억압장이 강해짐 (Potential well)
        let phi_field = Box::new(|x: &[f64]| -> f64 {
            let r2: f64 = x.iter().map(|v| v*v).sum();
            0.5 * r2 // Phi = 1/2 r^2 (Harmonic oscillator potential)
        });
        
        Self {
            manifold: Arc::new(SuppressionManifold::new(phi_field, dim)),
            eta: 0.1,
        }
    }

    /// SFE 통합 방정식의 단일 스텝 실행
    /// x: 현재 위치 (상태)
    /// p: 현재 운동량 (또는 의도)
    pub fn step(&self, x: &[f64], p: &[f64], dt: f64) -> (Vec<f64>, f64) {
        // 1. 곡률 R(x) 계산
        let r = self.manifold.ricci_scalar(x);
        
        // 2. 억압 계수 (Suppression Factor)
        // 곡률이 클수록(R > 0) 이동이 억제됨. Part8 핵심: e^(-R)
        let suppression = (-r).exp();
        
        // 3. 기하학적 구배 (Gradient Flow)
        // 여기서는 운동량 p가 구배 역할을 한다고 가정하거나, 별도의 포텐셜 구배를 사용
        // 식: exp_x(-eta * nabla_g Phi) -> 여기서는 p를 접벡터로 사용
        
        // p_eff = suppression * p (억압된 운동량)
        let p_eff: Vec<f64> = p.iter().map(|v| v * suppression).collect();
        
        // 4. Exponential Map (기하학적 이동)
        // 평탄한 공간에서는 x + v*dt 이지만, 굽은 공간에서는 측지선을 따라 이동
        let x_new = self.manifold.exp_map(x, &p_eff, dt);
        
        (x_new, suppression)
    }
    
    /// [NEW] Part8.1 통합 라그랑지안 계산
    /// L = R / 16pi G + 1/2 (del R)^2 - V(R)
    /// G_N은 단위계 상 1로 가정, V(R)은 단순 2차항 가정
    pub fn calculate_lagrangian(&self, x: &[f64]) -> f64 {
        let r = self.manifold.ricci_scalar(x);
        
        // R의 Gradient 계산 (수치 미분)
        let eps = 1e-6;
        let mut grad_r_sq = 0.0;
        let dim = x.len();
        
        for i in 0..dim {
            let mut x_plus = x.to_vec();
            x_plus[i] += eps;
            let r_plus = self.manifold.ricci_scalar(&x_plus);
            let dr = (r_plus - r) / eps;
            grad_r_sq += dr * dr;
        }
        
        // 라그랑지안 항들
        let term_gravity = r / (16.0 * std::f64::consts::PI);
        let term_kinetic = 0.5 * grad_r_sq;
        let term_potential = 0.5 * r * r; // V(R) ~ 1/2 R^2 (Massive field approximation)
        
        term_gravity + term_kinetic - term_potential
    }
    
    /// [NEW] 전체 억압 에너지 밀도 계산
    /// rho = e^-R(x) * L(x)
    pub fn calculate_suppression_energy_density(&self, x: &[f64]) -> f64 {
        let r = self.manifold.ricci_scalar(x);
        let lagrangian = self.calculate_lagrangian(x);
        (-r).exp() * lagrangian
    }
    
    /// SCQE (Self-Correcting Quantum Element) 시뮬레이션
    /// 큐비트가 로컬 곡률을 감지하여 경로를 수정하는지 테스트
    pub fn run_scqe_simulation(&self, steps: usize) -> Vec<(f64, f64, f64)> {
        let dim = self.manifold.dim;
        // [Fix] 초기 위치를 안전 구역(0.99)으로 이동 (기존 1.0은 0.81에서 시작함)
        let mut x = vec![0.5; dim]; 
        let mut p = vec![-1.0; dim]; // 원점으로 돌아가려는 복원력
        let dt = 0.01;
        
        let mut trajectory = Vec::new();
        
        // 목표 생존율 (Maginot Line) - 조기 경보를 위해 0.98로 상향
        let target_suppression = 0.98;
        
        for _ in 0..steps {
            // 1. 이동 후보 위치 계산 (Predictor)
            let (x_cand, _) = self.step(&x, &p, dt);
            
            // 2. 후보 위치의 안정성 검사 (Safety Check)
            let r_cand = self.manifold.ricci_scalar(&x_cand);
            let supp_cand = (-r_cand).exp();
            
            // 3. 마지노선(90%) 사수 로직 (Hard Constraint)
            // 90% 미만으로 떨어질 위치라면 이동을 거부하고 반사시킴 (Quantum Zeno / Hard Wall)
            if supp_cand < 0.90 {
                // [Reject] 이동 거부 및 운동량 반전
                for i in 0..dim {
                    // 벽에 부딪힌 것처럼 운동량 반전 (Energy dissipation included)
                    // 밖으로 나가려던 성분만 반전
                    if x[i] * p[i] > 0.0 {
                        p[i] *= -0.8; 
                    }
                }
                // 위치는 업데이트하지 않음 (x = x 유지) 또는 약간 후퇴
                // x = x; 
            } else {
                // [Accept] 안전하면 이동 확정
                x = x_cand;
            }
            
            // 현재 상태 재계산 (기록용)
            let r_curr = self.manifold.ricci_scalar(&x);
            let suppression = (-r_curr).exp();
            let energy_density = self.calculate_suppression_energy_density(&x);
            
            // [Fix] 자연적 동역학 (Harmonic Oscillator)
            for i in 0..dim {
                p[i] -= x[i] * dt; 
            }
            
            trajectory.push((r_curr, suppression, energy_density));
            
            // SCQE "Soft Feedback" (추가 안정화)
            // 마지노선 도달 전이라도 0.95 미만이면 부드럽게 복원력 가동
            if suppression < 0.95 {
                let gain = 50.0 * (0.95 - suppression);
                for i in 0..dim {
                    p[i] -= x[i] * gain * dt;
                }
            }
        }
        
        trajectory
    }
}
