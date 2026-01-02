use ndarray::{s, Array1};
use rayon::prelude::*;

const COUPLING_K: f64 = 50.0;
const DT: f64 = 0.01;

pub struct QSFEngine {
    pub phi: Array1<f64>,
    pub dphi: Array1<f64>,
    pub source_j: Array1<f64>,
    pub forces_buffer: Array1<f64>, // 할당 방지를 위한 캐시된 버퍼
    pub mu: f64,
    pub lam: f64,
    pub alpha2: f64, // Curvature suppression coupling (SFE Master Action)
}

impl QSFEngine {
    pub fn new(size: usize) -> Self {
        let mu: f64 = 1.0;
        let lam: f64 = 1.0;
        let alpha2: f64 = 0.0; // Default to 0, enable via Python
        let vacuum_vev = mu / lam.sqrt();
        let phi = Array1::from_elem(size, vacuum_vev);
        let dphi = Array1::zeros(size);
        let mut source_j = Array1::zeros(size);
        let forces_buffer = Array1::zeros(size);

        // 국소화된 소스 물질 생성
        let mid = size / 2;
        if size > 20 {
            let range = 10;
            let mut slice = source_j.slice_mut(s![mid - range..mid + range]);
            slice.fill(-5.0);
        }
        QSFEngine {
            phi,
            dphi,
            source_j,
            forces_buffer,
            mu,
            lam,
            alpha2,
        }
    }

    #[inline(always)]
    fn potential_force(phi_val: f64, mu: f64, lam: f64) -> f64 {
        // - dV/dphi = -(-mu^2 phi + lambda phi^3) = mu^2 phi - lambda phi^3
        // 최적화됨: phi * (mu^2 - lambda * phi^2)
        phi_val * (mu.powi(2) - lam * phi_val.powi(2))
    }

    pub fn step(&mut self) {
        let n = self.phi.len();

        // 스텐실 계산을 위해 phi에 대한 읽기 접근 권한과
        // forces_buffer에 대한 쓰기 접근 권한이 필요합니다.
        // Unsafe는 슬라이스나 반복자를 올바르게 사용하면 필요하지 않지만, 표준 반복자로는 이웃 요소에 쉽게 접근하기 어렵습니다.
        // Rust에서 안전한 원시 슬라이스(raw slice)와 병렬 반복자를 사용하겠습니다.

        let phi_slice = self.phi.as_slice().unwrap();
        let dphi_slice = self.dphi.as_slice().unwrap();
        let source_slice = self.source_j.as_slice().unwrap();
        let forces_slice = self.forces_buffer.as_slice_mut().unwrap();
        let mu = self.mu;
        let lam = self.lam;
        let alpha2 = self.alpha2;

        // 힘의 병렬 계산
        forces_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, force)| {
                // 경계 조건 (주기적 또는 디리클레? 코드는 0 왼쪽 인덱스 사용, 클램핑 또는 주기적 로직 의도됨)
                // 원본 코드: if i==0 { phi[0] } else { phi[i-1] } -> 사실상 노이만/디리클레 하이브리드?
                // 원래 로직 유지: 양끝 클램핑.
                let left = if i == 0 {
                    phi_slice[0]
                } else {
                    phi_slice[i - 1]
                };
                let right = if i == n - 1 {
                    phi_slice[n - 1]
                } else {
                    phi_slice[i + 1]
                };

                let laplacian = left + right - 2.0 * phi_slice[i];

                // SFE Master Action: Curvature Suppression (Biharmonic term)
                // -alpha2 * nabla^4 phi
                let biharmonic = if alpha2 != 0.0 {
                    let i_isize = i as isize;
                    let p_2l = if i_isize - 2 < 0 {
                        phi_slice[0]
                    } else {
                        phi_slice[i - 2]
                    };
                    let p_1l = left;
                    let p = phi_slice[i];
                    let p_1r = right;
                    let p_2r = if i_isize + 2 >= n as isize {
                        phi_slice[n - 1]
                    } else {
                        phi_slice[i + 2]
                    };

                    p_2l - 4.0 * p_1l + 6.0 * p - 4.0 * p_1r + p_2r
                } else {
                    0.0
                };

                let pot_f = Self::potential_force(phi_slice[i], mu, lam);
                let damping = -0.1 * dphi_slice[i];

                *force = pot_f + COUPLING_K * laplacian - alpha2 * biharmonic
                    + source_slice[i]
                    + damping;
            });

        // 상태 업데이트 (심플렉틱 오일러와 유사)
        // dphi += forces * DT
        // phi += dphi * DT
        // 이것도 병렬화할 수 있으며, ndarray의 벡터화 연산을 사용할 수도 있습니다.
        // 일관성과 캐시 지역성을 위해 병렬 반복자를 사용하겠습니다.

        let dphi_slice = self.dphi.as_slice_mut().unwrap();
        let phi_slice = self.phi.as_slice_mut().unwrap();
        let forces_slice = self.forces_buffer.as_slice().unwrap();

        dphi_slice
            .par_iter_mut()
            .zip(forces_slice.par_iter())
            .for_each(|(v, f)| {
                *v += f * DT;
            });

        phi_slice
            .par_iter_mut()
            .zip(dphi_slice.par_iter())
            .for_each(|(p, v)| {
                *p += v * DT;
            });
    }

    pub fn get_center_value(&self) -> f64 {
        self.phi[self.phi.len() / 2]
    }
}
