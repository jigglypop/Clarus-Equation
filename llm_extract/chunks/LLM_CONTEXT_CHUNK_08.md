# LLM Context Chunk

---
## File: `reality_stone/src/layers/bellman_lagrangian.rs`

```rust
// ============================================================================
// 파일: src/layers/bellman_lagrangian.rs
// 목적: 벨만 가치 함수 + 라그랑지안 에너지 시스템
// ============================================================================

use super::metric::{DiagonalMetric, MetricTensor, MetricType};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// 벨만 가치 함수 근사
pub struct ValueFunction {
    /// MLP 파라미터 (단순화: 선형 근사)
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

impl ValueFunction {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let weights =
            Array2::from_shape_fn((input_dim, hidden_dim), |_| rng.gen::<f32>() * 0.1 - 0.05);
        let bias = Array1::from_shape_fn(hidden_dim, |_| rng.gen::<f32>() * 0.1 - 0.05);

        Self { weights, bias }
    }

    /// V(x) 계산
    pub fn compute(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let hidden = x.dot(&self.weights);
        let mut output = Array1::zeros(x.nrows());

        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (j, &w) in hidden.row(i).iter().enumerate() {
                sum += (w + self.bias[j]).tanh(); // activation
            }
            *out = sum;
        }

        output
    }

    /// ∇V(x) 계산
    pub fn gradient(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let batch_size = x.nrows();
        let dim = x.ncols();
        let hidden = x.dot(&self.weights);

        let mut grad = Array2::zeros(x.raw_dim());

        for i in 0..batch_size {
            for j in 0..dim {
                let mut g = 0.0;
                for k in 0..self.bias.len() {
                    let h = hidden[[i, k]] + self.bias[k];
                    let tanh_h = h.tanh();
                    let sech_sq = 1.0 - tanh_h * tanh_h;
                    g += self.weights[[j, k]] * sech_sq;
                }
                grad[[i, j]] = g;
            }
        }

        grad
    }
}

/// 라그랑지안 파라미터
#[derive(Clone)]
pub struct LagrangianParams {
    pub kinetic_weight: f32,   // T의 가중치
    pub potential_weight: f32, // V의 가중치
    pub gamma: f32,            // 할인율
    pub regularization: RegularizationConfig,
}

#[derive(Clone)]
pub struct RegularizationConfig {
    pub attractor_weight: f32, // 기억 attractor β
    pub curvature_weight: f32, // 곡률 복잡도 γ
}

impl Default for LagrangianParams {
    fn default() -> Self {
        Self {
            kinetic_weight: 0.5,
            potential_weight: 1.0,
            gamma: 0.99,
            regularization: RegularizationConfig {
                attractor_weight: 0.01,
                curvature_weight: 0.001,
            },
        }
    }
}

/// 벨만 잠재 에너지 계산
/// V_Bell = (V(x) - (R + γV(x')))²
pub fn bellman_potential(
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    gamma: f32,
) -> Array1<f32> {
    let v_x = value_fn.compute(x);
    let v_x_next = value_fn.compute(x_next);

    // δ(x, x') = V(x) - (R + γ V(x'))
    let bellman_error = &v_x - &(reward + &(&v_x_next * gamma));

    // V_Bell = δ²
    bellman_error.mapv(|e| e * e)
}

/// 운동 에너지 계산: T = (1/2) g_ij v^i v^j
pub fn kinetic_energy(
    metric: &dyn MetricTensor,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
) -> Array1<f32> {
    let g = metric.compute_metric(x);

    // 대각 근사: T = (1/2) Σ g_ii v_i²
    let v_sq = v.mapv(|x| x * x);
    let weighted = &g * &v_sq;
    weighted.sum_axis(Axis(1)) * 0.5
}

/// 라그랑지안 계산: L = T - V
pub fn lagrangian(
    metric: &dyn MetricTensor,
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    params: &LagrangianParams,
) -> Array1<f32> {
    let kinetic = kinetic_energy(metric, x, v);
    let potential = bellman_potential(value_fn, x, x_next, reward, params.gamma);

    &kinetic * params.kinetic_weight - &potential * params.potential_weight
}

/// 표현 흐름 업데이트: x' = Exp_x(-η ∇_g V)
pub fn representation_flow(
    metric: &MetricType,
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    learning_rate: f32,
) -> Array2<f32> {
    // ∇V(x)
    let grad_v = value_fn.gradient(x);

    // ∇_g V = g^{-1} ∇V (리만 그래디언트)
    let g_inv = metric.as_trait().compute_inverse_metric(x);
    let riemannian_grad = &grad_v * &g_inv;

    // 자연 경사 방향으로 이동
    let direction = &riemannian_grad * (-learning_rate);

    // Exp_x(direction)
    crate::layers::geodesic::exponential_map(metric, x, &direction.view(), 1.0)
}

/// 메트릭 흐름 업데이트: g' = g + η ∂L/∂g
/// (대각 메트릭에 대해서만 구현)
pub fn metric_flow(
    metric: &mut DiagonalMetric,
    _: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    lagrangian_value: &ArrayView1<f32>,
    learning_rate: f32,
) {
    // ∂L/∂g_ii ≈ (1/2) v_i²  (T 항에서)
    let v_sq = v.mapv(|x| x * x);
    let grad_g = v_sq.mean_axis(Axis(0)).unwrap() * 0.5;

    // 가중치 업데이트
    let mean_lagrangian = lagrangian_value.mean().unwrap();
    for (i, &g) in grad_g.iter().enumerate() {
        metric.weights[i] += learning_rate * g * mean_lagrangian;
    }
}

/// 벨만 업데이트: V(x) ← V(x) + α [R + γV(x') - V(x)]
pub fn bellman_update(
    value_fn: &mut ValueFunction,
    x: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    gamma: f32,
    learning_rate: f32,
) {
    let batch_size = x.nrows();
    let hidden_dim = value_fn.weights.ncols();

    // 1. V(x) 계산 및 중간 값 저장 (Backprop을 위해)
    let hidden_pre = x.dot(&value_fn.weights); // (batch, hidden)
    let mut activations = Array2::zeros((batch_size, hidden_dim));
    let mut v_x = Array1::zeros(batch_size);

    for b in 0..batch_size {
        let mut sum_val = 0.0;
        for h in 0..hidden_dim {
            let val = hidden_pre[[b, h]] + value_fn.bias[h];
            let act = val.tanh();
            activations[[b, h]] = act;
            sum_val += act;
        }
        v_x[b] = sum_val;
    }

    // 2. TD Error 계산
    let v_x_next = value_fn.compute(x_next);
    let td_error = reward + &(&v_x_next * gamma) - &v_x; // (batch,)

    // 3. 파라미터 업데이트 (Semi-gradient descent)
    // Update rule: theta += alpha * td_error * ∇_theta V(x)

    // ∇_activations = (1 - tanh^2)
    // delta[b, h] = td_error[b] * (1 - activation[b, h]^2)
    let mut delta = Array2::zeros((batch_size, hidden_dim));
    for b in 0..batch_size {
        let err = td_error[b];
        for h in 0..hidden_dim {
            let act = activations[[b, h]];
            delta[[b, h]] = err * (1.0 - act * act);
        }
    }

    let lr_scaled = learning_rate / (batch_size as f32);

    // Bias 업데이트
    // ∇_bias[h] = sum_b(delta[b, h])
    let bias_grad = delta.sum_axis(Axis(0));
    // Gradient Clipping to prevent explosion
    let bias_grad_clipped = bias_grad.mapv(|g| g.clamp(-1.0, 1.0));
    value_fn.bias = &value_fn.bias + &(&bias_grad_clipped * lr_scaled);

    // Weights 업데이트
    // ∇_weights = x^T @ delta
    let weights_grad = x.t().dot(&delta);
    // Gradient Clipping
    let weights_grad_clipped = weights_grad.mapv(|g| g.clamp(-1.0, 1.0));
    value_fn.weights = &value_fn.weights + &(&weights_grad_clipped * lr_scaled);
}

/// 에너지 구성 요소
pub struct EnergyComponents {
    pub kinetic: Array1<f32>,
    pub potential: Array1<f32>,
    pub lagrangian: Array1<f32>,
    pub bellman_residual: Array1<f32>,
}

impl EnergyComponents {
    pub fn new(batch_size: usize) -> Self {
        Self {
            kinetic: Array1::zeros(batch_size),
            potential: Array1::zeros(batch_size),
            lagrangian: Array1::zeros(batch_size),
            bellman_residual: Array1::zeros(batch_size),
        }
    }
}

/// 전체 에너지 계산
pub fn compute_energy_components(
    metric: &dyn MetricTensor,
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    params: &LagrangianParams,
) -> EnergyComponents {
    let kinetic = kinetic_energy(metric, x, v);
    let potential = bellman_potential(value_fn, x, x_next, reward, params.gamma);

    let v_x = value_fn.compute(x);
    let v_x_next = value_fn.compute(x_next);
    let bellman_residual = &v_x - &(reward + &(&v_x_next * params.gamma));

    let lagrangian = &kinetic * params.kinetic_weight - &potential * params.potential_weight;

    EnergyComponents {
        kinetic,
        potential,
        lagrangian,
        bellman_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_value_function() {
        let vf = ValueFunction::new(4, 8);
        let x = arr2(&[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);

        let v = vf.compute(&x.view());
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|&x| x.is_finite()));

        let grad = vf.gradient(&x.view());
        assert_eq!(grad.shape(), x.shape());
        assert!(grad.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kinetic_energy() {
        use super::super::metric::DiagonalMetric;
        let metric = DiagonalMetric::new(3);
        let x = arr2(&[[0.1, 0.2, 0.3]]);
        let v = arr2(&[[1.0, 0.5, 0.2]]);

        let ke = kinetic_energy(&metric, &x.view(), &v.view());
        assert_eq!(ke.len(), 1);
        assert!(ke[0] > 0.0);
    }
}
```
---
## File: `reality_stone/src/layers/cuda/diffusion.cu`

```cpp
// ============================================================================
// 파일: src/layers/cuda/diffusion.cu
// 목적: 리만 라그랑지안 디퓨전 CUDA 커널
// ============================================================================

#include "mobius_common.cuh"
#include <cuda_runtime.h>
#include <math.h>

// Riemannian Step Kernel
// h_next = Exp_h ( (1-alpha) * (flow - h) )
// This is a retraction approximating the Lagrangian flow.
//
// Inputs:
//   h: [N, D] current state
//   flow: [N, D] target direction (tanh(hW))
//   output: [N, D] next state
//   alpha: scalar damping factor
//   c: curvature (fixed to 1.0 for now)
//   N, D: dimensions
extern "C" __global__ void riemannian_diffusion_step_kernel(
    const float* __restrict__ h,
    const float* __restrict__ flow,
    float* __restrict__ output,
    float alpha,
    float dt,
    int N,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = N * D;

    for (int i = idx; i < size; i += stride) {
        // 1. Compute tangent vector v in ambient space (Euclidean approx)
        // v = (1 - alpha) * (flow[i] - h[i])
        float h_val = h[i];
        float f_val = flow[i];
        float v = (1.0f - alpha) * (f_val - h_val);
        
        // 2. Update in Euclidean space first (Euler step)
        // h_new = h + v * dt
        float h_new = h_val + v * dt;
        
        // 3. Project back to Poincaré Ball (Manifold Constraint)
        // We apply a soft clipping to keep it within the ball (-1, 1)
        // Ideally, we should do proper Exp map, but component-wise clipping 
        // combined with norm projection is a valid retraction.
        
        // Simple robust projection
        if (h_new > 0.9999f) h_new = 0.9999f;
        if (h_new < -0.9999f) h_new = -0.9999f;
        
        output[i] = h_new;
    }
}

// Kernel Wrapper
extern "C" void riemannian_diffusion_step_cuda(
    const float* h,
    const float* flow,
    float* output,
    float alpha,
    float dt,
    int N,
    int D,
    cudaStream_t stream
) {
    int size = N * D;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    riemannian_diffusion_step_kernel<<<blocks, threads, 0, stream>>>(
        h, flow, output, alpha, dt, N, D
    );
}
```
---
## File: `reality_stone/src/layers/cuda/fast_metric_extraction.cu`

```cpp
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void init_random_basis_kernel(
    float* V,
    float* U,
    int in_dim,
    int out_dim,
    int k,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    int total_v = in_dim * k;
    int total_u = out_dim * k;
    
    if (idx < total_v) {
        float val = curand_normal(&state) / sqrtf((float)k);
        V[idx] = val;
    }
    
    if (idx < total_u) {
        float val = curand_normal(&state) / sqrtf((float)k);
        U[idx] = val;
    }
}

__global__ void compute_metric_from_weight_kernel(
    const float* W,
    const float* V,
    const float* U,
    float* G,
    int out_dim,
    int in_dim,
    int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= k || col >= k) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            float w_ij = W[i * in_dim + j];
            float u_ik = U[i * k + row];
            float v_jl = V[j * k + col];
            sum += u_ik * w_ij * v_jl;
        }
    }
    
    G[row * k + col] = sum;
}

__global__ void orthogonalize_basis_kernel(
    float* V,
    int dim,
    int k,
    int col_idx
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;
    
    __shared__ float dot_product;
    __shared__ float norm_sq;
    
    if (threadIdx.x == 0) {
        dot_product = 0.0f;
        norm_sq = 0.0f;
    }
    __syncthreads();
    
    for (int prev = 0; prev < col_idx; prev++) {
        float local_dot = V[row * k + col_idx] * V[row * k + prev];
        atomicAdd(&dot_product, local_dot);
        __syncthreads();
        
        V[row * k + col_idx] -= (dot_product / (norm_sq + 1e-8f)) * V[row * k + prev];
        __syncthreads();
    }
    
    float local_norm = V[row * k + col_idx] * V[row * k + col_idx];
    atomicAdd(&norm_sq, local_norm);
    __syncthreads();
    
    V[row * k + col_idx] /= sqrtf(norm_sq + 1e-8f);
}

extern "C" void fast_extract_metric_cuda(
    const float* W,
    float* U,
    float* G, 
    float* V,
    int out_dim,
    int in_dim,
    int k
) {
    float *d_W, *d_U, *d_G, *d_V;
    
    size_t w_size = out_dim * in_dim * sizeof(float);
    size_t u_size = out_dim * k * sizeof(float);
    size_t g_size = k * k * sizeof(float);
    size_t v_size = in_dim * k * sizeof(float);
    
    cudaMalloc(&d_W, w_size);
    cudaMalloc(&d_U, u_size);
    cudaMalloc(&d_G, g_size);
    cudaMalloc(&d_V, v_size);
    
    cudaMemcpy(d_W, W, w_size, cudaMemcpyHostToDevice);
    
    int max_dim = (in_dim > out_dim) ? in_dim : out_dim;
    int threads = 256;
    int blocks = (max_dim * k + threads - 1) / threads;
    
    unsigned long long seed = 42;
    init_random_basis_kernel<<<blocks, threads>>>(d_V, d_U, in_dim, out_dim, k, seed);
    cudaDeviceSynchronize();
    
    dim3 block_2d(16, 16);
    dim3 grid_2d((k + 15) / 16, (k + 15) / 16);
    compute_metric_from_weight_kernel<<<grid_2d, block_2d>>>(d_W, d_V, d_U, d_G, out_dim, in_dim, k);
    cudaDeviceSynchronize();
    
    cudaMemcpy(U, d_U, u_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(G, d_G, g_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, v_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_W);
    cudaFree(d_U);
    cudaFree(d_G);
    cudaFree(d_V);
}
```
---
## File: `reality_stone/src/layers/cuda/geodesic_topk_attention.cu`

```cpp
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

#define GEODESIC_EPS 1e-7f

namespace {
    const int MAX_THREADS = 256;

    // Warp-level reduction
    __device__ inline float warpReduceSum(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    __device__ inline float warpReduceMax(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        return val;
    }

    // Block-level reduction
    __device__ float blockReduceSum(float val) {
        __shared__ float shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        val = warpReduceSum(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        if (wid == 0) val = warpReduceSum(val);
        return val;
    }

    __device__ float blockReduceMax(float val) {
        __shared__ float shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        val = warpReduceMax(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e9f;
        if (wid == 0) val = warpReduceMax(val);
        return val;
    }

    // Poincare geodesic distance
    __device__ inline float poincare_distance(
        const float* q, const float* k, int d_h, float c
    ) {
        float dist_sq = 0.0f;
        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        
        for (int i = 0; i < d_h; ++i) {
            float diff = q[i] - k[i];
            dist_sq += diff * diff;
            q_norm_sq += q[i] * q[i];
            k_norm_sq += k[i] * k[i];
        }
        
        // d = arccosh(1 + 2c||x-y||² / ((1-c||x||²)(1-c||y||²)))
        float denom = fmaxf((1.0f - c*q_norm_sq) * (1.0f - c*k_norm_sq), GEODESIC_EPS);
        float arg = 1.0f + 2.0f * c * dist_sq / denom;
        return acoshf(fmaxf(arg, 1.0f + GEODESIC_EPS));
    }
}

/**
 * Fused Geodesic Top-k Attention Kernel
 *
 * This kernel performs the following in a single launch:
 * 1. Apply SPD metric (L @ Q, L @ K)
 * 2. Top-k key gathering
 * 3. Geodesic distance computation
 * 4. Softmax
 * 5. Weighted sum with values
 *
 * 1 thread block = 1 query token.
 * Uses shared memory to minimize global memory traffic.
 */
__global__ void geodesic_topk_attention_fused_kernel(
    // Inputs
    const float* __restrict__ Q,      // [B, H, T, d_h]
    const float* __restrict__ K,      // [B, H, S, d_h]
    const float* __restrict__ V,      // [B, H, S, d_v]
    const int64_t* __restrict__ idx,  // [B, T, K] top-k indices
    const float* __restrict__ L,      // [d_h, d_h] SPD Cholesky factor
    // Parameters
    const float c,                    // curvature
    const float tau,                  // temperature
    const int B, const int H,
    const int T, const int K_topk,
    const int d_h, const int d_v,
    // Output
    float* __restrict__ out           // [B, H, T, d_v]
) {
    // Block indices: (b, h, t)
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    
    if (b >= B || h >= H || t >= T) return;
    
    // Shared memory layout
    extern __shared__ float smem[];
    float* q_local = smem;                         // [d_h]
    float* k_local = smem + d_h;                   // [K_topk * d_h]
    float* v_local = smem + d_h + K_topk*d_h;      // [K_topk * d_v]
    float* scores = smem + d_h + K_topk*(d_h + d_v); // [K_topk]
    
    // ============================================================
    // Step 1: Load Q and apply SPD metric L
    // ============================================================
    // q' = L @ q
    for (int i = threadIdx.x; i < d_h; i += blockDim.x) {
        float q_transformed = 0.0f;
        const int q_offset = ((b*H + h)*T + t)*d_h;
        
        // Matrix-vector product: q'[i] = sum_j L[i,j] * q[j]
        for (int j = 0; j < d_h; ++j) {
            q_transformed += L[i*d_h + j] * Q[q_offset + j];
        }
        q_local[i] = q_transformed;
    }
    __syncthreads();
    
    // ============================================================
    // Step 2: Load top-K keys and apply SPD metric
    // ============================================================
    for (int k_idx = 0; k_idx < K_topk; ++k_idx) {
        const int64_t s = idx[(b*T + t)*K_topk + k_idx];
        
        for (int i = threadIdx.x; i < d_h; i += blockDim.x) {
            float k_transformed = 0.0f;
            const int k_offset = ((b*H + h)*s)*d_h;
            
            // k'[i] = sum_j L[i,j] * k[j]
            for (int j = 0; j < d_h; ++j) {
                k_transformed += L[i*d_h + j] * K[k_offset + j];
            }
            k_local[k_idx*d_h + i] = k_transformed;
        }
    }
    __syncthreads();
    
    // ============================================================
    // Step 3: Compute geodesic distances (parallel over K)
    // ============================================================
    for (int k_idx = threadIdx.x; k_idx < K_topk; k_idx += blockDim.x) {
        float dist = poincare_distance(
            q_local, 
            k_local + k_idx*d_h, 
            d_h, 
            c
        );
        
        // Score = -dist² / τ
        scores[k_idx] = -(dist * dist) / tau;
    }
    __syncthreads();
    
    // ============================================================
    // Step 4: Softmax (numerically stable)
    // ============================================================
    // 4a. Find max score
    float max_score = -1e9f;
    for (int k_idx = threadIdx.x; k_idx < K_topk; k_idx += blockDim.x) {
        max_score = fmaxf(max_score, scores[k_idx]);
    }
    max_score = blockReduceMax(max_score);
    if (threadIdx.x == 0) {
        scores[K_topk] = max_score;  // Store in extra slot
    }
    __syncthreads();
    max_score = scores[K_topk];
    
    // 4b. Compute exp and sum
    float sum_exp = 0.0f;
    for (int k_idx = threadIdx.x; k_idx < K_topk; k_idx += blockDim.x) {
        float exp_val = expf(scores[k_idx] - max_score);
        scores[k_idx] = exp_val;
        sum_exp += exp_val;
    }
    sum_exp = blockReduceSum(sum_exp);
    if (threadIdx.x == 0) {
        scores[K_topk] = sum_exp;  // Store in extra slot
    }
    __syncthreads();
    sum_exp = scores[K_topk];
    
    // 4c. Normalize
    for (int k_idx = threadIdx.x; k_idx < K_topk; k_idx += blockDim.x) {
        scores[k_idx] /= fmaxf(sum_exp, GEODESIC_EPS);
    }
    __syncthreads();
    
    // ============================================================
    // Step 5: Load values
    // ============================================================
    for (int k_idx = 0; k_idx < K_topk; ++k_idx) {
        const int64_t s = idx[(b*T + t)*K_topk + k_idx];
        const int v_offset = ((b*H + h)*s)*d_v;
        
        for (int i = threadIdx.x; i < d_v; i += blockDim.x) {
            v_local[k_idx*d_v + i] = V[v_offset + i];
        }
    }
    __syncthreads();
    
    // ============================================================
    // Step 6: Weighted sum (parallel over d_v)
    // ============================================================
    const int out_offset = ((b*H + h)*T + t)*d_v;
    for (int i = threadIdx.x; i < d_v; i += blockDim.x) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < K_topk; ++k_idx) {
            sum += scores[k_idx] * v_local[k_idx*d_v + i];
        }
        out[out_offset + i] = sum;
    }
}

/**
 * Host function to launch the kernel
 */
extern "C" void geodesic_topk_attention_cuda(
    const float* Q,
    const float* K,
    const float* V,
    const int64_t* idx,
    const float* L,
    float c,
    float tau,
    int B, int H, int T, int S, int K_topk,
    int d_h, int d_v,
    float* out
) {
    // Shared memory size
    // q_local[d_h] + k_local[K*d_h] + v_local[K*d_v] + scores[K+1]
    size_t smem_size = (d_h + K_topk*d_h + K_topk*d_v + K_topk + 1) * sizeof(float);
    
    // Grid: (T, H, B)
    dim3 grid(T, H, B);
    
    // Block: up to 256 threads
    int block_size = min(MAX_THREADS, max(d_h, max(d_v, K_topk)));
    block_size = (block_size + 31) / 32 * 32;  // Round up to warp size
    
    geodesic_topk_attention_fused_kernel<<<grid, block_size, smem_size>>>(
        Q, K, V, idx, L,
        c, tau,
        B, H, T, K_topk, d_h, d_v,
        out
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * FP16 version for Tensor Cores (future optimization)
 */
#ifdef ENABLE_FP16
#include <cuda_fp16.h>

__global__ void geodesic_topk_attention_fp16_kernel(
    const half* Q,
    const half* K,
    const half* V,
    const int64_t* idx,
    const half* L,
    float c, float tau,
    int B, int H, int T, int K, int d_h, int d_v,
    half* out
) {
    // TODO: Implement FP16 version with Tensor Cores
    // Use wmma API for matrix operations
}
#endif

// Batched Cholesky Decomposition
// Each block handles one matrix.
// Assumes d is small (e.g. <= 32 or 64).
// A: [B * T, d, d]
// L: [B * T, d, d]
__global__ void batched_cholesky_kernel(
    const float* __restrict__ A,
    float* __restrict__ L,
    int batch_count,
    int d
) {
    int b_idx = blockIdx.x;
    if (b_idx >= batch_count) return;

    const float* A_mat = A + b_idx * d * d;
    float* L_mat = L + b_idx * d * d;

    // Initialize L to 0
    for (int i = threadIdx.x; i < d * d; i += blockDim.x) {
        L_mat[i] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < d; ++k) {
        // Compute L_kk
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = 0; j < k; ++j) {
                float val = L_mat[k * d + j];
                sum += val * val;
            }
            float diag = A_mat[k * d + k] - sum;
            L_mat[k * d + k] = sqrtf(fmaxf(diag, 1e-6f));
        }
        __syncthreads();

        float l_kk = L_mat[k * d + k];

        // Compute L_ik for i > k
        for (int i = k + 1 + threadIdx.x; i < d; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < k; ++j) {
                sum += L_mat[i * d + j] * L_mat[k * d + j];
            }
            L_mat[i * d + k] = (A_mat[i * d + k] - sum) / l_kk;
        }
        __syncthreads();
    }
}

extern "C" void batched_cholesky_cuda(
    const float* A,
    float* L,
    int batch_count,
    int d
) {
    int block_size = 256;
    batched_cholesky_kernel<<<batch_count, block_size>>>(A, L, batch_count, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in batched_cholesky: %s\n", cudaGetErrorString(err));
    }
}
```
---
## File: `reality_stone/src/layers/cuda/klein.cu`

```cpp
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define KLEIN_EPS 1e-6f
#define BOUNDARY_EPS 1e-5f

__device__ static float norm_sq(const float* x, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

__device__ static float dot_product(const float* x, const float* y, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

__device__ static float klein_safe_acosh(float x) {
    return acoshf(fmaxf(x, 1.0f + KLEIN_EPS));
}

__device__ static float klein_safe_sqrt(float x) {
    return sqrtf(fmaxf(x, KLEIN_EPS));
}

__global__ void klein_distance_kernel(
    float* out, const float* u, const float* v, 
    float c, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float u2 = norm_sq(u_row, dim);
    float v2 = norm_sq(v_row, dim);
    float uv = dot_product(u_row, v_row, dim);
    
    float sqrt_c = klein_safe_sqrt(c);
    float numerator = 1.0f - c * uv;
    float denominator = klein_safe_sqrt((1.0f - c * u2) * (1.0f - c * v2));
    float arg = fmaxf(numerator / denominator, 1.0f + KLEIN_EPS);
    
    out[idx] = klein_safe_acosh(arg) / sqrt_c;
}

__device__ void klein_scalar_impl(
    const float* x, float* out, int dim, float c, float r
) {
    float norm_sq_val = norm_sq(x, dim);
    float norm_val = fmaxf(klein_safe_sqrt(norm_sq_val), KLEIN_EPS);
    float scaled_norm = fminf(norm_val * r, 1.0f / klein_safe_sqrt(c) - BOUNDARY_EPS);
    float scale = scaled_norm / norm_val;
    
    for (int i = 0; i < dim; ++i) {
        out[i] = scale * x[i];
    }
}

__device__ void klein_add_impl(
    const float* u, const float* v, float* out, int dim, float c
) {
    float u_norm_sq = norm_sq(u, dim);
    float uv_dot = dot_product(u, v, dim);
    
    float gamma_u = 1.0f / klein_safe_sqrt(1.0f - c * u_norm_sq);
    float denom = fmaxf(1.0f + c * uv_dot, KLEIN_EPS);
    float denom_inv = 1.0f / denom;
    
    float inv_gamma_u = 1.0f / gamma_u;
    float coeff_u_part = (c * gamma_u * uv_dot) / (1.0f + gamma_u);
    float coeff_u = 1.0f + coeff_u_part;
    
    for (int i = 0; i < dim; ++i) {
        out[i] = denom_inv * (coeff_u * u[i] + inv_gamma_u * v[i]);
    }
}

__global__ void klein_layer_forward_kernel(
    float* out, const float* u, const float* v, 
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    float* out_row = out + idx * dim;
    
    float u_prime[256];
    float v_prime[256];
    
    if (dim > 256) return;
    
    klein_scalar_impl(u_row, u_prime, dim, c, 1.0f - t);
    klein_scalar_impl(v_row, v_prime, dim, c, t);
    klein_add_impl(u_prime, v_prime, out_row, dim, c);
}

__device__ void klein_scalar_vjp_impl(
    const float* grad_output_prime, const float* x, 
    float c, float r, float* grad_x, int dim
) {
    float x_norm_sq = norm_sq(x, dim);
    float x_norm = klein_safe_sqrt(x_norm_sq);
    float x_norm_clamped = fmaxf(x_norm, KLEIN_EPS);
    
    float boundary = 1.0f / klein_safe_sqrt(c) - BOUNDARY_EPS;
    float scaled_norm = fminf(r * x_norm_clamped, boundary);
    float scale = scaled_norm / x_norm_clamped;
    
    float rn = r * x_norm_clamped;
    float d_scale_d_norm = (rn < boundary) ? 0.0f : -1.0f / fmaxf(x_norm_clamped * x_norm_clamped, KLEIN_EPS);
    
    float grad_norm_component = 0.0f;
    for (int i = 0; i < dim; ++i) {
        grad_norm_component += grad_output_prime[i] * x[i];
    }
    
    for (int i = 0; i < dim; ++i) {
        grad_x[i] = grad_output_prime[i] * scale 
                  + (grad_norm_component * d_scale_d_norm / x_norm_clamped) * x[i];
    }
}

__device__ void klein_add_vjp_impl(
    const float* grad_output, const float* u, const float* v,
    float c, float* grad_u, float* grad_v, int dim
) {
    float u_norm_sq = norm_sq(u, dim);
    float v_norm_sq = norm_sq(v, dim);
    float uv = dot_product(u, v, dim);
    
    float gamma_u = 1.0f / klein_safe_sqrt(1.0f - c * u_norm_sq);
    float denom = fmaxf(1.0f + c * uv, KLEIN_EPS);
    float denom_inv = 1.0f / denom;
    
    float inv_gamma_u = 1.0f / gamma_u;
    float coeff_u_part = (c * gamma_u * uv) / (1.0f + gamma_u);
    float coeff_u = 1.0f + coeff_u_part;
    
    float output_dot_grad = 0.0f;
    for (int j = 0; j < dim; ++j) {
        float out_j = denom_inv * (coeff_u * u[j] + inv_gamma_u * v[j]);
        output_dot_grad += out_j * grad_output[j];
    }
    
    float grad_denom = -output_dot_grad * denom_inv;
    
    for (int j = 0; j < dim; ++j) {
        float grad_num_u = coeff_u * grad_output[j] * denom_inv;
        float grad_num_v = inv_gamma_u * grad_output[j] * denom_inv;
        
        grad_u[j] = grad_num_u + c * grad_denom * v[j];
        grad_v[j] = grad_num_v + c * grad_denom * u[j];
    }
    
    float grad_coeff_u = 0.0f;
    float grad_inv_gamma_u = 0.0f;
    for (int j = 0; j < dim; ++j) {
        grad_coeff_u += (grad_output[j] * denom_inv) * u[j];
        grad_inv_gamma_u += (grad_output[j] * denom_inv) * v[j];
    }
    
    for (int j = 0; j < dim; ++j) {
        grad_u[j] -= u[j] * (grad_inv_gamma_u * c * gamma_u);
    }
    
    float d_coeff_u_d_uv = c * gamma_u / (1.0f + gamma_u);
    float d_coeff_u_d_gamma_u = (c * uv) / ((1.0f + gamma_u) * (1.0f + gamma_u));
    
    float grad_uv = grad_coeff_u * d_coeff_u_d_uv;
    float grad_gamma_u = grad_coeff_u * d_coeff_u_d_gamma_u;
    
    for (int j = 0; j < dim; ++j) {
        grad_u[j] += grad_uv * v[j] + grad_gamma_u * c * gamma_u * gamma_u * gamma_u * u[j];
        grad_v[j] += grad_uv * u[j];
    }
}

__global__ void klein_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    if (dim > 256) return;
    
    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    const float* grad_out = grad_output + idx * dim;
    float* gu = grad_u + idx * dim;
    float* gv = grad_v + idx * dim;
    
    float u_prime[256];
    float v_prime[256];
    float grad_u_prime[256];
    float grad_v_prime[256];
    
    klein_scalar_impl(u_row, u_prime, dim, c, 1.0f - t);
    klein_scalar_impl(v_row, v_prime, dim, c, t);
    
    klein_add_vjp_impl(grad_out, u_prime, v_prime, c, grad_u_prime, grad_v_prime, dim);
    
    klein_scalar_vjp_impl(grad_u_prime, u_row, c, 1.0f - t, gu, dim);
    klein_scalar_vjp_impl(grad_v_prime, v_row, c, t, gv, dim);
}

extern "C" {
    void klein_distance_cuda(
        float* out, const float* u, const float* v, 
        float c, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_distance_kernel<<<blocks, threads>>>(out, u, v, c, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void klein_layer_forward_cuda(
        float* out, const float* u, const float* v, 
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_layer_forward_kernel<<<blocks, threads>>>(out, u, v, c, t, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void klein_layer_backward_cuda(
        const float* grad_output, const float* u, const float* v,
        float* grad_u, float* grad_v,
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        klein_layer_backward_kernel<<<blocks, threads>>>(
            grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim
        );
        cudaDeviceSynchronize();
    }
}
```
---
## File: `reality_stone/src/layers/cuda/laplace_beltrami.cu`

```cpp
#include <cuda_runtime.h>

extern "C" __global__ void laplace_beltrami_apply_kernel(
    const float* __restrict__ lap,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * d;
    if (idx >= total) return;
    int i = idx / d;
    int k = idx % d;
    float acc = 0.0f;
    int row_offset = i * n;
    for (int j = 0; j < n; ++j) {
        float lij = lap[row_offset + j];
        if (lij != 0.0f) {
            acc += lij * x[j * d + k];
        }
    }
    out[idx] = acc;
}

extern "C" void laplace_beltrami_apply_cuda(
    const float* lap,
    const float* x,
    float* out,
    int n,
    int d,
    cudaStream_t stream
) {
    int total = n * d;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    laplace_beltrami_apply_kernel<<<grid_size, block_size, 0, stream>>>(lap, x, out, n, d);
}
```
---
## File: `reality_stone/src/layers/cuda/lorentz.cu`

```cpp
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define LORENTZ_EPS 1e-6f

__device__ static float lorentz_inner_product(const float* u, const float* v, int dim) {
    float inner = u[0] * v[0];
    for (int i = 1; i < dim; ++i) {
        inner -= u[i] * v[i];
    }
    return inner;
}

__device__ static float lorentz_safe_acosh(float x) {
    return acoshf(fmaxf(x, 1.0f + LORENTZ_EPS));
}

__device__ static float lorentz_safe_sqrt(float x) {
    return sqrtf(fmaxf(x, LORENTZ_EPS));
}

__global__ void lorentz_distance_kernel(
    float* out, const float* u, const float* v, 
    float c, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float inner = lorentz_inner_product(u_row, v_row, dim);
    float sqrt_c = lorentz_safe_sqrt(c);
    out[idx] = lorentz_safe_acosh(fmaxf(c * inner, 1.0f + LORENTZ_EPS)) / sqrt_c;
}

__global__ void lorentz_layer_forward_kernel(
    float* out, const float* u, const float* v, 
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* p = u + idx * dim;
    const float* q = v + idx * dim;
    float* result = out + idx * dim;
    
    float inner = lorentz_inner_product(p, q, dim);
    float theta = lorentz_safe_acosh(fmaxf(c * inner, 1.0f + LORENTZ_EPS));
    float sinh_theta = fmaxf(sinhf(theta), LORENTZ_EPS);
    
    float w1, w2;
    if (fabsf(theta) < 1e-6f) {
        w1 = 1.0f - t;
        w2 = t;
    } else {
        w1 = sinhf((1.0f - t) * theta) / sinh_theta;
        w2 = sinhf(t * theta) / sinh_theta;
    }
    
    for (int j = 0; j < dim; ++j) {
        result[j] = w1 * p[j] + w2 * q[j];
    }
}

__global__ void lorentz_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v,
    float c, float t, long long batch_size, long long dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* p = u + idx * dim;
    const float* q = v + idx * dim;
    const float* g = grad_output + idx * dim;
    float* gu = grad_u + idx * dim;
    float* gv = grad_v + idx * dim;
    
    float inner = lorentz_inner_product(p, q, dim);
    float alpha_arg = fmaxf(c * inner, 1.0f + LORENTZ_EPS);
    float alpha = acoshf(alpha_arg);
    float sinh_alpha = fmaxf(sinhf(alpha), LORENTZ_EPS);
    float cosh_alpha = coshf(alpha);
    
    float w1, w2, dw1_dalpha, dw2_dalpha;
    
    if (fabsf(alpha) < 1e-6f) {
        w1 = 1.0f - t;
        w2 = t;
        dw1_dalpha = 0.0f;
        dw2_dalpha = 0.0f;
    } else {
        w1 = sinhf((1.0f - t) * alpha) / sinh_alpha;
        w2 = sinhf(t * alpha) / sinh_alpha;
        
        float num1 = (1.0f - t) * coshf((1.0f - t) * alpha) * sinh_alpha 
                   - sinhf((1.0f - t) * alpha) * cosh_alpha;
        float num2 = t * coshf(t * alpha) * sinh_alpha 
                   - sinhf(t * alpha) * cosh_alpha;
        float denom = fmaxf(sinh_alpha * sinh_alpha, LORENTZ_EPS);
        
        dw1_dalpha = num1 / denom;
        dw2_dalpha = num2 / denom;
    }
    
    float scale = c / sinh_alpha;
    
    float g_dot_p = 0.0f;
    float g_dot_q = 0.0f;
    for (int j = 0; j < dim; ++j) {
        g_dot_p += g[j] * p[j];
        g_dot_q += g[j] * q[j];
    }
    
    float grad_term = g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha;
    
    for (int j = 0; j < dim; ++j) {
        float dalpha_dp_j = scale * ((j == 0) ? q[j] : -q[j]);
        float dalpha_dq_j = scale * ((j == 0) ? p[j] : -p[j]);
        
        gu[j] = w1 * g[j] + grad_term * dalpha_dp_j;
        gv[j] = w2 * g[j] + grad_term * dalpha_dq_j;
    }
}

extern "C" {
    void lorentz_distance_cuda(
        float* out, const float* u, const float* v, 
        float c, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_distance_kernel<<<blocks, threads>>>(out, u, v, c, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void lorentz_layer_forward_cuda(
        float* out, const float* u, const float* v, 
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_layer_forward_kernel<<<blocks, threads>>>(out, u, v, c, t, batch_size, dim);
        cudaDeviceSynchronize();
    }
    
    void lorentz_layer_backward_cuda(
        const float* grad_output, const float* u, const float* v,
        float* grad_u, float* grad_v,
        float c, float t, long long batch_size, long long dim
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        lorentz_layer_backward_kernel<<<blocks, threads>>>(
            grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim
        );
        cudaDeviceSynchronize();
    }
}
```
---
## File: `reality_stone/src/layers/cuda/mobius.cu`

```cpp
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

#include "mobius_common.cuh"

#define MIN_DENOMINATOR 1e-6f
#define EPS 1e-7f
#define BOUNDARY_EPS 1e-5f

__global__ void mobius_add_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        const float* v_row = v + i * dim;
        float* out_row = out + i * dim;
        mobius_add_point(u_row, v_row, out_row, dim, c, MIN_DENOMINATOR);
    }
}

extern "C" {
    void mobius_add_cuda(float* out, const float* u, const float* v, float c, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        mobius_add_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, v, c, batch_size, dim);
    }
}

// --- Mobius Scalar Multiplication ---

__global__ void mobius_scalar_kernel(float* out, const float* u, float c, float r, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        float* out_row = out + i * dim;
        mobius_scalar_point(u_row, out_row, dim, c, r, EPS, BOUNDARY_EPS);
    }
}

extern "C" {
    void mobius_scalar_cuda(float* out, const float* u, float c, float r, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        mobius_scalar_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, c, r, batch_size, dim);
    }
}
```
---
## File: `reality_stone/src/layers/cuda/mobius_common.cuh`

```cpp
#pragma once

// Common device helpers for Möbius operations shared by mobius.cu and poincare.cu
// We keep all math identical across kernels to avoid subtle numerical drift.

// Pointwise Möbius addition: out = u ⊕_c v for a single vector pair.
// Arguments:
// - u, v: input vectors (length dim)
// - out: output vector (length dim)
// - dim: spatial dimension
// - c: curvature parameter
// - min_den: minimum value to clamp the denominator for numerical stability
__device__ inline void mobius_add_point(
    const float* u,
    const float* v,
    float* out,
    int dim,
    float c,
    float min_den
) {
    float u2 = 0.0f;
    float v2 = 0.0f;
    float uv = 0.0f;

    for (int j = 0; j < dim; ++j) {
        float uj = u[j];
        float vj = v[j];
        u2 += uj * uj;
        v2 += vj * vj;
        uv += uj * vj;
    }

    float c2 = c * c;
    float den = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
    if (den < min_den) {
        den = min_den;
    }

    float coeff_u = (1.0f + 2.0f * c * uv + c * v2) / den;
    float coeff_v = (1.0f - c * u2) / den;

    for (int j = 0; j < dim; ++j) {
        out[j] = coeff_u * u[j] + coeff_v * v[j];
    }
}

// Pointwise Möbius scalar multiplication: out = r ⊗_c u for a single vector.
// Arguments:
// - u: input vector (length dim)
// - out: output vector (length dim)
// - dim: spatial dimension
// - c: curvature parameter
// - r: scalar multiplier
// - eps: small epsilon for handling very small norms / c ~ 0
// - boundary_eps: epsilon to keep arguments inside atanh / tanh domains
__device__ inline void mobius_scalar_point(
    const float* u,
    float* out,
    int dim,
    float c,
    float r,
    float eps,
    float boundary_eps
) {
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        norm_sq += u[j] * u[j];
    }

    // Very small vectors: fall back to simple scaling to keep gradients stable
    if (norm_sq < eps * eps) {
        for (int j = 0; j < dim; ++j) {
            out[j] = r * u[j];
        }
        return;
    }

    float norm = sqrtf(norm_sq);

    // c = 0: Euclidean case
    if (fabsf(c) < eps) {
        for (int j = 0; j < dim; ++j) {
            out[j] = r * u[j];
        }
        return;
    }

    float scale;
    if (c > 0.0f) {
        // Positive curvature
        float sqrt_c = sqrtf(c);
        float scn = fminf(sqrt_c * norm, 1.0f - boundary_eps);
        float alpha = atanhf(scn);
        float beta = tanhf(r * alpha);
        scale = beta / (sqrt_c * norm);
    } else {
        // Negative curvature (compute with real-valued formula)
        float sqrt_abs_c = sqrtf(-c);
        float scn = sqrt_abs_c * norm;
        float alpha = atanf(scn);
        float beta = tanf(r * alpha);
        scale = beta / (sqrt_abs_c * norm);
    }

    for (int j = 0; j < dim; ++j) {
        out[j] = scale * u[j];
    }
}
```
---
## File: `reality_stone/src/layers/cuda/poincare.cu`

```cpp
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "mobius_common.cuh"

#define POINCARE_EPS 1e-6f
#define POINCARE_BOUNDARY_EPS 1e-5f
#define POINCARE_MIN_DENOM 1e-6f
#define POINCARE_ATANH_CLAMP 1e-3f

__global__ void poincare_ball_layer_forward_kernel(const float* u, const float* v, float* out, float c, float t, long long batch_size, long long dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    const float* u_i = u + i * dim;
    const float* v_i = v + i * dim;
    float* out_i = out + i * dim;

    float u_prime[256]; // Max dim 256
    float v_prime[256];
    
    mobius_scalar_point(u_i, u_prime, dim, c, 1.0f - t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);
    mobius_scalar_point(v_i, v_prime, dim, c, t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);
    mobius_add_point(u_prime, v_prime, out_i, dim, c, POINCARE_EPS);
}

// Helper device function for mobius_scalar_vjp
__device__ void mobius_scalar_vjp(
    const float* grad_output_prime, const float* x, float c, float r,
    float* grad_x, int dim, float eps) {

    float x_norm_sq = 0;
    for (int i = 0; i < dim; ++i) {
        x_norm_sq += x[i] * x[i];
    }
    float x_norm = fmaxf(sqrtf(x_norm_sq), eps);
    
    if (fabsf(c) < eps) {
        // c = 0: Euclidean case
        for (int i = 0; i < dim; ++i) {
            grad_x[i] = r * grad_output_prime[i];
        }
        return;
    }
    
    float scale;
    float grad_scale_factor;
    
    if (c > 0.0f) {
        // Positive curvature
        float sqrt_c = sqrtf(c);
        float scn = fminf(sqrt_c * x_norm, 1.0f - eps);
        float alpha = atanhf(scn);
        float beta = tanhf(r * alpha);
        scale = beta / (sqrt_c * x_norm);
        
        float inner_deriv_atanh = r * (1.0f - beta * beta);
        float inner_deriv_norm = (1.0f / fmaxf(1.0f - scn * scn, eps)) * (sqrt_c / x_norm);
        grad_scale_factor = inner_deriv_atanh * inner_deriv_norm / (sqrt_c * x_norm) - scale / x_norm;
    } else {
        // Negative curvature
        float sqrt_abs_c = sqrtf(-c);
        float scn = sqrt_abs_c * x_norm;
        float alpha = atanf(scn);
        float beta = tanf(r * alpha);
        scale = beta / (sqrt_abs_c * x_norm);
        
        float inner_deriv_atan = r * (1.0f + beta * beta);
        float inner_deriv_norm = (1.0f / (1.0f + scn * scn)) * (sqrt_abs_c / x_norm);
        grad_scale_factor = inner_deriv_atan * inner_deriv_norm / (sqrt_abs_c * x_norm) - scale / x_norm;
    }

    float grad_scale = 0;
    for (int i = 0; i < dim; ++i) {
        grad_scale += grad_output_prime[i] * x[i];
    }

    for (int i = 0; i < dim; ++i) {
        grad_x[i] = scale * grad_output_prime[i] + grad_scale_factor * grad_scale * x[i];
    }
}

// Helper device function for mobius_add_vjp
__device__ void mobius_add_vjp(
    const float* grad_output, const float* x, const float* y, float c,
    float* grad_x, float* grad_y, int dim, float eps) {

    float x2 = 0, y2 = 0, xy = 0;
    for(int i=0; i<dim; ++i) {
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
        xy += x[i] * y[i];
    }

    float den = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    den = fmaxf(den, eps);

    float u_calc[256]; // Assuming max dim 256
    for(int i=0; i<dim; ++i) {
        u_calc[i] = (1.0f + 2.0f * c * xy + c * y2) * x[i] + (1.0f - c * x2) * y[i];
    }

    float output[256];
    for(int i=0; i<dim; ++i) {
        output[i] = u_calc[i] / den;
    }

    float grad_u[256];
    for(int i=0; i<dim; ++i) {
        grad_u[i] = grad_output[i] / den;
    }

    float grad_den_sum = 0;
    for(int i=0; i<dim; ++i) {
        grad_den_sum -= grad_output[i] * output[i] / den;
    }
    
    float grad_x_from_u[256], grad_y_from_u[256];
    float factor_x = 1.0f + 2.0f * c * xy + c * y2;
    float factor_y = 1.0f - c * x2;
    for(int i=0; i<dim; ++i) {
        grad_x_from_u[i] = grad_u[i] * factor_x;
        grad_y_from_u[i] = grad_u[i] * factor_y;
    }
    
    float grad_xy_from_u = 0, grad_x2_from_u = 0;
    for(int i=0; i<dim; ++i) {
        grad_xy_from_u += 2.0f * c * grad_u[i] * x[i];
        grad_x2_from_u -= c * grad_u[i] * y[i];
    }

    float grad_xy_from_den = 2.0f * c * grad_den_sum;
    float grad_x2_from_den = c * c * y2 * grad_den_sum;
    float grad_y2_from_den = c * c * x2 * grad_den_sum;

    float grad_xy_val = grad_xy_from_u + grad_xy_from_den;
    float grad_x2_val = grad_x2_from_u + grad_x2_from_den;
    float grad_y2_val = grad_y2_from_den;

    for(int i=0; i<dim; ++i) {
        grad_x[i] = grad_x_from_u[i] + 2.0f * grad_x2_val * x[i] + grad_xy_val * y[i];
        grad_y[i] = grad_y_from_u[i] + 2.0f * grad_y2_val * y[i] + grad_xy_val * x[i];
    }
}

__device__ float poincare_distance_impl(const float* x, const float* y, int dim, float c, float eps, float boundary_eps) {
    // Poincare distance: d = (2/sqrt(c)) * atanh(sqrt(c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))))
    float norm_sq_diff = 0.0f;  // ||x-y||²
    float x2 = 0.0f;            // ||x||²
    float y2 = 0.0f;            // ||y||²
    
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        norm_sq_diff += diff * diff;
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
    }
    
    // frac = c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))
    float den = (1.0f - c * x2) * (1.0f - c * y2);
    // Increased denominator clamp for numerical stability near boundary
    den = fmaxf(den, boundary_eps);
    float frac = (c * norm_sq_diff) / den;
    frac = fmaxf(frac, 0.0f);
    
    // d = (2/sqrt(c)) * atanh(sqrt(frac / (1 + frac)))
    float sqrtc = sqrtf(c);
    float arg = sqrtf(frac / (1.0f + frac));
    // More conservative atanh domain restriction
    arg = fminf(arg, 1.0f - boundary_eps);
    
    return (2.0f / sqrtc) * atanhf(arg);
}

    __global__ void poincare_distance_kernel(const float* x, const float* y, float* out, int batch_size, int dim, float c, float boundary_eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    const float* x_i = x + i * dim;
    const float* y_i = y + i * dim;
    
    out[i] = poincare_distance_impl(x_i, y_i, dim, c, POINCARE_EPS, boundary_eps);
}


// Backward Kernel for Poincare Ball Layer
__global__ void poincare_ball_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v, float c, float t, long long batch_size, long long dim) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    const float* u_i = u + i * dim;
    const float* v_i = v + i * dim;
    const float* grad_output_i = grad_output + i * dim;
    float* grad_u_i = grad_u + i * dim;
    float* grad_v_i = grad_v + i * dim;

    float u_prime[256], v_prime[256];
    mobius_scalar_point(u_i, u_prime, dim, c, 1.0f - t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);
    mobius_scalar_point(v_i, v_prime, dim, c, t, POINCARE_EPS, POINCARE_BOUNDARY_EPS);

    float grad_u_prime[256], grad_v_prime[256];
    mobius_add_vjp(grad_output_i, u_prime, v_prime, c, grad_u_prime, grad_v_prime, dim, POINCARE_EPS);
    
    mobius_scalar_vjp(grad_u_prime, u_i, c, 1.0f - t, grad_u_i, dim, POINCARE_EPS);
    mobius_scalar_vjp(grad_v_prime, v_i, c, t, grad_v_i, dim, POINCARE_EPS);
}

__device__ void exp_map_poincare_point(
    const float* x,
    const float* v,
    float* out,
    int dim,
    float c,
    float eps) {
    float x_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float xi = x[i];
        x_norm_sq += xi * xi;
    }
    float v_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float vi = v[i];
        v_norm_sq += vi * vi;
    }
    float v_norm = sqrtf(fmaxf(v_norm_sq, eps));
    if (fabsf(c) < eps) {
        for (int i = 0; i < dim; ++i) {
            out[i] = x[i] + v[i];
        }
        return;
    }
    float one_minus_cx2 = fmaxf(1.0f - c * x_norm_sq, eps);
    float lambda = 2.0f / one_minus_cx2;
    float sqrt_c = sqrtf(fabsf(c));
    float arg = 0.5f * lambda * sqrt_c * v_norm;
    float beta;
    if (c > 0.0f) {
        // Correct formula: beta = tanh(arg)
        // No atanh/clamp needed here as arg is in real domain
        beta = tanhf(arg);
    } else {
        // Correct formula: beta = tan(arg)
        beta = tanf(arg);
    }
    float scale = beta / (sqrt_c * v_norm);
    float u_temp[256];
    for (int i = 0; i < dim; ++i) {
        u_temp[i] = scale * v[i];
    }
    mobius_add_point(x, u_temp, out, dim, c, eps);
}

__global__ void poincare_riemannian_adam_kernel(
    float* x,
    const float* grad,
    float* m,
    float* v,
    float c,
    float lr,
    float beta1,
    float beta2,
    float eps,
    long long batch_size,
    long long dim,
    long long step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    float* x_i = x + i * dim;
    const float* g_i = grad + i * dim;
    float* m_i = m + i * dim;
    float* v_i = v + i * dim;
    float g_r[256];
    if (fabsf(c) < eps) {
        for (int j = 0; j < dim; ++j) {
            g_r[j] = g_i[j];
        }
    } else {
        float x_norm_sq = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float xi = x_i[j];
            x_norm_sq += xi * xi;
        }
        float one_minus_cx2 = fmaxf(1.0f - c * x_norm_sq, eps);
        float lambda = 2.0f / one_minus_cx2;
        float inv_lambda_sq = 1.0f / (lambda * lambda);
        for (int j = 0; j < dim; ++j) {
            g_r[j] = inv_lambda_sq * g_i[j];
        }
    }
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_b2 = 1.0f - beta2;
    for (int j = 0; j < dim; ++j) {
        float mj = m_i[j];
        float vj = v_i[j];
        float gr = g_r[j];
        mj = beta1 * mj + one_minus_b1 * gr;
        vj = beta2 * vj + one_minus_b2 * gr * gr;
        m_i[j] = mj;
        v_i[j] = vj;
    }
    float t = (float)step;
    float bias_c1 = 1.0f - powf(beta1, t);
    float bias_c2 = 1.0f - powf(beta2, t);
    float u[256];
    for (int j = 0; j < dim; ++j) {
        float m_hat = m_i[j] / bias_c1;
        float v_hat = v_i[j] / bias_c2;
        u[j] = -lr * m_hat / (sqrtf(v_hat) + eps);
    }
    float x_new[256];
    exp_map_poincare_point(x_i, u, x_new, dim, c, eps);
    float radius = c > 0.0f ? 1.0f / sqrtf(c) : 1.0f;
    float max_norm = radius - POINCARE_BOUNDARY_EPS;
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        norm_sq += x_new[j] * x_new[j];
    }
    float norm = sqrtf(fmaxf(norm_sq, eps));
    float scale = 1.0f;
    if (norm > max_norm) {
        scale = max_norm / norm;
    }
    for (int j = 0; j < dim; ++j) {
        x_i[j] = x_new[j] * scale;
    }
}



extern "C" {
    void poincare_ball_layer_cuda(float* out, const float* u, const float* v, float c, float t, long long batch_size, long long dim) {
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_ball_layer_forward_kernel<<<num_blocks, threads_per_block>>>(u, v, out, c, t, batch_size, dim);
    }
    
    void poincare_ball_layer_backward_cuda(
        const float* grad_output, const float* u, const float* v,
        float* grad_u, float* grad_v, float c, float t, long long batch_size, long long dim) {
        
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_ball_layer_backward_kernel<<<num_blocks, threads_per_block>>>(
            grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim);
    }

    void poincare_distance_cuda(float* out, const float* x, const float* y, float c, float boundary_eps, long long batch_size, long long dim) {
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_distance_kernel<<<num_blocks, threads_per_block>>>(x, y, out, batch_size, dim, c, boundary_eps);
    }

    void poincare_riemannian_adam_cuda(
        float* x,
        const float* grad,
        float* m,
        float* v,
        float c,
        float lr,
        float beta1,
        float beta2,
        float eps,
        long long batch_size,
        long long dim,
        long long step) {
        dim3 threads_per_block(256);
        dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x);
        poincare_riemannian_adam_kernel<<<num_blocks, threads_per_block>>>(
            x,
            grad,
            m,
            v,
            c,
            lr,
            beta1,
            beta2,
            eps,
            batch_size,
            dim,
            step);
    }
}
```
---
## File: `reality_stone/src/layers/cuda/rsulf_forward.cu`

```cpp
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

#define RSULF_EPS 1e-6f
#define WARP_SIZE 32

namespace {

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

}

__global__ void rsulf_forward_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v1,
    const float* __restrict__ s1,
    const float* __restrict__ u1,
    const float* __restrict__ v2,
    const float* __restrict__ s2,
    const float* __restrict__ u2,
    const float* __restrict__ g_inv,
    const float* __restrict__ v_mem,
    const float eta,
    const float alpha,
    const float gamma_param,
    const int batch,
    const int d,
    const int r,
    const int ffn_dim,
    float* __restrict__ x_out,
    float* __restrict__ v_out
) {
    extern __shared__ float smem[];
    
    const int b = blockIdx.x;
    if (b >= batch) return;
    
    float* x_local = smem;
    float* h1 = smem + d;
    float* h2 = smem + d + r;
    float* phi_grad = smem + d + r + ffn_dim;
    
    const float* x_row = x + b * d;
    float* x_out_row = x_out + b * d;
    
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        x_local[i] = x_row[i];
    }
    __syncthreads();
    
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += x_local[i] * v1[i * r + j];
        }
        h1[j] = sum * s1[j];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < ffn_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < r; ++j) {
            sum += h1[j] * u1[i * r + j];
        }
        h2[i] = silu(sum);
    }
    __syncthreads();
    
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            sum += h2[i] * v2[i * r + j];
        }
        h1[j] = sum * s2[j];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < r; ++j) {
            sum += h1[j] * u2[i * r + j];
        }
        phi_grad[i] = sum;
    }
    __syncthreads();
    
    float local_phi_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        local_phi_sq += phi_grad[i] * phi_grad[i];
    }
    float phi_val = blockReduceSum(local_phi_sq) * 0.5f;
    
    __shared__ float shared_mean[1];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        local_sum += x_local[i];
    }
    float mean_val = blockReduceSum(local_sum) / (float)d;
    if (threadIdx.x == 0) {
        shared_mean[0] = mean_val;
    }
    __syncthreads();
    mean_val = shared_mean[0];
    
    float v_prev = (v_mem != nullptr) ? v_mem[b] : 0.0f;
    float v_new = gamma_param * v_prev + (1.0f - gamma_param) * phi_val;
    
    if (threadIdx.x == 0) {
        v_out[b] = v_new;
    }
    
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g_i = (i < r) ? g_inv[i] : 1.0f;
        
        float term1 = -eta * g_i * phi_grad[i];
        float term2 = alpha * (x_local[i] - mean_val);
        float term3 = gamma_param * v_new;
        
        float velocity = term1 + term2 + term3;
        // Removed hard velocity clamp [-1, 1] to follow gradients
        // velocity = fmaxf(-1.0f, fminf(1.0f, velocity));
        
        float x_next = x_local[i] + velocity;
        // Removed hard clamp [-10, 10]
        x_out_row[i] = x_next;
    }
}

__global__ void rsulf_forward_vectorized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v1,
    const float* __restrict__ s1,
    const float* __restrict__ u1_t,
    const float* __restrict__ v2,
    const float* __restrict__ s2,
    const float* __restrict__ u2_t,
    const float* __restrict__ g_inv,
    const float* __restrict__ v_mem,
    const float eta,
    const float alpha,
    const float gamma_param,
    const int batch,
    const int d,
    const int r,
    const int ffn_dim,
    float* __restrict__ x_out,
    float* __restrict__ v_out
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= batch) return;
    
    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_h1 = smem + d;
    float* s_h2 = smem + d + r;
    float* s_phi = smem + d + r + ffn_dim;
    float* s_reduce = smem + 2 * d + r + ffn_dim;
    
    const float* x_in = x + bid * d;
    float* x_o = x_out + bid * d;
    
    for (int i = tid; i < d; i += blockDim.x) {
        s_x[i] = x_in[i];
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < d; i += 4) {
            if (i + 3 < d) {
                acc += s_x[i] * v1[i * r + j];
                acc += s_x[i+1] * v1[(i+1) * r + j];
                acc += s_x[i+2] * v1[(i+2) * r + j];
                acc += s_x[i+3] * v1[(i+3) * r + j];
            } else {
                for (int k = i; k < d; ++k) {
                    acc += s_x[k] * v1[k * r + j];
                }
            }
        }
        s_h1[j] = acc * s1[j];
    }
    __syncthreads();
    
    for (int i = tid; i < ffn_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u1_t[j * ffn_dim + i];
        }
        s_h2[i] = silu(acc);
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            acc += s_h2[i] * v2[i * r + j];
        }
        s_h1[j] = acc * s2[j];
    }
    __syncthreads();
    
    for (int i = tid; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u2_t[j * d + i];
        }
        s_phi[i] = acc;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    float local_phi_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        local_sum += s_x[i];
        local_phi_sq += s_phi[i] * s_phi[i];
    }
    
    s_reduce[tid] = local_sum;
    s_reduce[tid + blockDim.x] = local_phi_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
            s_reduce[tid + blockDim.x] += s_reduce[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    float mean_val = s_reduce[0] / (float)d;
    float phi_val = s_reduce[blockDim.x] * 0.5f;
    
    float v_prev = (v_mem != nullptr) ? v_mem[bid] : 0.0f;
    float v_new = gamma_param * v_prev + (1.0f - gamma_param) * phi_val;
    
    if (tid == 0) {
        v_out[bid] = v_new;
    }
    
    for (int i = tid; i < d; i += blockDim.x) {
        float g_i = (i < d) ? g_inv[i] : 1.0f;
        float velocity = -eta * g_i * s_phi[i] + alpha * (s_x[i] - mean_val) + gamma_param * v_new;
        // velocity = fmaxf(-1.0f, fminf(1.0f, velocity));
        x_o[i] = s_x[i] + velocity; // Removed clamping
    }
}

extern "C" void rsulf_forward_cuda(
    const float* x,
    const float* v1,
    const float* s1,
    const float* u1,
    const float* v2,
    const float* s2,
    const float* u2,
    const float* g_inv,
    const float* v_mem,
    float eta,
    float alpha,
    float gamma_param,
    int batch,
    int d,
    int r,
    int ffn_dim,
    float* x_out,
    float* v_out
) {
    size_t smem_size = (2 * d + r + ffn_dim + 512) * sizeof(float);
    
    int block_size = 256;
    dim3 grid(batch);
    dim3 block(block_size);
    
    if (d <= 1024 && r <= 256 && ffn_dim <= 4096) {
        rsulf_forward_vectorized_kernel<<<grid, block, smem_size>>>(
            x, v1, s1, u1, v2, s2, u2, g_inv, v_mem,
            eta, alpha, gamma_param,
            batch, d, r, ffn_dim,
            x_out, v_out
        );
    } else {
        rsulf_forward_fused_kernel<<<grid, block, smem_size>>>(
            x, v1, s1, u1, v2, s2, u2, g_inv, v_mem,
            eta, alpha, gamma_param,
            batch, d, r, ffn_dim,
            x_out, v_out
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rsulf_forward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void rsulf_batch_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v1,
    const float* __restrict__ s1,
    const float* __restrict__ u1_t,
    const float* __restrict__ v2,
    const float* __restrict__ s2,
    const float* __restrict__ u2_t,
    const float* __restrict__ g_inv,
    const float* __restrict__ v_mem,
    const float eta,
    const float alpha,
    const float gamma_param,
    const int batch,
    const int seq_len,
    const int d,
    const int r,
    const int ffn_dim,
    float* __restrict__ x_out,
    float* __restrict__ v_out
) {
    const int tid = threadIdx.x;
    const int total_tokens = batch * seq_len;
    const int token_idx = blockIdx.x;
    
    if (token_idx >= total_tokens) return;
    
    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_h1 = smem + d;
    float* s_h2 = smem + d + r;
    float* s_phi = smem + d + r + ffn_dim;
    float* s_reduce = smem + 2 * d + r + ffn_dim;
    
    const float* x_in = x + token_idx * d;
    float* x_o = x_out + token_idx * d;
    
    for (int i = tid; i < d; i += blockDim.x) {
        s_x[i] = x_in[i];
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < d; ++i) {
            acc += s_x[i] * v1[i * r + j];
        }
        s_h1[j] = acc * s1[j];
    }
    __syncthreads();
    
    for (int i = tid; i < ffn_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u1_t[j * ffn_dim + i];
        }
        s_h2[i] = silu(acc);
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            acc += s_h2[i] * v2[i * r + j];
        }
        s_h1[j] = acc * s2[j];
    }
    __syncthreads();
    
    for (int i = tid; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u2_t[j * d + i];
        }
        s_phi[i] = acc;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    float local_phi_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        local_sum += s_x[i];
        local_phi_sq += s_phi[i] * s_phi[i];
    }
    
    s_reduce[tid] = local_sum;
    s_reduce[tid + blockDim.x] = local_phi_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
            s_reduce[tid + blockDim.x] += s_reduce[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    float mean_val = s_reduce[0] / (float)d;
    float phi_val = s_reduce[blockDim.x] * 0.5f;
    
    float v_prev = (v_mem != nullptr) ? v_mem[token_idx] : 0.0f;
    float v_new = gamma_param * v_prev + (1.0f - gamma_param) * phi_val;
    
    if (tid == 0) {
        v_out[token_idx] = v_new;
    }
    
    for (int i = tid; i < d; i += blockDim.x) {
        float g_i = (i < d) ? g_inv[i] : 1.0f;
        float velocity = -eta * g_i * s_phi[i] + alpha * (s_x[i] - mean_val) + gamma_param * v_new;
        // velocity = fmaxf(-1.0f, fminf(1.0f, velocity));
        x_o[i] = s_x[i] + velocity; // Removed clamping
    }
}

extern "C" void rsulf_batch_forward_cuda(
    const float* x,
    const float* v1,
    const float* s1,
    const float* u1,
    const float* v2,
    const float* s2,
    const float* u2,
    const float* g_inv,
    const float* v_mem,
    float eta,
    float alpha,
    float gamma_param,
    int batch,
    int seq_len,
    int d,
    int r,
    int ffn_dim,
    float* x_out,
    float* v_out
) {
    int total_tokens = batch * seq_len;
    size_t smem_size = (2 * d + r + ffn_dim + 512) * sizeof(float);
    
    int block_size = 256;
    dim3 grid(total_tokens);
    dim3 block(block_size);
    
    rsulf_batch_forward_kernel<<<grid, block, smem_size>>>(
        x, v1, s1, u1, v2, s2, u2, g_inv, v_mem,
        eta, alpha, gamma_param,
        batch, seq_len, d, r, ffn_dim,
        x_out, v_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rsulf_batch_forward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void rsulf_unified_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v1,
    const float* __restrict__ s1,
    const float* __restrict__ u1_t,
    const float* __restrict__ v2,
    const float* __restrict__ s2,
    const float* __restrict__ u2_t,
    const float* __restrict__ g_inv,
    const float* __restrict__ laplacian,
    const float* __restrict__ v_mem,
    const float eta,
    const float alpha,
    const float beta,
    const float gamma_param,
    const float curvature,
    const int batch,
    const int seq_len,
    const int d,
    const int r,
    const int ffn_dim,
    const int window,
    float* __restrict__ x_out,
    float* __restrict__ v_out
) {
    const int tid = threadIdx.x;
    const int token_idx = blockIdx.x;
    const int total_tokens = batch * seq_len;
    
    if (token_idx >= total_tokens) return;
    
    const int seq_idx = token_idx % seq_len;
    
    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_h1 = smem + d;
    float* s_h2 = smem + d + r;
    float* s_phi = smem + d + r + ffn_dim;
    float* s_reduce = smem + 2 * d + r + ffn_dim;
    float* s_velocity = smem + 2 * d + r + ffn_dim + 512;
    float* s_xu2 = smem + 3 * d + r + ffn_dim + 512;
    
    const float* x_in = x + token_idx * d;
    float* x_o = x_out + token_idx * d;
    
    for (int i = tid; i < d; i += blockDim.x) {
        s_x[i] = x_in[i];
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < d; ++i) {
            acc += s_x[i] * v1[i * r + j];
        }
        s_h1[j] = acc * s1[j];
    }
    __syncthreads();
    
    for (int i = tid; i < ffn_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u1_t[j * ffn_dim + i];
        }
        s_h2[i] = silu(acc);
    }
    __syncthreads();
    
    for (int j = tid; j < r; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            acc += s_h2[i] * v2[i * r + j];
        }
        s_h1[j] = acc * s2[j];
    }
    __syncthreads();
    
    for (int i = tid; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            acc += s_h1[j] * u2_t[j * d + i];
        }
        s_phi[i] = acc;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    float local_phi_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        local_sum += s_x[i];
        local_phi_sq += s_phi[i] * s_phi[i];
    }
    
    s_reduce[tid] = local_sum;
    s_reduce[tid + blockDim.x] = local_phi_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
            s_reduce[tid + blockDim.x] += s_reduce[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    float mean_val = s_reduce[0] / (float)d;
    float phi_val = s_reduce[blockDim.x] * 0.5f;
    
    float v_prev = (v_mem != nullptr) ? v_mem[token_idx] : 0.0f;
    float v_new = gamma_param * v_prev + (1.0f - gamma_param) * phi_val;
    
    if (tid == 0) {
        v_out[token_idx] = v_new;
    }
    
    for (int i = tid; i < d; i += blockDim.x) {
        float g_i = (i < d) ? g_inv[i] : 1.0f;
        
        float term1 = -eta * g_i * s_phi[i];
        
        float term2 = alpha * (s_x[i] - mean_val);
        
        float term3 = 0.0f;
        if (beta != 0.0f && laplacian != nullptr && seq_idx < seq_len) {
            int start_j = (seq_idx > window) ? (seq_idx - window) : 0;
            for (int j = start_j; j < seq_idx; ++j) {
                float l_ij = laplacian[seq_idx * seq_len + j];
                const float* x_j = x + (token_idx - seq_idx + j) * d;
                term3 += l_ij * x_j[i];
            }
            float l_ii = laplacian[seq_idx * seq_len + seq_idx];
            term3 += l_ii * s_x[i];
            term3 *= beta;
        }
        
        s_velocity[i] = term1 + term2 + term3;
    }
    __syncthreads();
    
    float local_v_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        local_v_sq += s_velocity[i] * s_velocity[i];
    }
    s_reduce[tid] = local_v_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float v_norm_sq = s_reduce[0];
    
    float local_v_max = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float a = fabsf(s_velocity[i]);
        if (a > local_v_max) {
            local_v_max = a;
        }
    }
    s_reduce[tid] = local_v_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_reduce[tid + stride] > s_reduce[tid]) {
                s_reduce[tid] = s_reduce[tid + stride];
            }
        }
        __syncthreads();
    }
    float max_vel = s_reduce[0];
    if (max_vel > 5.0f) {
        float scale = 5.0f / max_vel;
        for (int i = tid; i < d; i += blockDim.x) {
            s_velocity[i] *= scale;
        }
        __syncthreads();
    }

    if (fabsf(curvature) > RSULF_EPS) {
        for (int j = tid; j < r; j += blockDim.x) {
            float acc_v1 = 0.0f;
            float acc_u2 = 0.0f;
            for (int i = 0; i < d; ++i) {
                float x_i = s_x[i];
                acc_v1 += x_i * v1[i * r + j];
                acc_u2 += x_i * u2_t[j * d + i];
            }
            s_h1[j] = acc_v1;
            s_xu2[j] = acc_u2;
        }
        __syncthreads();
    }
    
    for (int i = tid; i < d; i += blockDim.x) {
        float velocity = s_velocity[i];
        
        float delta = 0.0f;
        if (fabsf(curvature) > RSULF_EPS) {
            delta = -0.5f * curvature * v_norm_sq * s_x[i];
        }

        float gamma = 0.0f;
        if (fabsf(curvature) > RSULF_EPS) {
            for (int j = 0; j < r; ++j) {
                float z = s_h1[j] * s_xu2[j];
                gamma += z * v1[i * r + j];
            }
            gamma *= curvature * (1.0f / (float)r);
        }
        
        float x_next = s_x[i] + velocity + delta + gamma;
        x_o[i] = x_next;
    }
}

extern "C" void rsulf_unified_forward_cuda(
    const float* x,
    const float* v1,
    const float* s1,
    const float* u1,
    const float* v2,
    const float* s2,
    const float* u2,
    const float* g_inv,
    const float* laplacian,
    const float* v_mem,
    float eta,
    float alpha,
    float beta,
    float gamma_param,
    float curvature,
    int batch,
    int seq_len,
    int d,
    int r,
    int ffn_dim,
    int window,
    float* x_out,
    float* v_out
) {
    int total_tokens = batch * seq_len;
    size_t smem_size = (3 * d + 2 * r + ffn_dim + 512) * sizeof(float);
    int block_size = 256;
    dim3 grid(total_tokens);
    dim3 block(block_size);
    rsulf_unified_forward_kernel<<<grid, block, smem_size>>>(
        x,
        v1,
        s1,
        u1,
        v2,
        s2,
        u2,
        g_inv,
        laplacian,
        v_mem,
        eta,
        alpha,
        beta,
        gamma_param,
        curvature,
        batch,
        seq_len,
        d,
        r,
        ffn_dim,
        window,
        x_out,
        v_out
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rsulf_unified_forward: %s\n", cudaGetErrorString(err));
    }
}
```
---
## File: `reality_stone/src/layers/cuda/spline_kernel.cu`

```cpp
// src/layers/cuda/spline_kernel.cu

#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cstdio>

// Warp-level helper to sum all thread values in a warp.
// Must be defined before kernels that use it.
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Catmull-Rom spline interpolation kernel
__global__ void spline_interpolate_kernel(
    const float* __restrict__ control_points,  // (k+1) x in_features
    float* __restrict__ weights,               // out_features x in_features
    const int k,
    const int in_features,
    const int out_features
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_features * in_features;
    
    if (tid >= total_elements) return;
    
    const int out_idx = tid / in_features;
    const int in_idx = tid % in_features;
    
    // Normalize output index to [0, 1]
    const float t = (float)out_idx / (out_features - 1);
    const float t_scaled = t * k;
    
    // Compute control point index (with clamping)
    int j = (int)floorf(t_scaled);
    j = max(1, min(j, k - 2));
    
    const float t_local = t_scaled - j;
    
    // Load 4 control points
    const float p0 = control_points[(j - 1) * in_features + in_idx];
    const float p1 = control_points[j * in_features + in_idx];
    const float p2 = control_points[(j + 1) * in_features + in_idx];
    const float p3 = control_points[(j + 2) * in_features + in_idx];
    
    // Catmull-Rom coefficients
    const float t2 = t_local * t_local;
    const float t3 = t2 * t_local;
    
    const float c0 = -0.5f * t3 + t2 - 0.5f * t_local;
    const float c1 = 1.5f * t3 - 2.5f * t2 + 1.0f;
    const float c2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t_local;
    const float c3 = 0.5f * t3 - 0.5f * t2;
    
    // Compute interpolated value
    weights[out_idx * in_features + in_idx] = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;
}

// Spline-based GEMM kernel (matrix multiply of input and interpolated weights)
__global__ void spline_gemm_kernel(
    const float* __restrict__ input,           // batch_size x in_features
    const float* __restrict__ control_points,  // (k+1) x in_features
    float* __restrict__ output,                // batch_size x out_features
    const int batch_size,
    const int k,
    const int in_features,
    const int out_features
) {
    // Each block computes one output element
    const int batch_idx = blockIdx.y;
    const int out_idx = blockIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Load input into shared memory (tiling)
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    
    // Compute spline parameters
    const float t = (float)out_idx / (out_features - 1);
    const float t_scaled = t * k;
    int j = (int)floorf(t_scaled);
    j = max(1, min(j, k - 2));
    const float t_local = t_scaled - j;
    
    // Catmull-Rom 계수
    const float t2 = t_local * t_local;
    const float t3 = t2 * t_local;
    
    const float c0 = -0.5f * t3 + t2 - 0.5f * t_local;
    const float c1 = 1.5f * t3 - 2.5f * t2 + 1.0f;
    const float c2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t_local;
    const float c3 = 0.5f * t3 - 0.5f * t2;
    
    // Load input into shared memory
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        shared_input[i] = input[batch_idx * in_features + i];
    }
    __syncthreads();
    
    // Compute inner product
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        // 4개의 제어점에서 보간
        const float p0 = control_points[(j - 1) * in_features + i];
        const float p1 = control_points[j * in_features + i];
        const float p2 = control_points[(j + 1) * in_features + i];
        const float p3 = control_points[(j + 2) * in_features + i];
        
        const float weight = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;
        sum += shared_input[i] * weight;
    }
    
    // Warp reduction
    sum = warpReduceSum(sum);
    
    // First thread stores the result
    if (threadIdx.x == 0) {
        output[batch_idx * out_features + out_idx] = sum;
    }
}

// FP16 version (placeholder for future Tensor Core optimization)
__global__ void spline_gemm_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ control_points,
    half* __restrict__ output,
    const int batch_size,
    const int k,
    const int in_features,
    const int out_features
) {
    // TODO: FP16/Tensor Core 구현
}

// Backward kernel: compute gradients for control points
__global__ void spline_backward_kernel(
    const float* __restrict__ grad_output,     // (batch_size, out_features)
    const float* __restrict__ input,           // (batch_size, in_features)
    float* __restrict__ grad_control_points, // (k+1, in_features)
    const int batch_size,
    const int k,
    const int in_features,
    const int out_features
) {
    const int cp_idx = blockIdx.x; // (k+1) 제어점 중 하나
    const int in_f_idx = threadIdx.x; // in_features 중 하나

    if (cp_idx > k || in_f_idx >= in_features) return;

    float grad_sum = 0.0f;

    // Accumulate this control point's contribution over all (batch, out_feature)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features; ++j) {
            
            // 1. Check whether this control point (cp_idx) is used for output j
            const float t = (float)j / (out_features - 1);
            const float t_scaled = t * k;
            const int p1_idx = max(1, min((int)floorf(t_scaled), k - 2));

            // Catmull-Rom uses 4 control points (p0, p1, p2, p3)
            // p1 index is j, so p0=j-1, p2=j+1, p3=j+2
            if (cp_idx < p1_idx - 1 || cp_idx > p1_idx + 2) {
                continue; // this control point is not used
            }

            // 2. If used, compute Catmull-Rom coefficient for this control point
            const float t_local = t_scaled - p1_idx;
            const float t2 = t_local * t_local;
            const float t3 = t2 * t_local;
            float c = 0.0f;
            
            if (cp_idx == p1_idx - 1) {       // p0
                c = -0.5f * t3 + t2 - 0.5f * t_local;
            } else if (cp_idx == p1_idx) {    // p1
                c = 1.5f * t3 - 2.5f * t2 + 1.0f;
            } else if (cp_idx == p1_idx + 1) {  // p2
                c = -1.5f * t3 + 2.0f * t2 + 0.5f * t_local;
            } else {                          // p3
                c = 0.5f * t3 - 0.5f * t2;
            }
            
            // 3. Chain rule: dL/dCp = dL/dOut * dOut/dW * dW/dCp
            // dOut/dW = input,  dW/dCp = c
            grad_sum += grad_output[i * out_features + j] * input[i * in_features + in_f_idx] * c;
        }
    }
    
    // Atomically accumulate the computed gradient
    atomicAdd(&grad_control_points[cp_idx * in_features + in_f_idx], grad_sum);
}


// C++ interface
extern "C" {

void spline_interpolate_cuda(
    const float* control_points,
    float* weights,
    int k,
    int in_features,
    int out_features
) {
    const int total_elements = out_features * in_features;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    spline_interpolate_kernel<<<blocks, threads_per_block>>>(
        control_points, weights, k, in_features, out_features
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in spline_interpolate: %s\n", cudaGetErrorString(error));
    }
}

void spline_forward_cuda(
    const float* input,
    const float* control_points,
    float* output,
    int batch_size,
    int k,
    int in_features,
    int out_features
) {
    // 블록 구성: (out_features, batch_size)
    dim3 blocks(out_features, batch_size);
    const int threads_per_block = 128;
    const int shared_mem_size = in_features * sizeof(float);
    
    spline_gemm_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input, control_points, output, batch_size, k, in_features, out_features
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in spline_forward: %s\n", cudaGetErrorString(error));
    }
}

void spline_backward_cuda(
    const float* grad_output,
    const float* input,
    float* grad_control_points,
    int batch_size,
    int k,
    int in_features,
    int out_features
) {
    // Initialize control point gradients
    cudaMemset(grad_control_points, 0, (k + 1) * in_features * sizeof(float));

    // Blocks: number of control points (k+1)
    // Threads: in_features
    dim3 blocks(k + 1);
    dim3 threads(in_features);
    
    spline_backward_kernel<<<blocks, threads>>>(
        grad_output, input, grad_control_points,
        batch_size, k, in_features, out_features
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in spline_backward: %s\n", cudaGetErrorString(error));
    }
}

} // extern "C"
```
---
## File: `reality_stone/src/layers/cuda/test_kernels.cu`

```cpp
/**
 * CUDA kernel unit tests
 * Compile: nvcc -std=c++11 -arch=sm_70 -I.. test_kernels.cu poincare.cu lorentz.cu klein.cu mobius.cu -o test_kernels
 * Run:     ./test_kernels
 */

#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define EPSILON 1e-5f
#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        printf("FAIL: %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
        return false; \
    }

#define TEST_ASSERT_NEAR(a, b, eps, msg) \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s\n  expected=%.6f got=%.6f diff=%.6e (tol=%.6e)\n  at %s:%d\n", \
               msg, (b), (a), fabsf((a)-(b)), (eps), __FILE__, __LINE__); \
        return false; \
    }

extern "C" {
    void poincare_distance_cuda(float* out, const float* x, const float* y, float c, long long batch_size, long long dim);
    void poincare_ball_layer_cuda(float* out, const float* u, const float* v, float c, float t, long long batch_size, long long dim);
    
    void lorentz_distance_cuda(float* out, const float* u, const float* v, float c, int batch_size, int dim);
    void lorentz_layer_forward_cuda(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim);
    
    void klein_distance_cuda(float* out, const float* u, const float* v, float c, int batch_size, int dim);
    void klein_layer_forward_cuda(float* out, const float* u, const float* v, float c, float t, int batch_size, int dim);
    
    void mobius_add_cuda(float* out, const float* u, const float* v, float c, int64_t batch_size, int64_t dim);
    void mobius_scalar_cuda(float* out, const float* u, float c, float r, int64_t batch_size, int64_t dim);
    void poincare_riemannian_adam_cuda(
        float* x,
        const float* grad,
        float* m,
        float* v,
        float c,
        float lr,
        float beta1,
        float beta2,
        float eps,
        long long batch_size,
        long long dim,
        long long step);
}

// Helper: allocate and copy to GPU
float* to_gpu(const float* host, int size) {
    float* dev;
    cudaMalloc(&dev, size * sizeof(float));
    cudaMemcpy(dev, host, size * sizeof(float), cudaMemcpyHostToDevice);
    return dev;
}

// Helper: copy from GPU and free
void from_gpu(float* host, float* dev, int size) {
    cudaMemcpy(host, dev, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);
}

// ============================================================================
// Poincare Tests
// ============================================================================

bool test_poincare_distance_same_point() {
    printf("Poincare distance: same point ... ");
    
    float x[] = {0.1f, 0.2f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    poincare_distance_cuda(d_out, d_x, d_x, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-4f, "Poincare distance to self should be 0");
    printf("PASS\n");
    return true;
}

bool test_poincare_distance_origin() {
    printf("Poincare distance: origin ... ");
    
    float x[] = {0.0f, 0.0f};
    float y[] = {0.0f, 0.0f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    poincare_distance_cuda(d_out, d_x, d_y, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    cudaFree(d_y);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-5f, "Poincare distance at origin should be 0");
    printf("PASS\n");
    return true;
}

bool test_poincare_ball_layer_interpolation() {
    printf("Poincare layer: endpoints t=0, t=1 ... ");
    
    float u[] = {0.3f, 0.4f};
    float v[] = {-0.2f, 0.1f};
    float c = 1.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_v = to_gpu(v, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    // t=0 should return u
    poincare_ball_layer_cuda(d_out, d_u, d_v, c, 0.0f, 1, 2);
    float result_t0[2];
    cudaMemcpy(result_t0, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result_t0[0], u[0], EPSILON, "t=0: x component should match u");
    TEST_ASSERT_NEAR(result_t0[1], u[1], EPSILON, "t=0: y component should match u");
    
    // t=1 should return v
    poincare_ball_layer_cuda(d_out, d_u, d_v, c, 1.0f, 1, 2);
    float result_t1[2];
    cudaMemcpy(result_t1, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result_t1[0], v[0], EPSILON, "t=1: x component should match v");
    TEST_ASSERT_NEAR(result_t1[1], v[1], EPSILON, "t=1: y component should match v");
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

bool test_poincare_distance_symmetry() {
    printf("Poincare distance: symmetry d(x,y)=d(y,x) ... ");

    float x[] = {0.1f, 0.2f};
    float y[] = {0.2f, -0.1f};
    float c = 1.0f;

    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_out1;
    float* d_out2;
    cudaMalloc(&d_out1, sizeof(float));
    cudaMalloc(&d_out2, sizeof(float));

    poincare_distance_cuda(d_out1, d_x, d_y, c, 1, 2);
    poincare_distance_cuda(d_out2, d_y, d_x, c, 1, 2);

    float d_xy;
    float d_yx;
    from_gpu(&d_xy, d_out1, 1);
    from_gpu(&d_yx, d_out2, 1);
    cudaFree(d_x);
    cudaFree(d_y);

    TEST_ASSERT_NEAR(d_xy, d_yx, 1e-5f, "Poincare distance symmetry violated");
    printf("PASS\n");
    return true;
}

bool test_poincare_triangle_inequality() {
    printf("Poincare distance: triangle inequality ... ");

    float x[] = {0.0f, 0.0f};
    float y[] = {0.1f, 0.1f};
    float z[] = {0.2f, -0.05f};
    float c = 1.0f;

    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_z = to_gpu(z, 2);

    float* d_out;
    cudaMalloc(&d_out, sizeof(float));

    poincare_distance_cuda(d_out, d_x, d_y, c, 1, 2);
    float d_xy;
    from_gpu(&d_xy, d_out, 1);

    cudaMalloc(&d_out, sizeof(float));
    poincare_distance_cuda(d_out, d_y, d_z, c, 1, 2);
    float d_yz;
    from_gpu(&d_yz, d_out, 1);

    cudaMalloc(&d_out, sizeof(float));
    poincare_distance_cuda(d_out, d_x, d_z, c, 1, 2);
    float d_xz;
    from_gpu(&d_xz, d_out, 1);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    TEST_ASSERT(d_xz <= d_xy + d_yz + 1e-4f, "Poincare triangle inequality violated");
    printf("PASS\n");
    return true;
}

bool test_poincare_ball_layer_inside_ball() {
    printf("Poincare layer: t=0.5 stays inside ball ... ");

    float u[] = {0.3f, 0.4f};
    float v[] = {-0.2f, 0.1f};
    float c = 1.0f;

    float* d_u = to_gpu(u, 2);
    float* d_v = to_gpu(v, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    poincare_ball_layer_cuda(d_out, d_u, d_v, c, 0.5f, 1, 2);
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(result[0] * result[0] + result[1] * result[1]);
    float radius = 1.0f / sqrtf(c);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);

    TEST_ASSERT(norm < radius - 1e-3f, "Poincare interpolation exceeds ball boundary");
    printf("PASS\n");
    return true;
}

// ============================================================================
// Lorentz Tests
// ============================================================================

bool test_lorentz_distance_same_point() {
    printf("Lorentz distance: same point ... ");
    
    // Point on hyperboloid: x0 = sqrt(1/c + ||x||²)
    float c = 1.0f;
    float space_norm_sq = 0.1f * 0.1f + 0.2f * 0.2f;
    float x0 = sqrtf(1.0f / c + space_norm_sq);
    float x[] = {x0, 0.1f, 0.2f};
    
    float* d_x = to_gpu(x, 3);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    lorentz_distance_cuda(d_out, d_x, d_x, c, 1, 3);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-3f, "Lorentz distance to self should be 0");
    printf("PASS\n");
    return true;
}

bool test_lorentz_layer_interpolation() {
    printf("Lorentz layer: endpoints t=0, t=1 and constraint ... ");
    
    float c = 1.0f;
    float u_space_norm_sq = 0.3f * 0.3f + 0.4f * 0.4f;
    float u0 = sqrtf(1.0f / c + u_space_norm_sq);
    float u[] = {u0, 0.3f, 0.4f};
    float v_space_norm_sq = (-0.2f) * (-0.2f) + 0.1f * 0.1f;
    float v0 = sqrtf(1.0f / c + v_space_norm_sq);
    float v[] = {v0, -0.2f, 0.1f};
    
    float* d_u = to_gpu(u, 3);
    float* d_v = to_gpu(v, 3);
    float* d_out;
    cudaMalloc(&d_out, 3 * sizeof(float));
    
    lorentz_layer_forward_cuda(d_out, d_u, d_v, c, 0.0f, 1, 3);
    float result_t0[3];
    cudaMemcpy(result_t0, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result_t0[0], u[0], 1e-4f, "t=0: time component mismatch with u[0]");
    TEST_ASSERT_NEAR(result_t0[1], u[1], 1e-4f, "t=0: spatial component u[1] mismatch");
    TEST_ASSERT_NEAR(result_t0[2], u[2], 1e-4f, "t=0: spatial component u[2] mismatch");

    lorentz_layer_forward_cuda(d_out, d_u, d_v, c, 1.0f, 1, 3);
    float result_t1[3];
    cudaMemcpy(result_t1, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    TEST_ASSERT_NEAR(result_t1[0], v[0], 1e-4f, "t=1: time component mismatch with v[0]");
    TEST_ASSERT_NEAR(result_t1[1], v[1], 1e-4f, "t=1: spatial component v[1] mismatch");
    TEST_ASSERT_NEAR(result_t1[2], v[2], 1e-4f, "t=1: spatial component v[2] mismatch");

    float diff_u = fabsf(u[0] * u[0] - (u[1] * u[1] + u[2] * u[2]) - 1.0f / c);
    float diff_v = fabsf(v[0] * v[0] - (v[1] * v[1] + v[2] * v[2]) - 1.0f / c);
    float diff_t0 = fabsf(result_t0[0] * result_t0[0] - (result_t0[1] * result_t0[1] + result_t0[2] * result_t0[2]) - 1.0f / c);
    float diff_t1 = fabsf(result_t1[0] * result_t1[0] - (result_t1[1] * result_t1[1] + result_t1[2] * result_t1[2]) - 1.0f / c);

    TEST_ASSERT(diff_u < 1e-4f, "input u violates Lorentz hyperboloid constraint");
    TEST_ASSERT(diff_v < 1e-4f, "input v violates Lorentz hyperboloid constraint");
    TEST_ASSERT(diff_t0 < 1e-3f, "t=0 output violates Lorentz hyperboloid constraint");
    TEST_ASSERT(diff_t1 < 1e-3f, "t=1 output violates Lorentz hyperboloid constraint");
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

// ============================================================================
// Klein Tests
// ============================================================================

bool test_klein_distance_same_point() {
    printf("Klein distance: same point ... ");
    
    float x[] = {0.1f, 0.2f};
    float c = 1.0f;
    
    float* d_x = to_gpu(x, 2);
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    klein_distance_cuda(d_out, d_x, d_x, c, 1, 2);
    
    float result;
    from_gpu(&result, d_out, 1);
    cudaFree(d_x);
    
    TEST_ASSERT_NEAR(result, 0.0f, 1e-3f, "Klein distance to self should be 0");
    printf("PASS\n");
    return true;
}

bool test_klein_layer_inside_ball() {
    printf("Klein layer: t=0.5 stays inside ball ... ");

    float x[] = {0.1f, 0.2f};
    float y[] = {-0.1f, 0.1f};
    float c = 1.0f;

    float* d_x = to_gpu(x, 2);
    float* d_y = to_gpu(y, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    klein_layer_forward_cuda(d_out, d_x, d_y, c, 0.5f, 1, 2);
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(result[0] * result[0] + result[1] * result[1]);
    float radius = 1.0f / sqrtf(c);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    TEST_ASSERT(norm < radius - 1e-3f, "Klein layer output exceeds ball boundary");
    printf("PASS\n");
    return true;
}

// ============================================================================
// Möbius Tests
// ============================================================================

bool test_mobius_add_identity() {
    printf("Mobius add: identity u+0=u ... ");
    
    float u[] = {0.1f, 0.2f};
    float zero[] = {0.0f, 0.0f};
    float c = 1.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_zero = to_gpu(zero, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_add_cuda(d_out, d_u, d_zero, c, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result[0], u[0], EPSILON, "u+0: first component mismatch");
    TEST_ASSERT_NEAR(result[1], u[1], EPSILON, "u+0: second component mismatch");
    
    cudaFree(d_u);
    cudaFree(d_zero);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

bool test_mobius_scalar_zero() {
    printf("Mobius scalar: r=0 ... ");
    
    float u[] = {0.3f, 0.4f};
    float c = 1.0f;
    float r = 0.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result[0], 0.0f, EPSILON, "r=0: first component not zero");
    TEST_ASSERT_NEAR(result[1], 0.0f, EPSILON, "r=0: second component not zero");
    
    cudaFree(d_u);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

bool test_mobius_scalar_identity() {
    printf("Mobius scalar: r=1 ... ");
    
    float u[] = {0.1f, 0.2f};
    float c = 1.0f;
    float r = 1.0f;
    
    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    
    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);
    
    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    TEST_ASSERT_NEAR(result[0], u[0], 1e-3f, "r=1: first component mismatch");
    TEST_ASSERT_NEAR(result[1], u[1], 1e-3f, "r=1: second component mismatch");
    
    cudaFree(d_u);
    cudaFree(d_out);
    
    printf("PASS\n");
    return true;
}

bool test_mobius_add_inside_ball() {
    printf("Mobius add: stays inside ball ... ");

    float u[] = {0.2f, 0.1f};
    float v[] = {0.1f, -0.1f};
    float c = 1.0f;

    float* d_u = to_gpu(u, 2);
    float* d_v = to_gpu(v, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    mobius_add_cuda(d_out, d_u, d_v, c, 1, 2);

    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(result[0] * result[0] + result[1] * result[1]);
    float radius = 1.0f;

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);

    TEST_ASSERT(norm < radius - 1e-3f, "Mobius add result exceeds ball boundary");
    printf("PASS\n");
    return true;
}

bool test_mobius_scalar_euclidean_limit() {
    printf("Mobius scalar: c=0 Euclidean limit ... ");

    float u[] = {0.3f, -0.4f};
    float c = 0.0f;
    float r = 2.0f;

    float* d_u = to_gpu(u, 2);
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));

    mobius_scalar_cuda(d_out, d_u, c, r, 1, 2);

    float result[2];
    cudaMemcpy(result, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_out);

    TEST_ASSERT_NEAR(result[0], r * u[0], 1e-6f, "c=0 mode mobius_scalar[0] mismatch");
    TEST_ASSERT_NEAR(result[1], r * u[1], 1e-6f, "c=0 mode mobius_scalar[1] mismatch");
    printf("PASS\n");
    return true;
}

bool test_poincare_riemannian_adam_euclidean_limit() {
    printf("Poincare Riemannian Adam: c=0 Euclidean limit ... ");

    const int batch_size = 1;
    const int dim = 2;
    float x_host[batch_size * dim] = {0.5f, -0.3f};
    float g_host[batch_size * dim] = {0.5f, -0.3f};
    float m_host[batch_size * dim] = {0.0f, 0.0f};
    float v_host[batch_size * dim] = {0.0f, 0.0f};

    float lr = 0.1f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    long long step = 1;
    float c = 0.0f;

    float* d_x = to_gpu(x_host, batch_size * dim);
    float* d_g = to_gpu(g_host, batch_size * dim);
    float* d_m = to_gpu(m_host, batch_size * dim);
    float* d_v = to_gpu(v_host, batch_size * dim);

    poincare_riemannian_adam_cuda(
        d_x,
        d_g,
        d_m,
        d_v,
        c,
        lr,
        beta1,
        beta2,
        eps,
        batch_size,
        dim,
        step);

    float x_new[batch_size * dim];
    float m_new[batch_size * dim];
    float v_new[batch_size * dim];
    from_gpu(x_new, d_x, batch_size * dim);
    from_gpu(m_new, d_m, batch_size * dim);
    from_gpu(v_new, d_v, batch_size * dim);

    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_m);
    cudaFree(d_v);

    float m_ref[batch_size * dim];
    float v_ref[batch_size * dim];
    for (int i = 0; i < batch_size * dim; ++i) {
        float g = g_host[i];
        m_ref[i] = beta1 * m_host[i] + (1.0f - beta1) * g;
        v_ref[i] = beta2 * v_host[i] + (1.0f - beta2) * g * g;
    }
    float bias_c1 = 1.0f - powf(beta1, (float)step);
    float bias_c2 = 1.0f - powf(beta2, (float)step);
    float u[batch_size * dim];
    for (int i = 0; i < batch_size * dim; ++i) {
        float m_hat = m_ref[i] / bias_c1;
        float v_hat = v_ref[i] / bias_c2;
        u[i] = -lr * m_hat / (sqrtf(v_hat) + eps);
    }
    float x_ref[batch_size * dim];
    for (int i = 0; i < batch_size * dim; ++i) {
        x_ref[i] = x_host[i] + u[i];
    }

    for (int i = 0; i < batch_size * dim; ++i) {
        TEST_ASSERT_NEAR(x_new[i], x_ref[i], 1e-6f, "Poincare Riemannian Adam c=0 mismatch");
    }

    printf("PASS\n");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    printf("\n");
    printf("=======================================================\n");
    printf("        CUDA kernel unit tests\n");
    printf("=======================================================\n\n");
    
    int passed = 0;
    int total = 0;
    
    printf("Poincare Tests:\n");
    total++; if (test_poincare_distance_same_point()) passed++;
    total++; if (test_poincare_distance_origin()) passed++;
    total++; if (test_poincare_ball_layer_interpolation()) passed++;
    total++; if (test_poincare_distance_symmetry()) passed++;
    total++; if (test_poincare_triangle_inequality()) passed++;
    total++; if (test_poincare_ball_layer_inside_ball()) passed++;
    
    printf("\nLorentz Tests:\n");
    total++; if (test_lorentz_distance_same_point()) passed++;
    total++; if (test_lorentz_layer_interpolation()) passed++;
    
    printf("\nKlein Tests:\n");
    total++; if (test_klein_distance_same_point()) passed++;
    total++; if (test_klein_layer_inside_ball()) passed++;
    
    printf("\nMobius Tests:\n");
    total++; if (test_mobius_add_identity()) passed++;
    total++; if (test_mobius_scalar_zero()) passed++;
    total++; if (test_mobius_scalar_identity()) passed++;
    total++; if (test_mobius_add_inside_ball()) passed++;
    total++; if (test_mobius_scalar_euclidean_limit()) passed++;
    total++; if (test_poincare_riemannian_adam_euclidean_limit()) passed++;
    
    printf("\n=======================================================\n");
    printf("Result: %d/%d tests passed", passed, total);
    if (passed == total) {
        printf(" [OK]\n");
    } else {
        printf(" [FAIL]\n");
    }
    printf("=======================================================\n\n");
    
    return (passed == total) ? 0 : 1;
}
```
---
## File: `reality_stone/src/layers/decoder.rs`

```rust
use super::rsulf::{randomized_svd, GlobalBasis};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

pub struct RiemannianDecoder {
    pub d_model: usize,
    pub k: usize,
    pub r: usize,
    pub vocab: usize,
    pub u: Array2<f32>,
    pub a: Array2<f32>,
    pub bt: Array2<f32>,
    pub bias: Array1<f32>,
}

impl RiemannianDecoder {
    pub fn new(u: Array2<f32>, a: Array2<f32>, bt: Array2<f32>, bias: Array1<f32>) -> Self {
        let d_model = u.nrows();
        let k = u.ncols();
        let vocab = bt.nrows();
        let r = bt.ncols();
        Self {
            d_model,
            k,
            r,
            vocab,
            u,
            a,
            bt,
            bias,
        }
    }

    pub fn from_lm_head(
        w_lm: ArrayView2<f32>,
        b_lm: ArrayView1<f32>,
        global_basis: &GlobalBasis,
        target_rank: usize,
    ) -> Self {
        let vocab = w_lm.nrows();
        let d_model = w_lm.ncols();
        let u_basis = global_basis.u.view();
        let k_basis = global_basis.rank.min(d_model);
        let u = u_basis.slice(s![.., 0..k_basis]).to_owned();
        let w_tilde = w_lm.dot(&u);
        let max_rank = target_rank.min(k_basis).min(vocab);
        let (u_svd, s_svd, v_svd) = randomized_svd(&w_tilde, max_rank, 20, 5);
        let r = u_svd.ncols();
        let mut bt = Array2::<f32>::zeros((vocab, r));
        for j in 0..r {
            let s_val = s_svd[j].sqrt();
            for i in 0..vocab {
                bt[[i, j]] = u_svd[[i, j]] * s_val;
            }
        }
        let mut a = Array2::<f32>::zeros((r, k_basis));
        for j in 0..r {
            let s_val = s_svd[j].sqrt();
            for i in 0..k_basis {
                a[[j, i]] = s_val * v_svd[[i, j]];
            }
        }
        let bias = b_lm.to_owned();
        RiemannianDecoder::new(u, a, bt, bias)
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let c = x.dot(&self.u);
        let q = c.dot(&self.a.t());
        let mut logits = q.dot(&self.bt.t());
        for i in 0..self.vocab {
            let b = self.bias[i];
            for j in 0..logits.nrows() {
                logits[[j, i]] += b;
            }
        }
        logits
    }
}
```
---
## File: `reality_stone/src/layers/diffusion.rs`

```rust
// ============================================================================
// 파일: src/layers/diffusion.rs
// 목적: 리만 라그랑지안 디퓨전 (Riemannian Lagrangian Diffusion) 구현
// ============================================================================

use crate::layers::geodesic;
use crate::layers::metric::DiagonalMetric;
use ndarray::{Array2, ArrayView2};

/// 리만 라그랑지안 디퓨전 상태 관리
pub struct RiemannianDiffusion {
    pub metric: DiagonalMetric,
    pub alpha: f32, // 에너지 감쇠 계수 (0.0 ~ 1.0)
    pub dt: f32,    // 시간 간격
}

impl RiemannianDiffusion {
    pub fn new(dim: usize, alpha: f32, dt: f32) -> Self {
        Self {
            metric: DiagonalMetric::new(dim),
            alpha,
            dt,
        }
    }

    /// 디퓨전 스텝: h(t+1) = Exp_h(t) ( -∇E * dt )
    /// 여기서 에너지는 잠재 에너지(Potential)와 운동 에너지(Kinetic)의 상호작용으로 정의됩니다.
    /// 단순화된 모델: 흐름(Flow)을 접공간(Tangent Space)에서의 벡터장으로 해석하고,
    /// 지수 맵(Exponential Map)을 통해 다양체 위로 업데이트합니다.
    pub fn step(
        &self,
        h: &ArrayView2<f32>,          // 현재 상태 (Batch, Hidden)
        flow_field: &ArrayView2<f32>, // 흐름 벡터장 (Batch, Hidden) - 예를 들어 tanh(h @ W)
    ) -> Array2<f32> {
        // 1. 접공간에서의 업데이트 방향 계산
        // dH = -alpha * H + (1-alpha) * Flow
        // 여기서는 사용자 코드의 수식: h_new = alpha * h + (1-alpha) * tanh(flow) 를
        // 리만 관점에서 해석:
        // Tangent Vector v = (1-alpha) * (Flow - h)  (유클리드 근사)
        // 혹은 더 정확하게는, Flow가 목표 지점이라면 Geodesic 방향.

        // 사용자의 수식을 그대로 따르되, 리만 지수 맵을 사용하여 이동
        // h_next = h + (1-alpha) * (Flow - h) * dt  (유클리드 Euler)
        // -> v = (Flow - h) * (1-alpha)
        // -> h_next = Exp_h(v * dt)

        // Flow field는 이미 활성화 함수가 적용된 상태라고 가정 (외부에서 계산)
        let delta = flow_field - h;
        let tangent_vector = &delta * (1.0 - self.alpha);

        // 2. 지수 맵을 사용하여 업데이트 (Manifold 제약 조건 유지)
        // Diagonal Metric을 고려한 지수 맵 사용
        // 여기서는 간단히 유클리드에 가까운 근사를 사용하거나,
        // 실제 geodesic 모듈을 활용.

        // MetricTensor trait을 통해 exponential map 호출
        // geodesic::exponential_map expects &MetricType enum wrapper
        let metric_enum = crate::layers::metric::MetricType::Diagonal(self.metric.clone());
        geodesic::exponential_map(&metric_enum, h, &tangent_vector.view(), self.dt)
    }

    /// 가중치 기반 에너지 흐름 계산 (Rust 내부에서 처리할 경우)
    pub fn compute_flow(
        &self,
        h: &ArrayView2<f32>,
        weights: &ArrayView2<f32>, // (Hidden, Hidden)
    ) -> Array2<f32> {
        let linear = h.dot(weights);
        linear.mapv(|x| x.tanh())
    }
}
```
---
## File: `reality_stone/src/layers/geodesic.rs`

```rust
// ============================================================================
// 파일: src/layers/geodesic.rs
// 목적: 측지선 흐름 및 exponential/logarithmic map
// ============================================================================

use super::metric::MetricType;
use ndarray::{Array2, ArrayView2};

const EPS: f32 = 1e-7;
const MAX_GEODESIC_STEPS: usize = 100;

/// 지수 사상 (Exponential Map): Exp_x(v)
/// 점 x에서 tangent vector v 방향으로 이동
pub fn exponential_map(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    step_size: f32,
) -> Array2<f32> {
    match metric {
        MetricType::Poincare(m) => {
            crate::layers::poincare::poincare_exp_at(x, v, m.curvature, 1e-5)
        }
        MetricType::Lorentz(m) => {
            // Lorentz exp_0: v (tangent) → hyperboloid
            if is_at_origin(x) {
                crate::layers::lorentz::lorentz_exp0_space(v, m.curvature)
            } else {
                // General point: use geodesic flow
                exponential_map_generic(metric, x, v, step_size)
            }
        }
        MetricType::Klein(_) => {
            // Klein uses geodesic flow
            exponential_map_generic(metric, x, v, step_size)
        }
        MetricType::Diagonal(_) => {
            // Euclidean-like: Exp_x(v) ≈ x + v
            x + &(v * step_size)
        }
    }
}

/// 로그 사상 (Logarithmic Map): Log_x(y)
/// 두 점 x, y를 연결하는 tangent vector
pub fn logarithmic_map(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Array2<f32> {
    match metric {
        MetricType::Poincare(m) => {
            crate::layers::poincare::poincare_log_at(x, y, m.curvature, 1e-5)
        }
        MetricType::Lorentz(m) => {
            if is_at_origin(x) {
                crate::layers::lorentz::lorentz_log0_space(y, m.curvature)
            } else {
                logarithmic_map_generic(metric, x, y)
            }
        }
        MetricType::Klein(_) => logarithmic_map_generic(metric, x, y),
        MetricType::Diagonal(_) => {
            // Euclidean: Log_x(y) = y - x
            y - x
        }
    }
}

/// 일반적인 exponential map (측지선 방정식 수치 적분)
fn exponential_map_generic(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    step_size: f32,
) -> Array2<f32> {
    let metric_trait = metric.as_trait();
    let batch_size = x.nrows();
    let dim = x.ncols();

    let mut position = x.to_owned();
    let mut velocity = v * step_size;
    let dt = 0.01; // 작은 시간 스텝
    let num_steps = (step_size / dt).ceil() as usize;

    for _ in 0..num_steps.min(MAX_GEODESIC_STEPS) {
        // 측지선 방정식: d²x^k/dt² + Γ^k_ij dx^i/dt dx^j/dt = 0
        let christoffel = metric_trait.christoffel_symbols(&position.view());

        let mut acceleration = Array2::zeros((batch_size, dim));
        for b in 0..batch_size {
            for k in 0..dim {
                let mut acc = 0.0;
                // 대각 근사: Γ^k_ii만 고려
                for i in 0..dim {
                    acc -= christoffel[b][[k, i]] * velocity[[b, i]] * velocity[[b, i]];
                }
                acceleration[[b, k]] = acc;
            }
        }

        // Velocity Verlet integration
        velocity = &velocity + &(&acceleration * (dt * 0.5));
        position = &position + &(&velocity * dt);
        velocity = &velocity + &(&acceleration * (dt * 0.5));
    }

    position
}

/// 일반적인 logarithmic map (역문제)
fn logarithmic_map_generic(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Array2<f32> {
    // 초기 추정: v_0 = (y - x)
    let mut v = y - x;

    // Newton 방법으로 Exp_x(v) = y를 만족하는 v 찾기
    for _ in 0..10 {
        let exp_v = exponential_map(metric, x, &v.view(), 1.0);
        let residual = &exp_v - y;
        let residual_norm = crate::ops::norm_sq_batched(&residual.view()).mapv(|n| n.sqrt());

        if residual_norm.mean().unwrap() < EPS {
            break;
        }

        // v 업데이트: v -= learning_rate * residual
        v = &v - &(&residual * 0.5);
    }

    v
}

/// 측지선 보간: γ(t) = Exp_x(t * Log_x(y))
pub fn geodesic_interpolation(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    t: f32,
) -> Array2<f32> {
    match metric {
        MetricType::Poincare(m) => {
            crate::layers::poincare::poincare_ball_layer(x, y, m.curvature, t)
        }
        MetricType::Lorentz(m) => {
            crate::layers::lorentz::lorentz_layer_forward(x, y, m.curvature, t)
        }
        MetricType::Klein(m) => crate::layers::klein::klein_layer_forward(x, y, m.curvature, t),
        MetricType::Diagonal(_) => {
            // Linear interpolation
            x * (1.0 - t) + y * t
        }
    }
}

/// 측지선 경로 생성: x → y를 num_steps개 점으로 나눔
pub fn geodesic_path(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    num_steps: usize,
) -> Vec<Array2<f32>> {
    let mut path = Vec::with_capacity(num_steps);

    for i in 0..num_steps {
        let t = i as f32 / (num_steps - 1).max(1) as f32;
        let point = geodesic_interpolation(metric, x, y, t);
        path.push(point);
    }

    path
}

/// 평행 이동 (Parallel Transport)
/// tangent vector v를 x에서 y로 이동
pub fn parallel_transport(
    metric: &MetricType,
    v: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Array2<f32> {
    let metric_trait = metric.as_trait();

    // 측지선을 따라 v를 이동
    let path = geodesic_path(metric, x, y, 10);
    let mut transported_v = v.to_owned();

    for i in 0..(path.len() - 1) {
        let christoffel = metric_trait.christoffel_symbols(&path[i].view());
        let dx = &path[i + 1] - &path[i];

        // dv^k/dt = -Γ^k_ij v^i dx^j/dt (대각 근사)
        let batch_size = transported_v.nrows();
        let dim = transported_v.ncols();

        for b in 0..batch_size {
            for k in 0..dim {
                let mut correction = 0.0;
                for i in 0..dim {
                    correction -= christoffel[b][[k, i]] * transported_v[[b, i]] * dx[[b, i]];
                }
                transported_v[[b, k]] += correction;
            }
        }
    }

    transported_v
}

/// 측지선 거리 계산 (메트릭 기반)
pub fn geodesic_distance(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> ndarray::Array1<f32> {
    metric.as_trait().distance(x, y)
}

// 유틸리티
fn is_at_origin(x: &ArrayView2<f32>) -> bool {
    crate::ops::norm_sq_batched(x).iter().all(|&n| n < EPS)
}

#[cfg(test)]
mod tests {
    use super::super::metric::*;
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_geodesic_interpolation_euclidean() {
        let metric = MetricType::Diagonal(DiagonalMetric::new(2));
        let x = arr2(&[[0.0, 0.0]]);
        let y = arr2(&[[1.0, 1.0]]);

        let mid = geodesic_interpolation(&metric, &x.view(), &y.view(), 0.5);
        assert!((mid[[0, 0]] - 0.5).abs() < 1e-5);
        assert!((mid[[0, 1]] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_geodesic_path() {
        let metric = MetricType::Diagonal(DiagonalMetric::new(2));
        let x = arr2(&[[0.0, 0.0]]);
        let y = arr2(&[[1.0, 0.0]]);

        let path = geodesic_path(&metric, &x.view(), &y.view(), 5);
        assert_eq!(path.len(), 5);
        assert!((path[0][[0, 0]] - 0.0).abs() < 1e-5);
        assert!((path[4][[0, 0]] - 1.0).abs() < 1e-5);
    }
}
```
---
## File: `reality_stone/src/layers/human_decoder.rs`

```rust
use crate::layers::poincare::poincare_distance;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

const EPS: f32 = 1e-8;

#[derive(Clone, Copy)]
pub struct StageWeights {
    pub logit: f32,
    pub cosine: f32,
    pub geodesic: f32,
}

pub struct HumanStyleDecoder {
    embeddings: Array2<f32>,
    norms: Array1<f32>,
    skeleton: Vec<usize>,
    relation: Vec<usize>,
    object: Vec<usize>,
    relation_weights: StageWeights,
    object_weights: StageWeights,
    curvature: f32,
}

impl HumanStyleDecoder {
    pub fn new(
        embeddings: Array2<f32>,
        skeleton: Vec<usize>,
        relation: Vec<usize>,
        object: Vec<usize>,
        relation_weights: StageWeights,
        object_weights: StageWeights,
        curvature: f32,
    ) -> Self {
        let norm_vec = embeddings
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt().max(EPS))
            .collect::<Vec<f32>>();
        let norms = Array1::from_vec(norm_vec);
        Self {
            embeddings,
            norms,
            skeleton,
            relation,
            object,
            relation_weights,
            object_weights,
            curvature,
        }
    }

    fn masked_argmax(&self, logits: &ArrayView1<f32>, pool: &[usize]) -> Option<usize> {
        if pool.is_empty() {
            return None;
        }
        let mut best_idx = pool[0];
        let mut best_val = logits[best_idx];
        for &idx in pool.iter().skip(1) {
            let val = logits[idx];
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    fn select_topk(&self, logits: &ArrayView1<f32>, pool: &[usize], k: usize) -> Vec<usize> {
        if pool.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut pairs = pool
            .iter()
            .map(|idx| (*idx, logits[*idx]))
            .collect::<Vec<(usize, f32)>>();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k.min(pairs.len()));
        pairs.into_iter().map(|(idx, _)| idx).collect()
    }

    fn cosine_with_context(&self, idx: usize, context: &ArrayView1<f32>, ctx_norm: f32) -> f32 {
        let embed = self.embeddings.row(idx);
        let dot = embed.dot(context);
        dot / (self.norms[idx] * ctx_norm.max(EPS))
    }

    fn poincare_distance_single(&self, idx: usize, context: &ArrayView2<f32>) -> f32 {
        if self.curvature <= 0.0 {
            return self.euclidean_distance(idx, context);
        }
        let embed = self.embeddings.slice(s![idx..idx + 1, ..]);
        let dist = poincare_distance(&embed.view(), context, self.curvature, EPS);
        dist[0]
    }

    fn euclidean_distance(&self, idx: usize, context: &ArrayView2<f32>) -> f32 {
        let embed = self.embeddings.row(idx);
        let ctx = context.row(0);
        let diff = &embed - &ctx;
        diff.dot(&diff).sqrt()
    }

    fn select_relation(
        &self,
        logits: &ArrayView1<f32>,
        context: &ArrayView1<f32>,
        ctx_norm: f32,
        topk: usize,
    ) -> Option<usize> {
        let candidates = self.select_topk(logits, &self.relation, topk);
        if candidates.is_empty() {
            return None;
        }
        let mut best_idx = candidates[0];
        let mut best_score = f32::MIN;
        for idx in candidates {
            let logit_term = self.relation_weights.logit * logits[idx];
            let cos_term =
                self.relation_weights.cosine * self.cosine_with_context(idx, context, ctx_norm);
            let score = logit_term + cos_term;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    fn select_object(
        &self,
        logits: &ArrayView1<f32>,
        context_row: &ArrayView1<f32>,
        context_view: &ArrayView2<f32>,
        ctx_norm: f32,
        topk: usize,
    ) -> Option<usize> {
        let candidates = self.select_topk(logits, &self.object, topk);
        if candidates.is_empty() {
            return None;
        }
        let mut best_idx = candidates[0];
        let mut best_score = f32::MIN;
        for idx in candidates {
            let logit_term = self.object_weights.logit * logits[idx];
            let cos_term =
                self.object_weights.cosine * self.cosine_with_context(idx, context_row, ctx_norm);
            let geo_term =
                self.object_weights.geodesic * self.poincare_distance_single(idx, context_view);
            let score = logit_term + cos_term - geo_term;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    pub fn decode_batch(
        &self,
        logits: ArrayView2<f32>,
        relation_ctx: ArrayView2<f32>,
        object_ctx: ArrayView2<f32>,
        topk_relation: usize,
        topk_object: usize,
    ) -> Vec<usize> {
        assert_eq!(logits.nrows(), relation_ctx.nrows());
        assert_eq!(relation_ctx.nrows(), object_ctx.nrows());
        assert_eq!(relation_ctx.ncols(), self.embeddings.ncols());
        let batch = logits.nrows();
        let mut outputs = Vec::with_capacity(batch);
        for b in 0..batch {
            let log_row = logits.row(b);
            let rel_row = relation_ctx.row(b);
            let obj_row = object_ctx.row(b);
            let rel_norm = rel_row.dot(&rel_row).sqrt().max(EPS);
            let obj_norm = obj_row.dot(&obj_row).sqrt().max(EPS);
            let obj_view = object_ctx.slice(s![b..b + 1, ..]);
            let skel_choice = self.masked_argmax(&log_row, &self.skeleton);
            let rel_choice = self.select_relation(&log_row, &rel_row, rel_norm, topk_relation);
            let obj_choice =
                self.select_object(&log_row, &obj_row, &obj_view, obj_norm, topk_object);
            let token = obj_choice
                .or(rel_choice)
                .or(skel_choice)
                .unwrap_or_else(|| {
                    log_row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                });
            outputs.push(token);
        }
        outputs
    }
}
```
---
## File: `reality_stone/src/layers/hyper_metric.rs`

```rust
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct TinyMLP {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl TinyMLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            w1: Array2::zeros((input_dim, hidden_dim)),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::zeros((hidden_dim, output_dim)),
            b2: Array1::zeros(output_dim),
        }
    }

    pub fn from_weights(
        w1: Array2<f32>,
        b1: Array1<f32>,
        w2: Array2<f32>,
        b2: Array1<f32>,
    ) -> Self {
        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let h = x.dot(&self.w1) + &self.b1;
        let h_act = h.mapv(|v| v.max(0.0));

        h_act.dot(&self.w2) + &self.b2
    }
}

#[derive(Debug, Clone)]
pub struct HyperMetric {
    pub u_global: Array2<f32>,
    pub v_global: Array2<f32>,
    pub hypernet: TinyMLP,
    pub r: usize,
    pub d_model: usize,
}

impl HyperMetric {
    pub fn new(d_model: usize, r: usize, hyper_hidden: usize) -> Self {
        let hyper_input_dim = 64;
        let output_dim = r * r;

        Self {
            u_global: Array2::zeros((d_model, r)),
            v_global: Array2::zeros((d_model, r)),
            hypernet: TinyMLP::new(hyper_input_dim, hyper_hidden, output_dim),
            r,
            d_model,
        }
    }

    pub fn from_components(
        u_global: Array2<f32>,
        v_global: Array2<f32>,
        hypernet: TinyMLP,
    ) -> Self {
        let d_model = u_global.nrows();
        let r = u_global.ncols();
        Self {
            u_global,
            v_global,
            hypernet,
            r,
            d_model,
        }
    }

    pub fn generate_core(&self, layer_emb: &Array1<f32>) -> Array2<f32> {
        let flat_core = self.hypernet.forward(layer_emb);
        flat_core.into_shape((self.r, self.r)).unwrap().to_owned()
    }

    pub fn project_forward(&self, x: &Array2<f32>, layer_emb: &Array1<f32>) -> Array2<f32> {
        let core = self.generate_core(layer_emb);

        let x_proj = x.dot(&self.u_global);
        let x_core = x_proj.dot(&core);

        x_core.dot(&self.v_global.t())
    }
}
```
---
## File: `reality_stone/src/layers/klein.rs`

```rust
use crate::ops::{batch::EPS, dot_batched, norm_sq_batched};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

#[inline]
fn safe_sqrt(x: f32) -> f32 {
    x.max(EPS).sqrt()
}

#[inline]
fn safe_acosh(x: f32) -> f32 {
    (x.max(1.0 + EPS)).acosh()
}

const BOUNDARY_EPS: f32 = 1e-5;

/// 클라인 거리 (Klein Distance)
///
/// d_K(u,v) = (1/√c) * acosh((1 - c⟨u,v⟩) / √((1-c||u||²)(1-c||v||²)))
pub fn klein_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    let numerator = 1.0 - c * &uv;
    let denominator = ((1.0 - c * &u2) * (1.0 - c * &v2)).mapv(|z| z.max(EPS).sqrt());
    let arg = (&numerator / &denominator).mapv(|z| z.max(1.0 + EPS));
    arg.mapv(|r| safe_acosh(r) / sqrtc)
}

/// 클라인 덧셈 (Einstein Addition)
///
/// u (+) v = (1 / (1 + c<u,v>)) * (u + v/gamma_u + (c*gamma_u / (1+gamma_u)) * <u,v> * u)
/// where gamma_u = 1 / sqrt(1 - c|u|^2)
pub fn klein_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));

    let gamma_u = (1.0 - c * &u_norm_sq).mapv(|val| 1.0 / safe_sqrt(val));
    let denom = 1.0 + c * &uv;
    let denom_inv = denom.mapv(|val| 1.0 / val.max(EPS));

    let coeff_v = &gamma_u.mapv(|g| 1.0 / g); // 1/gamma_u = sqrt(1-c|u|^2)
    let coeff_u_part = (c * &gamma_u * &uv) / (1.0 + &gamma_u);
    let coeff_u = 1.0 + &coeff_u_part;

    // Result = denom_inv * (coeff_u * u + coeff_v * v)
    let mut result = u * &coeff_u;
    result = &result + &(v * coeff_v);
    result * &denom_inv
}

/// 클라인 덧셈의 역전파 (VJP)
pub fn klein_add_vjp(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
) -> (Array2<f32>, Array2<f32>) {
    // Forward 중간값 재계산
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));

    let gamma_u = (1.0 - c * &u_norm_sq).mapv(|val| 1.0 / safe_sqrt(val));
    let denom = 1.0 + c * &uv;
    let denom_inv = denom.mapv(|val| 1.0 / val.max(EPS));

    let inv_gamma_u = gamma_u.mapv(|g| 1.0 / g); // 1/gamma_u
    let coeff_u_part = (c * &gamma_u * &uv) / (1.0 + &gamma_u);
    let coeff_u = 1.0 + &coeff_u_part;

    let num = u * &coeff_u + v * &inv_gamma_u; // 분자
                                               // output = num / denom

    // 그라디언트 계산
    // dL/dNum = grad_output / denom
    let grad_num = grad_output * &denom_inv;

    // dL/dDenom = - <grad_output, num> / denom^2 = - <grad_output, output> / denom
    let output = &num * &denom_inv;
    let grad_denom = -(grad_output * &output)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1))
        * &denom_inv;

    // Denom = 1 + c <u,v>
    // dL/du += c * grad_denom * v
    // dL/dv += c * grad_denom * u
    let mut grad_u = v * (&grad_denom * c);
    let mut grad_v = u * (&grad_denom * c);

    // Num = coeff_u * u + inv_gamma_u * v
    // dL/du += coeff_u * grad_num
    // dL/dv += inv_gamma_u * grad_num
    grad_u = &grad_u + &(&grad_num * &coeff_u);
    grad_v = &grad_v + &(&grad_num * &inv_gamma_u);

    // dL/d_coeff_u = <grad_num, u>
    let grad_coeff_u = (&grad_num * u).sum_axis(Axis(1)).insert_axis(Axis(1));
    // dL/d_inv_gamma_u = <grad_num, v>
    let grad_inv_gamma_u = (&grad_num * v).sum_axis(Axis(1)).insert_axis(Axis(1));

    // inv_gamma_u = sqrt(1 - c|u|^2)
    // d_inv_gamma_u / du = -c * gamma_u * u
    grad_u = &grad_u - &(u * (&grad_inv_gamma_u * c * &gamma_u));

    // coeff_u 미분
    let d_coeff_u_d_uv = c * &gamma_u / (1.0 + &gamma_u);
    let d_coeff_u_d_gamma_u = (c * &uv) / ((1.0 + &gamma_u) * (1.0 + &gamma_u));

    let grad_uv = &grad_coeff_u * &d_coeff_u_d_uv;
    grad_u = &grad_u + &(v * &grad_uv);
    grad_v = &grad_v + &(u * &grad_uv);

    let grad_gamma_u = &grad_coeff_u * &d_coeff_u_d_gamma_u;
    grad_u = &grad_u + &(u * (&grad_gamma_u * c * &gamma_u * &gamma_u * &gamma_u));

    (grad_u, grad_v)
}

/// 클라인 스칼라 곱 (Klein Scalar Multiplication)
pub fn klein_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    let scaled_norm = (&norm_clamped * r).mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
    let scale = scaled_norm / &norm_clamped;

    u * scale
}

/// 클라인 -> 푸앵카레 변환
pub fn klein_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + (1.0 - c * x_norm_sq).mapv(|v| v.max(0.0).sqrt())).mapv(|v| v.max(EPS));
    x / &den
}

/// 클라인 -> 로렌츠 변환
pub fn klein_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let x0 = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS).sqrt());
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    result.slice_mut(s![.., 0..1]).assign(&x0);
    result.slice_mut(s![.., 1..]).assign(&(x * &x0));
    result
}

/// 클라인 스칼라 곱의 VJP
pub fn klein_scalar_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let norm = norm_sq_batched(x).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    let scaled_norm = (&norm_clamped * r).mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
    let scale = scaled_norm / &norm_clamped;

    let boundary = 1.0 / c.sqrt() - BOUNDARY_EPS;
    let d_scale_d_norm = (&norm_clamped).mapv(|n| {
        let rn = r * n;
        if rn < boundary {
            0.0
        } else {
            -1.0 / (n * n).max(EPS)
        }
    });

    let grad_norm_component = (grad_output * x).sum_axis(Axis(1)).insert_axis(Axis(1));
    let grad_x = grad_output * &scale + (grad_norm_component * d_scale_d_norm / &norm_clamped) * x;
    grad_x
}

/// 클라인 모델의 순전파 레이어
pub fn klein_layer_forward(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let u_prime = klein_scalar(u, c, 1.0 - t);
    let v_prime = klein_scalar(v, c, t);
    klein_add(&u_prime.view(), &v_prime.view(), c)
}

/// 클라인 모델의 역전파 레이어
pub fn klein_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_prime = klein_scalar(u, c, 1.0 - t);
    let v_prime = klein_scalar(v, c, t);
    let (grad_u_prime, grad_v_prime) =
        klein_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);
    let grad_u = klein_scalar_vjp(&grad_u_prime.view(), &u.view(), c, 1.0 - t);
    let grad_v = klein_scalar_vjp(&grad_v_prime.view(), &v.view(), c, t);
    (grad_u, grad_v)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn klein_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn klein_layer_forward_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn klein_layer_backward_cuda(
                grad_output: *const f32,
                u: *const f32,
                v: *const f32,
                grad_u: *mut f32,
                grad_v: *mut f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn klein_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::klein_distance_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn klein_layer_forward_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::klein_layer_forward_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn klein_layer_backward_cuda(
        grad_output: *const f32,
        u: *const f32,
        v: *const f32,
        grad_u: *mut f32,
        grad_v: *mut f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::klein_layer_backward_cuda(
                grad_output,
                u,
                v,
                grad_u,
                grad_v,
                c,
                t,
                batch_size,
                dim,
            );
        }
    }
}

/// 클라인 -> 푸앵카레 곡률 그라디언트
pub fn to_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = 1.0 + c * &x_norm_sq;
    let den_clamped = den.mapv_into(|v| v.max(EPS));

    let numerator = -2.0 * x * &x_norm_sq;
    let denominator = &den_clamped * &den_clamped;

    numerator / denominator
}

/// 푸앵카레 -> 클라인 변환
pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    // Poincaré -> Klein: 2x / (1 + c||x||^2)
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + c * &x_norm_sq).mapv(|v| v.max(EPS));
    (2.0 * x) / &den
}

/// 푸앵카레 -> 클라인 곡률 그라디언트
pub fn from_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let sqrt_expr = (1.0 - c * &x_norm_sq)
        .mapv_into(|v| v.max(EPS))
        .mapv(f32::sqrt);
    let den = 1.0 + &sqrt_expr;
    let den_clamped = den.mapv_into(|v| v.max(EPS));

    let d_sqrt_expr_dc = -0.5 * &x_norm_sq / &sqrt_expr;
    let d_den_dc = &d_sqrt_expr_dc;

    let numerator = -x * d_den_dc;
    let denominator = &den_clamped * &den_clamped;

    numerator / denominator
}
```
---
## File: `reality_stone/src/layers/lorentz.rs`

```rust
// 순수 로렌츠 구현 (Poincaré 폴백 없음)
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::ops::{batch::EPS, norm_sq_batched};

#[inline]
fn safe_sqrt(x: f32) -> f32 {
    x.max(EPS).sqrt()
}

#[inline]
fn safe_acosh(x: f32) -> f32 {
    if x <= 1.0 {
        0.0
    } else {
        x.acosh()
    }
}

/// 로렌츠 민코프스키 내적 (Minkowski Inner Product)
///
/// <u,v>_L = u0*v0 - u1*v1 - ... - un*vn
pub fn lorentz_inner(u: &ArrayView2<f32>, v: &ArrayView2<f32>) -> Array1<f32> {
    let batch_size = u.nrows();
    let mut result = Array1::zeros(batch_size);

    result
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, inner)| {
            let u_row = u.row(i);
            let v_row = v.row(i);

            // 민코프스키 내적 계산
            *inner = u_row[0] * v_row[0];
            for j in 1..u_row.len() {
                *inner -= u_row[j] * v_row[j];
            }
        });

    result
}

/// 원점 O = (1/√c, 0, ..., 0) 에서의 지수 맵 (Exponential Map)
///
/// 접벡터 u (R^d)를 쌍곡면(Hyperboloid) 위의 점으로 매핑합니다.
pub fn lorentz_exp0_space(u: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch = u.nrows();
    let dim = u.ncols();
    let sqrtc = c.sqrt();
    let u_norm = norm_sq_batched(u).mapv(f32::sqrt);
    let s = u_norm.mapv(|v| sqrtc * v);
    let mut out = Array2::<f32>::zeros((batch, dim + 1));

    // 시간 성분 (Time component)
    {
        let mut tcol = out.slice_mut(s![.., 0..1]);
        let mut idx = 0;
        for mut row in tcol.rows_mut() {
            let sv = s[idx];
            row[[0]] = sv.cosh() / sqrtc;
            idx += 1;
        }
    }
    // 공간 성분 (Space components)
    for i in 0..batch {
        let sv = s[i];
        let scale = if sv.abs() < 1e-6 {
            1.0 / sqrtc
        } else {
            sv.sinh() / (sv * sqrtc)
        };
        for j in 0..dim {
            out[[i, j + 1]] = u[[i, j]] * scale;
        }
    }
    out
}

/// lorentz_exp0_space의 정확한 역전파 (Gradient)
///
/// 입력 u(접벡터)에 대한 그라디언트를 반환합니다.
pub fn lorentz_exp0_space_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    c: f32,
) -> Array2<f32> {
    let batch = u.nrows();
    let d = u.ncols();
    let sqrtc = c.sqrt();
    let mut grad_input = Array2::<f32>::zeros(u.raw_dim());

    for i in 0..batch {
        // r = ||u||, s = sqrt(c) * r
        let mut r_sq = 0.0f32;
        for j in 0..d {
            r_sq += u[[i, j]] * u[[i, j]];
        }
        let r = r_sq.sqrt();
        let s = sqrtc * r;

        // f(s) = sinh(s)/(s*sqrt(c))
        let f = if s.abs() < 1e-6 {
            1.0 / sqrtc
        } else {
            s.sinh() / (s * sqrtc)
        };

        // f'(s) = (cosh(s)*s - sinh(s)) / (s^2 * sqrt(c))
        let fp = if s.abs() < 1e-6 {
            // s가 작을 때의 극한값 처리: g'(s) ~ s/3, f'(s) -> 0
            0.0
        } else {
            (s.cosh() * s - s.sinh()) / (s * s * sqrtc)
        };

        // ds/du_k = sqrt(c) * u_k / r
        let inv_r = if r < 1e-6 { 0.0 } else { 1.0 / r };

        // 출력 그라디언트 수집: 시간 성분 g_t, 공간 성분 g_s
        let g_t = grad_output[[i, 0]];
        let sinh_s = s.sinh();

        // dot(g_s, u)
        let mut g_s_dot_u = 0.0f32;
        for j in 0..d {
            g_s_dot_u += grad_output[[i, j + 1]] * u[[i, j]];
        }

        // 차원별 그라디언트 계산
        for k in 0..d {
            let u_k = u[[i, k]];
            let dsduk = sqrtc * u_k * inv_r;
            let term_from_space = f * grad_output[[i, k + 1]] + (g_s_dot_u * fp * dsduk);
            let term_from_time = g_t * (sinh_s * u_k * inv_r);
            grad_input[[i, k]] = term_from_space + term_from_time;
        }
    }

    grad_input
}

/// 원점에서의 로그 맵 (Logarithmic Map)
///
/// 쌍곡면 위의 점(시간+공간)을 원점의 접공간(R^d)으로 매핑합니다.
pub fn lorentz_log0_space(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch = x.nrows();
    let dim = x.ncols() - 1;
    let sqrtc = c.sqrt();
    let mut out = Array2::<f32>::zeros((batch, dim));
    for i in 0..batch {
        let x0 = x[[i, 0]];
        // s = arcosh(√c x0)
        let s = (sqrtc * x0).acosh();
        let denom = s.sinh().max(EPS);
        let scale = if s.abs() < 1e-6 {
            1.0
        } else {
            s / (denom * sqrtc)
        };
        for j in 0..dim {
            out[[i, j]] = x[[i, j + 1]] * scale;
        }
    }
    out
}

/// lorentz_log0_space의 정확한 역전파 (Gradient)
///
/// 입력 x(시간+공간)에 대한 그라디언트를 반환합니다.
pub fn lorentz_log0_space_backward(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
) -> Array2<f32> {
    let batch = x.nrows();
    let dim = x.ncols();
    let space_dim = dim - 1;
    let sqrtc = c.sqrt();
    let mut grad_input = Array2::<f32>::zeros(x.raw_dim());

    for i in 0..batch {
        let x0 = x[[i, 0]];
        // s = acosh( sqrt(c) * x0 )
        let ax0 = (sqrtc * x0).max(1.0 + EPS);
        let s = ax0.acosh();
        let sinh_s = s.sinh().max(EPS);
        let cosh_s = s.cosh();

        // scale = s / (sinh(s) * sqrt(c))
        let scale = s / (sinh_s * sqrtc);

        // d scale / d x0 = (sinh(s) - s cosh(s)) / sinh(s)^3
        let dscale_dx0 = (sinh_s - s * cosh_s) / (sinh_s * sinh_s * sinh_s);

        // g_xspace = grad_output_space * scale
        for j in 0..space_dim {
            grad_input[[i, j + 1]] = grad_output[[i, j]] * scale;
        }

        // 시간 성분: scale을 통해서만 영향 받음
        let mut dot_gspace_xspace = 0.0f32;
        for j in 0..space_dim {
            dot_gspace_xspace += grad_output[[i, j]] * x[[i, j + 1]];
        }
        grad_input[[i, 0]] = dot_gspace_xspace * dscale_dx0;
    }

    grad_input
}

/// 로렌츠 거리 (Lorentz Distance)
///
/// cosh(√c d) = c <u,v>_L
pub fn lorentz_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let inner = lorentz_inner(u, v);
    let sqrtc = c.sqrt();
    inner.mapv(|x| safe_acosh((c * x).max(1.0)) / sqrtc)
}

/// 로렌츠 덧셈 (Lorentz Addition)
///
/// 자이로벡터 공간(Gyrovector space)에서의 덧셈 연산을 수행합니다.
pub fn lorentz_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let v_row = v.row(i);

            // 내적 계산 (Inner products)
            let mut uu = u_row[0] * u_row[0];
            let mut vv = v_row[0] * v_row[0];
            let mut uv = u_row[0] * v_row[0];

            for j in 1..dim {
                uu -= u_row[j] * u_row[j];
                vv -= v_row[j] * v_row[j];
                uv -= u_row[j] * v_row[j];
            }

            let beta_u = (-uu / c).max(EPS);
            let beta_v = (-vv / c).max(EPS);
            let gamma_u = 1.0 / safe_sqrt(beta_u);
            let gamma_v = 1.0 / safe_sqrt(beta_v);
            let gamma_uv = -uv / (c * (beta_u * beta_v).sqrt());

            for j in 0..dim {
                let denom_u = (1.0 + gamma_u).max(EPS);
                let denom_v = (1.0 + gamma_v).max(EPS);
                row[j] = gamma_uv * (gamma_u * u_row[j] / denom_u + gamma_v * v_row[j] / denom_v)
                    + u_row[j]
                    + v_row[j];
            }
        });

    result
}

/// 로렌츠 스칼라 곱 (Lorentz Scalar Multiplication)
pub fn lorentz_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let time_comp = u_row[0];

            let mut space_norm_sq = 0.0;
            for j in 1..dim {
                space_norm_sq += u_row[j] * u_row[j];
            }

            // 쌍곡면 제약조건: time^2 - ||x||^2 = 1/c
            let denom = (time_comp * time_comp - 1.0 / c).max(EPS);
            let norm = (space_norm_sq / denom).sqrt();
            let theta = norm.min(1.0 - EPS).atanh() * r;
            let scale = theta.tanh() / norm.max(EPS);

            // 공간 성분 설정
            let mut scaled_space_norm_sq = 0.0;
            for j in 1..dim {
                row[j] = u_row[j] * scale;
                scaled_space_norm_sq += row[j] * row[j];
            }
            // 시간 성분 재계산: x0 = sqrt(1/c + ||x||^2)
            row[0] = (1.0 / c + scaled_space_norm_sq).sqrt();
        });

    result
}

/// lorentz_scalar의 정확한 역전파 (Gradient)
///
/// 입력 u(time + space)에 대한 그라디언트를 반환합니다.
pub fn lorentz_scalar_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let space_dim = dim - 1;
    let mut grad_input = Array2::<f32>::zeros(u.raw_dim());

    for i in 0..batch_size {
        let t = u[[i, 0]];
        // 공간 벡터 s
        let mut space_norm_sq = 0.0f32;
        for j in 0..space_dim {
            let v = u[[i, j + 1]];
            space_norm_sq += v * v;
        }

        // 순전파 계산 재현
        let denom = (t * t - 1.0 / c).max(EPS);
        let ns = space_norm_sq.sqrt();
        let ns_safe = ns.max(EPS);
        let sqrt_denom = denom.sqrt();
        let norm = (ns / sqrt_denom).max(0.0);

        let norm_clamp_top = 1.0 - EPS;
        let scn = norm.min(norm_clamp_top);
        let alpha = scn.atanh();
        let theta = r * alpha;
        let beta = theta.tanh();
        let scale = if norm < EPS { r } else { beta / norm };

        // s' 와 t'
        let mut s_prime_sq = 0.0f32;
        for j in 0..space_dim {
            let sp = u[[i, j + 1]] * scale;
            s_prime_sq += sp * sp;
        }
        let t_prime = (1.0 / c + s_prime_sq).sqrt();

        // s' 에 대한 유효 그라디언트 (시간 성분 경로 포함)
        let g_tprime = grad_output[[i, 0]];
        // d t' / d s'_j = s'_j / t' 미리 계산
        // dot(g_sprime_eff, s) 누적
        let mut dot_gs_eff_s = 0.0f32;

        let mut g_sprime_eff: Vec<f32> = vec![0.0; space_dim];
        for j in 0..space_dim {
            let s_j = u[[i, j + 1]];
            let s_prime_j = s_j * scale;
            let g_sprime_j = grad_output[[i, j + 1]];
            let g_eff = g_sprime_j + g_tprime * (s_prime_j / t_prime.max(EPS));
            g_sprime_eff[j] = g_eff;
            dot_gs_eff_s += g_eff * s_j;
        }

        // norm의 미분
        // d norm / d s_k = s_k / (ns * sqrt(denom))
        // d norm / d t = - (t / denom) * norm
        let inv_sqrt_denom = 1.0 / sqrt_denom;
        let dnorm_dt = -(t / denom) * norm;

        // d beta / d norm = r * (1 - beta^2) / (1 - scn^2) * dscn/dnorm
        let dscn_dnorm = if norm <= norm_clamp_top { 1.0 } else { 0.0 };
        let one_minus_beta_sq = 1.0 - beta * beta;
        let one_minus_scn_sq = (1.0 - scn * scn).max(EPS);
        let dbeta_dnorm = r * one_minus_beta_sq / one_minus_scn_sq * dscn_dnorm;

        // d scale / d norm = (norm * dbeta_dnorm - beta) / norm^2
        let dscale_dnorm = if norm < 1e-6 {
            0.0
        } else {
            (norm * dbeta_dnorm - beta) / (norm * norm)
        };

        // d scale / d s_k 및 d scale / d t
        let mut dscale_ds: Vec<f32> = vec![0.0; space_dim];
        for k in 0..space_dim {
            let dnorm_dsk = if ns_safe < 1e-6 {
                0.0
            } else {
                u[[i, k + 1]] * (1.0 / (ns_safe)) * inv_sqrt_denom
            };
            dscale_ds[k] = dscale_dnorm * dnorm_dsk;
        }
        let dscale_dt = dscale_dnorm * dnorm_dt;

        // 최종 그라디언트 누적
        // 공간 성분에 대한 그라디언트
        for k in 0..space_dim {
            let gk = g_sprime_eff[k] * scale + dot_gs_eff_s * dscale_ds[k];
            grad_input[[i, k + 1]] = gk;
        }

        // 시간 성분에 대한 그라디언트
        let mut g_time = 0.0f32;
        g_time += dot_gs_eff_s * dscale_dt;
        grad_input[[i, 0]] = g_time;
    }

    grad_input
}

/// 로렌츠 -> 클라인 변환
pub fn lorentz_to_klein(x: &ArrayView2<f32>, _: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols() - 1;
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let x0 = x_row[0].max(EPS);

            for j in 0..dim {
                row[j] = x_row[j + 1] / x0;
            }
        });

    result
}

/// 로렌츠 -> 푸앵카레 변환
pub fn lorentz_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let klein = lorentz_to_klein(x, c);
    crate::layers::klein::klein_to_poincare(&klein.view(), c)
}

/// 로렌츠 모델의 순전파 레이어 계산
///
/// 파라미터 t에 따라 u와 v 사이를 지오데식(Geodesic)으로 보간합니다.
pub fn lorentz_layer_forward(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::<f32>::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let p = u.row(i);
            let q = v.row(i);
            // 민코프스키 내적
            let mut inner = p[0] * q[0];
            for j in 1..dim {
                inner -= p[j] * q[j];
            }
            let theta = safe_acosh((c * inner).max(1.0 + EPS));
            let sinh_theta = theta.sinh().max(EPS);

            // 가중치 w1, w2 계산
            let w1 = if theta.abs() < 1e-6 {
                1.0 - t
            } else {
                ((1.0 - t) * theta).sinh() / sinh_theta
            };
            let w2 = if theta.abs() < 1e-6 {
                t
            } else {
                (t * theta).sinh() / sinh_theta
            };

            // 선형 결합 (시간 성분 포함)
            for j in 0..dim {
                row[j] = w1 * p[j] + w2 * q[j];
            }
        });

    result
}

/// 로렌츠 모델의 역전파 레이어 계산
pub fn lorentz_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut gu = Array2::<f32>::zeros(u.raw_dim());
    let mut gv = Array2::<f32>::zeros(v.raw_dim());

    for i in 0..batch_size {
        let p = u.row(i);
        let q = v.row(i);
        let g = grad_output.row(i);

        let mut inner = p[0] * q[0];
        for j in 1..dim {
            inner -= p[j] * q[j];
        }

        let alpha_arg = (c * inner).max(1.0 + EPS);
        let alpha = alpha_arg.acosh();
        let sinh_alpha = alpha.sinh().max(EPS);
        let cosh_alpha = alpha.cosh();

        // 가중치 w1, w2
        let w1 = if alpha.abs() < 1e-6 {
            1.0 - t
        } else {
            ((1.0 - t) * alpha).sinh() / sinh_alpha
        };
        let w2 = if alpha.abs() < 1e-6 {
            t
        } else {
            (t * alpha).sinh() / sinh_alpha
        };

        // dw/dalpha
        let num1 = (1.0 - t) * ((1.0 - t) * alpha).cosh() * sinh_alpha
            - ((1.0 - t) * alpha).sinh() * cosh_alpha;
        let num2 = t * (t * alpha).cosh() * sinh_alpha - (t * alpha).sinh() * cosh_alpha;
        let denom = (sinh_alpha * sinh_alpha).max(EPS);
        let dw1_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num1 / denom
        };
        let dw2_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num2 / denom
        };

        // d alpha / d p = (c / sinh(alpha)) * G q  (G는 민코프스키 메트릭)
        let scale = c / sinh_alpha;
        let mut dalpha_dp = vec![0.0f32; dim];
        let mut dalpha_dq = vec![0.0f32; dim];
        dalpha_dp[0] = scale * q[0];
        dalpha_dq[0] = scale * p[0];
        for j in 1..dim {
            dalpha_dp[j] = scale * (-q[j]);
            dalpha_dq[j] = scale * (-p[j]);
        }

        // g dot p, g dot q (유클리드)
        let mut g_dot_p = 0.0f32;
        let mut g_dot_q = 0.0f32;
        for j in 0..dim {
            g_dot_p += g[j] * p[j];
            g_dot_q += g[j] * q[j];
        }

        for j in 0..dim {
            gu[[i, j]] = w1 * g[j] + (g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha) * dalpha_dp[j];
            gv[[i, j]] = w2 * g[j] + (g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha) * dalpha_dq[j];
        }
    }

    (gu, gv)
}

fn acosh_derivative(z: f32) -> f32 {
    // d/dz acosh(z) = 1 / (sqrt(z-1) * sqrt(z+1))
    let zp = (z + 1.0).max(1.0 + EPS);
    let zm = (z - 1.0).max(EPS);
    1.0 / (zp.sqrt() * zm.sqrt())
}

/// 동적 곡률을 사용한 로렌츠 레이어 순전파
pub fn lorentz_layer_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let y = lorentz_layer_forward(u, v, c, t);
    (y, c)
}

/// 동적 곡률을 사용한 로렌츠 레이어 역전파 (정석 미분)
pub fn lorentz_layer_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let (grad_u, grad_v) = lorentz_layer_backward(grad_output, u, v, c, t);

    // Chain Rule을 통한 grad_c 계산
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut grad_c = 0.0f32;
    for i in 0..batch_size {
        let p = u.row(i);
        let q = v.row(i);
        // 민코프스키 내적
        let mut inner = p[0] * q[0];
        for j in 1..dim {
            inner -= p[j] * q[j];
        }
        let z = (c * inner).max(1.0 + EPS);
        let alpha = z.acosh();
        let sinh_alpha = alpha.sinh().max(EPS);
        let cosh_alpha = alpha.cosh();

        // dw/dalpha
        let num1 = (1.0 - t) * ((1.0 - t) * alpha).cosh() * sinh_alpha
            - ((1.0 - t) * alpha).sinh() * cosh_alpha;
        let num2 = t * (t * alpha).cosh() * sinh_alpha - (t * alpha).sinh() * cosh_alpha;
        let denom = (sinh_alpha * sinh_alpha).max(EPS);
        let dw1_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num1 / denom
        };
        let dw2_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num2 / denom
        };

        // dalpha/dc = (d acosh(z)/dz) * dz/dc, where z = c * inner
        let dalpha_dz = acosh_derivative(z);
        let dz_dc = inner;
        let dalpha_dc = dalpha_dz * dz_dc;

        let dw1_dc = dw1_dalpha * dalpha_dc;
        let dw2_dc = dw2_dalpha * dalpha_dc;

        // dy/dc = dw1_dc * p + dw2_dc * q; grad_c 누적
        for j in 0..dim {
            let d_yj_dc = dw1_dc * p[j] + dw2_dc * q[j];
            grad_c += grad_output[[i, j]] * d_yj_dc;
        }
    }

    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;
    (grad_u, grad_v, grad_kappa)
}

/// 레이어별 곡률을 사용한 로렌츠 레이어 순전파
pub fn lorentz_layer_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let y = lorentz_layer_forward(u, v, c, t);
    (y, c)
}

/// 레이어별 곡률을 사용한 로렌츠 레이어 역전파
pub fn lorentz_layer_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let (grad_u, grad_v) = lorentz_layer_backward(grad_output, u, v, c, t);

    // grad_c 누적 (dynamic 버전과 동일한 로직)
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut grad_c = 0.0f32;
    for i in 0..batch_size {
        let p = u.row(i);
        let q = v.row(i);
        let mut inner = p[0] * q[0];
        for j in 1..dim {
            inner -= p[j] * q[j];
        }
        let z = (c * inner).max(1.0 + EPS);
        let alpha = z.acosh();
        let sinh_alpha = alpha.sinh().max(EPS);
        let cosh_alpha = alpha.cosh();

        let num1 = (1.0 - t) * ((1.0 - t) * alpha).cosh() * sinh_alpha
            - ((1.0 - t) * alpha).sinh() * cosh_alpha;
        let num2 = t * (t * alpha).cosh() * sinh_alpha - (t * alpha).sinh() * cosh_alpha;
        let denom = (sinh_alpha * sinh_alpha).max(EPS);
        let dw1_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num1 / denom
        };
        let dw2_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num2 / denom
        };

        let dalpha_dz = acosh_derivative(z);
        let dz_dc = inner;
        let dalpha_dc = dalpha_dz * dz_dc;
        let dw1_dc = dw1_dalpha * dalpha_dc;
        let dw2_dc = dw2_dalpha * dalpha_dc;

        for j in 0..dim {
            let d_yj_dc = dw1_dc * p[j] + dw2_dc * q[j];
            grad_c += grad_output[[i, j]] * d_yj_dc;
        }
    }

    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c * dc_dkappa;
    (grad_u, grad_v, grad_kappa)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn lorentz_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn lorentz_layer_forward_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn lorentz_layer_backward_cuda(
                grad_output: *const f32,
                u: *const f32,
                v: *const f32,
                grad_u: *mut f32,
                grad_v: *mut f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn lorentz_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_distance_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn lorentz_layer_forward_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_layer_forward_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn lorentz_layer_backward_cuda(
        grad_output: *const f32,
        u: *const f32,
        v: *const f32,
        grad_u: *mut f32,
        grad_v: *mut f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_layer_backward_cuda(
                grad_output,
                u,
                v,
                grad_u,
                grad_v,
                c,
                t,
                batch_size,
                dim,
            );
        }
    }
}

/// 푸앵카레 -> 로렌츠 변환
pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let factor = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS));

    result
        .slice_mut(s![.., 0..1])
        .assign(&(&factor * (1.0 + c * &x_norm_sq) / c.sqrt()));
    result
        .slice_mut(s![.., 1..])
        .assign(&(&factor * 2.0 * x / c.sqrt()));
    result
}

/// 푸앵카레 -> 로렌츠 변환의 곡률 그라디언트
pub fn from_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let mut grad_result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS));
    let sqrt_c = c.sqrt();

    // 시간 성분 그라디언트
    let d_time_den_dc = -&x_norm_sq;
    let d_time_num_dc = &x_norm_sq;
    let time_num = 1.0 + c * &x_norm_sq;
    let d_time_dc = (d_time_num_dc * &den - &time_num * d_time_den_dc) / (&den * &den);
    grad_result
        .slice_mut(s![.., 0..1])
        .assign(&(&d_time_dc / sqrt_c - &time_num / (2.0 * c * sqrt_c * &den)));

    // 공간 성분 그라디언트
    let d_factor_dc = &x_norm_sq / (&den * &den);
    grad_result
        .slice_mut(s![.., 1..])
        .assign(&(x * (&d_factor_dc / sqrt_c - 1.0 / (c * sqrt_c * &den))));

    grad_result
}
```
---
## File: `reality_stone/src/layers/memory.rs`

```rust
use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct ControlPoint {
    pub t: usize,
    pub x: Array1<f32>,
    pub v: Array1<f32>,
}

pub struct GeodesicMemory {
    pub d_model: usize,
    pub threshold: f32,
    pub control_points: VecDeque<ControlPoint>,
    buffer: Vec<(usize, Array1<f32>)>,
    last_t: usize,
}

impl GeodesicMemory {
    pub fn new(d_model: usize, threshold: f32) -> Self {
        Self {
            d_model,
            threshold,
            control_points: VecDeque::new(),
            buffer: Vec::new(),
            last_t: 0,
        }
    }

    pub fn push(&mut self, t: usize, x: ArrayView1<f32>) -> bool {
        let x_owned = x.to_owned();
        self.buffer.push((t, x_owned));
        self.last_t = t;

        if self.buffer.len() < 3 {
            if self.control_points.is_empty() {
                self.add_control_point(t);
                return true;
            }
            return false;
        }

        let len = self.buffer.len();
        let (t_curr, x_curr) = &self.buffer[len - 1];
        let (_t_prev, x_prev) = &self.buffer[len - 2];
        let (_t_prev2, x_prev2) = &self.buffer[len - 3];

        let acc = x_curr - x_prev * 2.0 + x_prev2;
        let acc_norm = acc.dot(&acc).sqrt();

        if acc_norm > self.threshold {
            self.add_control_point(*t_curr);
            let last = self.buffer.pop().unwrap();
            self.buffer.clear();
            self.buffer.push(last);
            return true;
        }

        false
    }

    fn add_control_point(&mut self, t: usize) {
        let v = if let Some((_, x_last)) = self.buffer.last() {
            if self.buffer.len() >= 2 {
                let x_prev = &self.buffer[self.buffer.len() - 2].1;
                x_last - x_prev
            } else {
                Array1::zeros(self.d_model)
            }
        } else {
            Array1::zeros(self.d_model)
        };

        let x = if let Some((_, x_val)) = self.buffer.last() {
            x_val.clone()
        } else {
            Array1::zeros(self.d_model)
        };

        self.control_points.push_back(ControlPoint { t, x, v });
    }

    pub fn query(&self, t: f32) -> Array1<f32> {
        if self.control_points.is_empty() {
            return Array1::zeros(self.d_model);
        }

        let idx = self
            .control_points
            .binary_search_by(|cp| cp.t.partial_cmp(&(t as usize)).unwrap())
            .unwrap_or_else(|x| x);

        if idx == 0 {
            if self.control_points.len() == 1 && !self.buffer.is_empty() {
                if let Some((t_buf, _x_buf)) = self.buffer.last() {
                    if t > (self.control_points[0].t as f32) && t <= (*t_buf as f32) {}
                }
            }
            return self.control_points[0].x.clone();
        }
        if idx >= self.control_points.len() {
            if !self.buffer.is_empty() {
                let (t_last, x_last) = self.buffer.last().unwrap();
                let cp_last = self.control_points.back().unwrap();
                if t > (cp_last.t as f32) && t <= (*t_last as f32) {
                    let p0 = cp_last;
                    let v_tip = if self.buffer.len() >= 2 {
                        &self.buffer[self.buffer.len() - 1].1
                            - &self.buffer[self.buffer.len() - 2].1
                    } else {
                        Array1::zeros(self.d_model)
                    };

                    let t0 = p0.t as f32;
                    let t1 = *t_last as f32;
                    let s = (t - t0) / (t1 - t0);
                    let s2 = s * s;
                    let s3 = s2 * s;

                    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
                    let h10 = s3 - 2.0 * s2 + s;
                    let h01 = -2.0 * s3 + 3.0 * s2;
                    let h11 = s3 - s2;

                    let dt = t1 - t0;
                    let m0 = &p0.v * dt;
                    let m1 = &v_tip * dt;
                    return &p0.x * h00 + &m0 * h10 + x_last * h01 + &m1 * h11;
                }
            }
            return self.control_points.back().unwrap().x.clone();
        }

        let p0 = &self.control_points[idx - 1];
        let p1 = &self.control_points[idx];

        let t0 = p0.t as f32;
        let t1 = p1.t as f32;

        if t1 == t0 {
            return p0.x.clone();
        }

        let s = (t - t0) / (t1 - t0);
        let s2 = s * s;
        let s3 = s2 * s;

        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;

        let dt = t1 - t0;
        let m0 = &p0.v * dt;
        let m1 = &p1.v * dt;

        let x_t = &p0.x * h00 + &m0 * h10 + &p1.x * h01 + &m1 * h11;

        x_t
    }

    pub fn get_compression_stats(&self) -> (usize, usize, f32) {
        let stored = self.control_points.len();
        let covered = if self.control_points.is_empty() {
            0
        } else {
            self.last_t + 1
        };
        let ratio = if stored > 0 {
            covered as f32 / stored as f32
        } else {
            0.0
        };
        (stored, covered, ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_spline_compression_sine_wave() {
        let seq_len = 100;
        let d_model = 16;
        let mut trajectory = Vec::new();

        for t in 0..seq_len {
            let mut state = Array1::<f32>::zeros(d_model);
            for i in 0..d_model {
                state[i] = ((t as f32) * 0.1 + (i as f32)).sin();
            }
            trajectory.push(state);
        }

        let mut memory = GeodesicMemory::new(d_model, 0.01);

        for (t, state) in trajectory.iter().enumerate() {
            memory.push(t, state.view());
        }

        let (stored, covered, ratio) = memory.get_compression_stats();
        println!(
            "Compression: Stored {} / Covered {} (Ratio {:.2}x)",
            stored, covered, ratio
        );

        assert!(stored < seq_len, "Should compress somewhat");
        assert!(stored > 2, "Should have more than start/end points");

        let mut total_mse = 0.0;
        let mut max_mse = 0.0;

        for (t, original) in trajectory.iter().enumerate() {
            let reconstructed = memory.query(t as f32);
            let diff = original - &reconstructed;
            let mse = diff.dot(&diff) / d_model as f32;

            total_mse += mse;
            if mse > max_mse {
                max_mse = mse;
            }
        }
        let avg_mse = total_mse / seq_len as f32;
        println!(
            "Reconstruction Error: Avg MSE {:.6}, Max MSE {:.6}",
            avg_mse, max_mse
        );

        assert!(avg_mse < 0.01, "Average reconstruction error too high");
    }

    #[test]
    fn test_linear_trajectory_compression() {
        let seq_len = 50;
        let d_model = 4;
        let mut trajectory = Vec::new();

        for t in 0..seq_len {
            let mut state = Array1::<f32>::zeros(d_model);
            for i in 0..d_model {
                state[i] = (t as f32) * 0.1;
            }
            trajectory.push(state);
        }

        let mut memory = GeodesicMemory::new(d_model, 0.001);

        for (t, state) in trajectory.iter().enumerate() {
            memory.push(t, state.view());
        }

        let (stored, _, _) = memory.get_compression_stats();
        println!("Linear Compression: Stored {}", stored);
    }
}
```
---
## File: `reality_stone/src/layers/metric.rs`

```rust
// ============================================================================
// 파일: src/layers/metric.rs
// 목적: 리만 메트릭 텐서의 추상화 및 구현
// ============================================================================

use ndarray::{Array1, Array2, ArrayView2, Axis};

const EPS: f32 = 1e-7;

/// 리만 메트릭 텐서의 공통 인터페이스
pub trait MetricTensor: Send + Sync {
    /// 메트릭 텐서 g_ij(x) 계산 (대각 원소만 반환, batch x dim)
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32>;

    /// 역메트릭 g^ij(x) 계산 (대각 원소만)
    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32>;

    /// 크리스토펠 기호 Γ^k_ij 계산 (대각 근사, batch별)
    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>>;

    /// 리만 거리 d_g(x, y)
    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32>;

    /// 메트릭의 행렬식 det(g)
    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32>;

    /// 곡률 스칼라
    fn curvature(&self) -> f32;
}

/// 대각 메트릭 (구현 효율성을 위한 근사)
/// g_ij(x) = w_i(x) δ_ij
#[derive(Clone)]
pub struct DiagonalMetric {
    /// 각 차원의 가중치를 계산하는 함수 파라미터
    pub weights: Array1<f32>, // learnable parameters
    pub base_weight: f32,
}

impl DiagonalMetric {
    pub fn new(dim: usize) -> Self {
        Self {
            weights: Array1::ones(dim),
            base_weight: 1.0,
        }
    }

    /// w_i(x) = softplus(weights[i] * x[i]) + ε
    fn compute_weights(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros(x.raw_dim());
        for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                let z = self.weights[j] * x[[i, j]];
                *val = softplus(z) + EPS;
            }
        }
        result
    }
}

impl MetricTensor for DiagonalMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        self.compute_weights(x)
    }

    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let weights = self.compute_weights(x);
        weights.mapv(|w| 1.0 / w.max(EPS))
    }

    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // 대각 메트릭에서 Γ^i_ii = (1/2w_i) * dw_i/dx_i
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut symbols = Vec::new();

        for i in 0..batch_size {
            let mut gamma = Array2::zeros((dim, dim));
            for j in 0..dim {
                let x_val = x[[i, j]];
                let w = softplus(self.weights[j] * x_val) + EPS;
                // d(softplus(z))/dz = sigmoid(z)
                let dw_dx = self.weights[j] * sigmoid(self.weights[j] * x_val);
                gamma[[j, j]] = 0.5 * dw_dx / w;
            }
            symbols.push(gamma);
        }
        symbols
    }

    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        // 유클리드 거리의 메트릭 가중 버전
        let diff = x - y;
        let weights = self.compute_weights(x);
        let weighted_sq = &diff * &diff * &weights;
        weighted_sq.sum_axis(Axis(1)).mapv(|s| s.sqrt())
    }

    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let weights = self.compute_weights(x);
        weights
            .axis_iter(Axis(0))
            .map(|row| row.iter().product())
            .collect()
    }

    fn curvature(&self) -> f32 {
        0.0 // 대각 메트릭의 곡률은 0 (국소적으로 평탄)
    }
}

/// 푸앵카레 메트릭
/// g_ij(x) = (2/(1-c||x||²))² δ_ij
#[derive(Clone)]
pub struct PoincareMetric {
    pub curvature: f32,
}

impl PoincareMetric {
    pub fn new(curvature: f32) -> Self {
        Self { curvature }
    }

    fn conformal_factor(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        let denom = (1.0 - self.curvature * &x_norm_sq).mapv(|v| v.max(EPS));
        (2.0 / denom).mapv(|v| v * v)
    }
}

impl MetricTensor for PoincareMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let lambda_sq = self.conformal_factor(x);
        // g_ij = λ² δ_ij, 대각만 저장
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut metric = Array2::zeros((batch_size, dim));
        for i in 0..batch_size {
            for j in 0..dim {
                metric[[i, j]] = lambda_sq[i];
            }
        }
        metric
    }

    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let lambda_sq = self.conformal_factor(x);
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut inv_metric = Array2::zeros((batch_size, dim));
        for i in 0..batch_size {
            for j in 0..dim {
                inv_metric[[i, j]] = 1.0 / lambda_sq[i].max(EPS);
            }
        }
        inv_metric
    }

    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // Poincaré: Γ^k_ij = (2c/(1-c||x||²)) * (δ_ik x_j + δ_jk x_i - δ_ij x_k)
        let batch_size = x.nrows();
        let dim = x.ncols();
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);

        let mut symbols = Vec::new();
        for b in 0..batch_size {
            let coeff = 2.0 * c / (1.0 - c * x_norm_sq[b]).max(EPS);
            let mut gamma = Array2::zeros((dim, dim));

            // 대각 근사: i=j=k만 고려
            for i in 0..dim {
                gamma[[i, i]] = coeff * x[[b, i]];
            }
            symbols.push(gamma);
        }
        symbols
    }

    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        crate::layers::poincare::poincare_distance(x, y, self.curvature, 1e-5)
    }

    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let lambda_sq = self.conformal_factor(x);
        let dim = x.ncols() as f32;
        lambda_sq.mapv(|l| l.powf(dim))
    }

    fn curvature(&self) -> f32 {
        -self.curvature
    }
}

/// 로렌츠 (Hyperboloid) 메트릭
/// Minkowski inner product: ⟨u,v⟩ = u₀v₀ - Σᵢ uᵢvᵢ
#[derive(Clone)]
pub struct LorentzMetric {
    pub curvature: f32,
}

impl LorentzMetric {
    pub fn new(curvature: f32) -> Self {
        Self { curvature }
    }
}

impl MetricTensor for LorentzMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        // Minkowski metric: diag(1, -1, -1, ...)
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut metric = Array2::zeros((batch_size, dim));

        for i in 0..batch_size {
            metric[[i, 0]] = 1.0;
            for j in 1..dim {
                metric[[i, j]] = -1.0;
            }
        }
        metric
    }

    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        // Minkowski 메트릭은 자기역원
        self.compute_metric(x)
    }

    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // 민코프스키 공간에서 크리스토펠 기호는 0 (평탄)
        let batch_size = x.nrows();
        let dim = x.ncols();
        vec![Array2::zeros((dim, dim)); batch_size]
    }

    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        crate::layers::lorentz::lorentz_distance(x, y, self.curvature)
    }

    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        Array1::from_elem(x.nrows(), -1.0) // det(η) = -1
    }

    fn curvature(&self) -> f32 {
        -self.curvature
    }
}

/// Klein 메트릭 (projective model)
#[derive(Clone)]
pub struct KleinMetric {
    pub curvature: f32,
}

impl KleinMetric {
    pub fn new(curvature: f32) -> Self {
        Self { curvature }
    }
}

impl MetricTensor for KleinMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        // Klein: g_ij = (1/(1-c||x||²)) * (δ_ij + c xᵢxⱼ/(1-c||x||²))
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        let factor = (1.0 - c * &x_norm_sq).mapv(|v| 1.0 / v.max(EPS));

        // 대각 근사: g_ii = factor * (1 + c x_i²/(1-c||x||²))
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut metric = Array2::zeros((batch_size, dim));

        for i in 0..batch_size {
            for j in 0..dim {
                metric[[i, j]] = factor[i] * (1.0 + c * x[[i, j]] * x[[i, j]] * factor[i]);
            }
        }
        metric
    }

    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let metric = self.compute_metric(x);
        metric.mapv(|g| 1.0 / g.max(EPS))
    }

    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // Klein 모델의 크리스토펠 기호 (대각 근사)
        let batch_size = x.nrows();
        let dim = x.ncols();
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);

        let mut symbols = Vec::new();
        for b in 0..batch_size {
            let denom = (1.0 - c * x_norm_sq[b]).max(EPS);
            let coeff = c / denom;
            let mut gamma = Array2::zeros((dim, dim));

            for i in 0..dim {
                gamma[[i, i]] = coeff * x[[b, i]];
            }
            symbols.push(gamma);
        }
        symbols
    }

    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        crate::layers::klein::klein_distance(x, y, self.curvature)
    }

    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        let dim = x.ncols() as f32;
        let factor = 1.0 - c * &x_norm_sq;
        factor.mapv(|f| f.powf(-dim))
    }

    fn curvature(&self) -> f32 {
        -self.curvature
    }
}

// 유틸리티 함수
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // 수치 안정성
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 메트릭 타입 열거형 (런타임 선택용)
pub enum MetricType {
    Diagonal(DiagonalMetric),
    Poincare(PoincareMetric),
    Lorentz(LorentzMetric),
    Klein(KleinMetric),
}

impl MetricType {
    pub fn as_trait(&self) -> &dyn MetricTensor {
        match self {
            MetricType::Diagonal(m) => m,
            MetricType::Poincare(m) => m,
            MetricType::Lorentz(m) => m,
            MetricType::Klein(m) => m,
        }
    }

    pub fn as_trait_mut(&mut self) -> &mut dyn MetricTensor {
        match self {
            MetricType::Diagonal(m) => m,
            MetricType::Poincare(m) => m,
            MetricType::Lorentz(m) => m,
            MetricType::Klein(m) => m,
        }
    }
}
```
---
## File: `reality_stone/src/layers/mod.rs`

```rust
// src/layers/mod.rs

//! # Reality Stone 레이어 모듈
//!
//! 리만 기하학에 최적화된 다양한 하이퍼볼릭 레이어를 제공합니다.

// 기하학적 레이어들
pub mod bellman;
pub mod klein;
pub mod lorentz;
pub mod memory;
pub mod poincare;
pub mod riemann;
pub mod spline;
pub mod spline_cache;
pub mod suppression;
pub mod utils;

// 통합 리만 시스템
pub mod bellman_lagrangian;
pub mod decoder;
pub mod diffusion;
pub mod geodesic;
pub mod human_decoder;
pub mod hyper_metric;
pub mod metric;
pub mod rsulf;
pub mod symplectic;
pub mod unified_riemannian;

pub use self::poincare::{
    poincare_ball_layer, poincare_ball_layer_backward, poincare_distance, poincare_exp_at,
    poincare_log_at, poincare_to_klein, poincare_to_lorentz,
};

pub use self::bellman_lagrangian::{
    bellman_potential, kinetic_energy, representation_flow, EnergyComponents, LagrangianParams,
    ValueFunction,
};
pub use self::decoder::RiemannianDecoder;
pub use self::diffusion::RiemannianDiffusion; // Export diffusion
pub use self::geodesic::{exponential_map, geodesic_interpolation, geodesic_path, logarithmic_map};
pub use self::human_decoder::{HumanStyleDecoder, StageWeights};
pub use self::metric::{
    DiagonalMetric, KleinMetric, LorentzMetric, MetricTensor, MetricType, PoincareMetric,
};
pub use self::unified_riemannian::{
    LayerCache, LayerGradients, LayerOutput, UnifiedRiemannianLayer,
};
```
---
## File: `reality_stone/src/layers/poincare.rs`

```rust
use crate::ops::{batch::EPS, dot_batched, mobius, norm_sq_batched};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

// 매직 넘버 사용을 피하기 위한 공통 상수
const BOUNDARY_EPS: f32 = 1e-5;

/// 푸앵카레 공 레이어의 역전파 (Backward Pass)
///
/// 입력:
/// - grad_output: 출력에 대한 그라디언트
/// - u, v: 순전파 시 입력 벡터들
/// - c: 곡률
/// - t: 보간 파라미터 (0.0 ~ 1.0)
///
/// 출력:
/// - (grad_u, grad_v): 입력 u, v에 대한 그라디언트
pub fn poincare_ball_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    // 뫼비우스 덧셈의 VJP
    let (grad_u_prime, grad_v_prime) =
        mobius::mobius_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);

    // 스칼라 곱의 VJP
    let grad_u = mobius::mobius_scalar_vjp(&grad_u_prime.view(), &u.view(), c, 1.0 - t);
    let grad_v = mobius::mobius_scalar_vjp(&grad_v_prime.view(), &v.view(), c, t);

    (grad_u, grad_v)
}

/// 푸앵카레 거리 (Poincaré Distance) 계산
///
/// d(u, v) = (2/√c) * atanh(√(c * ||u-v||² / ((1-c||u||²)(1-c||v||²))))
pub fn poincare_distance(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    boundary_eps: f32,
) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    let norm_sq_diff = (&u2 + &v2 - 2.0 * &uv).mapv_into(|val| val.max(0.0));
    let den = (1.0 - c * &u2) * (1.0 - c * &v2);

    // 경계 근처에서의 수치적 안정성을 위해 분모 제한
    let den_clamped = den.mapv_into(|val| val.max(boundary_eps));

    let frac = norm_sq_diff / den_clamped;
    // 수치 안정성을 고려한 변형 공식 사용:
    // arg = √(c * frac / (1 + c * frac))
    // d = (2/sqrt(c)) * atanh(sqrt(delta / (2 + delta))) 와 유사
    frac.mapv_into(|val| {
        let cf = c * val;
        let arg = (cf / (1.0 + cf)).sqrt().min(1.0 - boundary_eps);
        (2.0 / sqrtc) * arg.atanh()
    })
}

/// 푸앵카레 공 -> 로렌츠(Lorentz) 모델 변환
///
/// x_L = ( (1+c|x|^2)/(1-c|x|^2) * 1/√c,  2x / (1-c|x|^2) * 1/√c )
pub fn poincare_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 - c * &x_norm_sq).mapv_into(|v| v.max(EPS));
    let sqrtc = c.sqrt();

    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let time_component = (1.0 + c * &x_norm_sq) / (&den * sqrtc);
    let space_components = (2.0 * x) / (&den * sqrtc);

    result.slice_mut(s![.., 0..1]).assign(&time_component);
    result.slice_mut(s![.., 1..]).assign(&space_components);
    result
}

/// 푸앵카레 공 -> 클라인(Klein) 모델 변환
///
/// x_K = 2x / (1 + c|x|^2)
pub fn poincare_to_klein(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = 1.0 + c * &x_norm_sq;
    let den_clamped = den.mapv_into(|v| v.max(EPS));
    (2.0 * x) / &den_clamped
}

/// 푸앵카레 공의 지수 맵 (Exponential Map)
///
/// 점 x에서의 접벡터 v를 푸앵카레 공 위의 점으로 매핑합니다.
/// Exp_x(v) = x ⊕_c (tanh( (λ_x * √c * |v|)/2 ) * v / (√c * |v|))
pub fn poincare_exp_at(x: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, _: f32) -> Array2<f32> {
    // 컨포멀 팩터 λ_x = 2 / (1 - c ||x||^2)
    let x2 = norm_sq_batched(x).insert_axis(Axis(1));
    let one_minus_cx2 = (1.0 - c * &x2).mapv(|z| z.max(EPS));
    let lambda_x = 2.0 / &one_minus_cx2;

    // |v| 계산 및 안전한 스케일링
    let vnorm = norm_sq_batched(v).mapv(f32::sqrt).insert_axis(Axis(1));
    let vnorm_safe = vnorm.mapv(|z| z.max(EPS));

    if c.abs() < EPS {
        // 유클리드 극한: Exp ≈ x + v
        return x + v;
    }

    let sqrtc = c.sqrt();
    // u = tanh( (λ_x * √c * |v|)/2 ) / (√c * |v|) * v
    let arg = (&lambda_x * sqrtc * &vnorm_safe) * 0.5;
    let coeff = arg.mapv(|a| a.tanh()) / (sqrtc * &vnorm_safe);
    let u = &coeff * v;

    // 뫼비우스 덧셈으로 이동
    mobius::mobius_add(x, &u.view(), c)
}

/// 푸앵카레 공의 로그 맵 (Logarithmic Map)
///
/// 두 점 x, y에 대해 x에서 y로 향하는 접벡터를 계산합니다.
/// Log_x(y) = (2 / (√c λ_x)) * atanh( √c |(-x) ⊕_c y| ) * ((-x) ⊕_c y) / |(-x) ⊕_c y|
pub fn poincare_log_at(
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    c: f32,
    boundary_eps: f32,
) -> Array2<f32> {
    // 컨포멀 팩터 λ_x
    let x2 = norm_sq_batched(x).insert_axis(Axis(1));
    let one_minus_cx2 = (1.0 - c * &x2).mapv(|z| z.max(EPS));
    let lambda_x = 2.0 / &one_minus_cx2;

    if c.abs() < EPS {
        // 유클리드 극한: Log ≈ y - x
        return y - x;
    }

    // z = (-x) ⊕_c y (x를 원점으로 이동시켰을 때의 y의 위치)
    let neg_x = -x;
    let z = mobius::mobius_add(&neg_x.view(), y, c);
    let znorm = norm_sq_batched(&z.view())
        .mapv(f32::sqrt)
        .insert_axis(Axis(1));

    // 수치적 안정성을 위해 norm 클리핑
    let znorm_clip = znorm.mapv(|r| r.min(1.0 - boundary_eps).max(EPS));

    let sqrtc = c.sqrt();
    let atanh_term = (&znorm_clip * sqrtc).mapv(|u| u.atanh());
    let scale = (2.0 / (sqrtc * &lambda_x)) * &atanh_term / &znorm_clip;
    &scale * &z
}

/// 푸앵카레 공 레이어 (순전파)
///
/// 두 입력 u, v를 뫼비우스 덧셈으로 결합합니다.
/// 파라미터 t에 의해 가중치가 조절됩니다: ((1-t) ⊗ u) ⊕ (t ⊗ v)
pub fn poincare_ball_layer(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    mobius::mobius_add(&u_prime.view(), &v_prime.view(), c)
}

/// 동적 곡률을 사용하는 푸앵카레 레이어
pub fn poincare_ball_layer_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (result, _) = mobius::mobius_add_dynamic(&u_prime.view(), &v_prime.view(), dynamic_c);
    (result, c)
}

/// 동적 곡률 푸앵카레 레이어의 역전파
pub fn poincare_ball_layer_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    let (grad_u_prime, grad_v_prime, grad_kappa) = mobius::mobius_add_dynamic_backward(
        grad_output,
        &u_prime.view(),
        &v_prime.view(),
        dynamic_c,
    );

    let grad_u = mobius::mobius_scalar_vjp(&grad_u_prime.view(), u, c, 1.0 - t);
    let grad_v = mobius::mobius_scalar_vjp(&grad_v_prime.view(), v, c, t);

    (grad_u, grad_v, grad_kappa)
}

/// 레이어별 동적 곡률을 사용하는 푸앵카레 레이어
pub fn poincare_ball_layer_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (result, _) = mobius::mobius_add_layerwise(
        &u_prime.view(),
        &v_prime.view(),
        layer_curvatures,
        layer_idx,
    );
    (result, c)
}

/// 레이어별 동적 곡률 푸앵카레 레이어의 역전파
pub fn poincare_ball_layer_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    let (grad_u_prime, grad_v_prime) =
        mobius::mobius_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);

    let grad_u = mobius::mobius_scalar_vjp(&grad_u_prime.view(), u, c, 1.0 - t);
    let grad_v = mobius::mobius_scalar_vjp(&grad_v_prime.view(), v, c, t);

    // 곡률 c에 대한 그라디언트 계산 (Chain Rule)
    let grad_c_from_add_tensor = mobius::mobius_add_grad_c(&u_prime.view(), &v_prime.view(), c);
    let grad_c_add = (grad_output * &grad_c_from_add_tensor).sum();

    let grad_c_from_scalar_u_tensor = mobius::mobius_scalar_grad_c(u, c, 1.0 - t);
    let grad_c_scalar_u = (&grad_u_prime * &grad_c_from_scalar_u_tensor).sum();

    let grad_c_from_scalar_v_tensor = mobius::mobius_scalar_grad_c(v, c, t);
    let grad_c_scalar_v = (&grad_v_prime * &grad_c_from_scalar_v_tensor).sum();

    let grad_c_total = grad_c_add + grad_c_scalar_u + grad_c_scalar_v;

    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c_total * dc_dkappa;

    (grad_u, grad_v, grad_kappa)
}

/// 리만 아담 (Riemannian Adam) 옵티마이저 스텝
///
/// 푸앵카레 공 위에서의 파라미터 업데이트를 수행합니다.
/// 1. 그라디언트를 리만 그라디언트로 변환 (스케일링)
/// 2. 모멘텀 업데이트
/// 3. 접공간에서의 업데이트 방향 계산
/// 4. 지수 맵(Exponential Map)을 통한 파라미터 갱신
pub fn poincare_riemannian_adam_step(
    x: &ArrayView2<f32>,
    grad: &ArrayView2<f32>,
    m: &mut Array2<f32>,
    v: &mut Array2<f32>,
    step: u64,
    c: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    max_norm_eps: f32,
) -> Array2<f32> {
    let mut g_r: Array2<f32>;

    // 리만 그라디언트 변환: grad_R = grad_E / (lambda_x)^2
    if c.abs() < EPS {
        g_r = grad.to_owned();
    } else {
        let norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
        let one_minus_cx2 = (1.0 - c * &norm_sq).mapv(|z| z.max(EPS));
        let lambda = 2.0 / &one_minus_cx2;
        let inv_lambda_sq = 1.0 / (&lambda * &lambda);

        g_r = grad.to_owned();
        for (mut row, factor) in g_r
            .axis_iter_mut(Axis(0))
            .zip(inv_lambda_sq.axis_iter(Axis(0)))
        {
            let f = factor[0];
            for val in row.iter_mut() {
                *val *= f;
            }
        }
    }

    let one_minus_b1 = 1.0 - beta1;
    let one_minus_b2 = 1.0 - beta2;

    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_r
    ndarray::Zip::from(&mut *m)
        .and(&g_r)
        .for_each(|m_elt, g_elt| {
            *m_elt = beta1 * *m_elt + one_minus_b1 * *g_elt;
        });

    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_r^2
    ndarray::Zip::from(&mut *v)
        .and(&g_r)
        .for_each(|v_elt, g_elt| {
            *v_elt = beta2 * *v_elt + one_minus_b2 * (*g_elt * *g_elt);
        });

    // Bias correction
    let t = step as f32;
    let bias_c1 = 1.0 - beta1.powf(t);
    let bias_c2 = 1.0 - beta2.powf(t);
    let m_hat = m.mapv(|val| val / bias_c1);
    let v_hat = v.mapv(|val| val / bias_c2);

    // 업데이트 벡터 u (접공간)
    let mut u = m_hat.clone();
    ndarray::Zip::from(&mut u)
        .and(&v_hat)
        .for_each(|u_elt, v_elt| {
            *u_elt = -*u_elt * lr / (v_elt.sqrt() + eps);
        });

    if c.abs() < EPS {
        &x.to_owned() + &u
    } else {
        // 사용자 정의 max_norm_eps 또는 기본 안전값 사용
        let safe_eps = if max_norm_eps > 0.0 {
            max_norm_eps
        } else {
            BOUNDARY_EPS
        };
        // Exponential Map으로 업데이트 적용
        let x_new = poincare_exp_at(x, &u.view(), c, safe_eps);

        // 투영(Project) 로직 인라인 구현 (공 밖으로 나가지 않도록)
        let mut out = x_new.to_owned();
        let mut norms = norm_sq_batched(&out.view())
            .mapv(f32::sqrt)
            .insert_axis(Axis(1));
        let radius = if c > 0.0 { 1.0 / c.sqrt() } else { 1.0 };
        let max_norm = radius - safe_eps;

        for (mut row, mut norm) in out.axis_iter_mut(Axis(0)).zip(norms.axis_iter_mut(Axis(0))) {
            let n = norm[0].max(EPS);
            if n > max_norm {
                let scale = max_norm / n;
                row *= scale;
                norm[0] = max_norm;
            }
        }
        out
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn poincare_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                boundary_eps: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn poincare_ball_layer_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn poincare_ball_layer_backward_cuda(
                grad_output: *const f32,
                u: *const f32,
                v: *const f32,
                grad_u: *mut f32,
                grad_v: *mut f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn poincare_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        boundary_eps: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_distance_cuda(out, u, v, c, boundary_eps, batch_size, dim);
        }
    }

    pub fn poincare_ball_layer_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_ball_layer_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn poincare_ball_layer_backward_cuda(
        grad_output: *const f32,
        u: *const f32,
        v: *const f32,
        grad_u: *mut f32,
        grad_v: *mut f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_ball_layer_backward_cuda(
                grad_output,
                u,
                v,
                grad_u,
                grad_v,
                c,
                t,
                batch_size,
                dim,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::lorentz;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_mobius_add_identity() {
        let c = 1.0;
        let x = arr2(&[[0.1, 0.2]]);
        let z = arr2(&[[0.0, 0.0]]);
        let result = mobius::mobius_add(&x.view(), &z.view(), c);
        assert_relative_eq!(result, x, epsilon = EPSILON);
    }

    #[test]
    fn test_poincare_to_lorentz_and_back() {
        let c = 1.0;
        let x_poincare = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        let x_lorentz = poincare_to_lorentz(&x_poincare.view(), c);
        let x_poincare_restored = lorentz::lorentz_to_poincare(&x_lorentz.view(), c);

        assert_relative_eq!(x_poincare, x_poincare_restored, epsilon = EPSILON);
    }

    #[test]
    fn test_poincare_ball_layer_interpolation() {
        let c = 1.0;
        let u = arr2(&[[0.5, 0.5]]);
        let v = arr2(&[[-0.5, -0.5]]);

        // t=0 이면 u와 같아야 함
        let result_t0 = poincare_ball_layer(&u.view(), &v.view(), c, 0.0);
        assert_relative_eq!(result_t0, u, epsilon = EPSILON);

        // t=1 이면 v와 같아야 함
        let result_t1 = poincare_ball_layer(&u.view(), &v.view(), c, 1.0);
        assert_relative_eq!(result_t1, v, epsilon = EPSILON);

        // t=0.5 대칭성
        let result_t05 = poincare_ball_layer(&u.view(), &v.view(), c, 0.5);
        let result_t05_sym = poincare_ball_layer(&v.view(), &u.view(), c, 0.5);
        assert_relative_eq!(result_t05, result_t05_sym, epsilon = 1e-5);
    }

    #[test]
    fn test_distance_is_zero_for_same_point() {
        let c = 1.0;
        let x = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        let dist = poincare_distance(&x.view(), &x.view(), c, 1e-5);

        for val in dist.iter() {
            // 수치적 클램프로 인해 0이 아닌 매우 작은 값이 나올 수 있음
            assert!((*val).abs() < 1e-3);
        }
    }

    #[test]
    fn test_poincare_to_klein_then_back_shape_and_finiteness() {
        let c = 0.7_f32;
        let x = arr2(&[[0.1, -0.2], [0.3, 0.1]]);
        let k = poincare_to_klein(&x.view(), c);
        assert_eq!(k.ncols(), x.ncols());
        assert_eq!(k.nrows(), x.nrows());
        assert!(k.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_riemannian_adam_matches_euclidean_when_c_zero() {
        let x = arr2(&[[0.5f32, -0.3f32]]);
        let grad = arr2(&[[0.5f32, -0.3f32]]);
        let mut m = Array2::<f32>::zeros((1, 2));
        let mut v = Array2::<f32>::zeros((1, 2));
        let step = 1;
        let c = 0.0;
        let lr = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let x_view = x.view();
        let grad_view = grad.view();
        let x_new = poincare_riemannian_adam_step(
            &x_view, &grad_view, &mut m, &mut v, step, c, lr, beta1, beta2, eps, 1e-5,
        );
        let mut m_e = Array2::<f32>::zeros((1, 2));
        let mut v_e = Array2::<f32>::zeros((1, 2));
        let g = grad.clone();
        m_e = m_e * beta1 + &g * (1.0 - beta1);
        v_e = v_e * beta2 + &g.mapv(|x| x * x) * (1.0 - beta2);
        let m_hat = &m_e / (1.0 - beta1);
        let v_hat = &v_e / (1.0 - beta2);
        let mut u = m_hat.clone();
        ndarray::Zip::from(&mut u)
            .and(&v_hat)
            .for_each(|u_elt, v_elt| {
                *u_elt = -*u_elt * lr / (v_elt.sqrt() + eps);
            });
        let x_expected = &x + &u;
        assert_relative_eq!(x_new, x_expected, epsilon = 1e-6);
    }

    #[test]
    fn test_riemannian_adam_poincare_stays_inside_ball() {
        let x = arr2(&[[0.5f32, 0.4f32]]);
        let grad = arr2(&[[0.5f32, 0.4f32]]);
        let mut m = Array2::<f32>::zeros((1, 2));
        let mut v = Array2::<f32>::zeros((1, 2));
        let step = 1;
        let c = 1.0;
        let lr = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let x_view = x.view();
        let grad_view = grad.view();
        let x_new = poincare_riemannian_adam_step(
            &x_view, &grad_view, &mut m, &mut v, step, c, lr, beta1, beta2, eps, 1e-5,
        );
        let norms = norm_sq_batched(&x_new.view());
        let n = norms[0].sqrt();
        assert!(n < 1.0 - 1e-3);
    }
}
```
---
## File: `reality_stone/src/layers/riemann.rs`

```rust
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

use crate::layers::poincare::{poincare_exp_at, poincare_log_at};
use crate::ops::project_to_ball;

fn zeros_like(x: &ArrayView2<f32>) -> Array2<f32> {
    Array2::<f32>::zeros((x.nrows(), x.ncols()))
}

/// Riemann low-rank forward on Poincaré ball (tangent at origin):
/// y = Exp_0( ((Log_0(Proj(x)) @ P) @ Sigma^T) @ Q^T + b_tan, c )
pub fn riemann_lowrank_forward(
    x: &ArrayView2<f32>,     // [B, in]
    p: &ArrayView2<f32>,     // [in, r]
    sigma: &ArrayView2<f32>, // [r, r]
    q: &ArrayView2<f32>,     // [out, r]
    b_tan: &ArrayView1<f32>, // [out]
    c: f32,
    epsilon: f32,
) -> Array2<f32> {
    // 1) Project x to ball
    let x_proj = project_to_ball(&x, epsilon);

    // 2) v = Log_0(x_proj)
    let zeros = zeros_like(&x.view());
    let v = poincare_log_at(&zeros.view(), &x_proj.view(), c, epsilon);

    // 3) low-rank linear in tangent
    // z1 = v @ P  [B, r]
    let z1 = v.dot(p);
    // z2 = z1 @ Sigma^T
    let z2 = z1.dot(&sigma.t());
    // y_tan = z2 @ Q^T + b_tan
    let mut y_tan = z2.dot(&q.t());
    // add tangent bias row-wise
    // add tangent bias row-wise safely
    let b = b_tan.to_owned();
    for mut row in y_tan.axis_iter_mut(Axis(0)) {
        row += &b.view();
    }

    // 4) y = Exp_0(y_tan)
    let zeros_out = Array2::<f32>::zeros((y_tan.nrows(), y_tan.ncols()));
    let y = poincare_exp_at(&zeros_out.view(), &y_tan.view(), c, epsilon);
    y
}
```
