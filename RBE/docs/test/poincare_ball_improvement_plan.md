# 푸앵카레 볼 비트레이어 개선 계획

## 1. 현재 문제점 분석

### 1.1 encode_vector의 근본적 오류
현재 구현은 푸앵카레 볼의 핵심을 완전히 무시하고 있습니다:

```rust
// ❌ 현재: 단순한 Fourier 분석
pub fn encode_vector(&mut self, vector_data: &[f32]) -> HybridEncodedBlock {
    // 평균, 분산, 주파수 성분만 계산
    // 푸앵카레 볼 구조 전혀 없음!
}
```

### 1.2 누락된 핵심 요소들
1. **128비트 Packed128 구조 미사용**
2. **CORDIC 쌍곡회전 미적용**
3. **11비트 미분 사이클 시스템 없음**
4. **푸앵카레 메트릭 무시**
5. **상태-전이 미분 미구현**

## 2. 푸앵카레 볼 비트레이어 핵심 개념

### 2.1 128비트 구조
```rust
pub struct Packed128 {
    pub hi: u64,  // 푸앵카레 상태 코어
    pub lo: u64,  // 연속 파라미터 코어
}
```

**hi 필드 (64비트)**:
- `[63:62]`: 푸앵카레 사분면 (sinh/cosh/tanh/sech²)
- `[61:50]`: 쌍곡주파수 (12비트)
- `[49:38]`: 측지선 진폭 (12비트)
- `[37:32]`: 기저함수 선택 (6비트)
- `[31:0]`: CORDIC 회전 시퀀스 (32비트)

**lo 필드 (64비트)**:
- `[63:32]`: r_poincare (푸앵카레 반지름)
- `[31:0]`: theta_poincare (푸앵카레 각도)

### 2.2 CORDIC 쌍곡회전
```rust
// 푸앵카레 볼 내부로 매핑하는 핵심 알고리즘
for k in 0..20 {
    let sigma = if (cordic_rotations >> k) & 1 == 1 { 1.0 } else { -1.0 };
    // 쌍곡회전
    x = x - sigma * y * 2^(-k);
    y = y + sigma * x * 2^(-k);
    
    // 푸앵카레 볼 경계 처리
    if k % 4 == 0 {
        let r = sqrt(x² + y²);
        if r > ε {
            x *= tanh(r) / r;  // 핵심!
            y *= tanh(r) / r;
        }
    }
}
```

### 2.3 11비트 미분 사이클
- sinh/cosh: 4사이클 (2비트)
- tanh/sech²: 2사이클 (1비트)  
- sin/cos: 4사이클 (2비트)
- exp: 1사이클 (1비트)
- 구분비트로 분리

## 3. 개선 구현 방안

### 3.1 올바른 encode_vector
```rust
pub fn encode_vector_poincare(&mut self, vector_data: &[f32]) -> HybridEncodedBlock {
    let size = vector_data.len();
    
    // 1. 데이터를 푸앵카레 볼로 매핑
    let (r_poincare, theta_poincare) = map_to_poincare_ball(vector_data);
    
    // 2. 최적 사분면 선택
    let quadrant = select_optimal_quadrant(vector_data);
    
    // 3. CORDIC 회전 시퀀스 학습
    let cordic_seq = learn_cordic_sequence(vector_data, r_poincare, theta_poincare);
    
    // 4. 쌍곡 주파수/진폭 분석
    let (hyperbolic_freq, geodesic_amp) = analyze_hyperbolic_components(vector_data);
    
    // 5. 기저함수 선택
    let basis_func = select_basis_function(vector_data);
    
    // 6. Packed128 생성
    let packed = PoincarePackedBit128::new(
        quadrant,
        hyperbolic_freq,
        geodesic_amp,
        basis_func,
        cordic_seq,
        r_poincare,
        theta_poincare,
    );
    
    // 7. HybridEncodedBlock으로 변환
    convert_to_hybrid_block(packed, size)
}
```

### 3.2 푸앵카레 볼 매핑
```rust
fn map_to_poincare_ball(data: &[f32]) -> (f32, f32) {
    // 1. 데이터 정규화
    let norm = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    // 2. 푸앵카레 반지름 (항상 < 1)
    let r = norm.tanh();  // 핵심: tanh로 [0,1) 매핑
    
    // 3. 주 방향 각도
    let theta = compute_principal_angle(data);
    
    (r, theta)
}
```

### 3.3 CORDIC 시퀀스 학습
```rust
fn learn_cordic_sequence(data: &[f32], r: f32, theta: f32) -> u32 {
    let mut cordic_bits = 0u32;
    let mut x = r * theta.cos();
    let mut y = r * theta.sin();
    
    // 20회 반복으로 최적 회전 찾기
    for k in 0..20 {
        // 목표와의 거리 계산
        let error_pos = compute_error(data, x, y, 1.0);
        let error_neg = compute_error(data, x, y, -1.0);
        
        // 더 나은 방향 선택
        let sigma = if error_pos < error_neg { 1 } else { 0 };
        cordic_bits |= sigma << k;
        
        // 회전 적용
        let sign = if sigma == 1 { 1.0 } else { -1.0 };
        apply_cordic_rotation(&mut x, &mut y, k, sign);
    }
    
    cordic_bits
}
```

### 3.4 상태-전이 미분
```rust
impl BitDifferentiable for PoincarePackedBit128 {
    fn compute_gradient(&self, loss: f32) -> u128 {
        let mut grad_bits = 0u128;
        
        // 11비트 미분 사이클 적용
        let cycle_state = extract_cycle_state(self.hi);
        let next_state = advance_cycle(cycle_state);
        
        // 비트별 그래디언트 계산
        for bit in 0..128 {
            let perturbed = self.flip_bit(bit);
            let delta_loss = compute_loss_delta(perturbed);
            
            if delta_loss * loss < 0.0 {  // 개선 방향
                grad_bits |= 1 << bit;
            }
        }
        
        grad_bits
    }
}
```

## 4. 기대 효과

### 4.1 정확도 개선
- 현재: RMSE > 0.3 (D 등급)
- 목표: RMSE < 0.01 (B 등급)
- 이유: 푸앵카레 볼의 무한 표현력 활용

### 4.2 압축률 유지
- 128비트로 임의 크기 벡터 표현
- CORDIC로 즉석 가중치 생성
- 메모리 대역폭 93.75% 절약

### 4.3 학습 가능성
- 11비트 미분 사이클로 해석적 미분
- 상태-전이로 비트 단위 학습
- 리만 기하학적 최적화

## 5. 구현 우선순위

1. **Phase 1**: Packed128 기반 encode_vector 재구현
2. **Phase 2**: CORDIC 시퀀스 학습 알고리즘
3. **Phase 3**: 11비트 미분 사이클 통합
4. **Phase 4**: 푸앵카레 메트릭 최적화
5. **Phase 5**: 비트 단위 역전파 구현

## 6. 검증 방법

```rust
#[test]
fn test_poincare_encoding_accuracy() {
    let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let encoded = encode_vector_poincare(&data);
    let decoded = decode_poincare(&encoded);
    
    let rmse = compute_rmse(&data, &decoded);
    assert!(rmse < 0.01);  // B 등급 이상
    
    // 푸앵카레 볼 내부 확인
    let r = extract_r_poincare(&encoded);
    assert!(r < 1.0);  // 항상 볼 내부
}
``` 